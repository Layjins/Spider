"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
import webdataset as wds
# from memory_profiler import profile

import torch
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from spider.common.dist_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from spider.common.registry import registry
from spider.common.logger import MetricLogger, SmoothedValue
from spider.datasets import *
from spider.datasets.utils.data_utils import prepare_sample
from collections import OrderedDict
from transformers.deepspeed import is_deepspeed_zero3_enabled





@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self.torch_model = model

        self._wrapped_model = None
        self._optimizer = None
        self._scaler = None
        # self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.iters_per_epoch = self.config.run.get("iters_per_epoch", None)
        self.max_epoch = int(self.config.run.get("max_epoch", None))
        self.device = torch.device(self.config.run.get("device", "cuda"))
        self.cuda_enabled = self.device.type == "cuda"
        self.use_distributed = self.config.run.get("distributed", False)

        self.log_freq = self.config.run.get("log_freq", 50)
        self.init_lr = float(self.config.run.get("init_lr", 0.00005))
        self.min_lr = float(self.config.run.get("min_lr", 0.0))
        self.accum_grad_iters = int(self.config.run.get("accum_grad_iters", 1))

        # self.train_splits = self.config.run.get("train_splits", [])
        # if len(self.train_splits) == 0:
        #     logging.info("Empty train splits.")

        # self.valid_splits = self.config.run.get("valid_splits", [])
        # if len(self.valid_splits) == 0:
        #     logging.info("No validation splits found.")

        # self.test_splits = self.config.run.get("test_splits", [])

        self.only_evaluate = self.config.run.get("evaluate_only", False)

        self.use_dist_eval_sampler = self.config.run.get("use_dist_eval_sampler", True)

        self.pretrained_ckpt_path = self.config.pretrained_ckpt_path
        self.resume_ckpt_path = self.config.run.get("resume_ckpt_path", None)

        self.is_evaluate = True if len(self.config.datasets.val) > 0 else False


        self.result_dir = registry.mapping['paths']['result_dir']
        self.output_dir = registry.mapping['paths']['output_dir']

        if is_main_process():
            self.writer = SummaryWriter(self.output_dir)

        ds_params = json.load(open('train_configs/ds_config.json'))
        self.deepspeed_model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            # args=types.SimpleNamespace(**args)
        )

        self.dataloaders = self._dataloaders


    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        # load from checkpoint if specified
        if self.pretrained_ckpt_path is not None:
            self._load_checkpoint_weight(self.pretrained_ckpt_path)

        # resume from checkpoint if specified
        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)


        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # import pdb
            # pdb.set_trace()
            # training phase
            logging.info("Start training")
            self.train_epoch(cur_epoch)
            # self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if self.is_evaluate:
                val_loaders = self.dataloaders["val"]
                for name, val_loader in val_loaders.items():
                    logging.info("Evaluating on {}.".format(name))

                    val_log = self.eval_epoch(
                        name=name, val_loader=val_loader, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric:
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            # self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.only_evaluate:
                    if cur_epoch % 1 == 0:
                        self._save_checkpoint(cur_epoch, is_best=False)

            if self.only_evaluate:
                break

            if self.config.run.distributed:
                dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.only_evaluate)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))


    def train_epoch(self, epoch):
        self.deepspeed_model.module.train()

        train_loader = self.dataloaders["train"]
        if not hasattr(train_loader, "__next__"):
            # convert to iterator if not already
            train_loader = iter(train_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("gen_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("mse_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("bce_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("dice_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("l1_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("giou_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("gen_acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        for i in metric_logger.log_every(range(self.iters_per_epoch), self.log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= self.iters_per_epoch:
                break

            samples = next(train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
            samples.update(
                {
                    "epoch": epoch,
                    "iters": i,
                    "num_iters_per_epoch": self.iters_per_epoch,
                }
            )

            with torch.cuda.amp.autocast(enabled=True):
                res = self.deepspeed_model(samples)
                loss = res['loss']

            # for name, param in self.deepspeed_model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.requires_grad)  # 至少应该有一个 True
            if loss.requires_grad:
                self.deepspeed_model.backward(loss)
                # self.deepspeed_model.backward(loss, retain_graph=True)

            # update gradients every accum_grad_iters iterations
            if (i + 1) % self.accum_grad_iters == 0:
                self.deepspeed_model.step()

            # for key, value in res.items():
            #     metric_logger.update(loss=loss.item())
            # import pdb
            # pdb.set_trace()
            metric_logger.update(**res)
            # metric_logger.update(gen_acc=gen_acc)

            if is_main_process():
                for key, value in res.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.writer.add_scalar(key, value, epoch * self.iters_per_epoch + i)

            # self.writer.add_scalar('loss', loss.item(), epoch*self.iters_per_epoch+i)
            # self.writer.add_scalar('gen_acc', gen_acc, epoch * self.iters_per_epoch + i)

            #####################################################################
            # import pdb
            # pdb.set_trace()
            # freeze pretrained embed_tokens and lm_head
            if self.torch_model.freeze_tokens:
                if is_deepspeed_zero3_enabled():
                    import deepspeed
                    if self.torch_model.using_lora:
                        params = [self.torch_model.old_embed_tokens, self.torch_model.llama_model.base_model.model.model.embed_tokens.weight, self.torch_model.old_lm_head, self.torch_model.llama_model.base_model.model.lm_head.weight]
                        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                            # freeze pretrained embed_tokens and lm_head
                            new_embed_device = self.torch_model.llama_model.base_model.model.model.embed_tokens.weight.device
                            old_embed_tokens_data = self.torch_model.old_embed_tokens.data[:self.torch_model.num_old_embed_tokens, :]
                            old_embed_tokens_data = old_embed_tokens_data.to(new_embed_device)
                            self.torch_model.llama_model.base_model.model.model.embed_tokens.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_embed_tokens_data

                            # freeze pretrained lm_head
                            new_lm_head_device = self.torch_model.llama_model.base_model.model.lm_head.weight.device
                            old_lm_head_data = self.torch_model.old_lm_head.data[:self.torch_model.num_old_embed_tokens, :]
                            old_lm_head_data = old_lm_head_data.to(new_lm_head_device)
                            self.torch_model.llama_model.base_model.model.lm_head.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_lm_head_data
                    else:
                        params = [self.torch_model.old_embed_tokens, self.torch_model.llama_model.model.embed_tokens.weight, self.torch_model.old_lm_head, self.torch_model.llama_model.lm_head.weight]
                        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                            # freeze pretrained embed_tokens and lm_head
                            new_embed_device = self.torch_model.llama_model.model.embed_tokens.weight.device
                            old_embed_tokens_data = self.torch_model.old_embed_tokens.data[:self.torch_model.num_old_embed_tokens, :]
                            old_embed_tokens_data = old_embed_tokens_data.to(new_embed_device)
                            self.torch_model.llama_model.model.embed_tokens.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_embed_tokens_data

                            # freeze pretrained lm_head
                            new_lm_head_device = self.torch_model.llama_model.lm_head.weight.device
                            old_lm_head_data = self.torch_model.old_lm_head.data[:self.torch_model.num_old_embed_tokens, :]
                            old_lm_head_data = old_lm_head_data.to(new_lm_head_device)
                            self.torch_model.llama_model.lm_head.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_lm_head_data
                else:
                    if self.torch_model.using_lora:
                        # freeze pretrained embed_tokens and lm_head
                        new_embed_device = self.torch_model.llama_model.base_model.model.model.embed_tokens.weight.device
                        old_embed_tokens_data = self.torch_model.old_embed_tokens.data[:self.torch_model.num_old_embed_tokens, :]
                        old_embed_tokens_data = old_embed_tokens_data.to(new_embed_device)
                        self.torch_model.llama_model.base_model.model.model.embed_tokens.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_embed_tokens_data

                        # freeze pretrained lm_head
                        new_lm_head_device = self.torch_model.llama_model.base_model.model.lm_head.weight.device
                        old_lm_head_data = self.torch_model.old_lm_head.data[:self.torch_model.num_old_embed_tokens, :]
                        old_lm_head_data = old_lm_head_data.to(new_lm_head_device)
                        self.torch_model.llama_model.base_model.model.lm_head.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_lm_head_data
                    else:
                        # freeze pretrained embed_tokens and lm_head
                        new_embed_device = self.torch_model.llama_model.model.embed_tokens.weight.device
                        old_embed_tokens_data = self.torch_model.old_embed_tokens.data[:self.torch_model.num_old_embed_tokens, :]
                        old_embed_tokens_data = old_embed_tokens_data.to(new_embed_device)
                        self.torch_model.llama_model.model.embed_tokens.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_embed_tokens_data

                        # freeze pretrained lm_head
                        new_lm_head_device = self.torch_model.llama_model.lm_head.weight.device
                        old_lm_head_data = self.torch_model.old_lm_head.data[:self.torch_model.num_old_embed_tokens, :]
                        old_lm_head_data = old_lm_head_data.to(new_lm_head_device)
                        self.torch_model.llama_model.lm_head.weight.data[:self.torch_model.num_old_embed_tokens, :] = old_lm_head_data
            # import pdb
            # pdb.set_trace()
            #####################################################################

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))


    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    @torch.no_grad()
    def eval_epoch(self, name, val_loader, cur_epoch, skip_reload=False):
        # import pdb
        # pdb.set_trace()
        self.deepspeed_model.module.eval()

        metric_logger = MetricLogger(delimiter="  ")
        # metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        # metric_logger.add_meter("gen_acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        header = f"Evaluate: data epoch: [{cur_epoch}] at dataset {name}"

        for i in metric_logger.log_every(val_loader, self.log_freq, header):
            samples = next(val_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)
            answers = self.deepspeed_model.generate(samples)



        # for images, questions, img_ids in tqdm(eval_dataloader):
        #     texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #     answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        #     for answer, img_id, question in zip(answers, img_ids, questions):
        #         answer = answer.replace("<unk>","").replace(" ","").strip()
        #         pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
        #         if re.match(pattern, answer):
        #             minigpt4_predict[img_id].append(answer)
        #         else:
        #             resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] give me the location of','').strip()]})
        # if args.resample:



        # # data_loader = self.dataloaders.get(split_name, None)
        # # assert data_loader, "data_loader for split {} is None.".format(split_name)
        #
        # # TODO In validation, you need to compute loss as well as metrics
        # # TODO consider moving to model.before_evaluation()
        #
        # model = self.unwrap_dist_model(self.deepspeed_model)
        # if not skip_reload and cur_epoch == "best":
        #     model = self._reload_best_model(model)
        # model.eval()
        #
        # self.task.before_evaluation(
        #     model=model,
        #     dataset=self.datasets[split_name],
        # )
        # results = self.task.evaluation(model, data_loader)
        #
        # if results is not None:
        #     return self.task.after_evaluation(
        #         val_result=results,
        #         split_name=split_name,
        #         epoch=cur_epoch,
        #     )

    @property
    def _dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """

        # concatenate map-style datasets and chain wds.DataPipe datasets separately
        # training set becomes a tuple (ConcatDataset, ChainDataset), both are
        # optional but at least one of them is required. The resultant ConcatDataset
        # and ChainDataset will be sampled evenly.
        dataloaders = dict()
        for split, split_datasets in self.datasets.items():
            is_train = split == 'train'
            split_loaders = dict()
            for name, dataset in split_datasets.items():
                # statistic
                if hasattr(dataset, "__len__"):
                    # a single map-style dataset
                    num_records = len(dataset)
                    logging.info(
                        f"Loaded {num_records} records for {split} split from the dataset {name}."
                    )
                else:
                    # a single wds.DataPipeline
                    num_records = -1
                    logging.info(
                        f"Dataset {name} for {split} split is a wds.DataPipeline dataset, no __len__ attribute."
                    )

                collate_fn = getattr(dataset, "collater", None)

                loader = self._create_loader(dataset, self.config.run.num_workers, dataset.batch_size, is_train, collate_fn)
                split_loaders[name] = loader

            if is_train:
                loaders = []
                dataset_ratios = []
                for name, loader in split_loaders.items():
                    loaders.append(loader)
                    dataset_ratios.append(split_datasets[name].sample_ratio)
                dataloaders[split] = MultiIterLoader(loaders=loaders, ratios=dataset_ratios)
            else:
                dataloaders[split] = split_loaders
        return dataloaders

    def _create_loader(self, dataset, num_workers, bsz, is_train, collate_fn):
        # create a single dataloader for each split
        if isinstance(dataset, ChainDataset) or isinstance(
            dataset, wds.DataPipeline
        ):
            # wds.WebdDataset instance are chained together
            # webdataset.DataPipeline has its own sampler and collate_fn
            loader = iter(
                DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )
        else:
            # map-style dataset are concatenated together
            # setup distributed sampler

            if self.use_distributed:
                sampler = DistributedSampler(
                    dataset,
                    shuffle=is_train,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                )
                sampler = sampler if is_train else None
            else:
                sampler = None

            loader = DataLoader(
                dataset,
                batch_size=bsz,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=sampler is None and is_train,
                collate_fn=collate_fn,
                drop_last=True if is_train else False,
            )
            loader = PrefetchLoader(loader)

            # if is_train:
            loader = IterLoader(loader, use_distributed=self.use_distributed)

        return loader


    # @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        logging.info("Saving checkpoint at epoch {}.".format(cur_epoch))
        client_state = {"epoch": cur_epoch}
        self.deepspeed_model.save_checkpoint(self.output_dir, cur_epoch, client_state)
        # logging.info("self.deepspeed_model has been saved!")
        # save tokenizer
        self.torch_model.llama_tokenizer.save_pretrained(self.output_dir)
        # logging.info("llama_tokenizer has been saved!")
        # save configuration
        self.torch_model.llama_model.config.save_pretrained(self.output_dir)
        logging.info(f'[!] save model into {self.output_dir}')


    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, ckpt_dir):
        """
        Resume from a checkpoint.
        """
        load_dir = os.path.dirname(ckpt_dir)
        ckpt_id = int(os.path.basename(ckpt_dir))
        _, client_state = self.deepspeed_model.load_checkpoint(load_dir, ckpt_id)

        # state_dict = checkpoint["model"]
        # message = self.unwrap_dist_model(self.deepspeed_model).load_state_dict(state_dict, strict=False)
        #
        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        # if self.scaler and "scaler" in checkpoint:
        #     self.scaler.load_state_dict(checkpoint["scaler"])

        # self.start_epoch = client_state["epoch"] + 1
        self.start_epoch = ckpt_id + 1
        logging.info("Resumed checkpoint from {}".format(ckpt_dir))

    def _load_checkpoint_weight(self, ckpt_dir):
        """
        load from a checkpoint.
        """
        load_dir = os.path.dirname(ckpt_dir)
        ckpt_id = int(os.path.basename(ckpt_dir))
        # _, client_state = self.deepspeed_model.load_checkpoint(load_dir, ckpt_id, load_module_strict=True, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        _, client_state = self.deepspeed_model.load_checkpoint(load_dir, ckpt_id, load_module_strict=False, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        logging.info("Load pretrained checkpoint from {}".format(ckpt_dir))

