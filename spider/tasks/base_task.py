"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist

from spider.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from spider.common.logger import MetricLogger
from spider.common.registry import registry
from spider.datasets.utils.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.cfg = ""

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        self.cfg = cfg
        model_config = cfg.model

        # import pdb
        # pdb.set_trace()

        model_cls = registry.get_model_class(model_config.type)
        model_config.pop('type')
        return model_cls(**model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """



        # import pdb
        # pdb.set_trace()

        dataset_configs = cfg.datasets

        assert len(dataset_configs.train) > 0, "At least one train dataset has to be specified."

        # for dataset_name, dataset_config in dataset_configs.items():
        #     # dataset_config = datasets_config[name]
        #
        #     builder = registry.get_builder_class(dataset_name)(dataset_config)
        #     dataset = builder.build_datasets()
        #
        #     # import pdb
        #     # pdb.set_trace()
        #
        #     dataset['train'].name = dataset_name
        #     dataset['train'].batch_size = dataset_config.batch_size
        #     dataset['train'].sample_ratio = dataset_config.sample_ratio
        #     datasets[dataset_name] = dataset
        datasets = dict()
        for split, dataset_configs in cfg.datasets.items():
            split_datasets = dict()
            for dataset_name, dataset_config in dataset_configs.items():
                # dataset_config = datasets_config[name]

                builder = registry.get_builder_class(dataset_name)(dataset_config)
                dataset = builder.build_datasets()

                # import pdb
                # pdb.set_trace()

                dataset.name = dataset_name
                dataset.batch_size = dataset_config.batch_size
                dataset.sample_ratio = dataset_config.get("sample_ratio", 50)
                split_datasets[dataset_name] = dataset

                # dataset[split].name = dataset_name
                # dataset[split].batch_size = dataset_config.batch_size
                # dataset[split].sample_ratio = dataset_config.get("sample_ratio", 50)
                # datasets[dataset_name] = dataset
            datasets[split] = split_datasets

        return datasets

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            logging.info("result file saved to %s" % final_result_file)

        return final_result_file
