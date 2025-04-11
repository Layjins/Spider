import logging
import random
import webdataset as wds

from torch.utils.data.dataloader import default_collate

from spider.processors import *
from spider.common.registry import registry



class T2ICCSBUDataset:
    def __init__(self, webdataset_path):
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(webdataset_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "txt", handler=wds.warn_and_continue),
            wds.map_tuple(torch_transform, text_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

        self.instruction_pool = [
            "{}",
            "Generate {}",
            "Please generate an image based on the following text: {}",
            "Could you create an image from this text: {}",
            "I would like you to generate an image based on this text: {}",
            "Please create an image from the following text: {}",
            "Could you generate an image based on this text: {}",
            "I need your help in creating an image from this text: {}",
            "Please create an image from the following text: {}",
            "I would like you to create an image from this text: {}",
        ]

    def collater(self, samples):
        return default_collate(samples)

    def to_dict(self, sample):
        instruction = random.choice(self.instruction_pool)

        question = instruction.format(sample[1])
        # answer = "<IMAGE><IMAGE-Placeholder></IMAGE>"
        # answer = "<IMAGE><IMAGE-Placeholder></IMAGE> {} ".format(sample[1])
        answer = "<IMAGE>{}<IMAGE-Placeholder></IMAGE>".format(sample[1])

        return {
            "Question": question,
            "TaskPrompt": "[IMAGE]",
            "Answer": answer,
            "IMAGE": sample[0],
            "Caption": sample[1]
        }


@registry.register_builder("t2i_cc_sbu")
class T2ICCSBUBuilder:
    train_dataset_cls = T2ICCSBUDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building T2ICCSBU datasets...")

        build_info = self.config.build_info

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            webdataset_path=build_info.webdataset_path,
        ).inner_dataset