import logging
import random
import webdataset as wds

from torch.utils.data.dataloader import default_collate
from spider.processors import *
from spider.common.registry import registry



class I2TCCSBUDataset:
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
            "What is this image",
            "Please generate a caption for this image",
            "Could you provide a descriptive caption for this image",
            "I need your help in generating a caption for this image",
            "Could you describe this image for me",
            "I would like you to generate a caption for this image",
            "Please provide a descriptive caption for this image",
        ]

    def collater(self, samples):
        return default_collate(samples)

    def to_dict(self, sample):
        instruction = random.choice(self.instruction_pool)

        question = "<IMAGE><IMAGE-Placeholder></IMAGE> {} ".format(instruction)

        return {
            "Question": question,
            "TaskPrompt": "[TEXT]", # "TaskPrompt": "[CAPTION]",
            "Answer": sample[1],
            "IMAGE": sample[0],
            "Caption": sample[1]
        }


@registry.register_builder("i2t_cc_sbu")
class I2TCCSBUBuilder:
    train_dataset_cls = I2TCCSBUDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building I2TCCSBU datasets...")

        build_info = self.config.build_info

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            webdataset_path=build_info.webdataset_path,
        ).inner_dataset

