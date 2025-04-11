import logging
import random
import webdataset as wds

from torch.utils.data.dataloader import default_collate
from spider.processors import *
from spider.common.registry import registry



class T2ITsCCSBUDataset:
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
            # "an image of {}, a video of {}, an audio of {}",
            # "Generate an image of {}, a video of {}, an audio of {}",
            # "Please generate an image based on the following text: {}, a video based on the following text: {}, an audio based on the following text: {}",
            # "Could you create an image from this text: {}, a video from this text: {}, an audio from this text: {}",
            # "I would like you to generate an image based on this text: {}, a video based on this text: {}, an audio based on this text: {}",
            # "Please create an image from the following text: {}, a video from the following text: {}, an audio from the following text: {}",
            # "Could you generate an image based on this text: {}, a video based on this text: {}, an audio based on this text: {}",
            # "I need your help in creating an image from this text: {}, a video from this text: {}, an audio from this text: {}",
            # "Please create an image from the following text: {}, a video from the following text: {}, an audio from the following text: {}",
            # "I would like you to create an image from this text: {}, a video from this text: {}, an audio from this text: {}",
            "Generate {}",
            "Please generate {}",
            "Could you create {}",
            "I would like you to generate {}",
            "Please create {}",
            "Could you generate {}",
            "I need your help in creating {}",
            "Please create {}",
            "I would like you to create {}",
            "Please generate the content based on the following text: {}",
            "Could you create the content from this text: {}",
            "I would like you to generate the content based on this text: {}",
            "Please create the content from the following text: {}",
            "Could you generate the content based on this text: {}",
            "I need your help in creating the content from this text: {}",
            "Please create the content from the following text: {}",
            "I would like you to create the content from this text: {}",
            "Please generate an <IMAGE> based on the following text: {}",
            "Please generate a <VIDEO> based on the following text: {}",
            "Please generate an <AUDIO> based on the following text: {}",
            "Please generate an <IMAGE> and a <VIDEO> based on the following text: {}",
            "Please generate an <IMAGE> and an <AUDIO> based on the following text: {}",
            "Please generate a <VIDEO> and an <AUDIO> based on the following text: {}",
            "Please generate an <IMAGE>, a <VIDEO>, and an <AUDIO> based on the following text: {}",
        ]

    def collater(self, samples):
        return default_collate(samples)

    def to_dict(self, sample):
        instruction = random.choice(self.instruction_pool)
        instruction_q = instruction

        # Determine which modalities to include based on the instruction
        modalities = {
            "<IMAGE>": "<IMAGE>{}<IMAGE-Placeholder></IMAGE>",
            "<VIDEO>": "<VIDEO>{}<VIDEO-Placeholder></VIDEO>",
            "<AUDIO>": "<AUDIO>{}<AUDIO-Placeholder></AUDIO>",
        }

        # Build the question and answer based on the selected instruction
        answer_components = []
        if "<IMAGE>" in instruction:
            instruction_q = instruction_q.replace("<IMAGE>", "image")
            answer_components.append(modalities["<IMAGE>"].format(sample[1]))
        if "<VIDEO>" in instruction:
            instruction_q = instruction_q.replace("<VIDEO>", "video")
            answer_components.append(modalities["<VIDEO>"].format(sample[1]))
        if "<AUDIO>" in instruction:
            instruction_q = instruction_q.replace("<AUDIO>", "audio")
            answer_components.append(modalities["<AUDIO>"].format(sample[1]))
        if ("<IMAGE>" not in instruction) and ("<VIDEO>" not in instruction) and ("<AUDIO>" not in instruction):
            answer_components.append(modalities["<IMAGE>"].format(sample[1]))
            answer_components.append(modalities["<VIDEO>"].format(sample[1]))
            answer_components.append(modalities["<AUDIO>"].format(sample[1]))

        question = instruction_q.format(sample[1])
        answer = "{}{}".format(sample[1], "".join(answer_components))

        return {
            "Question": question,
            "TaskPrompt": "[SMARTMULTIMODAL]",
            "Answer": answer,
            "IMAGE": sample[0],
            "Caption": sample[1]
        }


@registry.register_builder("t2i_ts_cc_sbu")
class T2ITsCCSBUBuilder:
    train_dataset_cls = T2ITsCCSBUDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building T2ITsCCSBU datasets...")

        build_info = self.config.build_info

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            webdataset_path=build_info.webdataset_path,
        ).inner_dataset