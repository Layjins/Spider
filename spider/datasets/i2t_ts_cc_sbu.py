import logging
import random
import webdataset as wds

from torch.utils.data.dataloader import default_collate
from spider.processors import *
from spider.common.registry import registry



class I2TTsCCSBUDataset:
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
            "Generate similar contents to this image",
            "Please generate a multimodal caption for this image",
            "Please generate the contents that are similar to this image",
            "Please generate an <IMAGE> that is similar to this image",
            "Please generate a <VIDEO> that is similar to this image",
            "Please generate an <AUDIO> that is similar to this image",
            "Please generate an <IMAGE> and a <VIDEO> that are similar to this image",
            "Please generate an <IMAGE> and an <AUDIO> that are similar to this image",
            "Please generate a <VIDEO> and an <AUDIO> that are similar to this image",
            "Please generate an <IMAGE>, a <VIDEO>, and an <AUDIO> that are similar to this image",
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

        question = "<IMAGE><IMAGE-Placeholder></IMAGE> {}".format(instruction_q)
        answer = "{}{}".format(sample[1], "".join(answer_components))

        return {
            "Question": question,
            "TaskPrompt": "[SMARTMULTIMODAL]",
            "Answer": answer,
            "IMAGE": sample[0],
            "Caption": sample[1]
        }


@registry.register_builder("i2t_ts_cc_sbu")
class I2TTsCCSBUBuilder:
    train_dataset_cls = I2TTsCCSBUDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building I2TTsCCSBU datasets...")

        build_info = self.config.build_info

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            webdataset_path=build_info.webdataset_path,
        ).inner_dataset

