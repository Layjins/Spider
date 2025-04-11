import os
import json
import random
import logging
import warnings

import cv2
from PIL import Image
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from spider.processors import *
from spider.common.registry import registry


class PororoDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, h5_path, subset):
        super(PororoDataset, self).__init__()
        self.h5_file = h5_path
        self.subset = subset

        self.instruction_pool = [
            "{}",
            "Create a text-and-image story based on the following text: {}",
            "Generate a story using both the text and image provided below: {}",
            "Help me create a visual and textual narrative from the following image: {}",
            "Based on the text provided, generate a story with accompanying images: {}",
            "Create a narrative combining the text and image provided below: {}",
            "Write a story using the following text and image as references: {}",
            "Help me turn this image and text into a compelling story: {}",
            "Combine the text and image to create a storytelling piece: {}",
            "Use the text and image provided to generate a complete story: {}",
            "Craft a story that integrates both the text and image shown below: {}",
        ]

    def open_h5(self):
        h5 = h5py.File(self.h5_file, "r")
        self.h5 = h5[self.subset]

    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        return len(self.h5['text'])

    def preprocess(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        # image process
        images = list() # (5, 3, 224, 224)
        for i in range(5):
            im = self.h5['image{}'.format(i)][index]
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            im = torch_transform(im)
            images.append(im)
        # images = torch.stack(images) # torch.Size([5, 3, 224, 224])

        # text process
        texts_aug = list()  # 去除所有标点符号的texts
        text_all = self.h5['text'][index].decode('utf-8') # 整段text为1句，不同帧的text用'|'分隔
        texts = text_all.split('|')  # 整段text分成5句
        for text_i in texts:
            text_i = text_processor(text_i) # 去除所有标点符号
            texts_aug.append(text_i)
        
        return images, texts, texts_aug, text_all

    def __getitem__(self, index):
        images, texts, texts_aug, text_all = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(texts[0])
        question = "<IMAGE><IMAGE-Placeholder></IMAGE> {} ".format(instruction)
        answer = "{}".format(text_all) # 整段text为1句，不同帧的text用'|'分隔

        # import pdb
        # pdb.set_trace()

        return {
            "Question": question,
            "TaskPrompt": "[STORY]",
            "Answer": answer,
            "IMAGE": images[0],
            "images": images,
            "texts": texts
        }


@registry.register_builder("pororo")
class PororoBuilder:
    train_dataset_cls = PororoDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building Pororo datasets...")

        build_info = self.config.build_info
        h5_path = build_info.h5_path
        subset = build_info.subset

        if not os.path.exists(h5_path):
            warnings.warn(f"h5 path {h5_path} does not exist.")
            return None

        if subset not in ['train', 'val', 'test']:
            warnings.warn(f"Invalid subset {subset}. Subset must be one of 'train', 'val', 'test'.")
            return None

        # Create datasets
        dataset_cls = self.train_dataset_cls
        return dataset_cls(h5_path=h5_path, subset=subset)
