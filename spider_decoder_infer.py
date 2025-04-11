from spider.common.registry import registry
import deepspeed
import torch
import os
import json
from deepspeed.module_inject.auto_tp import ReplaceWithTensorSlicing
from deepspeed.module_inject.load_checkpoint import load_model_with_checkpoint
from deepspeed.module_inject.replace_module import GroupQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import random
from collections import defaultdict
import cv2
import re
import ast
import numpy as np
from PIL import Image
import html
import math
import tempfile
import imageio
import scipy
from datetime import datetime
import torchvision.transforms as T
import torchvision.io as io
import torch.backends.cudnn as cudnn
from pytorchvideo import transforms as pv_transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from mmengine import Config
import copy
from StoryDiffusion.Comic_Generation import init_story_generation, story_generation



class SpiderDecoderInfer(object):
    def __init__(self, cfg):
        self.model_config = cfg.model
        model_cls = registry.get_model_class(self.model_config.type)
        self.model_config.pop('type')
        self.spider_decoder = model_cls(**self.model_config)
        self.spider_decoder = self.spider_decoder.eval()
        if self.model_config.story_generation != None:
            self.story_diffusion = init_story_generation(model_name=self.model_config.story_generation.model_name, device="cuda")
        else:
            self.story_diffusion = None

    def __call__(self, samples):
        ### inference outputs ###
        answers = []
        predictions = dict(
            IMAGE=[],
            VIDEO=[],
            AUDIO=[],
            MASK=[],
            BOX=dict(bboxes=[],label_names=[],scores=[]),
            IMAGESTORY=[],
        )
        predictions_text = dict(
            IMAGE=[],
            VIDEO=[],
            AUDIO=[],
            MASK=[],
            BOX=[],
            IMAGESTORY=[],
            IMAGESTORY_prompts=[],
        )
        ##########################
        # inference
        answers, predictions, predictions_text = self.spider_decoder.generate(samples, answers, predictions, predictions_text)
        # image story
        if len(predictions_text["IMAGESTORY"]) > 0:
            output_texts = predictions_text["IMAGESTORY"][0]
            print("story_texts:", output_texts)
            general_prompt, prompt_array, style_name = self.extract_story_elements(output_texts)
            print("General Prompt:", general_prompt)
            print("Prompt Array:", prompt_array)
            print("Style Name:", style_name)
            if (self.story_diffusion != None) and general_prompt and prompt_array and isinstance(prompt_array, list) and len(prompt_array) > 0 and style_name:
                preds = story_generation(self.story_diffusion, general_prompt=general_prompt, prompt_array=prompt_array, style_name=style_name)
                predictions["IMAGESTORY"].append(preds)
                predictions_text["IMAGESTORY_prompts"].append(prompt_array)
            else:
                print("Error: One or more required inputs for story_generation are empty!")
        return answers, predictions, predictions_text

    def clean_prompt_array(self, prompt_str):
        """ 解析 Prompt Array，兼容 Python 列表、JSON 数组、换行格式、HTML/XML 等 """
        if not prompt_str.strip():
            return []  # 直接返回空列表
        # **去除 HTML/XML 标签**
        prompt_str = re.sub(r"<.*?>", "", prompt_str).strip()
        # **尝试用 `ast.literal_eval` 解析（适用于 Python 列表）**
        try:
            parsed_array = ast.literal_eval(prompt_str)
            if isinstance(parsed_array, list):
                return [str(item).strip() for item in parsed_array if item]  # 确保元素是字符串
        except (SyntaxError, ValueError):
            pass  # 解析失败，尝试其他方法
        # **尝试用 `json.loads` 解析（适用于 JSON 数组）**
        try:
            parsed_array = json.loads(prompt_str)
            if isinstance(parsed_array, list):
                return [str(item).strip() for item in parsed_array if item]
        except json.JSONDecodeError:
            pass  # 解析失败，继续其他方式
        # **手动解析：去除 `[]` 并按 `', '` 或换行拆分**
        prompt_str = re.sub(r"^\[|\]$", "", prompt_str.strip())  # 去除 `[` 和 `]`
        prompts = re.split(r"'\s*,\s*'|\"\s*,\s*\"|\n", prompt_str)  # 按 `', '` `", "` 或换行分割
        # **移除额外的引号和空白**
        cleaned_prompts = [p.strip(" '\"") for p in prompts if p.strip()]
        return cleaned_prompts
    
    def extract_story_elements(self, output_texts):
        """ 提取 General Prompt、Prompt Array、Style Name（取最后一组），如果出现 </think>，则只处理其后的内容 """
        # 查找 </think> 并截取其后的内容
        think_split = output_texts.split("</think>", 1)
        if len(think_split) > 1:
            output_texts = think_split[1]  # 取 </think> 之后的部分
        # 提取所有 General Prompt（忽略单双引号）
        general_prompt_matches = re.findall(r"<GENERALPROMPT>\s*(.*?)\s*</GENERALPROMPT>", output_texts, re.DOTALL)
        general_prompt = general_prompt_matches[-1].strip() if general_prompt_matches else ""
        # 提取所有 Prompt Array（适配各种格式）
        prompt_array_matches = re.findall(r"<PROMPTARRAY>\s*(.*?)\s*</PROMPTARRAY>", output_texts, re.DOTALL)
        prompt_array_str = prompt_array_matches[-1].strip() if prompt_array_matches else "[]"
        prompt_array = self.clean_prompt_array(prompt_array_str)
        # 提取所有 Style Name
        style_name_matches = re.findall(r"<STYLENAME>\s*(.*?)\s*</STYLENAME>", output_texts, re.DOTALL)
        style_name = style_name_matches[-1].strip() if style_name_matches else ""
        return general_prompt, prompt_array, style_name



if __name__ == "__main__":
    cfg_root = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/train_configs"
    cfg_path = os.path.join(cfg_root, "spider_decoder_cfg.py")
    cfg = Config.fromfile(cfg_path)
    spider_decoder_infer = SpiderDecoderInfer(cfg)
    ask_info = {}
    ask_info['llm_text_all'] = ["<IMAGE>apple</IMAGE><VIDEO>dog</VIDEO><AUDIO>cat</AUDIO>"] # output text of llm
    answers, predictions, predictions_text = spider_decoder_infer(ask_info)
    # answers = ['<IMAGE>apple</IMAGE><VIDEO>dog</VIDEO><AUDIO>cat</AUDIO>']
    # predictions_text = {'IMAGE': ['apple'], 'VIDEO': ['dog'], 'AUDIO': ['cat'], 'MASK': [], 'BOX': [], 'IMAGESTORY': [], 'IMAGESTORY_prompts': []}