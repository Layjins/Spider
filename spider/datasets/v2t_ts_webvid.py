import logging
import math
import random
import webdataset as wds

import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from pytorchvideo import transforms as pv_transforms
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from spider.common.registry import registry


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T_I_V_A.txt, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T_I_V_A.txt, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T_I_V_A.txt,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


class ProcessVideos:
    def __init__(self, clips_per_video=5, clip_frames=2):
        self.clip_len = clips_per_video * clip_frames
        self.clips_per_video = clips_per_video
        self.clip_frames = clip_frames

        self.video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(224),
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, data):
        # video
        video = data["mp4"][0]
        video_length = video.shape[0]
        skip_frames = video_length // self.clip_len
        video_clips = video[::skip_frames][:self.clip_len]  # 10, 3, H, W
        video_clips = video_clips.permute(0, 3, 1, 2) / 255.
        video_clips = video_clips.view(self.clip_frames, self.clips_per_video, *video_clips.shape[1:])   # (2, 5, 3, H, W)
        video_clips = video_clips.permute(1, 2, 0, 3, 4)
        video_clips = [self.video_transform(clip) for clip in video_clips]
        video_clips = SpatialCrop(224, num_crops=3)(video_clips)
        video_clips = torch.stack(video_clips, dim=0)
        data['video'] = video_clips
        # text
        data['text'] = data["json"]["text"]
        return data



class V2TTsWebVid:
    def __init__(self, webdataset_path):
        self.inner_dataset = wds.WebDataset(webdataset_path, resampled=True, handler=wds.warn_and_continue)\
            .decode(wds.torch_video, handler=wds.warn_and_continue)\
            .map(ProcessVideos(), handler=wds.warn_and_continue) \
            .to_tuple("video", "text", handler=wds.warn_and_continue)\
            .shuffle(100, handler=wds.warn_and_continue) \
            .map(self.to_dict, handler=wds.warn_and_continue)

        self.instruction_pool = [
            "What is this video",
            "Please generate a caption for this video",
            "Could you provide a descriptive caption for this video",
            "I need your help in generating a caption for this video",
            "Could you describe this video for me",
            "I would like you to generate a caption for this video",
            "Please provide a descriptive caption for this video",
            "Generate similar contents to this video",
            "Please generate a multimodal caption for this video",
            "Please generate the contents that are similar to this video",
            "Please generate an <IMAGE> that is similar to this video",
            "Please generate a <VIDEO> that is similar to this video",
            "Please generate an <AUDIO> that is similar to this video",
            "Please generate an <IMAGE> and a <VIDEO> that are similar to this video",
            "Please generate an <IMAGE> and an <AUDIO> that are similar to this video",
            "Please generate a <VIDEO> and an <AUDIO> that are similar to this video",
            "Please generate an <IMAGE>, a <VIDEO>, and an <AUDIO> that are similar to this video",
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

        question = "<VIDEO><VIDEO-Placeholder></VIDEO> {}".format(instruction_q)
        answer = "{}{}".format(sample[1], "".join(answer_components))

        return {
            "Question": question,
            "TaskPrompt": "[SMARTMULTIMODAL]",
            "Answer": answer,
            "VIDEO": sample[0],
            "Caption": sample[1]
        }


@registry.register_builder("v2t_ts_webvid")
class V2TTsWebVidBuilder:
    train_dataset_cls = V2TTsWebVid

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building V2TTsWebVid datasets...")

        build_info = self.config.build_info

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            webdataset_path=build_info.webdataset_path,
        ).inner_dataset
