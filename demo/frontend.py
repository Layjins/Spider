import argparse
import os
import random
from collections import defaultdict
import cv2
import re
import json
import ast
import numpy as np
from PIL import Image
import torch
import html
import math
import gradio as gr
import tempfile
import imageio
import scipy
from datetime import datetime
import torchvision.transforms as T
import torchvision.io as io
import torch.backends.cudnn as cudnn
from pytorchvideo import transforms as pv_transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from spider.common.config import parse_args
from spider.processors import *
from demo.inference_api import SpiderInference
from mmengine import Config
import copy


args = parse_args()
cfg = Config.fromfile(args.config)
infer = SpiderInference(cfg)



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


class SpatialCrop(torch.nn.Module):
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
            flipped_video = T.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res

class ProcessVideos:
    def __init__(self, clips_per_video=5, clip_frames=2):
        """
        初始化视频处理参数
        clips_per_video: 每个视频分成几个片段。如果clips_per_video增加，模型将学习更多的小片段，而不是长时间序列。
        clip_frames: 每个片段包含多少帧。如果 clip_frames 增加，每个 clip 包含更多的时间信息，更有利于长时间序列建模。
        """
        self.clip_len = clips_per_video * clip_frames # 每个视频的总帧数
        self.clips_per_video = clips_per_video # 每个视频分割成几个片段
        self.clip_frames = clip_frames # 每个片段包含多少帧
        # 视频预处理（尺寸调整和归一化）
        self.video_transform = T.Compose(
            [
                pv_transforms.ShortSideScale(224), # 缩放视频的短边至224像素
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, video_path):
        # import pdb
        # pdb.set_trace()
        video, _, _ = io.read_video(video_path, pts_unit="sec")  # 读取视频
        video_length = video.shape[0]
        # 计算跳帧间隔, 以保证最终采样 self.clip_len 帧
        skip_frames = video_length // self.clip_len
        # 采样帧，每隔 `skip_frames` 取一帧，总共 self.clip_len 帧
        video_clips = video[::skip_frames][:self.clip_len]  # 10, 3, H, W
        video_clips = video_clips.permute(0, 3, 1, 2) / 255.
        video_clips = video_clips.view(self.clip_frames, self.clips_per_video, *video_clips.shape[1:])   # (2, 5, 3, H, W)
        video_clips = video_clips.permute(1, 2, 0, 3, 4)
        video_clips = [self.video_transform(clip) for clip in video_clips]
        video_clips = SpatialCrop(224, num_crops=3)(video_clips)
        video_clips = torch.stack(video_clips, dim=0)
        return video_clips


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox


def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text


def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
    (0, 160, 255),
    (0, 80, 255),
    (0, 215, 160),
    (0, 215, 80),
    (255, 215, 160),
    (255, 215, 80),
    (255, 160, 200),
    (255, 80, 200),
    (255, 160, 160),
    (255, 80, 80),
    (215, 255, 160),
    (215, 255, 80),
    (160, 255, 200),
    (80, 255, 200),
    (160, 255, 160),
    (80, 255, 80),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

used_colors = colors


def save_tmp_img(visual_img):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"/root/autodl-tmp/exp_story/results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    filename = os.path.join(dir_path, next(tempfile._get_candidate_names()) + '.jpg')
    visual_img.save(filename)
    return filename

def save_image_to_local(image: Image.Image):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"/root/autodl-tmp/exp_story/results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    filename = os.path.join(dir_path, next(tempfile._get_candidate_names()) + '.jpg')
    image.save(filename)
    return filename

def save_video_to_local(video):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"/root/autodl-tmp/exp_story/results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    filename = os.path.join(dir_path, next(tempfile._get_candidate_names()) + '.mp4')
    # writer = imageio.get_writer(filename, format='FFMPEG', fps=8)
    # for frame in video:
    #     writer.append_data(frame)
    # writer.close()
    from diffusers.utils import export_to_video
    export_to_video(video, filename, fps=8)
    return filename

def save_audio_to_local(audio):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"/root/autodl-tmp/exp_story/results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    filename = os.path.join(dir_path, next(tempfile._get_candidate_names()) + '.wav')
    scipy.io.wavfile.write(filename, rate=16000, data=audio)
    return filename

def save_imagestory_to_local(img, idx):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"/root/autodl-tmp/exp_story/results/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    # file_path = os.path.join(dir_path, f"image_{idx}.png")
    file_path = os.path.join(dir_path, f"image_{idx}.jpg")
    img.save(file_path)
    return file_path


def visualize_all_mask_together(image, masks):
    if (image is None) or (len(masks) == 0):
        return None

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        # reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        # reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        reverse_norm_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        reverse_norm_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        # pil_img = T.ToPILImage()(image_tensor)
        # image_h = pil_img.height
        # image_w = pil_img.width
        # image = np.array(pil_img)[:, :, [2, 1, 0]]
        image = np.array(image_tensor.permute(1,2,0), dtype=np.uint8)
        image_h = image.shape[0]
        image_w = image.shape[1]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    new_image = image.copy()
    if (image_h != masks[0].size()[0]) or (image_w != masks[0].size()[1]):
        print("The size of image and mask is different")
        new_image = cv2.resize(new_image, (masks[0].size()[1],masks[0].size()[0]))

    used_colors = colors

    # 生成不同颜色的mask
    for i, mask in enumerate(masks):
        mask = mask.cpu() # (224,224)
        color_mask = np.zeros_like(new_image, dtype=np.uint8) # (224,224,3)
        color_mask[mask > 0] = used_colors[i] # mask取阈值
        new_image = cv2.addWeighted(new_image, 0.7, color_mask, 0.3, 0)

    pil_image = Image.fromarray(new_image)
    return pil_image


def visualize_all_bbox_together(image, boxes):
    if (image is None) or (len(boxes) == 0):
        return None

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        # reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        # reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        reverse_norm_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        reverse_norm_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        # pil_img = T.ToPILImage()(image_tensor)
        # image_h = pil_img.height
        # image_w = pil_img.width
        # image = np.array(pil_img)[:, :, [2, 1, 0]]
        image = np.array(image_tensor.permute(1,2,0), dtype=np.uint8)
        image_h = image.shape[0]
        image_w = image.shape[1]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    new_image = image.copy()


    box_line = 2
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    valid_box = 0
    for box_idx, box in enumerate(boxes):
        if len(box) != 4:
            continue
        valid_box = valid_box + 1
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        # draw box
        color = used_colors[box_idx % len(used_colors)]  # tuple(np.random.randint(0, 255, size=3).tolist())
        new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), color, box_line)
        # resize for chatbot visualization
        long_sight = image_w
        if image_h > long_sight:
            long_sight = image_h
        resize_scale = float(long_sight) / 512.0
        new_image = cv2.resize(new_image, (int(float(image_w)/resize_scale), int(float(image_h)/resize_scale)))

    if valid_box == 0:
        return None

    pil_image = Image.fromarray(new_image)
    return pil_image



def update_input_state(input_modality):
    imagebox = gr.Image(type="pil", visible=input_modality=="Image")
    audiobox = gr.Audio(type="filepath", visible=input_modality=="Audio")
    videobox = gr.Video(visible=input_modality=="Video")

    if input_modality == "Image":
        location_modalities = gr.Radio(label='Select Location Modality',
                                       choices=['None', 'Box', 'Mask'],
                                       value='None',
                                       visible=True,
                                       interactive=True)
    else:
        location_modalities = gr.Radio(label='Select Location Modality',
                                       choices=['None', 'Box', 'Mask'],
                                       value='None',
                                       visible=False,
                                       interactive=True)
    return imagebox, audiobox, videobox, location_modalities


def update_output_state(output_modality):
    visible_flag = False
    if visible_flag:
        imagebox = gr.Image(type="pil",
                            visible=(output_modality=="Image" or
                                    output_modality=="Box" or
                                    output_modality=="Mask"))
        audiobox = gr.Audio(type="filepath", visible=output_modality=="Audio")
        videobox = gr.Video(visible=output_modality=="Video")
    else:
        imagebox = gr.Image(type="pil", visible=False)
        audiobox = gr.Audio(type="filepath", visible=False)
        videobox = gr.Video(visible=False)

    return imagebox, audiobox, videobox

from spider.processors.vision_processor import *
import numpy as np
from spider.models.segment_anything.utils.transforms import ResizeLongestSide
def gradio_ask(input_modalities, output_modalities, input_imagebox, input_audiobox, input_videobox,
                        output_imagebox, output_audiobox, output_videobox, input_textbox, chatbot):
    # import pdb
    # pdb.set_trace()

    # input modalities
    ask_info = {}
    if input_modalities == "Text":
        question = f"{input_textbox}"
    elif input_modalities == "Image":
        question = f"<IMAGE><IMAGE-Placeholder></IMAGE> {input_textbox}"
        assert input_imagebox is not None, 'Please upload image or close input image modality!'
        # image = np.array(input_imagebox['image'])
        image = np.array(input_imagebox)
        image_aug = vision_aug_transform(image=image)
        image_aug = vision_tensor_transform(image_aug)
        ask_info['IMAGE'] = [image_aug.cuda()]
    elif input_modalities == "Audio":
        question = f"<AUDIO><AUDIO-Placeholder></AUDIO> {input_textbox}"
        # todo: ask_info['AUDIO']
    elif input_modalities == "Video":
        question = f"<VIDEO><VIDEO-Placeholder></VIDEO> {input_textbox}"
        # todo: ask_info['VIDEO']
        assert input_videobox is not None, 'Please upload video or close input video modality!'
        # imagebind只能处理clips_per_video=5, clip_frames=2
        video_processor = ProcessVideos(clips_per_video=5, clip_frames=2) # clip_len = clips_per_video * clip_frames # 每个视频采样后的总帧数
        processed_video = video_processor(input_videobox)
        ask_info['VIDEO'] = [processed_video.cuda()]
    ask_info['Question'] = [question]
    # import pdb
    # pdb.set_trace()

    # output modalities
    if output_modalities == "Text":
        task_prompt = "[TEXT]"
        if input_imagebox is not None:
            # image = np.array(input_imagebox['image'])
            image = np.array(input_imagebox)
            # detr aug
            detr_transform = ResizeLongestSide(512)
            image_detr = detr_transform.apply_image(image) # np.array
            pil_image_vis_text = image_detr.copy()
            pil_image_vis_text = Image.fromarray(pil_image_vis_text) # for visualization
    elif output_modalities == "Image":
        task_prompt = "[IMAGE]"
        # text-controal generation according to the contents in input_textbox
        input_textbox_controal_gen = False
        if input_textbox_controal_gen:
            question = f"{input_textbox}"
            ask_info['Caption'] = [question]
    elif output_modalities == "Audio":
        task_prompt = "[AUDIO]"
    elif output_modalities == "Video":
        task_prompt = "[VIDEO]"
    elif output_modalities == "SmartMultimodal":
        task_prompt = "[SMARTMULTIMODAL]"
        # text-controal generation according to the contents in input_textbox
        input_textbox_controal_gen = False
        if input_textbox_controal_gen:
            question = f"{input_textbox}"
            ask_info['Caption'] = [question]

        # box and mask
        if input_imagebox is not None:
            # box
            # image = np.array(input_imagebox['image'])
            image = np.array(input_imagebox)
            ask_info['Image_ori_array'] = [image]
            # pil_image_vis_box = copy.deepcopy(input_imagebox['image']) # PIL.Image.Image
            pil_image_vis_box = copy.deepcopy(input_imagebox) # PIL.Image.Image

            # mask
            # image = np.array(input_imagebox['image'])
            image = np.array(input_imagebox)
            # llama aug
            image_aug = vision_aug_transform(image=image)
            image_aug = vision_tensor_transform(image_aug)
            # sam aug
            sam_transform = ResizeLongestSide(1024)
            image_sam = sam_transform.apply_image(image) # np.array
            image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
            pil_image_vis_mask = copy.deepcopy(image_sam) # tensor for visualization
            ask_info['IMAGE_SAM'] = [image_sam.cuda()]
            meta_info = {}
            meta_info['original_shape'] = [torch.tensor(image.shape[0:2])]
            meta_info['aug_shape'] = [torch.tensor(image_aug.shape[1::])]
            meta_info['sam_shape'] = [torch.tensor(image_sam.shape[1::])]
            ask_info["Meta_info"] = meta_info
    elif output_modalities == "SpecificMultimodal":
        task_prompt = "[SPECIFICMULTIMODAL]"
        # text-controal generation according to the contents in input_textbox
        input_textbox_controal_gen = False
        if input_textbox_controal_gen:
            question = f"{input_textbox}"
            ask_info['Caption'] = [question]
    elif (output_modalities == "Box") or (output_modalities == "Mask"):
        task_prompt = "[BOX]"
        assert input_imagebox is not None, 'Please upload image or close input image modality!'
        # image = np.array(input_imagebox['image'])
        image = np.array(input_imagebox)
        ask_info['Image_ori_array'] = [image]
        # pil_image_vis_box = copy.deepcopy(input_imagebox['image']) # PIL.Image.Image
        pil_image_vis_box = copy.deepcopy(input_imagebox) # PIL.Image.Image
        ########################################
        # 验证：训练时图像由iio.imread读取 和 前端输入的np.array(input_imagebox['image'])，两者是一致的
        # import imageio.v3 as iio
        # image_iio = iio.imread('/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/apple.jpg', pilmode="RGB")
        # print(np.array_equal(image, image_iio)) # True
        # 验证：mmcv读取，和iio.imread读取，的检测结果，有些区别
        # from mmdet.apis import init_detector, inference_detector
        # config_file = '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
        # checkpoint_file = '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
        # model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0' or device='cpu'
        # res_image_path = inference_detector(model, '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/apple.jpg', text_prompt='apple')
        # res_image_iio = inference_detector(model, image_iio, text_prompt='apple')
        # res_image = inference_detector(model, image, text_prompt='apple')
        ########################################
        # detr aug
        # detr_transform = ResizeLongestSide(512)
        # image_detr = detr_transform.apply_image(image) # np.array
        # # pil_image_vis_box = image_detr.copy()
        # # pil_image_vis_box = Image.fromarray(pil_image_vis_box) # for visualization
        # image_detr = detr_preprocess(torch.from_numpy(image_detr).permute(2, 0, 1).contiguous())
        # pil_image_vis_box = copy.deepcopy(image_detr) # tensor for visualization
        # ask_info['IMAGE_DETR'] = [image_detr.cuda()]

        if output_modalities == "Mask":
            task_prompt = "[MASK]"
            assert input_imagebox is not None, 'Please upload image or close input image modality!'
            # image = np.array(input_imagebox['image'])
            image = np.array(input_imagebox)
            # llama aug
            image_aug = vision_aug_transform(image=image)
            image_aug = vision_tensor_transform(image_aug)
            # sam aug
            sam_transform = ResizeLongestSide(1024)
            image_sam = sam_transform.apply_image(image) # np.array
            image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
            pil_image_vis_mask = copy.deepcopy(image_sam) # tensor for visualization
            ask_info['IMAGE_SAM'] = [image_sam.cuda()]
            meta_info = {}
            meta_info['original_shape'] = [torch.tensor(image.shape[0:2])]
            meta_info['aug_shape'] = [torch.tensor(image_aug.shape[1::])]
            meta_info['sam_shape'] = [torch.tensor(image_sam.shape[1::])]
            ask_info["Meta_info"] = meta_info
    elif output_modalities == "ImageStory":
        task_prompt = "[IMAGESTORY]"
        ask_info['SystemPrompt'] = [cfg.model.system_prompt]
    ask_info['TaskPrompt'] = [task_prompt]

    # import pdb
    # pdb.set_trace()
    answers, predictions, predictions_text = infer(ask_info)

    # predictions
    modality = task_prompt[1:-1]
    output_text = answers[0]
    # 添加问题部分
    chatbot.append((question, "[OUTPUT]"))
    # 初始化用于追踪每个模态的计数器
    image_idx = 0
    video_idx = 0
    audio_idx = 0
    # 使用正则表达式提取模态部分，包括普通文本
    for match in re.finditer(r'(<IMAGE>.*?</IMAGE>|<VIDEO>.*?</VIDEO>|<AUDIO>.*?</AUDIO>|<MASK>.*?</MASK>|<BOX>.*?</BOX>|\[END\]|[^<]+)', output_text):
        matched_text = match.group(0)
        # 处理模态内容
        if matched_text == "[END]":
            chatbot.append((None, matched_text))
        elif "<IMAGE>" in matched_text:
            chatbot.append((None, matched_text))
            image_file_path = save_image_to_local(predictions["IMAGE"][image_idx])
            chatbot.append([None, (image_file_path,)])
            image_idx += 1
        elif "<VIDEO>" in matched_text:
            chatbot.append((None, matched_text))
            video_file_path = save_video_to_local(predictions["VIDEO"][video_idx])
            chatbot.append([None, (video_file_path,)])
            video_idx += 1
        elif "<AUDIO>" in matched_text:
            chatbot.append((None, matched_text))
            audio_file_path = save_audio_to_local(predictions["AUDIO"][audio_idx])
            chatbot.append([None, (audio_file_path,)])
            audio_idx += 1
        elif "<MASK>" in matched_text:
            chatbot.append((None, matched_text))
            if len(predictions["MASK"]) > 0:
                visual_img = visualize_all_mask_together(pil_image_vis_mask, predictions["MASK"])
                if visual_img is not None:
                    image_file_path = save_tmp_img(visual_img)
                    chatbot.append([None, (image_file_path,)])
        elif "<BOX>" in matched_text:
            chatbot.append((None, matched_text))
            if len(predictions["BOX"]['bboxes']) > 0:
                for box_idx, bbox in enumerate(predictions["BOX"]['bboxes']):
                    output_name_box = str(predictions["BOX"]['label_names'][box_idx]) + str(predictions["BOX"]['bboxes'][box_idx].int().tolist())
                    chatbot.append((None, 'Box coordinate: ' + str(output_name_box)))
                visual_img = visualize_all_bbox_together(pil_image_vis_box, predictions["BOX"]['bboxes'])
                if visual_img is not None:
                    image_file_path = save_tmp_img(visual_img)
                    chatbot.append([None, (image_file_path,)])
        else:
            # 普通文本内容
            chatbot.append((None, matched_text))

    # image story
    if len(predictions["IMAGESTORY"]) > 0:
        for img_idx, img_story in enumerate(predictions["IMAGESTORY"][0]):
            image_file_path = save_imagestory_to_local(img_story, img_idx)
            chatbot.append([None, (image_file_path,)])
            chatbot.append([None, predictions_text["IMAGESTORY_prompts"][0][img_idx]])


    # modality = task_prompt[1:-1]
    # chatbot.append((question, answers[0]))
    # if modality == "TEXT":
    #     # print("TEXT")
    #     # chatbot.append((question, answers[0]))
    #     if input_imagebox is not None:
    #         image_file_path = save_tmp_img(pil_image_vis_text)
    #         chatbot.append([None, (image_file_path,)])
    # # import pdb
    # # pdb.set_trace()
    # if len(predictions["IMAGE"]) > 0:
    #     # print("IMAGE")
    #     # chatbot.append((question, answers[0]))
    #     for pred in predictions["IMAGE"]:
    #         image_file_path = save_image_to_local(pred)
    #         chatbot.append([None, (image_file_path,)])
    # if len(predictions["AUDIO"]) > 0:
    #     # print("AUDIO")
    #     # chatbot.append((question, answers[0]))
    #     for pred in predictions["AUDIO"]:
    #         audio_file_path = save_audio_to_local(pred)
    #         chatbot.append([None, (audio_file_path,)])
    # if len(predictions["VIDEO"]) > 0:
    #     # print("VIDEO")
    #     # chatbot.append((question, answers[0]))
    #     for pred in predictions["VIDEO"]:
    #         video_file_path = save_video_to_local(pred)
    #         chatbot.append([None, (video_file_path,)])
    # if len(predictions["BOX"]['bboxes']) > 0:
    #     # print("BOX")
    #     # print(predictions["BOX"])
    #     # predictions["BOX"]['bboxes']
    #     # predictions["BOX"]['label_names']
    #     # predictions["BOX"]['scores']
    #     # chatbot.append((question, answers[0]))
    #     if len(predictions["BOX"]['bboxes']) > 0:
    #         for box_idx, bbox in enumerate(predictions["BOX"]['bboxes']):
    #             output_name_box = str(predictions["BOX"]['label_names'][box_idx]) + str(predictions["BOX"]['bboxes'][box_idx].int().tolist())
    #             chatbot.append((question, answers[0] + '. Box coordinate: ' + str(output_name_box)))
    #         visual_img = visualize_all_bbox_together(pil_image_vis_box, predictions["BOX"]['bboxes'])
    #         if visual_img is not None:
    #             image_file_path = save_tmp_img(visual_img)
    #             chatbot.append([None, (image_file_path,)])
    # if len(predictions["MASK"]) > 0:
    #     # print("MASK") 
    #     # chatbot.append((question, answers[0]))
    #     if len(predictions["MASK"]) > 0:
    #         visual_img = visualize_all_mask_together(pil_image_vis_mask, predictions["MASK"])
    #         if visual_img is not None:
    #             image_file_path = save_tmp_img(visual_img)
    #             chatbot.append([None, (image_file_path,)])

    return chatbot



def build_chat():
    with gr.Blocks() as demo:
        gr.Markdown("""<h1 align="center">Spider Demo</h1>""")
        gr.Markdown("""<h3 align="center">Welcome to Our Spider, a multimodal LLM!</h3>""")

        with gr.Row():
            # 左侧
            with gr.Column(scale=2):
                # input
                with gr.Group():
                    input_modalities = gr.Radio(label='Select Input Modality',
                                                choices=['Text', 'Image', 'Audio', 'Video'],
                                                value='Text')
                input_imagebox = gr.Image(type="pil", visible=True)
                input_audiobox = gr.Audio(type="filepath", visible=False)
                input_videobox = gr.Video(visible=False)

                # output
                with gr.Group():
                    output_modalities = gr.Radio(label='Select Output Modality',
                                                 choices=['Text', 'ImageStory', 'Image', 'Box', 'Mask', 'Audio', 'Video', 'SmartMultimodal', 'SpecificMultimodal'],
                                                 value='Text',
                                                 interactive=True)
                output_imagebox = gr.Image(type="pil", visible=False)
                output_audiobox = gr.Audio(type="filepath", visible=False)
                output_videobox = gr.Video(visible=False)


            # 中间
            with gr.Column(scale=6):
                # 对话框
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="Chatbot",
                        visible=True,
                        height=1070,
                    )

                with gr.Row():
                    # 输入框
                    with gr.Column(scale=8):
                        input_textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and add to prompt",
                            visible=True,
                            container=False,
                        )

                    # 发送按钮
                    with gr.Column(scale=1, min_width=60):
                        send_btn = gr.Button(value="Send")

                    # 清空历史
                    with gr.Column(scale=1, min_width=60):
                        clear_btn = gr.Button(value="Clear")

            # # 右侧
            # with gr.Column(scale=2):
            #     with gr.Group():
            #         output_modalities = gr.Radio(label='Select Output Modality',
            #                                      choices=['Text', 'Image', 'Box', 'Mask', 'Audio', 'Video'],
            #                                      value='Text',
            #                                      interactive=True)
            #     output_imagebox = gr.Image(type="pil", visible=False)
            #     output_audiobox = gr.Audio(type="filepath", visible=False)
            #     output_videobox = gr.Video(visible=False)

        input_modalities.change(fn=update_input_state,
                                inputs=[input_modalities],
                                outputs=[input_imagebox, input_audiobox, input_videobox])
        output_modalities.change(fn=update_output_state,
                                inputs=[output_modalities],
                                outputs=[output_imagebox, output_audiobox, output_videobox])


        # send按钮
        send_btn.click(gradio_ask,
                       inputs=[input_modalities, output_modalities, input_imagebox, input_audiobox, input_videobox,
                        output_imagebox, output_audiobox, output_videobox, input_textbox, chatbot,],
                       outputs=[chatbot])
        # # 回车，send的快捷方式
        # textbox.submit(submit,
        #                inputs=[state, textbox, imagebox, audiobox, videobox],
        #                outputs=[chatbot, state, textbox, imagebox, audiobox, videobox])
        # clear_btn.click(build_chat, inputs=None, outputs=None)



    # demo.launch(share=True, server_port=8081, server_name='11.198.63.54') # 运行机器的ip
    # demo.launch(share=True, server_port=8000, server_name='0.0.0.0')
    demo.launch(
        allowed_paths=[
            "/root/autodl-tmp",
            "/root/autodl-tmp/exp_story/results",
        ],
        share=True, server_port=6006) # autodl


build_chat()
