import io
import os
import ffmpeg
import numpy as np
import gradio as gr
import soundfile as sf 
import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info
from argparse import ArgumentParser
import json
import re
import ast
from datetime import datetime
from PIL import Image
import imageio
import scipy
import tempfile
import cv2
import torch
from mmengine import Config
import copy
import random
from spider.processors import *
from spider.processors.vision_processor import *
from spider.models.segment_anything.utils.transforms import ResizeLongestSide
from spider_decoder_infer import SpiderDecoderInfer



#####################################################
#####################################################
MODEL_NAME = "spider_free_qwen"
# MODEL_NAME = "spider_story_free_qwen"
DEFAULT_CKPT_PATH = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/Qwen2.5-Omni-7B"
# SYSTEM_PROMPT
USER_PROMPT = ""
ASSISTANT_PROMPT = ""
if MODEL_NAME == "spider_free_qwen":
    cfg_root = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/train_configs"
    cfg_path = os.path.join(cfg_root, "spider_decoder_cfg.py")
    cfg = Config.fromfile(cfg_path)
    spider_decoder_infer = SpiderDecoderInfer(cfg)
    # system prompt
    # user input example: A red apple<IMAGE>a red apple</IMAGE> is on the table. The cat is running<VIDEO>the cat is running</VIDEO>. You can hear a dog's bark<AUDIO>dog's bark</AUDIO>.
    # user input example: A red apple<IMAGE>a red apple</IMAGE> is on the table. The cat is running<VIDEO>the cat is running</VIDEO>. You can hear a dog's bark<AUDIO>dog's bark</AUDIO>. Here's a short story: <IMAGESTORY><GENERALPROMPT> 'a man with a black suit' </GENERALPROMPT>, <PROMPTARRAY> ['wake up in the bed', 'have breakfast', 'work in the company', 'reading book in the home'] </PROMPTARRAY>, <STYLENAME> 'Comic book' </STYLENAME></IMAGESTORY>.
    # user input example: generate an image of a car, segment the car, detect the car, generate a video of a car is driving on the road, an audio of a car engine, a story about a car.
    # user input example: generate an image of a car, a video of a car is driving on the road, a audio of a car engine.
    # user input example: generate an image of a car, a video of a car is driving on the road, a audio of a car engine, and a story about a car.
    # user input example: an image of a dog, a video of a dog is running, an audio of a dog's bark.

    SYSTEM_PROMPT = cfg.model.system_prompt
    if 'IMAGE' in cfg.model.diffusion_modules:
        SYSTEM_PROMPT = SYSTEM_PROMPT + cfg.model.system_prompt_image
    if 'VIDEO' in cfg.model.diffusion_modules:
        SYSTEM_PROMPT = SYSTEM_PROMPT + cfg.model.system_prompt_video
    if 'AUDIO' in cfg.model.diffusion_modules:
        SYSTEM_PROMPT = SYSTEM_PROMPT + cfg.model.system_prompt_audio
    if cfg.model.mask_decoder_modules != None:
        SYSTEM_PROMPT = SYSTEM_PROMPT + cfg.model.system_prompt_mask
    if cfg.model.box_decoder_modules != None:
        SYSTEM_PROMPT = SYSTEM_PROMPT + cfg.model.system_prompt_box
    if cfg.model.story_generation != None:
        SYSTEM_PROMPT = SYSTEM_PROMPT + cfg.model.system_prompt_story
    USER_PROMPT = cfg.model.user_prompt
    ASSISTANT_PROMPT = cfg.model.assistant_prompt
elif MODEL_NAME == "spider_story_free_qwen":
    from StoryDiffusion.Comic_Generation import init_story_generation, story_generation
    story_diffusion = init_story_generation(model_name="Unstable", device="cuda")
    SYSTEM_PROMPT = "You are Spider-Story, an AI assistant that generates structured story descriptions for visual storytelling." \
    "Your task is to output a well-formatted response with the following structure:" \
    "1. **General Prompt**: A brief description of the main character or setting. User may provide corresponding content for it." \
    "2. **Prompt Array**: A sequence of key moments in the story, each describing a separate scene (formatted as a Python list). User may provide corresponding content for it." \
    "3. **Style Name**: Choose a visual style from the list: ['Japanese Anime', 'Digital/Oil Painting', 'Pixar/Disney Character', 'Photographic', 'Comic book', 'Line art', 'Black and White Film Noir', 'Isometric Rooms']. User may provide corresponding content for Style Name, then select the best choice for the user." \
    "### **Example Output Format**" \
    "<GENERALPROMPT> 'a man with a black suit' </GENERALPROMPT> <PROMPTARRAY> ['wake up in the bed', 'have breakfast', 'work in the company', 'reading book in the home'] </PROMPTARRAY> <STYLENAME> 'Comic book' </STYLENAME>" \
    "### **Instructions**" \
    "- `<GENERALPROMPT>` must contain a **quoted string** describing the character or setting." \
    "- `<PROMPTARRAY>` must be a **valid Python list** of quoted strings. Recheck the format of <PROMPTARRAY>, which must be a Python list!" \
    "- `<STYLENAME>` must be a **quoted string** chosen from the predefined list." \
    "- The response **must strictly follow** the above format with XML-like tags." \
    "- **Example Output Format** is the example. The specific content should generate according to the user demand." \
    "Now, generate a structured story description in this format. And carefully recheck the formats of <GENERALPROMPT>, <PROMPTARRAY>, <STYLENAME>."
else:
    SYSTEM_PROMPT = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=True,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=6006, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='en', help='Display language for the UI.')

    args = parser.parse_args()
    return args

#####################################################
#####################################################

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

def clean_prompt_array(prompt_str):
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

def extract_answer(output_texts):
    """ 如果出现 </think>，则只提取其后的内容 """
    # 查找 </think> 并截取其后的内容
    think_split = output_texts.split("</think>", 1)
    if len(think_split) > 1:
        output_texts = think_split[1]  # 取 </think> 之后的部分
    return output_texts

def extract_story_elements(output_texts):
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
    prompt_array = clean_prompt_array(prompt_array_str)
    # 提取所有 Style Name
    style_name_matches = re.findall(r"<STYLENAME>\s*(.*?)\s*</STYLENAME>", output_texts, re.DOTALL)
    style_name = style_name_matches[-1].strip() if style_name_matches else ""
    return general_prompt, prompt_array, style_name


def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        model = Qwen2_5OmniModel.from_pretrained(args.checkpoint_path,
                                                    torch_dtype='auto',
                                                    attn_implementation='flash_attention_2',
                                                    device_map=device_map)
    else:
        model = Qwen2_5OmniModel.from_pretrained(args.checkpoint_path, device_map=device_map)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
    return model, processor

def _launch_demo(args, model, processor):
    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan']
    DEFAULT_VOICE = 'Chelsie'

    # default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs. Only generates text, no audio output.'
    default_system_prompt = SYSTEM_PROMPT

    language = args.ui_language

    def get_text(text: str, cn_text: str):
        if language == 'en':
            return text
        if language == 'zh':
            return cn_text
        return text
    
    def convert_webm_to_mp4(input_file, output_file):
        try:
            (
                ffmpeg
                .input(input_file)
                .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
                .run(quiet=True, overwrite_output=True)
            )
            print(f"Conversion successful: {output_file}")
        except ffmpeg.Error as e:
            print("An error occurred during conversion.")
            print(e.stderr.decode('utf-8'))

    def format_history(history: list, system_prompt: str):
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": USER_PROMPT})
        messages.append({"role": "assistant", "content": ASSISTANT_PROMPT})
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("image"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "image",
                            "image": file_path
                        }]
                    })
                elif mime_type.startswith("video"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "video",
                            "video": file_path
                        }]
                    })
                elif mime_type.startswith("audio"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "audio",
                            "audio": file_path,
                        }]
                    })
        return messages

    def predict(messages, voice=DEFAULT_VOICE):
        print('predict history: ', messages)    

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audios, images, videos = process_mm_info(messages, True)

        inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, audio = model.generate(**inputs, spk=voice, use_audio_in_video=True)

        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = response[0].split("\n")[-1]
        yield {"type": "text", "data": response}

        # decoders
        print("output_texts:", response)
        if MODEL_NAME == "spider_story_free_qwen": # image story
            general_prompt, prompt_array, style_name = extract_story_elements(response)
            print("General Prompt:", general_prompt)
            print("Prompt Array:", prompt_array)
            print("Style Name:", style_name)
            if general_prompt and prompt_array and isinstance(prompt_array, list) and len(prompt_array) > 0 and style_name:
                preds = story_generation(story_diffusion, general_prompt=general_prompt, prompt_array=prompt_array, style_name=style_name)
                story_images_path = []
                for img_idx, img_story in enumerate(preds):
                    image_file_path = save_imagestory_to_local(img_story, img_idx)
                    story_images_path.append(image_file_path)
                yield {"type": "story_images", "data": story_images_path}
                yield {"type": "story_prompts", "data": prompt_array}
            else:
                print("Error: One or more required inputs for story_generation are empty!")
        elif MODEL_NAME == "spider_free_qwen": # spider decoder
            # input
            ask_info = {}
            # ask_info['llm_text_all'] = ["<IMAGE>a red apple</IMAGE><VIDEO>cat is running</VIDEO><AUDIO>dog's bark</AUDIO>"] # output text of llm
            ask_info['llm_text_all'] = [extract_answer(response)]
            # input for box and mask
            if images is not None: # check the type of images, maybe need to adjust images
                input_image = copy.deepcopy(images[0]) # PIL.Image.Image
                # box
                image = np.array(input_image)
                ask_info['Image_ori_array'] = [image]
                pil_image_vis_box = copy.deepcopy(input_image) # PIL.Image.Image

                # mask
                image = np.array(input_image)
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
            # generation
            answers, predictions, predictions_text = spider_decoder_infer(ask_info)
            print("answers:", answers)
            print("predictions_text:", predictions_text)
            # for gradio
            output_text = answers[0]
            # 初始化用于追踪每个模态的计数器
            image_idx = 0
            video_idx = 0
            audio_idx = 0
            mask_idx = 0
            box_idx = 0
            story_idx = 0
            # 使用正则表达式提取模态部分，包括普通文本
            for match in re.finditer(r'(<IMAGE>.*?</IMAGE>|<VIDEO>.*?</VIDEO>|<AUDIO>.*?</AUDIO>|<MASK>.*?</MASK>|<BOX>.*?</BOX>|<IMAGESTORY>.*?</IMAGESTORY>|[^<]+)', output_text):
                matched_text = match.group(0)
                # 处理模态内容
                if "<IMAGE>" in matched_text:
                    if len(predictions["IMAGE"]) > 0:
                        image_file_path = save_image_to_local(predictions["IMAGE"][image_idx])
                        yield {"type": "spider_image", "data": image_file_path}
                        yield {"type": "image_prompt", "data": matched_text}
                        image_idx += 1
                elif "<VIDEO>" in matched_text:
                    if len(predictions["VIDEO"]) > 0:
                        video_file_path = save_video_to_local(predictions["VIDEO"][video_idx])
                        yield {"type": "spider_video", "data": video_file_path}
                        yield {"type": "video_prompt", "data": matched_text}
                        video_idx += 1
                elif "<AUDIO>" in matched_text:
                    if len(predictions["AUDIO"]) > 0:
                        audio_file_path = save_audio_to_local(predictions["AUDIO"][audio_idx])
                        yield {"type": "spider_audio", "data": audio_file_path}
                        yield {"type": "audio_prompt", "data": matched_text}
                        audio_idx += 1
                elif "<MASK>" in matched_text:
                    if len(predictions["MASK"]) > 0:
                        visual_img = visualize_all_mask_together(pil_image_vis_mask, predictions["MASK"][mask_idx])
                        mask_idx += 1
                        if visual_img is not None:
                            image_file_path = save_tmp_img(visual_img)
                            yield {"type": "mask_image", "data": image_file_path}
                            yield {"type": "mask_prompt", "data": matched_text}
                elif "<BOX>" in matched_text:
                    if len(predictions["BOX"]['bboxes']) > 0:
                        # for idx, bbox in enumerate(predictions["BOX"]['bboxes'][box_idx]):
                        #     output_name_box = str(predictions["BOX"]['label_names'][box_idx][idx]) + str(predictions["BOX"]['bboxes'][box_idx][idx].int().tolist())
                        #     chatbot.append((None, 'Box coordinate: ' + str(output_name_box)))
                        visual_img = visualize_all_bbox_together(pil_image_vis_box, predictions["BOX"]['bboxes'][box_idx])
                        box_idx += 1
                        if visual_img is not None:
                            image_file_path = save_tmp_img(visual_img)
                            yield {"type": "box_image", "data": image_file_path}
                            yield {"type": "box_prompt", "data": matched_text}
                elif "<IMAGESTORY>" in matched_text:
                    if len(predictions["IMAGESTORY"]) > 0:
                        story_images_path = []
                        for img_idx, img_story in enumerate(predictions["IMAGESTORY"][story_idx]):
                            image_file_path = save_imagestory_to_local(img_story, img_idx)
                            story_images_path.append(image_file_path)
                        prompt_array = predictions_text["IMAGESTORY_prompts"][story_idx]
                        yield {"type": "story_images", "data": story_images_path}
                        yield {"type": "story_prompts", "data": prompt_array}
                        story_idx += 1
                else:
                    # 普通文本内容
                    yield {"type": "text", "data": matched_text}

        # audio
        audio = np.array(audio * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, samplerate=24000, format="WAV")
        wav_io.seek(0)
        wav_bytes = wav_io.getvalue()
        audio_path = processing_utils.save_bytes_to_cache(
            wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
        yield {"type": "audio", "data": audio_path}

    def media_predict(audio, video, history, system_prompt, voice_choice):
        # First yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        if video is not None:
            convert_webm_to_mp4(video, video.replace('.webm', '.mp4'))
            video = video.replace(".webm", ".mp4")
        files = [audio, video]

        for f in files:
            if f:
                history.append({"role": "user", "content": (f, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt,)


        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice):
            show_modality_prompt = False # 是否显示模态对应的提示文本
            # text
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    None,  # webcam
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            # image
            if chunk["type"] == "spider_image":
                image = chunk["data"]  # 暂存图片
            if chunk["type"] == "image_prompt" and image:
                # 将图片和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Image(image)
                })
                if show_modality_prompt:
                    history.append({
                        "role": "assistant",
                        "content": chunk["data"]
                    })
                yield (
                    None, None, history,
                    gr.update(visible=False), gr.update(visible=True),
                )
            # video
            if chunk["type"] == "spider_video":
                video = chunk["data"]  # 暂存视频
            if chunk["type"] == "video_prompt" and video:
                # 将视频和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Video(video)
                })
                if show_modality_prompt:
                    history.append({
                        "role": "assistant",
                        "content": chunk["data"]
                    })
                yield (
                    None, None, history,
                    gr.update(visible=False), gr.update(visible=True),
                )
            # audio
            if chunk["type"] == "spider_audio":
                audio = chunk["data"]  # 暂存音频
            if chunk["type"] == "audio_prompt" and audio:
                # 将音频和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(audio)
                })
                if show_modality_prompt:
                    history.append({
                        "role": "assistant",
                        "content": chunk["data"]
                    })
                yield (
                    None, None, history,
                    gr.update(visible=False), gr.update(visible=True),
                )
            # mask
            if chunk["type"] == "mask_image":
                mask_image = chunk["data"]  # 暂存图片
            if chunk["type"] == "mask_prompt" and mask_image:
                # 将图片和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Image(mask_image)
                })
                if show_modality_prompt:
                    history.append({
                        "role": "assistant",
                        "content": chunk["data"]
                    })
                yield (
                    None, None, history,
                    gr.update(visible=False), gr.update(visible=True),
                )
            # box
            if chunk["type"] == "box_image":
                box_image = chunk["data"]  # 暂存图片
            if chunk["type"] == "box_prompt" and box_image:
                # 将图片和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Image(box_image)
                })
                if show_modality_prompt:
                    history.append({
                        "role": "assistant",
                        "content": chunk["data"]
                    })
                yield (
                    None, None, history,
                    gr.update(visible=False), gr.update(visible=True),
                )
            # story
            if chunk["type"] == "story_images":
                images = chunk["data"]  # 暂存图片列表
            if chunk["type"] == "story_prompts" and images:
                # 将图片和描述插入聊天记录
                for img, prompt in zip(images, chunk["data"]):
                    history.append({
                        "role": "assistant",
                        "content": gr.Image(img)
                    })
                    history.append({
                        "role": "assistant",
                        "content": prompt
                    })
                yield (
                    None, None, history,
                    gr.update(visible=False), gr.update(visible=True),
                )
            # audio
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        # Final yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    def chat_predict(text, audio, image, video, history, system_prompt, voice_choice):
        # Process text input
        if text:
            history.append({"role": "user", "content": text})

        # Process audio input
        if audio:
            history.append({"role": "user", "content": (audio, )})

        # Process image input
        if image:
            history.append({"role": "user", "content": (image, )})

        # Process video input
        if video:
            history.append({"role": "user", "content": (video, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt)

        yield None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice):
            # text
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(
                ), history
            # image
            if chunk["type"] == "spider_image":
                image = chunk["data"]  # 暂存图片
            if chunk["type"] == "image_prompt" and image:
                # 将图片和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Image(image)
                })
                history.append({
                    "role": "assistant",
                    "content": chunk["data"]
                })
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            # video
            if chunk["type"] == "spider_video":
                video = chunk["data"]  # 暂存视频
            if chunk["type"] == "video_prompt" and video:
                # 将视频和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Video(video)
                })
                history.append({
                    "role": "assistant",
                    "content": chunk["data"]
                })
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            # audio
            if chunk["type"] == "spider_audio":
                audio = chunk["data"]  # 暂存音频
            if chunk["type"] == "audio_prompt" and audio:
                # 将音频和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(audio)
                })
                history.append({
                    "role": "assistant",
                    "content": chunk["data"]
                })
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            # mask
            if chunk["type"] == "mask_image":
                mask_image = chunk["data"]  # 暂存图片
            if chunk["type"] == "mask_prompt" and mask_image:
                # 将图片和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Image(mask_image)
                })
                history.append({
                    "role": "assistant",
                    "content": chunk["data"]
                })
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            # box
            if chunk["type"] == "box_image":
                box_image = chunk["data"]  # 暂存图片
            if chunk["type"] == "box_prompt" and box_image:
                # 将图片和描述插入聊天记录
                history.append({
                    "role": "assistant",
                    "content": gr.Image(box_image)
                })
                history.append({
                    "role": "assistant",
                    "content": chunk["data"]
                })
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            # story
            if chunk["type"] == "story_images":
                images = chunk["data"]  # 暂存图片列表
            if chunk["type"] == "story_prompts" and images:
                # 将图片和描述插入聊天记录
                for img, prompt in zip(images, chunk["data"]):
                    history.append({
                        "role": "assistant",
                        "content": gr.Image(img)
                    })
                    history.append({
                        "role": "assistant",
                        "content": prompt
                    })
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history
            # audio
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history # chatbot

    with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt",
                                            value=default_system_prompt)
        with antd.Flex(gap="small", justify="center", align="center"):
            with antd.Flex(vertical=True, gap="small", align="center"):
                antd.Typography.Title("SpiderFree (Qwen2.5-Omni) Demo",
                                    level=1,
                                    elem_style=dict(margin=0, fontSize=28))
                # Using gr.Row for horizontal layout (Instructions and Citation side by side)
                with gr.Row(equal_height=True):
                    # Instructions Section (Left side)
                    with gr.Column():
                        with antd.Flex(vertical=True, gap="small", align="start"):
                            antd.Typography.Text(get_text("🎯 Instructions for Online use:", "🎯 使用说明："), strong=True)
                            antd.Typography.Text(get_text("1️⃣ Click the Audio Record button or the Camera Record button.",                                                            "1️⃣ 点击音频录制按钮，或摄像头-录制按钮"))
                            antd.Typography.Text(get_text("2️⃣ Input audio or video.", "2️⃣ 输入音频或者视频"))
                            antd.Typography.Text(get_text("3️⃣ Click the submit button and wait for the model's response.",                                                            "3️⃣ 点击提交并等待模型的回答"))
                    
                    # Citation Section (Right side)
                    with gr.Column():
                        with antd.Flex(vertical=True, gap="small", align="start"):
                            antd.Typography.Text("📚 Citation: If you use this code for your research, please cite our paper:", strong=True)
                            antd.Typography.Text(
                                "```bibtex\n"
                                "@article{lai2024spider,\n"
                                "  title={Spider: Any-to-Many Multimodal LLM},\n"
                                "  author={Lai, Jinxiang and Zhang, Jie and Liu, Jun and Li, Jian and Lu, Xiaocheng and Guo, Song},\n"
                                "  journal={arXiv preprint arXiv:2411.09439},\n"
                                "  year={2024}\n"
                                "}\n"
                                "```"
                            )
                            # Contact Section
                            antd.Typography.Text("📞 Contact: Jinxiang Lai: layjins1994@gmail.com", strong=True)
        
        voice_choice = gr.Dropdown(label="Voice Choice",
                                choices=VOICE_LIST,
                                value=DEFAULT_VOICE)
        with gr.Tabs():
            with gr.Tab("Offline"):
                chatbot = gr.Chatbot(type="messages", height=650)

                # Media upload section in one row
                with gr.Row(equal_height=True):
                    audio_input = gr.Audio(sources=["upload"],
                                        type="filepath",
                                        label="Upload Audio",
                                        elem_classes="media-upload",
                                        scale=1)
                    image_input = gr.Image(sources=["upload"],
                                        type="filepath",
                                        label="Upload Image",
                                        elem_classes="media-upload",
                                        scale=1)
                    video_input = gr.Video(sources=["upload"],
                                        label="Upload Video",
                                        elem_classes="media-upload",
                                        scale=1)

                # Text input section
                text_input = gr.Textbox(show_label=False,
                                        placeholder="Enter text here...")

                # Control buttons
                with gr.Row():
                    submit_btn = gr.Button(get_text("Submit", "提交"),
                                        variant="primary",
                                        size="lg")
                    stop_btn = gr.Button(get_text("Stop", "停止"),
                                        visible=False,
                                        size="lg")
                    clear_btn = gr.Button(get_text("Clear History", "清除历史"),
                                        size="lg")

                def clear_chat_history():
                    return [], gr.update(value=None), gr.update(
                        value=None), gr.update(value=None), gr.update(value=None)

                submit_event = gr.on(
                    triggers=[submit_btn.click, text_input.submit],
                    fn=chat_predict,
                    inputs=[
                        text_input, audio_input, image_input, video_input, chatbot,
                        system_prompt_textbox, voice_choice
                    ],
                    outputs=[
                        text_input, audio_input, image_input, video_input, chatbot
                    ])

                stop_btn.click(fn=lambda:
                            (gr.update(visible=True), gr.update(visible=False)),
                            inputs=None,
                            outputs=[submit_btn, stop_btn],
                            cancels=[submit_event],
                            queue=False)

                clear_btn.click(fn=clear_chat_history,
                                inputs=None,
                                outputs=[
                                    chatbot, text_input, audio_input, image_input,
                                    video_input
                                ])

                # Add some custom CSS to improve the layout
                gr.HTML("""
                    <style>
                        .media-upload {
                            margin: 10px;
                            min-height: 160px;
                        }
                        .media-upload > .wrap {
                            border: 2px dashed #ccc;
                            border-radius: 8px;
                            padding: 10px;
                            height: 100%;
                        }
                        .media-upload:hover > .wrap {
                            border-color: #666;
                        }
                        /* Make upload areas equal width */
                        .media-upload {
                            flex: 1;
                            min-width: 0;
                        }
                    </style>
                """)

            with gr.Tab("Online"):
                with gr.Row():
                    with gr.Column(scale=1):
                        microphone = gr.Audio(sources=['microphone'],
                                            type="filepath")
                        webcam = gr.Video(sources=['webcam'],
                                        height=400,
                                        include_audio=True)
                        submit_btn = gr.Button(get_text("Submit", "提交"),
                                            variant="primary")
                        stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                        clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                    with gr.Column(scale=2):
                        media_chatbot = gr.Chatbot(height=650, type="messages")

                    def clear_history():
                        return [], gr.update(value=None), gr.update(value=None)

                    submit_event = submit_btn.click(fn=media_predict,
                                                    inputs=[
                                                        microphone, webcam,
                                                        media_chatbot,
                                                        system_prompt_textbox,
                                                        voice_choice
                                                    ],
                                                    outputs=[
                                                        microphone, webcam,
                                                        media_chatbot, submit_btn,
                                                        stop_btn
                                                    ])
                    stop_btn.click(
                        fn=lambda:
                        (gr.update(visible=True), gr.update(visible=False)),
                        inputs=None,
                        outputs=[submit_btn, stop_btn],
                        cancels=[submit_event],
                        queue=False)
                    clear_btn.click(fn=clear_history,
                                    inputs=None,
                                    outputs=[media_chatbot, microphone, webcam])
 
    demo.queue(default_concurrency_limit=100, max_size=100).launch(
        allowed_paths=[
            "/root/autodl-tmp",
            "/root/autodl-tmp/exp_story/results",
        ],
        max_threads=100,
        ssr_mode=False,
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,)


if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)