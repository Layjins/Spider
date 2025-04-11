import re
import os
from typing import List
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from .layers import *
from spider.common.utils import *
from spider.common.registry import registry
from .custom_sd import StableDiffusionPipeline
from .custom_vd import TextToVideoSDPipeline
from .custom_ad import AudioLDMPipeline
from mmdet.apis import init_detector, inference_detector


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.cuda.current_device()

# init Grounding DINO
init_dino_flag = True
# init_dino_flag = False
if init_dino_flag:
    config_file = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
    checkpoint_file = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
    grounding_dino_model = init_detector(config_file, checkpoint_file, device='cpu') # for training, to save gpu memory
else:
    grounding_dino_model = None



@registry.register_model("spider_decoder")
class SpiderDecoder(BaseModel):
    def __init__(self,
                name="spider_decoder",
                system_prompt='You are Spider, an AI assistant that can understand and generate many modalities.',
                get_prompt_embed_for_diffusion=False, # If True: get prompt embedding for diffusion
                diffusion_modules=dict(
                    IMAGE=dict(type="sd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/stable-diffusion-v1-5'),
                    VIDEO=dict(type="vd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/zeroscope_v2_576w'),
                    AUDIO=dict(type="ad", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/audioldm-l-full'),
                ),
                system_prompt_image='You are Spider, an AI assistant that can understand and generate many modalities.',
                system_prompt_video='You are Spider, an AI assistant that can understand and generate many modalities.',
                system_prompt_audio='You are Spider, an AI assistant that can understand and generate many modalities.',
                mask_decoder_modules=dict(
                    sam_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/sam_vit_h_4b8939.pth",
                    freeze_mask_decoder=True,
                ),
                system_prompt_mask='You are Spider, an AI assistant that can understand and generate many modalities.',
                box_decoder_modules=dict(
                    # grounding DINO
                    config_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py',
                    checkpoint_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
                ),
                system_prompt_box='You are Spider, an AI assistant that can understand and generate many modalities.',
                story_generation=dict(
                    model_name="Unstable",
                ),
                system_prompt_story='You are Spider, an AI assistant that can understand and generate many modalities.',
                max_context_len=4096,
                ):
        super().__init__()
        self.model_name = name
        self.max_context_len = max_context_len
        self.device = torch.cuda.current_device()
        self.box_decoder_modules = box_decoder_modules
        self.mask_decoder_modules = mask_decoder_modules

        self.get_prompt_embed_for_diffusion = get_prompt_embed_for_diffusion
        self.sd_ckpt_path = None
        self.vd_ckpt_path = None
        self.ad_ckpt_path = None
        self.diffusion_pipes = None
        self.mask_decoder_sam = None
        if 'IMAGE' in diffusion_modules:
            self.sd_ckpt_path = diffusion_modules['IMAGE']['ckpt']
        if 'VIDEO' in diffusion_modules:
            self.vd_ckpt_path = diffusion_modules['VIDEO']['ckpt']
        if 'AUDIO' in diffusion_modules:
            self.ad_ckpt_path = diffusion_modules['AUDIO']['ckpt']
        if self.get_prompt_embed_for_diffusion and ('IMAGE' in diffusion_modules) or ('VIDEO' in diffusion_modules) or ('AUDIO' in diffusion_modules):
            self.diffusion_pipes = self.init_diffusion_model(diffusion_modules)
        if mask_decoder_modules != None:
            self.mask_decoder_sam = self.init_mask_decoder_sam(mask_decoder_modules)

        self.decode_modality = dict(
            IMAGE=self.decode_image,
            VIDEO=self.decode_video,
            AUDIO=self.decode_audio,
            MASK=self.decode_mask,
            BOX=self.decode_box,
            IMAGESTORY=None,
        )

    ##############################################
    #          Decoder Side Modules              #
    ##############################################
    def decode_image(self, samples, guidance_scale=7.5, num_inference_steps=40):
        image_outputs = None
        if ("llm_text_res" in samples) and (self.sd_ckpt_path != None): # llm-text-control generation, for inference
            if self.get_prompt_embed_for_diffusion: # text prompt embed control
                captions = samples['llm_text_res']
                image_diffusion_pipe = self.diffusion_pipes['IMAGE']
                condiction_embeds = image_diffusion_pipe(captions, return_prompts_only=True).detach()
                condiction_embeds = condiction_embeds.to(self.device)
                # generate image by text embedding
                generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path, torch_dtype=torch.float16).to(self.device)
                image_outputs = generation_model(prompt_embeds=condiction_embeds,
                                                guidance_scale=guidance_scale,
                                                num_inference_steps=num_inference_steps).images
            else: # text prompt control
                generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path, torch_dtype=torch.float16).to(self.device)
                image_outputs = generation_model(prompt=samples['llm_text_res'],
                                                guidance_scale=guidance_scale,
                                                num_inference_steps=num_inference_steps).images
        else:
            print("no input text prompt for image generation. or no image generation model.")
        return image_outputs

    def decode_video(self, samples, guidance_scale=7.5, num_inference_steps=40, height=320, width=576, num_frames=16):
        video_outputs = None
        if ("llm_text_res" in samples) and (self.vd_ckpt_path != None): # llm-text-control generation, for inference
            if self.get_prompt_embed_for_diffusion: # text prompt embed control
                captions = samples['llm_text_res']
                video_diffusion_pipe = self.diffusion_pipes['VIDEO']
                condiction_embeds = video_diffusion_pipe(captions, return_prompts_only=True).detach()
                condiction_embeds = condiction_embeds.to(self.device)
                generation_model = TextToVideoSDPipeline.from_pretrained(self.vd_ckpt_path, torch_dtype=torch.float16).to(self.device)
                video_outputs = generation_model(prompt_embeds=condiction_embeds,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps, height=height,
                                            width=width, num_frames=num_frames).frames
            else: # text prompt control
                generation_model = TextToVideoSDPipeline.from_pretrained(self.vd_ckpt_path, torch_dtype=torch.float16).to(self.device)
                video_outputs = generation_model(prompt=samples['llm_text_res'],
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps, height=height,
                                            width=width, num_frames=num_frames).frames
        else:
            print("no input text prompt for video generation. or no video generation model.")
        return video_outputs

    def decode_audio(self, samples, guidance_scale=7.5, num_inference_steps=40, audio_length_in_s=5.0):
        audio_outputs = None
        if ("llm_text_res" in samples) and (self.ad_ckpt_path != None): # llm-text-control generation, for inference
            if self.get_prompt_embed_for_diffusion: # text prompt embed control
                captions = samples['llm_text_res']
                audio_diffusion_pipe = self.diffusion_pipes['AUDIO']
                condiction_embeds = audio_diffusion_pipe(captions, return_prompts_only=True).detach()
                condiction_embeds = condiction_embeds.to(self.device)
                generation_model = AudioLDMPipeline.from_pretrained(self.ad_ckpt_path, torch_dtype=torch.float16).to(self.device)
                audio_outputs = generation_model(prompt_embeds=condiction_embeds,
                                                guidance_scale=guidance_scale,
                                                num_inference_steps=num_inference_steps,
                                                audio_length_in_s=audio_length_in_s).audios
            else: # text prompt control
                generation_model = AudioLDMPipeline.from_pretrained(self.ad_ckpt_path, torch_dtype=torch.float16).to(self.device)
                audio_outputs = generation_model(prompt=samples['llm_text_res'],
                                                guidance_scale=guidance_scale,
                                                num_inference_steps=num_inference_steps,
                                                audio_length_in_s=audio_length_in_s).audios
        else:
            print("no input text prompt for audio generation. or no audio generation model.")
        return audio_outputs

    def decode_mask(self, samples):
        if ("IMAGE_SAM" not in samples) or (self.mask_decoder_sam == None):
            print("no input image for seg. or no seg model.")
            return None
        # read image
        images = []
        if isinstance(samples["IMAGE_SAM"][0], list):
            images.append(samples["IMAGE_SAM"][-1][0])
        else:
            images.append(samples["IMAGE_SAM"][0])
        images = torch.stack(images, dim=0)
        images = images.to(self.mask_decoder_sam.device)
        # SAM encoder
        image_embeddings = self.get_visual_embs(images)
        
        # pre box by Grounding DINO
        outputs_det = self.decode_box(samples)
        if outputs_det == None:
            print("no object detected.")
            return None
        # outputs_bboxes对应原图大小，需要resize到IMAGE_SAM图像大小
        original_h, original_w = samples["Meta_info"]['original_shape'][0]
        sam_h, sam_w = samples["Meta_info"]['sam_shape'][0]
        for box_idx, box in enumerate(outputs_det["outputs_bboxes"][0]):
            outputs_det["outputs_bboxes"][0][box_idx][0] = box[0]/original_w*sam_w
            outputs_det["outputs_bboxes"][0][box_idx][1] = box[1]/original_h*sam_h
            outputs_det["outputs_bboxes"][0][box_idx][2] = box[2]/original_w*sam_w
            outputs_det["outputs_bboxes"][0][box_idx][3] = box[3]/original_h*sam_h

        multimask_output = False
        pred_masks = []
        for i in range(len(image_embeddings)):
            # get box of mask
            box_for_sam = None
            box_for_sam = outputs_det["outputs_bboxes"][i][0] # top1 box
            box_for_sam = box_for_sam.unsqueeze(0)
            box_for_sam = box_for_sam.to(image_embeddings[i].device)
            # start segment
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.mask_decoder_sam.prompt_encoder(
                points=None,
                boxes=box_for_sam,
                masks=None,
                text_embeds=None,
            )
            sparse_embeddings = sparse_embeddings.to(image_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.mask_decoder_sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.mask_decoder_sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.mask_decoder_sam.postprocess_masks(
                low_res_masks,
                input_size=(samples['Meta_info']['sam_shape'][i][0], samples['Meta_info']['sam_shape'][i][1]),
                original_size=(samples['Meta_info']['sam_shape'][i][0], samples['Meta_info']['sam_shape'][i][1]),
            )
            pred_masks.append(pred_mask[0])
        return pred_masks

    def decode_box(self, samples):
        if ("Image_ori_array" not in samples) or (grounding_dino_model == None):
            print("no input image for det. or no det model.")
            return None
        # text_prompt
        if "llm_text_res" in samples: # llm-text-control, for inference
            captions = samples['llm_text_res']
        else:
            print("no input text prompt for det.")
            return None
        # read image
        images = []
        if isinstance(samples["Image_ori_array"][0], list):
            images.append(samples["Image_ori_array"][-1][0])
        else:
            images.append(samples["Image_ori_array"][0])

        outputs_det = dict(
            outputs_bboxes=[],
            outputs_label_names=[],
            outputs_scores=[],
        )
        for i in range(len(images)):
            # inference
            if isinstance(images[i], torch.Tensor):
                res = inference_detector(grounding_dino_model, np.array(images[i].cpu()), text_prompt=captions[i])
            else:
                res = inference_detector(grounding_dino_model, images[i], text_prompt=captions[i])
            # print(res)
            # res.pred_instances.bboxes # [300, 4], all boxes. The size of the box is the same as the original image
            # res.pred_instances.bboxes[0] # the top1 box, tensor([ 185.9219, 69.1232, 1067.4197, 1093.7174])
            # pred_score_thr: filter out the low score pre box
            res_bboxes = []
            res_label_names = []
            res_scores = []
            pred_score_thr = 0.3
            for box_idx, score in enumerate(res.pred_instances.scores):
                if score < pred_score_thr:
                    break
                res_bboxes.append(res.pred_instances.bboxes[box_idx])
                res_label_names.append(res.pred_instances.label_names[box_idx])
                res_scores.append(res.pred_instances.scores[box_idx])
            outputs_det["outputs_bboxes"].append(res_bboxes)
            outputs_det["outputs_label_names"].append(res_label_names)
            outputs_det["outputs_scores"].append(res_scores)
        return outputs_det



    ##############################################
    #                 Generating                 #
    ##############################################
    def get_llm_text_res(self, string, modality):
        # llm_text_res = self.get_llm_text_res("<MASK>apple</MASK>", "MASK")
        # llm_text_res: apple
        # 构建正则表达式模式
        pattern = rf"<{modality}>(.*?)</{modality}>"
        # 使用正则表达式查找所有匹配的内容
        matches = re.findall(pattern, string)
        # 返回所有匹配的内容
        return matches

    def get_llm_text_modality(self, string, modality_keys):
        # 输入：string = "<IMAGE>a</IMAGE><VIDEO>b</VIDEO><AUDIO>c</AUDIO>", modality_keys = ["IMAGE", "VIDEO", "AUDIO", "MASK", "BOX"]
        # 输出：["IMAGE", "VIDEO", "AUDIO"]，即提取modality_keys中出现在string中的字符串列表
        # 初始化一个空列表来存储匹配的模态
        matched_modalities = []
        # 遍历所有的模态键
        for modality in modality_keys:
            # 构建正则表达式模式来匹配 <MODALITY>...</MODALITY>
            pattern = rf"<{modality}>.*?</{modality}>"
            # 使用正则表达式查找匹配的内容
            if re.search(pattern, string):
                matched_modalities.append(modality)
        # 返回所有匹配的模态
        return matched_modalities


    @torch.no_grad()
    def generate(self, samples, answers, predictions, predictions_text):
        output_texts = samples['llm_text_all'][0]
        # get modality from llm output texts
        modality_list = self.get_llm_text_modality(output_texts, self.decode_modality.keys())
        for modality in modality_list:
            llm_text_res_list = self.get_llm_text_res(output_texts, modality) # ['caption1', 'caption2']
            modality_i = 0
            for llm_text_res in llm_text_res_list:
                samples['llm_text_res'] = [llm_text_res]
                modality_i += 1
                # gather predictions_text
                predictions_text[modality].append(llm_text_res)
                # gather predictions
                if modality == 'IMAGE':
                    preds = self.decode_modality[modality](samples)
                    if preds is not None:
                        predictions[modality].append(preds[0])
                elif modality == 'VIDEO':
                    preds = self.decode_modality[modality](samples)
                    if preds is not None:
                        predictions[modality].append(preds)
                elif modality == 'AUDIO':
                    preds = self.decode_modality[modality](samples)
                    if preds is not None:
                        predictions[modality].append(preds[0])
                elif modality == 'BOX':
                    outputs_det = self.decode_modality[modality](samples)
                    if outputs_det is not None:
                        predictions[modality]['bboxes'].append(outputs_det["outputs_bboxes"][0])
                        predictions[modality]['label_names'].append(outputs_det["outputs_label_names"][0])
                        predictions[modality]['scores'].append(outputs_det["outputs_scores"][0])
                        # DETR: boxes = tensor([[266.7737, 301.7450,  70.3573,  70.3573]], device='cuda:0')
                        # Grounding DINO:  size = [300, 4], all boxes
                elif modality == 'MASK':
                    preds = self.decode_modality[modality](samples)
                    if preds is not None:
                        predictions[modality].append(preds[0])
        answers.append(output_texts)
        return answers, predictions, predictions_text
