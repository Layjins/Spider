import re
import os
import logging
from typing import List
import gc
import random
import json
import ast

from transformers import StoppingCriteria, StoppingCriteriaList
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
from transformers import AutoTokenizer, CLIPModel

from copy import deepcopy


# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = torch.cuda.current_device()

# init CLIP for SAM
sam_clip_flag = False
if sam_clip_flag:
    CLIP_model = CLIPModel.from_pretrained("/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/clip-vit-base-patch32")
    CLIP_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/clip-vit-base-patch32")

# init Grounding DINO
# init_dino_flag = True
init_dino_flag = False
if init_dino_flag:
    config_file = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py"
    checkpoint_file = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"
    # init Grounding DINO for industry
    # config_file = "/nickccnie/checkpoints/general_det/grounding_dino_swin-t_NG_en/grounding_dino_swin-t_NG_en.py"
    # checkpoint_file = "/nickccnie/checkpoints/general_det/grounding_dino_swin-t_NG_en/epoch_12.pth"
    # grounding_dino_model = init_detector(config_file, checkpoint_file, device=device) # for inference, to speed-up inference time
    grounding_dino_model = init_detector(config_file, checkpoint_file, device='cpu') # for training, to save gpu memory
else:
    grounding_dino_model = None



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


@registry.register_model("spider")
class Spider(BaseModel):
    """LoRA for LLaMa model"""

    def __init__(self,
                 name="spider",
                 encoder_modules=dict(
                     imagebind_ckpt_path=None
                 ),
                 input_proj_modules=dict(
                     freeze_input_proj=False,
                 ),
                 use_embed_align_loss=False,
                 only_embed_align_loss=False,
                 word_align_loss=False,
                 only_llm_gen_loss=False,
                 llm_modules=dict(
                     vicuna_ckpt_path=None,
                     using_lora=False,
                     freeze_lm=True,
                     freeze_tokens=True,
                     lora_r=32,
                     lora_alpha=21,
                     lora_dropout=0.1,
                     lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
                 ),
                 tokenizer_modules=dict(
                     tokenizer_path=None,
                     new_modality_tokens={"IMAGE": 4, "VIDEO": 24, "AUDIO": 8, "MASK": 1, "BOX": 1},
                     new_special_tokens=['[INPUT]', '[OUTPUT]', '[END]', '[IMAGE]', '[VIDEO]', '[AUDIO]', '[BOX]', '[MASK]', '[SMARTMULTIMODAL]', '[SPECIFICMULTIMODAL]'],
                     bbox_bins=1000,
                 ),
                 system_prompt=None,
                 output_alignment_modules=dict(
                     IMAGE=dict(alignment_module_mode='transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                              alignment_input_tokens=4, alignment_output_tokens=77, alignment_output_dim=768),
                     VIDEO=dict(alignment_module_mode='transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                              alignment_input_tokens=24, alignment_output_tokens=77, alignment_output_dim=1024),
                     AUDIO=dict(alignment_module_mode='transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                              alignment_input_tokens=8, alignment_output_tokens=1, alignment_output_dim=512),
                     MASK=dict(alignment_module_mode='transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                               alignment_input_tokens=1, alignment_output_tokens=1, alignment_output_dim=256),
                     # BOX=dict(alignment_module_mode='transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                     #          alignment_input_tokens=1, alignment_output_tokens=1, alignment_output_dim=256),
                ),
                 diffusion_modules=dict(
                     IMAGE=dict(type="sd", ckpt='runwayml/stable-diffusion-v1-5'),
                     VIDEO=dict(type="vd", ckpt='cerspense/zeroscope_v2_576w'),
                     AUDIO=dict(type="ad", ckpt='cvssp/audioldm-l-full'),
                 ),
                 mask_decoder_modules=dict(
                     sam_path="/youtu_fuxi-team2-1/Public_Dataset/Pretrain_model/sam_vit_h_4b8939.pth",
                     freeze_mask_decoder=False,
                 ),
                 box_decoder_modules=dict(
                     config_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py',
                     checkpoint_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
                 ),
                 max_context_len=100,
                 ):
        super().__init__()

        # import pdb
        # pdb.set_trace()

        self.model_name = name
        self.modality_tokens = tokenizer_modules['new_modality_tokens']
        self.max_context_len = max_context_len
        self.device = torch.cuda.current_device()
        self.output_alignment_modules = output_alignment_modules
        self.box_decoder_modules = box_decoder_modules
        self.mask_decoder_modules = mask_decoder_modules
        self.use_embed_align_loss = use_embed_align_loss
        self.only_embed_align_loss = only_embed_align_loss
        self.word_align_loss = word_align_loss
        self.only_llm_gen_loss = only_llm_gen_loss
        self.using_lora = llm_modules['using_lora']
        self.freeze_lm = llm_modules['freeze_lm']
        self.freeze_tokens = llm_modules['freeze_tokens']

        # 无效代码，用来提高gpu利用率
        self.gpu_eff = True
        if self.gpu_eff:
            self.a = torch.randn([1, 1000, 500, 500]).float().to(self.device)
            self.b = torch.randn([1, 1000, 500, 500]).float().to(self.device)

        self.visual_encoder, self.visual_hidden_size = self.init_imagebind_encoder(**encoder_modules)
        self.llama_model = self.init_llm(**llm_modules)
        if self.using_lora:
            self.num_old_embed_tokens = self.llama_model.base_model.model.model.embed_tokens.weight.size()[0]
            self.old_embed_tokens = deepcopy(self.llama_model.base_model.model.model.embed_tokens.weight)
            self.old_lm_head = deepcopy(self.llama_model.base_model.model.lm_head.weight)
        else:
            self.num_old_embed_tokens = self.llama_model.model.embed_tokens.weight.size()[0]
            self.old_embed_tokens = deepcopy(self.llama_model.model.embed_tokens.weight)
            self.old_lm_head = deepcopy(self.llama_model.lm_head.weight)
        
        # import pdb
        # pdb.set_trace()

        self.llama_tokenizer, new_modality_idxs = self.init_tokenizer(**tokenizer_modules)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        # import pdb
        # pdb.set_trace()

        # input alignment
        self.llama_proj = self.init_input_llama_proj(
            in_dim=self.visual_hidden_size,
            out_dim=self.llama_model.config.hidden_size,
            **input_proj_modules)

        # output alignment
        self.output_alignment_MoE_mode = None
        self.reconstruct_loss = False
        MoE_modes = ['moe_transformer', 'moe_aligner']
        for modality, output_alignment_module in output_alignment_modules.items():
            if output_alignment_module['alignment_module_mode'] in MoE_modes:
                self.output_alignment_MoE_mode = output_alignment_module['alignment_module_mode']
                self.reconstruct_loss = output_alignment_module['reconstruct_loss']
                self.alignment_layer_moe = output_alignment_module['alignment_layer']
                self.freeze_output_alignment_proj_moe = output_alignment_module['freeze_output_alignment_proj']
                break

        if self.output_alignment_MoE_mode == None:
            self.alignment_projs = {}
            for modality, output_alignment_module in output_alignment_modules.items():
                logging.info(f'Initializing Text2{modality.lower().capitalize()} alignment proj ...')
                self.alignment_projs[modality] = \
                    self.init_output_alignment_proj(
                        llama_hidden_layers=self.llama_model.config.num_hidden_layers,
                        llama_hidden_size=self.llama_model.config.hidden_size,
                        **output_alignment_module)
                logging.info(f'Text2{modality.lower().capitalize()} alignment proj initialized.')
        else: # MOE mode
            logging.info(f'Initializing moe alignment proj ...')
            self.alignment_projs = self.init_output_alignment_proj_moe(
                        output_alignment_modules,
                        llama_hidden_layers=self.llama_model.config.num_hidden_layers,
                        llama_hidden_size=self.llama_model.config.hidden_size,
                        alignment_module_mode=self.output_alignment_MoE_mode,
                        reconstruct_loss=self.reconstruct_loss,
                        alignment_layer=self.alignment_layer_moe,
                        freeze_output_alignment_proj=self.freeze_output_alignment_proj_moe,)
            logging.info(f'moe alignment proj initialized.')

        # for clip align loss
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # CLIP output embedding alignment for SAM
        if sam_clip_flag:
            self.alignment_projs_CLIP_SAM = TextFcLayer(512, 256, num_input_tokens=1, num_output_tokens=1, mode='transformer', device=self.device)

        if self.model_name == "spider_story":
            self.sd_ckpt_path = None
            self.vd_ckpt_path = None
            self.ad_ckpt_path = None
            self.diffusion_pipes = None
            self.mask_decoder_sam = None

            self.encode_modality = dict(
                IMAGE=self.encode_image,
                VIDEO=self.encode_video,
                AUDIO=self.encode_audio,
            )
            self.decode_modality = dict()
            self.loss_modality = dict()
        else: # "spider"
            self.sd_ckpt_path = diffusion_modules['IMAGE']['ckpt']
            self.vd_ckpt_path = diffusion_modules['VIDEO']['ckpt']
            self.ad_ckpt_path = diffusion_modules['AUDIO']['ckpt']
            self.diffusion_pipes = self.init_diffusion_model(diffusion_modules)
            self.mask_decoder_sam = self.init_mask_decoder_sam(mask_decoder_modules)

            # init Grounding DINO
            # config_file = self.box_decoder_modules['config_file']
            # checkpoint_file = self.box_decoder_modules['checkpoint_file']
            # self.grounding_dino_model = init_detector(config_file, checkpoint_file, device=self.device)

            self.encode_modality = dict(
                IMAGE=self.encode_image,
                VIDEO=self.encode_video,
                AUDIO=self.encode_audio,
                # DEPTH=self.encode_depth,
                # THERMAL=self.encode_thermal,
                # SURFACE=self.encode_surface,
                # BOX=self.encode_box,
                # MASK=self.encode_mask,
                # POINT=self.encode_polygon
            )

            self.decode_modality = dict(
                IMAGE=self.decode_image,
                VIDEO=self.decode_video,
                AUDIO=self.decode_audio,
                MASK=self.decode_mask,
                BOX=self.decode_box,
            )

            self.loss_modality = dict(
                IMAGE=self.loss_image,
                VIDEO=self.loss_video,
                AUDIO=self.loss_audio,
                MASK=self.loss_mask,
                BOX=self.loss_box,
            )


    ##############################################
    #                                            #
    #          Encoder Side Modules              #
    #                                            #
    ##############################################
    def encode_image(self, input):
        # import pdb
        # pdb.set_trace()
        # convert into visual dtype
        inputs = {'vision': input.to(self.llama_model.dtype)}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision']  # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_video(self, input):
        # import pdb
        # pdb.set_trace()
        inputs = {'vision': input.to(self.llama_model.dtype)}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings['vision']  # bsz x 1024
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, input):
        # import pdb
        # pdb.set_trace()
        inputs = {'audio': input.to(self.llama_model.dtype)}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings['audio']  # bsz x 1024
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama


    #
    # def encode_point(self, point):
    #     pass
    # def encode_box(self, box):
    #     pass
    # def encode_polygon(self, polygon):
    #     pass
    # def encode_mask(self, mask):
    #     pass
    # def encode_depth(self, depth):
    #     pass
    # def encode_thermal(self, thermal):
    #     pass
    # def encode_surface(self, surface):
    #     pass


    ##############################################
    #                                            #
    #          Decoder Side Modules              #
    #                                            #
    ##############################################
    def decode_image(self, samples, image_hidden_embeds_list, image_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True, guidance_scale=7.5, num_inference_steps=40):
        if self.output_alignment_MoE_mode == None:
            assert "IMAGE" in self.alignment_projs, "IMAGE alignment projections do not exist !"
            alignment_projs = self.alignment_projs['IMAGE']
        else:
            alignment_projs = self.alignment_projs
        # for semantic-align loss and inference
        proj_image_hidden_embeds_list = []
        rec_loss_list = []
        for layer_idx, fc_layer in enumerate(alignment_projs):
            ##### modify the fc_layer to MoE-like module #####
            # fuse hidden_embeds and hidden_embeds_text
            # torch.mean(input_embeds_text_list[layer_idx], dim=1, keepdim=True) # (batch_size, 1, dim)
            image_hidden_embeds = image_hidden_embeds_list[layer_idx] + image_input_embeds_list[layer_idx] # (batch_size, new_modality_tokens["IMAGE"]=1, dim)
            hidden_embeds_text = hidden_embeds_text_list[layer_idx] + input_embeds_text_list[layer_idx] # (batch_size, text_token_len, dim)
            # hidden_embeds_text = hidden_embeds_text.mean(dim=1, keepdim=True) # (batch_size, 1, dim)
            # hidden_embeds = image_hidden_embeds + hidden_embeds_text # (batch_size, new_modality_tokens["IMAGE"]=1 + text_token_len=1, dim)
            hidden_embeds = image_hidden_embeds
            # hidden_embeds = hidden_embeds_text
            # hidden_embeds = torch.cat([image_hidden_embeds, hidden_embeds_text], dim=1) # (batch_size, new_modality_tokens["IMAGE"]=1 + text_token_len, dim)

            # project hidden_embeds to the task-encoder space
            if self.reconstruct_loss:
                proj_emb, rec_proj_emb = fc_layer(hidden_embeds, modality='IMAGE')
                # reconstruction loss
                rec_loss = self.l2_loss(rec_proj_emb, hidden_embeds.detach())
                rec_loss = rec_loss.mean()
                # clip loss
                rec_clip_loss = self.clip_align_loss(rec_proj_emb, hidden_embeds.detach())
                rec_clip_loss = rec_clip_loss.mean()
                rec_loss = rec_loss + rec_clip_loss
                rec_loss_list.append(rec_loss)
                # project emb
                proj_image_hidden_embeds_list.append(proj_emb)
            else:
                # proj_image_hidden_embeds_list.append(fc_layer(image_hidden_embeds, modality='IMAGE') + fc_layer(hidden_embeds_text, modality='IMAGE')) # loss=nan
                # proj_image_hidden_embeds_list.append(fc_layer(hidden_embeds, modality='IMAGE'))
                # proj_image_hidden_embeds_list.append(fc_layer(image_hidden_embeds, modality='IMAGE'))
                proj_image_hidden_embeds_list.append(fc_layer(hidden_embeds, modality='IMAGE'))
        proj_image_hidden_embeds = torch.stack(proj_image_hidden_embeds_list, dim=-1).sum(dim=-1)  # (N, 77, 768)
        # proj_image_hidden_embeds = proj_image_hidden_embeds / proj_image_hidden_embeds.norm(dim=-1, keepdim=True)
        if self.reconstruct_loss:
            rec_losses = torch.stack(rec_loss_list, dim=-1).sum(dim=-1)

        # for embed-align loss
        # if "Caption" in samples: # for training
        #     captions = samples['Caption']
        #     captions_tokens = self.llama_tokenizer(captions, return_tensors="pt", add_special_tokens=False).to(self.device)
        #     captions_embeds = self.embed_tokens(captions_tokens.input_ids, using_lora=self.using_lora)
        #     # import pdb
        #     # pdb.set_trace()

        #     # projection
        #     proj_image_captions_embeds_list = []
        #     for layer_idx, fc_layer in enumerate(alignment_projs):
        #         proj_image_captions_embeds_list.append(fc_layer(captions_embeds, modality='IMAGE'))
        #     proj_image_captions_embeds = torch.stack(proj_image_captions_embeds_list, dim=-1).sum(dim=-1)  # (N, 77, 768)


        # for training
        if return_embeds_only:
            if self.reconstruct_loss:
                return proj_image_hidden_embeds, rec_losses
            else:
                return proj_image_hidden_embeds
            # if "Caption" in samples: # for training
            #     return proj_image_hidden_embeds, proj_image_captions_embeds
            # else:
            #     return proj_image_hidden_embeds
            

        # generation, for inference
        # hidden_embeds_scale = 0.0 # when hidden_embeds_scale = 0, only using condiction_embeds; when hidden_embeds_scale = 1, not using condiction_embeds
        hidden_embeds_scale = 0.1
        if "Caption" in samples: # text-control generation, for inference
            # text
            captions = samples['Caption']
            image_diffusion_pipe = self.diffusion_pipes['IMAGE']
            condiction_embeds = image_diffusion_pipe(captions, return_prompts_only=True).detach() # get text embedding by the diffusioin text-encoder
            # fuse
            # proj_image_hidden_embeds = proj_image_hidden_embeds
            # proj_image_hidden_embeds = condiction_embeds.to(self.device)
            proj_image_hidden_embeds = hidden_embeds_scale * proj_image_hidden_embeds + (1 - hidden_embeds_scale) * (condiction_embeds.to(self.device))
            # if return_embeds_only:
            #     return proj_image_hidden_embeds
            # generate image by text embedding
            generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path, torch_dtype=torch.float16).to(self.device)
            image_outputs = generation_model(prompt_embeds=proj_image_hidden_embeds,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps).images
        elif "llm_text_res" in samples: # llm-text-control generation, for inference
            # text
            captions = samples['llm_text_res']
            image_diffusion_pipe = self.diffusion_pipes['IMAGE']
            condiction_embeds = image_diffusion_pipe(captions, return_prompts_only=True).detach()
            # fuse
            # proj_image_hidden_embeds = proj_image_hidden_embeds # bad
            # proj_image_hidden_embeds = condiction_embeds.to(self.device) # good
            proj_image_hidden_embeds = hidden_embeds_scale * proj_image_hidden_embeds + (1 - hidden_embeds_scale) * (condiction_embeds.to(self.device)) # good
            # if return_embeds_only:
            #     return proj_image_hidden_embeds
            # generate image by text embedding
            generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path, torch_dtype=torch.float16).to(self.device)
            image_outputs = generation_model(prompt_embeds=proj_image_hidden_embeds,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps).images
        else: # llm-control generation, for inference
            # if return_embeds_only:
            #     return proj_image_hidden_embeds
            # generate image by llm embedding
            generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path, torch_dtype=torch.float16).to(self.device)
            image_outputs = generation_model(prompt_embeds=proj_image_hidden_embeds,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps).images
        return image_outputs

    def decode_video(self, samples, video_hidden_embeds_list, video_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True,
                     guidance_scale=7.5, num_inference_steps=40, height=320, width=576, num_frames=16):
        if self.output_alignment_MoE_mode == None:
            assert "VIDEO" in self.alignment_projs, "VIDEO alignment projections do not exist !"
            alignment_projs = self.alignment_projs['VIDEO']
        else:
            alignment_projs = self.alignment_projs
        proj_video_hidden_embeds_list = []
        for layer_idx, fc_layer in enumerate(alignment_projs):
            proj_video_hidden_embeds_list.append(
                fc_layer(video_hidden_embeds_list[layer_idx] + video_input_embeds_list[layer_idx], modality='VIDEO'))
        proj_video_hidden_embeds = torch.stack(proj_video_hidden_embeds_list, dim=-1).sum(dim=-1)  # (N, 77, 768)

        if return_embeds_only:
            return proj_video_hidden_embeds

        # hidden_embeds_scale = 0.0 # when hidden_embeds_scale = 0, only using condiction_embeds; when hidden_embeds_scale = 1, not using condiction_embeds
        hidden_embeds_scale = 0.1
        if "llm_text_res" in samples: # llm-text-control generation, for inference
            # text
            captions = samples['llm_text_res']
            video_diffusion_pipe = self.diffusion_pipes['VIDEO']
            condiction_embeds = video_diffusion_pipe(captions, return_prompts_only=True).detach()
            # fuse
            proj_video_hidden_embeds = hidden_embeds_scale * proj_video_hidden_embeds + (1 - hidden_embeds_scale) * (condiction_embeds.to(self.device))
            generation_model = TextToVideoSDPipeline.from_pretrained(self.vd_ckpt_path, torch_dtype=torch.float16).to(self.device)
            video_outputs = generation_model(prompt_embeds=proj_video_hidden_embeds,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps, height=height,
                                         width=width, num_frames=num_frames).frames
        else:
            generation_model = TextToVideoSDPipeline.from_pretrained(self.vd_ckpt_path, torch_dtype=torch.float16).to(self.device)
            video_outputs = generation_model(prompt_embeds=proj_video_hidden_embeds,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps, height=height,
                                         width=width, num_frames=num_frames).frames
        return video_outputs

    def decode_audio(self, samples, audio_hidden_embeds_list, audio_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True,
                     guidance_scale=7.5, num_inference_steps=40, audio_length_in_s=5.0):
        if self.output_alignment_MoE_mode == None:
            assert "AUDIO" in self.alignment_projs, "AUDIO alignment projections do not exist !"
            alignment_projs = self.alignment_projs['AUDIO']
        else:
            alignment_projs = self.alignment_projs
        proj_audio_hidden_embeds_list = []
        for layer_idx, fc_layer in enumerate(alignment_projs):
            proj_audio_hidden_embeds_list.append(
                fc_layer(audio_hidden_embeds_list[layer_idx] + audio_input_embeds_list[layer_idx], modality='AUDIO'))
        proj_audio_hidden_embeds = torch.stack(proj_audio_hidden_embeds_list, dim=-1).sum(dim=-1)  # (N, 77, 768)

        if return_embeds_only:
            return proj_audio_hidden_embeds

        # hidden_embeds_scale = 0.0 # when hidden_embeds_scale = 0, only using condiction_embeds; when hidden_embeds_scale = 1, not using condiction_embeds
        hidden_embeds_scale = 0.1
        if "llm_text_res" in samples: # llm-text-control generation, for inference
            # text
            captions = samples['llm_text_res']
            audio_diffusion_pipe = self.diffusion_pipes['AUDIO']
            condiction_embeds = audio_diffusion_pipe(captions, return_prompts_only=True).detach()
            # fuse
            proj_audio_hidden_embeds = hidden_embeds_scale * proj_audio_hidden_embeds + (1 - hidden_embeds_scale) * (condiction_embeds.to(self.device))
            generation_model = AudioLDMPipeline.from_pretrained(self.ad_ckpt_path, torch_dtype=torch.float16).to(self.device)
            audio_outputs = generation_model(prompt_embeds=proj_audio_hidden_embeds,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps,
                                            audio_length_in_s=audio_length_in_s).audios
        else:    
            generation_model = AudioLDMPipeline.from_pretrained(self.ad_ckpt_path, torch_dtype=torch.float16).to(self.device)
            audio_outputs = generation_model(prompt_embeds=proj_audio_hidden_embeds,
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=num_inference_steps,
                                            audio_length_in_s=audio_length_in_s).audios
        return audio_outputs

    def decode_mask(self, samples, image_hidden_embeds_list, image_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True):
        if self.output_alignment_MoE_mode == None:
            assert "MASK" in self.alignment_projs, "MASK alignment projections do not exist !"
            alignment_projs = self.alignment_projs['MASK']
        else:
            alignment_projs = self.alignment_projs
        proj_image_hidden_embeds_list = []
        for layer_idx, fc_layer in enumerate(alignment_projs):
            proj_image_hidden_embeds_list.append(fc_layer(image_hidden_embeds_list[layer_idx] + image_input_embeds_list[layer_idx], modality='MASK'))
        proj_image_hidden_embeds = torch.stack(proj_image_hidden_embeds_list, dim=-1).sum(dim=-1)  # (N, 1, 256)

        # proj_image_hidden_embeds 和CLIP输出特征对齐
        # get_text_features by CLIP
        if "Caption" in samples: # text-control generation, for training
            captions = samples['Caption']
        elif "llm_text_res" in samples: # llm-text-control, for inference
            captions = samples['llm_text_res']
        if sam_clip_flag:
            inputs = CLIP_tokenizer(captions, padding=True, return_tensors="pt")#.to(self.device)
            text_features = CLIP_model.get_text_features(**inputs).detach() # text_features.size() = torch.Size([N, 512])
            # alignment to SAM decoder
            text_features = text_features.unsqueeze(1).to(self.device) # text_features.size() = torch.Size([N, 1, 512])
            proj_text_features = self.alignment_projs_CLIP_SAM(text_features) # torch.Size([N, 1, 512])
            # feature fusion
            # proj_image_hidden_embeds = proj_text_features
            proj_image_hidden_embeds = proj_image_hidden_embeds + proj_text_features

        # segmentation
        images = []
        for idx in range(len(samples['Question'])):
            if isinstance(samples["IMAGE_SAM"][idx], list):
                images.append(samples["IMAGE_SAM"][-1][idx])
            else:
                images.append(samples["IMAGE_SAM"][idx])
        images = torch.stack(images, dim=0)
        # SAM encoder
        image_embeddings = self.get_visual_embs(images)
        
        # pre box by Grounding DINO
        if return_embeds_only == True:
            proj_image_hidden_embeds_box = self.decode_box(samples, image_hidden_embeds_list, image_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
        else:
            outputs_bboxes, outputs_label_names, outputs_scores = self.decode_box(samples, image_hidden_embeds_list, image_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
            # outputs_bboxes对应原图大小，需要resize到IMAGE_SAM图像大小
            original_h, original_w = samples["Meta_info"]['original_shape'][0]
            sam_h, sam_w = samples["Meta_info"]['sam_shape'][0]
            for box_idx, box in enumerate(outputs_bboxes[0]):
                outputs_bboxes[0][box_idx][0] = box[0]/original_w*sam_w
                outputs_bboxes[0][box_idx][1] = box[1]/original_h*sam_h
                outputs_bboxes[0][box_idx][2] = box[2]/original_w*sam_w
                outputs_bboxes[0][box_idx][3] = box[3]/original_h*sam_h


        multimask_output = False
        pred_masks = []
        for i in range(len(proj_image_hidden_embeds)):
            # get box of mask
            box_for_sam = None
            if "BOX_of_MASK" in samples: # for training
                box_for_sam = samples['BOX_of_MASK'][i]
                box_for_sam = box_for_sam.unsqueeze(0)
                # box_for_sam = box_for_sam.to(proj_image_hidden_embeds[i].device)
            elif return_embeds_only == False: # for inference
                box_for_sam = outputs_bboxes[i][0] # top1 box
                box_for_sam = box_for_sam.unsqueeze(0)
                box_for_sam = box_for_sam.to(proj_image_hidden_embeds[i].device)

            # start segment
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.mask_decoder_sam.prompt_encoder(
                points=None,
                boxes=box_for_sam,
                masks=None,
                text_embeds=proj_image_hidden_embeds[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(proj_image_hidden_embeds[i].dtype)
            low_res_masks, iou_predictions = self.mask_decoder_sam.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.mask_decoder_sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            # import pdb
            # pdb.set_trace()
            pred_mask = self.mask_decoder_sam.postprocess_masks(
                low_res_masks,
                input_size=(samples['Meta_info']['sam_shape'][i][0], samples['Meta_info']['sam_shape'][i][1]),
                original_size=(samples['Meta_info']['sam_shape'][i][0], samples['Meta_info']['sam_shape'][i][1]),
            )
            # import pdb
            # pdb.set_trace()
            pred_masks.append(pred_mask[0])

        gc.collect()
        if return_embeds_only == True:
            return pred_masks, proj_image_hidden_embeds_box
        else:
            return pred_masks

    def decode_box(self, samples, image_hidden_embeds_list, image_input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True):
        if return_embeds_only: # don't train the Grounding DINO
            return 0

        # init Grounding DINO
        # config_file = self.box_decoder_modules['config_file']
        # checkpoint_file = self.box_decoder_modules['checkpoint_file']
        # init Grounding DINO for industry
        # config_file = "/nickccnie/checkpoints/general_det/grounding_dino_swin-t_NG_en/grounding_dino_swin-t_NG_en.py"
        # checkpoint_file = "/nickccnie/checkpoints/general_det/grounding_dino_swin-t_NG_en/epoch_12.pth"
        # self.grounding_dino_model = init_detector(config_file, checkpoint_file, device=self.device)

        # if self.output_alignment_MoE_mode == None:
        #     assert "BOX" in self.alignment_projs, "BOX alignment projections do not exist !"
        #     alignment_projs = self.alignment_projs['BOX']
        # else:
        #     alignment_projs = self.alignment_projs
        # proj_image_hidden_embeds_list = []
        # for layer_idx, fc_layer in enumerate(alignment_projs):
        #     proj_image_hidden_embeds_list.append(fc_layer(image_hidden_embeds_list[layer_idx] + image_input_embeds_list[layer_idx], modality='BOX'))
        # proj_image_hidden_embeds = torch.stack(proj_image_hidden_embeds_list, dim=-1).sum(dim=-1)  # (N, 1, 256)

        # if return_embeds_only:
        #     return proj_image_hidden_embeds
        # from transformers import logging
        # logging.set_verbosity_error() # 取消DINO不完整加载模型的warning
        
        # detection
        if "Caption" in samples: # text-control generation, for training
            # text_prompt
            captions = samples['Caption']
            # if return_embeds_only:
            #     return proj_image_hidden_embeds
        elif "llm_text_res" in samples: # llm-text-control, for inference
            # text_prompt
            captions = samples['llm_text_res']
            # if return_embeds_only:
            #     return proj_image_hidden_embeds

        # decode, for inference
        images = []
        for idx in range(len(samples['Question'])):
            if isinstance(samples["Image_ori_array"][idx], list):
                images.append(samples["Image_ori_array"][-1][idx])
            else:
                images.append(samples["Image_ori_array"][idx])

        # import pdb
        # pdb.set_trace()
        outputs_bboxes = []
        outputs_label_names = []
        outputs_scores = []
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
            outputs_bboxes.append(res_bboxes)
            outputs_label_names.append(res_label_names)
            outputs_scores.append(res_scores)
        return outputs_bboxes, outputs_label_names, outputs_scores




    ##############################################
    #                                            #
    #          Preparing_Input Embedding         #
    #                                            #
    ##############################################
    def split_placeholder(self, string):
        # input: <IMAGE><IMAGE-Placeholder></IMAGE> a dog
        # output: ['<IMAGE>', '<IMAGE-Placeholder>', '</IMAGE> a dog']

        pattern = r'<[A-Z]+-Placeholder>'
        # 找到所有匹配的标签及其位置
        matches = list(re.finditer(pattern, string))
        # 根据匹配的位置将字符串切分
        split_list = []
        start = 0
        for match in matches:
            split_list.append(string[start:match.start()])
            split_list.append(match.group())
            start = match.end()
        split_list.append(string[start:])
        return split_list

    def get_modality(self, string):
        match = re.search(r'<([A-Z]+)-Placeholder>', string)
        return match.group(1)

    def get_llm_text_res_20240810(self, string, modality):
        # llm_text_res = self.get_llm_text_res("[OUTPUT]<MASK>[MASK0]</MASK>apple[END]", "MASK")
        # llm_text_res: apple
        pattern = '</' + modality + '>(.*?)\[END\]'
        match = re.search(pattern, string)
        if match:
            return match.group(1).strip()
        else:
            return None
        
    def get_llm_text_res(self, string, modality):
        # llm_text_res = self.get_llm_text_res("[OUTPUT]<MASK>apple[MASK0]</MASK>[END]", "MASK")
        # llm_text_res: apple
        # 构建正则表达式模式
        pattern = rf"<{modality}>(.*?)\[{modality}0\]"
        # 使用正则表达式查找所有匹配的内容
        matches = re.findall(pattern, string)
        # 返回所有匹配的内容
        return matches

    def get_llm_text_modality(self, string, modality_keys):
        # 输入：string = "<IMAGE>a[IMAGE0]</IMAGE><VIDEO>b[VIDEO0]</VIDEO><AUDIO>c[AUDIO0]</AUDIO>", modality_keys = ["IMAGE", "VIDEO", "AUDIO", "MASK", "BOX"]
        # 输出：["IMAGE", "VIDEO", "AUDIO"]，即提取modality_keys中出现在string中的字符串列表
        # 初始化一个空列表来存储匹配的模态
        matched_modalities = []
        # 遍历所有的模态键
        for modality in modality_keys:
            # 构建正则表达式模式
            pattern = rf"<{modality}>(.*?)\[{modality}0\]"
            # 使用正则表达式查找匹配的内容
            if re.search(pattern, string):
                matched_modalities.append(modality)
        # 返回所有匹配的模态
        return matched_modalities

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
        """ 提取 General Prompt、Prompt Array、Style Name """
        # 提取 General Prompt（忽略单双引号）
        general_prompt_match = re.search(r"<GENERALPROMPT>\s*(.*?)\s*</GENERALPROMPT>", output_texts, re.DOTALL)
        general_prompt = general_prompt_match.group(1).strip() if general_prompt_match else ""
        # 提取 Prompt Array（适配各种格式）
        prompt_array_match = re.search(r"<PROMPTARRAY>\s*(.*?)\s*</PROMPTARRAY>", output_texts, re.DOTALL)
        prompt_array_str = prompt_array_match.group(1).strip() if prompt_array_match else "[]"
        prompt_array = clean_prompt_array(prompt_array_str)
        # 提取 Style Name
        style_name_match = re.search(r"<STYLENAME>\s*(.*?)\s*</STYLENAME>", output_texts, re.DOTALL)
        style_name = style_name_match.group(1).strip() if style_name_match else ""
        return general_prompt, prompt_array, style_name

    def concat_embed_question_answer(self, question_embeds, question_atts, answer_embeds, answer_atts):
        question_lens = []
        question_answer_embeds = []
        question_answer_atts = []
        for i in range(question_embeds.size(0)):
            input_len = question_atts[i].sum()
            question_lens.append(input_len)
            question_answer_embeds.append(
                torch.cat([
                    question_embeds[i][:input_len],
                    answer_embeds[i],
                    question_embeds[i][input_len:]
                ])
            )
            question_answer_atts.append(
                torch.cat([
                    question_atts[i][:input_len],
                    answer_atts[i],
                    question_atts[i][input_len:]
                ])
            )
        question_answer_embeds = torch.stack(question_answer_embeds)
        question_answer_atts = torch.stack(question_answer_atts)
        return question_answer_embeds, question_answer_atts, question_lens

    def preparing_input_embedding(self, samples):
        # import pdb
        # pdb.set_trace()

        # preparing question embedding
        question_embeds = []
        for idx, question in enumerate(samples['Question']):
            frequency = 0
            question_embed = []
            question_splits = self.split_placeholder(question)
            question_splits.insert(0, '[INPUT]')
            question_splits.append(samples['TaskPrompt'][idx])
            if 'SystemPrompt' in samples:
                question_splits.append(samples['SystemPrompt'][idx])
            for question_split in question_splits:
                if "Placeholder" not in question_split:
                    question_split_tokens = self.llama_tokenizer(question_split, return_tensors="pt",
                                                                 add_special_tokens=False).to(self.device)
                    question_split_embed = self.embed_tokens(question_split_tokens.input_ids, using_lora=self.using_lora)  # 1, s1, c
                else:
                    modality = self.get_modality(question_split)
                    if isinstance(samples[modality][idx], list):
                        modality_input = samples[modality][frequency][idx][None]
                        frequency += 1
                    else:
                        modality_input = samples[modality][idx][None]
                    question_split_embed, _ = self.encode_modality[modality](modality_input)  # 1, s2, c

                question_embed.append(question_split_embed)
            question_embed = torch.cat(question_embed, dim=1)  # 1 x (1+s1+1+s2) x embed_dim
            question_embeds.append(question_embed)

        question_embed_lens = [question_embed.shape[1] for question_embed in question_embeds]
        pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=self.device), using_lora=self.using_lora)

        max_length = max(question_embed_lens) if max(question_embed_lens) < self.max_context_len else self.max_context_len
        wrapped_question_embeds = pad_emb.expand(len(question_embed_lens), max_length, -1).clone()
        wrapped_question_atts = torch.zeros([len(question_embed_lens), max_length], dtype=torch.int, device=self.device)

        for i, question_embed in enumerate(question_embeds):
            length = question_embed_lens[i] if question_embed_lens[i] < self.max_context_len else self.max_context_len
            wrapped_question_embeds[i, :length] = question_embed[:, :length]
            wrapped_question_atts[i, :length] = 1


        # preparing answer embedding
        # import pdb
        # pdb.set_trace()
        answers = []
        for idx, answer in enumerate(samples['Answer']):
            answer_splits = self.split_placeholder(answer)
            # ['<IMAGE>a dog', '<IMAGE-Placeholder>', '</IMAGE><VIDEO>a dog', '<VIDEO-Placeholder>', '</VIDEO><AUDIO>a dog', '<AUDIO-Placeholder>', '</AUDIO>']
            answer_splits.insert(0, '[OUTPUT]')
            answer_splits.append('[END]')
            for split_idx, answer_split in enumerate(answer_splits):
                if "Placeholder" in answer_split:
                    modality = self.get_modality(answer_split)
                    answer_splits[split_idx] = ''.join([f'[{modality}{i}]' for i in range(self.modality_tokens[modality])])
            answers.append(''.join(answer_splits))

        answer_tokens = self.llama_tokenizer(answers, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_context_len, add_special_tokens=False).to(self.device)

        answer_token_ids = answer_tokens.input_ids
        answer_atts = answer_tokens.attention_mask
        answer_embeds = self.embed_tokens(answer_token_ids, using_lora=self.using_lora)


        # concat question and answer
        question_answer_embeds, question_answer_atts, question_lens = \
            self.concat_embed_question_answer(wrapped_question_embeds, wrapped_question_atts,
                                              answer_embeds, answer_atts)


        # add bos token at the begining
        batch_size = question_answer_embeds.shape[0]
        bos_tokens = torch.ones([batch_size, 1], dtype=answer_token_ids.dtype,
                                device=self.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        bos_embeds = self.embed_tokens(bos_tokens, using_lora=self.using_lora)
        bos_atts = wrapped_question_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, question_answer_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, question_answer_atts], dim=1)

        # import pdb
        # pdb.set_trace()

        # preparing target
        part_targets = answer_token_ids.masked_fill(answer_token_ids == self.llama_tokenizer.pad_token_id, -100)
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]], dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, question_lens[i] + 1:question_lens[i] + len(target) + 1] = target  # plus 1 for bos

        return inputs_embeds, attention_mask, targets

    ##############################################
    #                                            #
    #                   Forward                  #
    #                                            #
    ##############################################
    def forward(self, samples):
        # logging.info('222222222222222222222222222222222222')
        # for k, v in samples.items():
        #     if k in ['IMAGE', 'VIDEO', "AUDIO"]:
        #         logging.info(v.shape)
        #     else:
        #         logging.info(v)
        # import pdb
        # pdb.set_trace()

        # embed_align_mse_loss to align the text-encoders of llm and tasks
        if "Caption" in samples:
            captions = samples['Caption']
            if self.use_embed_align_loss:
                embed_align_mse_loss = self.loss_text_encoder_align(captions)
            else:
                embed_align_mse_loss = 0
            # word_align_loss = True # Local text alignment data，是Global text alignment data长文本中随机采样的words
            if self.word_align_loss:
                word_count = len(re.findall(r'\b\w+\b', captions[0]))
                select_word_num = int(word_count/2)
                if select_word_num < 1: 
                    select_word_num = 1
                word_align_mse_loss_total = 0
                for i in range(select_word_num):
                    captions_word = []
                    for cap_i, caption in enumerate(captions):
                        words = re.findall(r'\b\w+\b', caption)
                        random_word = random.choice(words) # 长文本中随机采样word
                        captions_word.append(random_word)
                    word_align_mse_loss = self.loss_text_encoder_align(captions_word)
                    word_align_mse_loss_total = word_align_mse_loss_total + word_align_mse_loss
                embed_align_mse_loss = (embed_align_mse_loss + word_align_mse_loss_total / (float(select_word_num))) / 2.0
            if self.only_embed_align_loss:
                loss = embed_align_mse_loss
                return {"loss": loss, "gen_loss": loss, "gen_acc": 0}

        inputs_embeds, attention_mask, targets = self.preparing_input_embedding(samples)
        # print(inputs_embeds.shape)
        # print(targets.shape)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                output_hidden_states=True,
                # reduction=reduction
            )
        # import pdb
        # pdb.set_trace()
            # outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=targets, output_hidden_states=True,)
        gen_loss = outputs.loss

        gen_acc = self.cal_acc(outputs, targets)

        # calculate decoder loss
        if self.only_llm_gen_loss == False:
            if samples['TaskPrompt'][0] == "[IMAGE]":
                # import pdb
                # pdb.set_trace()
                modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds(samples, outputs, targets)
                rec_loss = 0
                if self.reconstruct_loss:
                    hidden_embeds, rec_loss = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
                else:
                    hidden_embeds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
                semantic_align_mse_loss = self.loss_modality[modality](hidden_embeds, samples)
                loss = gen_loss + semantic_align_mse_loss + embed_align_mse_loss + rec_loss
                return {"loss": loss, "gen_loss": gen_loss, "IMAGE_semantic_align_mse_loss": semantic_align_mse_loss, "IMAGE_embed_align_mse_loss": embed_align_mse_loss, "rec_loss": rec_loss, "gen_acc": gen_acc}

            if samples['TaskPrompt'][0] == "[VIDEO]":
                # import pdb
                # pdb.set_trace()
                modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds(samples, outputs, targets)
                hidden_embeds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
                mse_loss = self.loss_modality[modality](hidden_embeds, samples)
                loss = gen_loss + mse_loss + embed_align_mse_loss
                return {"loss": loss, "gen_loss": gen_loss, "VIDEO_mse_loss": mse_loss, "VIDEO_embed_align_mse_loss": embed_align_mse_loss, "gen_acc": gen_acc}

            if samples['TaskPrompt'][0] == "[AUDIO]":
                # import pdb
                # pdb.set_trace()
                modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds(samples, outputs, targets)
                hidden_embeds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
                mse_loss = self.loss_modality[modality](hidden_embeds, samples)
                loss = gen_loss + mse_loss + embed_align_mse_loss
                return {"loss": loss, "gen_loss": gen_loss, "AUDIO_mse_loss": mse_loss, "AUDIO_embed_align_mse_loss": embed_align_mse_loss, "gen_acc": gen_acc}

            if samples['TaskPrompt'][0] == "[MASK]":
                # import pdb
                # pdb.set_trace()
                modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds(samples, outputs, targets)
                pred_masks, hidden_embeds_box = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
                mask_bce_loss, mask_dice_loss = self.loss_modality[modality](pred_masks, samples)
                mask_loss = mask_bce_loss + mask_dice_loss
                # loss = gen_loss + mask_loss
                # mse_loss = self.loss_modality['BOX'](hidden_embeds_box, samples)
                # mse_loss = 0
                # loss = gen_loss + mask_loss + mse_loss
                # return {"loss": loss, "gen_loss": gen_loss, "bce_loss": mask_bce_loss, "dice_loss": mask_dice_loss, "BOX_mse_loss": mse_loss, "gen_acc": gen_acc}
                loss = gen_loss + mask_loss + embed_align_mse_loss
                return {"loss": loss, "gen_loss": gen_loss, "bce_loss": mask_bce_loss, "dice_loss": mask_dice_loss, "MASK_embed_align_mse_loss": embed_align_mse_loss, "gen_acc": gen_acc}

            if samples['TaskPrompt'][0] == "[BOX]":
                # modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds(samples, outputs, targets)
                # hidden_embeds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=True)
                # mse_loss = self.loss_modality[modality](hidden_embeds, samples)
                # mse_loss = 0
                # loss = gen_loss + mse_loss
                # return {"loss": loss, "gen_loss": gen_loss, "BOX_mse_loss": mse_loss, "gen_acc": gen_acc}
                loss = gen_loss + embed_align_mse_loss

                # 无效代码，用来提高gpu利用率
                if self.gpu_eff:
                    for i in range(200):
                        c = torch.matmul(self.a, self.b)

                return {"loss": loss, "gen_loss": gen_loss, "BOX_embed_align_mse_loss": embed_align_mse_loss, "gen_acc": gen_acc}

        loss = gen_loss
        gc.collect()
        return {"loss": loss, "gen_loss": gen_loss, "gen_acc": gen_acc}


    ##############################################
    #                                            #
    #         Preparing Output Embedding         #
    #                                            #
    ##############################################
    def preparing_output_embeds(self, samples, outputs, targets, modality=None, modality_i=0):
        # "<IMAGE>a[IMAGE0]</IMAGE><VIDEO>b[VIDEO0]</VIDEO><AUDIO>c[AUDIO0]</AUDIO>"
        # import pdb
        # pdb.set_trace()

        if modality == None:
            modality = samples['TaskPrompt'][0][1:-1]
        begin_signal = f"<{modality}>"
        end_signal = f"</{modality}>"
        begin_signal_idx = self.llama_tokenizer(begin_signal, return_tensors="pt",
                                                add_special_tokens=False).to(self.device).input_ids
        end_signal_idx = self.llama_tokenizer(end_signal, return_tensors="pt",
                                              add_special_tokens=False).to(self.device).input_ids

        # import pdb
        # pdb.set_trace()
        # start_pos = (targets == begin_signal_idx).nonzero(as_tuple=False)[:, 1].tolist()
        # end_pos = (targets == end_signal_idx).nonzero(as_tuple=False)[:, 1].tolist()

        start_pos_indice = (targets == begin_signal_idx).nonzero(as_tuple=False)
        end_pos_indice = (targets == end_signal_idx).nonzero(as_tuple=False)
        # Group start and end positions by batch
        start_pos = [[] for _ in range(targets.size(0))]  # Initialize list for each batch
        end_pos = [[] for _ in range(targets.size(0))]    # Initialize list for each batch
        # Populate start_pos and end_pos
        for idx in range(start_pos_indice.size(0)):
            batch_idx = start_pos_indice[idx, 0].item()
            token_idx = start_pos_indice[idx, 1].item()
            start_pos[batch_idx].append(token_idx)
        for idx in range(end_pos_indice.size(0)):
            batch_idx = end_pos_indice[idx, 0].item()
            token_idx = end_pos_indice[idx, 1].item()
            end_pos[batch_idx].append(token_idx)
        # examples
        # start_pos = [[0], [0, 6]]  # Start positions of <IMAGE> for each batch
        # end_pos = [[4], [4, 10]]   # End positions of </IMAGE> for each batch


        hidden_embeds_list = []
        input_embeds_list = []
        hidden_embeds_text_list = []
        input_embeds_text_list = []
        if modality in self.output_alignment_modules.keys():
            for layer_idx in self.output_alignment_modules[modality]['alignment_layer']:
                hidden_embeds = []
                input_embeds = []
                hidden_embeds_text = []
                input_embeds_text = []
                for batch_idx, (s, e) in enumerate(zip(start_pos, end_pos)):
                    # import pdb
                    # pdb.set_trace()
                    text_s_i = s[modality_i]
                    s_i = e[modality_i]-self.modality_tokens[modality]
                    e_i = e[modality_i]
                    hidden_embeds.append(outputs.hidden_states[layer_idx][batch_idx, s_i:e_i, :])
                    input_embeds.append(self.embed_tokens(targets[batch_idx, s_i:e_i], using_lora=self.using_lora))
                    # output text embeddings
                    hidden_embeds_text.append(outputs.hidden_states[layer_idx][batch_idx, text_s_i+1:s_i, :])
                    input_embeds_text.append(self.embed_tokens(targets[batch_idx, text_s_i+1:s_i], using_lora=self.using_lora)) 
                hidden_embeds = torch.stack(hidden_embeds, dim=0)
                input_embeds = torch.stack(input_embeds, dim=0)
                hidden_embeds_list.append(hidden_embeds)
                input_embeds_list.append(input_embeds)
                # output text embeddings
                hidden_embeds_text = torch.stack(hidden_embeds_text, dim=0) # (batch_size, text_token_len, dim)
                input_embeds_text = torch.stack(input_embeds_text, dim=0)
                hidden_embeds_text_list.append(hidden_embeds_text)
                input_embeds_text_list.append(input_embeds_text)

        return modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list

    ##############################################
    #                                            #
    #                 Loss modules               #
    #                                            #
    ##############################################
    def contrastive_loss(self, logits):
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity):
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    def clip_align_loss(self, text_embeds, image_embeds):
        text_embeds = text_embeds.float()
        image_embeds = image_embeds.float()
        # text_embeds, image_embeds: (b,n,c)
        text_embeds = text_embeds.view(-1, text_embeds.size(2)) # (b*n,c)
        image_embeds = image_embeds.view(-1, image_embeds.size(2)) # (b*n,c)
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        loss = self.clip_loss(logits_per_text)
        return loss

    def loss_text_encoder_align(self, captions):
        # assert "Caption" in samples, "Caption not in samples!"
        # captions = samples['Caption']
        # llm text-encoder
        captions_tokens = self.llama_tokenizer(captions, return_tensors="pt", add_special_tokens=False).to(self.device)
        captions_embeds = self.embed_tokens(captions_tokens.input_ids, using_lora=self.using_lora) # (batch_size, text_token_len, dim)
        # captions_embeds = captions_embeds.mean(dim=1, keepdim=True) # (batch_size, 1, dim)

        mse_loss = 0
        #### IMAGE ####
        # if samples['TaskPrompt'][0] == "[IMAGE]":
        # diffusion text-encoder
        image_diffusion_pipe = self.diffusion_pipes['IMAGE']
        condiction_embeds = image_diffusion_pipe(captions, return_prompts_only=True).detach()  # (N, 77, 768)
        if self.output_alignment_MoE_mode == None:
            alignment_projs = self.alignment_projs['IMAGE']
        else:
            alignment_projs = self.alignment_projs
        # projection for image diffusion
        proj_captions_embeds_list = []
        rec_loss_list = []
        for layer_idx, fc_layer in enumerate(alignment_projs):
            if self.reconstruct_loss:
                proj_emb, rec_proj_emb = fc_layer(captions_embeds, modality='IMAGE')
                # reconstruction loss
                rec_loss = self.l2_loss(rec_proj_emb, captions_embeds.detach())
                rec_loss = rec_loss.mean()
                # clip loss
                rec_clip_loss = self.clip_align_loss(rec_proj_emb, captions_embeds.detach())
                rec_clip_loss = rec_clip_loss.mean()
                rec_loss = rec_loss + rec_clip_loss
                rec_loss_list.append(rec_loss)
                # project emb
                proj_captions_embeds_list.append(proj_emb)
            else:
                proj_captions_embeds_list.append(fc_layer(captions_embeds, modality='IMAGE'))
        proj_captions_embeds = torch.stack(proj_captions_embeds_list, dim=-1).sum(dim=-1)  # (N, 77, 768)
        if self.reconstruct_loss:
            rec_losses = torch.stack(rec_loss_list, dim=-1).sum(dim=-1)
        # import pdb
        # pdb.set_trace()
        # Embed-align loss
        image_mse_loss = self.l2_loss(proj_captions_embeds, condiction_embeds)
        image_mse_loss = image_mse_loss.mean()
        mse_loss = mse_loss + image_mse_loss
        # clip loss
        image_clip_loss = self.clip_align_loss(proj_captions_embeds, condiction_embeds)
        image_clip_loss = image_clip_loss.mean()
        mse_loss = mse_loss + image_clip_loss
        # rec loss
        if self.reconstruct_loss:
            mse_loss = mse_loss + rec_losses

        #### VIDEO ####
        # if samples['TaskPrompt'][0] == "[VIDEO]":
        
        #### AUDIO ####
        # if samples['TaskPrompt'][0] == "[AUDIO]":
        
        return mse_loss

    def loss_image(self, hidden_embeds, samples, loss_scale=1.0):
        # import pdb
        # pdb.set_trace()
        image_diffusion_pipe = self.diffusion_pipes['IMAGE']
        assert "Caption" in samples, "Caption not in samples!"
        captions = samples['Caption']
        condiction_embeds = image_diffusion_pipe(captions, return_prompts_only=True).detach()
        # import pdb
        # pdb.set_trace()
        # Semantic-align loss
        mse_loss = self.l2_loss(hidden_embeds, condiction_embeds)
        mse_loss = mse_loss.mean() * loss_scale
        # clip loss
        clip_loss = self.clip_align_loss(hidden_embeds, condiction_embeds)
        clip_loss = clip_loss.mean()
        mse_loss = mse_loss + clip_loss
        return mse_loss

    def loss_video(self, hidden_embeds, samples, loss_scale=1.0):
        # import pdb
        # pdb.set_trace()
        video_diffusion_pipe = self.diffusion_pipes['VIDEO']
        assert "Caption" in samples, "Caption not in samples!"
        captions = samples['Caption']
        condiction_embeds = video_diffusion_pipe(captions, return_prompts_only=True).detach()

        mse_loss = self.l2_loss(hidden_embeds, condiction_embeds)
        mse_loss = mse_loss.mean() * loss_scale
        # clip loss
        clip_loss = self.clip_align_loss(hidden_embeds, condiction_embeds)
        clip_loss = clip_loss.mean()
        mse_loss = mse_loss + clip_loss
        return mse_loss

    def loss_audio(self, hidden_embeds, samples, loss_scale=1.0):
        # import pdb
        # pdb.set_trace()
        audio_diffusion_pipe = self.diffusion_pipes['AUDIO']
        assert "Caption" in samples, "Caption not in samples!"
        captions = samples['Caption']
        condiction_embeds = audio_diffusion_pipe(captions, return_prompts_only=True).detach()

        assert len(condiction_embeds.shape) == 2, condiction_embeds.shape
        condiction_embeds = condiction_embeds.view(condiction_embeds.size(0), 1, condiction_embeds.size(1))

        mse_loss = self.l2_loss(hidden_embeds, condiction_embeds)
        mse_loss = mse_loss.mean() * loss_scale
        # clip loss
        clip_loss = self.clip_align_loss(hidden_embeds, condiction_embeds)
        clip_loss = clip_loss.mean()
        mse_loss = mse_loss + clip_loss
        return mse_loss

    def loss_mask(self, pred_masks, samples, bce_loss_scale=2.0, dice_loss_scale=0.5):
        # import pdb
        # pdb.set_trace()

        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = samples['MASK'][batch_idx][None].float()
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                self.sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                self.dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = bce_loss_scale * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = dice_loss_scale * mask_dice_loss / (num_masks + 1e-8)
        return mask_bce_loss, mask_dice_loss

    def loss_box(self, hidden_embeds, samples, loss_scale=1.0):
        # from transformers import logging
        # logging.set_verbosity_error() # 取消DINO不完整加载模型的warning
        # import pdb
        # pdb.set_trace()
        assert "Caption" in samples, "Caption not in samples!"
        captions = samples['Caption']
        # get grounding DINI text encoder
        condiction_embeds = hidden_embeds

        # import pdb
        # pdb.set_trace()
        mse_loss = self.l2_loss(hidden_embeds, condiction_embeds)
        mse_loss = mse_loss.mean() * loss_scale
        return mse_loss


    def cal_acc(self, outputs, targets):
        # import pdb
        # pdb.set_trace()
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, :-1]  # [B, S-1]
        labels = targets[:, 1:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return gen_acc

    def l2_loss(self, u: torch.Tensor, v: torch.Tensor):
        """
        Args:
          u: (N, T_I_V_A.txt, D) tensor.
          v: (N, T_I_V_A.txt, D) tensor.
        Returns:
          l1_loss: (N,) tensor of summed L1 loss.
        """
        assert u.shape == v.shape, (u.shape, v.shape)
        u = u.float()
        v = v.float()
        return ((u - v) ** 2).sum(dim=-1) ** 0.5
        # return ((u - v) ** 2).mean()

    def dice_loss(self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            num_masks: float,
            scale=1000,  # 100000.0,
            eps=1e-6,
    ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.float()
        targets = targets.float()
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        numerator = 2 * (inputs / scale * targets).sum(-1)
        denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
        loss = 1 - (numerator + eps) / (denominator + eps)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss

    def sigmoid_ce_loss(self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            num_masks: float,
    ):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        inputs = inputs.float()
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
        return loss

    
    def preparing_output_embeds_infer(self, samples, outputs, modality=None, targets=None, modality_i=0):
        # "<IMAGE>a[IMAGE0]</IMAGE><VIDEO>b[VIDEO0]</VIDEO><AUDIO>c[AUDIO0]</AUDIO>"
        # import pdb
        # pdb.set_trace()
        
        if modality == None:
            modality = samples['TaskPrompt'][0][1:-1]
        begin_signal = f"<{modality}>"
        end_signal = f"</{modality}>"
        begin_signal_idx = self.llama_tokenizer(begin_signal, return_tensors="pt",
                                                add_special_tokens=False).to(self.device).input_ids
        end_signal_idx = self.llama_tokenizer(end_signal, return_tensors="pt",
                                              add_special_tokens=False).to(self.device).input_ids

        if targets == None:
            targets = outputs.sequences[0][1:]
        # import pdb
        # pdb.set_trace()
        start_pos = (targets == begin_signal_idx).nonzero(as_tuple=False)[:, 1].tolist()
        end_pos = (targets == end_signal_idx).nonzero(as_tuple=False)[:, 1].tolist()

        hidden_embeds_list = []
        input_embeds_list = []
        hidden_embeds_text_list = []
        input_embeds_text_list = []
        if modality in self.output_alignment_modules.keys():
            for layer_idx in self.output_alignment_modules[modality]['alignment_layer']:
                hidden_embeds = []
                for _hidden_states in outputs.hidden_states[end_pos[modality_i]-self.modality_tokens[modality]:end_pos[modality_i]]:
                    hidden_embeds.append(_hidden_states[layer_idx]) # _hidden_states[layer_idx].shape = [1, 1, 4096]
                hidden_embeds = torch.cat(hidden_embeds, dim=1) # hidden_embeds.shape = [1, 1, 4096]
                input_embeds = self.embed_tokens(targets[end_pos[modality_i]-self.modality_tokens[modality]:end_pos[modality_i]].unsqueeze(0), using_lora=self.using_lora) # input_embeds.shape = [1, 1, 4096]
                # output text embeddings
                hidden_embeds_text = []
                for _hidden_states in outputs.hidden_states[start_pos[modality_i]+1:end_pos[modality_i]-self.modality_tokens[modality]]:
                    hidden_embeds_text.append(_hidden_states[layer_idx]) # _hidden_states[layer_idx].shape = [1, 1, 4096]
                hidden_embeds_text = torch.cat(hidden_embeds_text, dim=1) # hidden_embeds_text.shape = [1, text_token_len, 4096]
                input_embeds_text = self.embed_tokens(targets[start_pos[modality_i]+1:end_pos[modality_i]-self.modality_tokens[modality]].unsqueeze(0), using_lora=self.using_lora) # input_embeds.shape = [1, text_token_len, 4096]

                # import pdb
                # pdb.set_trace()
                # output_texts = self.llama_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                # output_texts_modality = self.llama_tokenizer.decode(targets[start_pos[modality_i]+1:end_pos[modality_i]-self.modality_tokens[modality]], skip_special_tokens=True)

                hidden_embeds_list.append(hidden_embeds)
                input_embeds_list.append(input_embeds)
                # output text embeddings
                hidden_embeds_text_list.append(hidden_embeds_text)
                input_embeds_text_list.append(input_embeds_text)

        return modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list

    @torch.no_grad()
    def generate(self, samples,
                 answers,
                 predictions,
                 predictions_text,
                 stop_words_ids=[[2]],
                 num_beams=1,
                 min_length=1,
                 top_p=0.9,
                 repetition_penalty=1,
                 length_penalty=1,
                 temperature=1,
                 do_sample=False):

        inputs_embeds, attention_mask = self.prepare_generation_embedding(samples)

        # import pdb
        # pdb.set_trace()

        # get stop flag
        end_ids_tokens = self.llama_tokenizer('[END]', return_tensors="pt", add_special_tokens=False).to(self.device)
        end_ids = end_ids_tokens.input_ids
        stop_words_ids = [[2]]
        stop_words_ids.append([end_ids])
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=1)])
        # llm inference
        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_context_len,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_attentions=True
            )

            # outputs = self.llama_model.generate(inputs_embeds=inputs_embeds,attention_mask=attention_mask,max_new_tokens=self.max_context_len,num_beams=num_beams,length_penalty=length_penalty,temperature=temperature,do_sample=do_sample, min_length=min_length,top_p=top_p,repetition_penalty=repetition_penalty,use_cache=True,stopping_criteria=stopping_criteria,output_hidden_states=True,return_dict_in_generate=True,output_attentions=True)


        # # modalities inference
        # answers = []
        # predictions = dict(
        #     IMAGE=[],
        #     VIDEO=[],
        #     AUDIO=[],
        #     MASK=[],
        #     BOX=dict(bboxes=[],label_names=[],scores=[]),
        #     IMAGESTORY=[],
        # )
        # predictions_text = dict(
        #     IMAGE=[],
        #     VIDEO=[],
        #     AUDIO=[],
        #     MASK=[],
        #     BOX=[],
        #     IMAGESTORY=[],
        # )
        for i in range(outputs.sequences.shape[0]):
            # if output_token[0] == 0:
            #     output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
            # output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            # output_texts = output_texts.replace("<s>", "")
            # output_texts = output_texts.split(r'[/INST]')[-1].strip()
            # if output_texts == '[BOX0]':
            #     box_prompt = outputs.hidden_states[i]

            # import pdb
            # pdb.set_trace()
            with self.maybe_autocast():
                modality = samples['TaskPrompt'][0][1:-1]
                ## Single Modal ##
                if modality in self.decode_modality.keys():
                    llm_text_res_list = self.get_llm_text_res(output_texts, modality) # ['caption1', 'caption2']
                    modality_i = 0
                    for llm_text_res in llm_text_res_list:
                        samples['llm_text_res'] = [llm_text_res]
                        modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds_infer(samples, outputs, modality=modality, targets=None, modality_i=modality_i)
                        modality_i += 1
                        # gather predictions
                        predictions_text[modality].append(llm_text_res)
                        if modality == 'BOX':
                            outputs_bboxes, outputs_label_names, outputs_scores = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                            predictions[modality]['bboxes'].append(outputs_bboxes[0])
                            predictions[modality]['label_names'].append(outputs_label_names[0])
                            predictions[modality]['scores'].append(outputs_scores[0])
                            # DETR: boxes = tensor([[266.7737, 301.7450,  70.3573,  70.3573]], device='cuda:0')
                            # Grounding DINO:  size = [300, 4], all boxes
                        elif modality == 'MASK':
                            preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                            predictions[modality].append(preds[0])
                        elif modality == 'IMAGE':
                            preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                            predictions[modality].append(preds[0])
                        elif modality == 'VIDEO':
                            preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                            predictions[modality].append(preds)
                        elif modality == 'AUDIO':
                            preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                            predictions[modality].append(preds[0])
                elif modality == 'SMARTMULTIMODAL' or modality == 'SPECIFICMULTIMODAL':
                    ## MultiModal ##
                    # get modality from llm output texts
                    modality_list = self.get_llm_text_modality(output_texts, self.decode_modality.keys())
                    for modality in modality_list:
                        llm_text_res_list = self.get_llm_text_res(output_texts, modality) # ['caption1', 'caption2']
                        modality_i = 0
                        for llm_text_res in llm_text_res_list:
                            samples['llm_text_res'] = [llm_text_res]
                            modality, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list = self.preparing_output_embeds_infer(samples, outputs, modality=modality, targets=None, modality_i=modality_i)
                            modality_i += 1
                            # gather predictions
                            predictions_text[modality].append(llm_text_res)
                            if modality == 'BOX':
                                outputs_bboxes, outputs_label_names, outputs_scores = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                                predictions[modality]['bboxes'].append(outputs_bboxes[0])
                                predictions[modality]['label_names'].append(outputs_label_names[0])
                                predictions[modality]['scores'].append(outputs_scores[0])
                                # DETR: boxes = tensor([[266.7737, 301.7450,  70.3573,  70.3573]], device='cuda:0')
                                # Grounding DINO:  size = [300, 4], all boxes
                            elif modality == 'MASK':
                                preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                                predictions[modality].append(preds[0])
                            elif modality == 'IMAGE':
                                preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                                predictions[modality].append(preds[0])
                            elif modality == 'VIDEO':
                                preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                                predictions[modality].append(preds)
                            elif modality == 'AUDIO':
                                preds = self.decode_modality[modality](samples, hidden_embeds_list, input_embeds_list, hidden_embeds_text_list, input_embeds_text_list, return_embeds_only=False)
                                predictions[modality].append(preds[0])
                elif modality == 'IMAGESTORY':
                    # gather predictions
                    predictions_text[modality].append(output_texts)
            answers.append(output_texts)

        return answers, predictions, predictions_text

    def prepare_generation_embedding(self, samples):
        # import pdb
        # pdb.set_trace()
        # preparing question embedding
        question_embeds = []
        for idx, question in enumerate(samples['Question']):
            frequency = 0
            question_embed = []
            question_splits = self.split_placeholder(question)
            question_splits.insert(0, '[INPUT]')
            question_splits.append(samples['TaskPrompt'][idx])
            if 'SystemPrompt' in samples:
                question_splits.append(samples['SystemPrompt'][idx])
            for question_split in question_splits:
                if "Placeholder" not in question_split:
                    question_split_tokens = self.llama_tokenizer(question_split, return_tensors="pt",
                                                                 add_special_tokens=False).to(self.device)
                    question_split_embed = self.embed_tokens(question_split_tokens.input_ids, using_lora=self.using_lora)  # 1, s1, c
                else:
                    modality = self.get_modality(question_split)
                    if isinstance(samples[modality][idx], list):
                        modality_input = samples[modality][frequency][idx][None]
                        frequency += 1
                    else:
                        modality_input = samples[modality][idx][None]
                    question_split_embed, _ = self.encode_modality[modality](modality_input)  # 1, s2, c

                question_embed.append(question_split_embed)
            question_embed = torch.cat(question_embed, dim=1)  # 1 x (1+s1+1+s2) x embed_dim
            question_embeds.append(question_embed)
        # import pdb
        # pdb.set_trace()
        question_embed_lens = [question_embed.shape[1] for question_embed in question_embeds]
        pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=self.device), using_lora=self.using_lora)

        max_length = max(question_embed_lens) if max(
            question_embed_lens) < self.max_context_len else self.max_context_len
        wrapped_question_embeds = pad_emb.expand(len(question_embed_lens), max_length, -1).clone()
        wrapped_question_atts = torch.zeros([len(question_embed_lens), max_length], dtype=torch.int, device=self.device)

        # for i, question_embed in enumerate(question_embeds):
        #     length = question_embed_lens[i] if question_embed_lens[i] < self.max_context_len else self.max_context_len
        #     wrapped_question_embeds[i, :length] = question_embed[:, :length]
        #     wrapped_question_atts[i, :length] = 1

        for i, question_embed in enumerate(question_embeds):
            length = question_embed_lens[i] if question_embed_lens[i] < self.max_context_len else self.max_context_len
            wrapped_question_embeds[i, -length:] = question_embed[:, :length]
            wrapped_question_atts[i, -length:] = 1

        # add bos token at the begining
        batch_size = wrapped_question_embeds.shape[0]
        bos_tokens = torch.ones([batch_size, 1], dtype=torch.int64,
                                device=self.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        bos_embeds = self.embed_tokens(bos_tokens, using_lora=self.using_lora)
        bos_atts = wrapped_question_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, wrapped_question_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, wrapped_question_atts], dim=1)

        return inputs_embeds, attention_mask

