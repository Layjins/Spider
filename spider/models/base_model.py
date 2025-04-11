import logging
import argparse

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM

from .ImageBind import *
from .modeling_llama import LlamaForCausalLM
from .layers import *

from spider.common.registry import registry
from spider.models.segment_anything import build_sam_vit_h

# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class BaseModel(nn.Module):
    """
    Base class for models
    """

    def __init__(self):
        super().__init__()

    def init_imagebind_encoder(self, imagebind_ckpt_path):
        # imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt',
        #                                    self.args['imagebind_version'])
        logging.info(f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        visual_encoder, visual_hidden_size = imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        # free vision encoder
        for name, param in visual_encoder.named_parameters():
            param.requires_grad = False
        visual_encoder.eval()
        logging.info('Visual encoder initialized.')
        return visual_encoder, visual_hidden_size

    def init_llm(self, vicuna_ckpt_path, using_lora, freeze_lm, freeze_tokens, lora_r, lora_alpha, lora_dropout,
                 lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        logging.info(f'Initializing language decoder from {vicuna_ckpt_path} ...')
        if "DeepSeek-R1-Distill-Llama-8B" in vicuna_ckpt_path: # llama3
            from transformers import AutoModelForCausalLM
            llama_model = AutoModelForCausalLM.from_pretrained(vicuna_ckpt_path)

            # from .modeling_llama3 import LlamaForCausalLM
            # llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)

            # from transformers import AutoConfig, AutoModelForCausalLM
            # from accelerate import init_empty_weights
            # mconfig = AutoConfig.from_pretrained(vicuna_ckpt_path, trust_remote_code=True)
            # with init_empty_weights():
            #     llama_model = AutoModelForCausalLM.from_config(mconfig, trust_remote_code=True)
        else:
            llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)

        if using_lora:
            logging.info("Instruct tuning the LLaMa with lora ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )

            llama_model = get_peft_model(llama_model, peft_config)
            llama_model.print_trainable_parameters()

        if freeze_lm:
            logging.info("Freezing the LLaMa ...")
            for param in llama_model.parameters():
                param.requires_grad = False
            llama_model.eval()
        logging.info('Language decoder initialized.')
        return llama_model

    def init_tokenizer(self, tokenizer_path, new_modality_tokens=None,
                       new_special_tokens=['[INPUT]', '[OUTPUT]', '[END]', '[IMAGE]',
                                           '[VIDEO]', '[AUDIO]', '[BOX]', '[MASK]', '[SMARTMULTIMODAL]', '[SPECIFICMULTIMODAL]'],
                       bbox_bins=1000):
        # use the new trained tokenizer
        logging.info(f'Initializing tokenizer from {tokenizer_path} ...')
        if "DeepSeek-R1-Distill-Llama-8B" in tokenizer_path: # llama3
            from transformers import AutoTokenizer
            llama_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        else:
            llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_tokenizer.padding_side = "right"

        new_modality_idxs = {}
        if new_modality_tokens is not None:
            for modality, count in new_modality_tokens.items():
                # add special image token to tokenizer
                llama_tokenizer.add_tokens(f"<{modality}>")
                llama_tokenizer.add_tokens(f"</{modality}>")

                cur_modality_idxs = []
                for i in range(count):
                    cur_token = f"[{modality}{i}]"
                    # logging.info(f'Adding {cur_token} token to vocabulary.')
                    # logging.info(f'Before adding new token, tokenizer("{cur_token}") =',
                    #       llama_tokenizer(f'{cur_token}', add_special_tokens=False))
                    num_added_tokens = llama_tokenizer.add_tokens(f'{cur_token}')
                    # logging.info(f"After adding {num_added_tokens} new tokens, tokenizer('{cur_token}') =",
                    #       llama_tokenizer(f'{cur_token}', add_special_tokens=False))
                    cur_modality_idx = llama_tokenizer(f'{cur_token}', add_special_tokens=False).input_ids
                    assert len(cur_modality_idx) == 1, cur_modality_idx
                    cur_modality_idxs.append(cur_modality_idx[0])
                new_modality_idxs[modality] = cur_modality_idxs

        # add special tokens
        llama_tokenizer.add_tokens(new_special_tokens)

        # add tokens for location x, y.  bins=1000
        for i in range(bbox_bins):
            llama_tokenizer.add_tokens([f"<Loc{i}>"])
        logging.info('Tokenizer initialized.')
        return llama_tokenizer, new_modality_idxs

    def init_input_llama_proj(self, in_dim, out_dim, freeze_input_proj):
        llama_proj = nn.Linear(in_dim, out_dim)
        if freeze_input_proj:
            for param in llama_proj.parameters():
                param.requires_grad = False
        return llama_proj

    def init_output_alignment_proj(
            self,
            llama_hidden_layers=None,
            llama_hidden_size=4096,
            alignment_module_mode='transformer',
            reconstruct_loss=False,
            alignment_layer=[-1],
            alignment_input_tokens=4,
            alignment_output_tokens=77,
            alignment_output_dim=768,
            freeze_output_alignment_proj=False
    ):

        alignment_proj = nn.ModuleList([])
        for layer_idx in alignment_layer:
            if layer_idx == -1 or layer_idx == llama_hidden_layers:
                in_dim = llama_hidden_size

                alignment_proj.append(
                    TextFcLayer(in_dim, alignment_output_dim,
                                num_input_tokens=alignment_input_tokens,
                                num_output_tokens=alignment_output_tokens,
                                mode=alignment_module_mode,
                                device=self.device))
            # self.sd_pipe.text_encoder.config.hidden_size
            elif layer_idx < llama_hidden_layers:
                alignment_proj.append(
                    TextFcLayer(llama_hidden_size, alignment_output_dim,
                                num_input_tokens=alignment_input_tokens,
                                num_output_tokens=alignment_output_tokens,
                                mode=alignment_module_mode,
                                device=self.device))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {llama_hidden_layers} layers.')

        if freeze_output_alignment_proj:
            for name, param in alignment_proj.named_parameters():
                param.requires_grad = False
        return alignment_proj.to(self.device)

    def init_output_alignment_proj_moe(
            self,
            output_alignment_modules,
            llama_hidden_layers=None,
            llama_hidden_size=4096,
            alignment_module_mode='transformer',
            reconstruct_loss=False,
            alignment_layer=[-1],
            freeze_output_alignment_proj=False
    ):

        alignment_proj = nn.ModuleList([])
        for layer_idx in alignment_layer:
            if layer_idx == -1 or layer_idx == llama_hidden_layers:
                alignment_proj.append(
                    TextFcLayerMoE(llama_hidden_size, output_alignment_modules,
                                mode=alignment_module_mode,
                                reconstruct_loss=reconstruct_loss,
                                device=self.device))
            elif layer_idx < llama_hidden_layers:
                alignment_proj.append(
                    TextFcLayerMoE(llama_hidden_size, output_alignment_modules,
                                mode=alignment_module_mode,
                                reconstruct_loss=reconstruct_loss,
                                device=self.device))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {llama_hidden_layers} layers.')

        if freeze_output_alignment_proj:
            for name, param in alignment_proj.named_parameters():
                param.requires_grad = False
        return alignment_proj.to(self.device)

    def init_diffusion_model(self, diffusion_modules):
        # import pdb
        # pdb.set_trace()
        diffusion_pipes = {}
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        for modality, diffusion_module in diffusion_modules.items():
            pipe = registry.get_model_class(diffusion_module['type']).from_pretrained(diffusion_module['ckpt'], torch_dtype=dtype)
            if not torch.cuda.is_available():
                logging.info('WARNING: using CPU, this will be slow!')
            else:
                pipe = pipe.to("cuda")
            diffusion_pipes[modality] = pipe
        return diffusion_pipes

    def init_mask_decoder_sam(self, mask_decoder_modules):
        mask_decoder_sam = build_sam_vit_h(mask_decoder_modules['sam_path'])
        for param in mask_decoder_sam.parameters():
            param.requires_grad = False
        mask_decoder_sam.mask_decoder.train()
        if mask_decoder_modules['freeze_mask_decoder'] == True:
            for param in mask_decoder_sam.mask_decoder.parameters():
                param.requires_grad = False
        else:
            for param in mask_decoder_sam.mask_decoder.parameters():
                param.requires_grad = True

        return mask_decoder_sam

    def get_visual_embs(self, images: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(images.shape[0]):
                torch.cuda.empty_cache()
                image = images[i]
                # h, w = image.shape[-2:]
                # padh = 1024 - h
                # padw = 1024 - w
                # image = F.pad(image, (0, padw, 0, padh))
                image_embeddings = self.mask_decoder_sam.image_encoder(
                    image.unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def embed_tokens(self, token_ids, using_lora=False):
        if using_lora:
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids) # lora wrapped model
        else:
            embeds = self.llama_model.model.embed_tokens(token_ids)
        return embeds

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(enabled=True)
        else:
            return contextlib.nullcontext()

