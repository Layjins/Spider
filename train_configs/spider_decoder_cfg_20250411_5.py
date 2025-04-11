model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt = "You are Spider, an AI assistant that understands and generates multimodal content.\n" \
        "You must output responses using some of the following tags:\n" \
        "- <IMAGE>...</IMAGE>\n" \
        "- <VIDEO>...</VIDEO>\n" \
        "- <AUDIO>...</AUDIO>\n" \
        "- <MASK>...</MASK>\n" \
        "- <BOX>...</BOX>\n" \
        "- <IMAGESTORY><GENERALPROMPT>...</GENERALPROMPT>, <PROMPTARRAY>[...]</PROMPTARRAY>, <STYLENAME>...</STYLENAME></IMAGESTORY>\n\n" \
        "## Examples:\n" \
        "User: An image of a tiger, its segmentation, and its roar.\n" \
        "Output: Tiger<IMAGE>Tiger</IMAGE>, Tiger<MASK>Tiger</MASK>, Tiger Roar<AUDIO>Tiger Roar</AUDIO>\n\n" \
        "User: A video of a cat jumping and the sound it makes.\n" \
        "Output: A cat jumping<VIDEO>A cat jumping</VIDEO>, Cat sound<AUDIO>Cat sound</AUDIO>\n\n" \
        "User: Detect and segment a red car.\n" \
        "Output: Red Car<BOX>Red Car</BOX>, Red Car<MASK>Red Car</MASK>\n\n" \
        "User: Tell a comic-style story about a robot.\n" \
        "Output: <IMAGESTORY><GENERALPROMPT>'a robot in the future'</GENERALPROMPT>, <PROMPTARRAY>['explores a city', 'meets a friend', 'saves the day']</PROMPTARRAY>, <STYLENAME>'Comic book'</STYLENAME></IMAGESTORY>\n\n" \
        "User: Generate an image of a dog and a video of it running.\n" \
        "Output: Dog<IMAGE>Dog</IMAGE>, Dog running<VIDEO>Dog running</VIDEO>\n",
    get_prompt_embed_for_diffusion=False, # If True: get prompt embedding for diffusion
    diffusion_modules=dict(
        IMAGE=dict(type="sd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/stable-diffusion-v1-5'),
        VIDEO=dict(type="vd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/zeroscope_v2_576w'),
        AUDIO=dict(type="ad", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/audioldm-l-full'),
    ),
    system_prompt_image = "",
    system_prompt_video = "",
    system_prompt_audio = "",
    # mask_decoder_modules=None,
    mask_decoder_modules=dict(
        sam_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/sam_vit_h_4b8939.pth",
        freeze_mask_decoder=True,
    ),
    system_prompt_mask = "",
    # box_decoder_modules=None,
    box_decoder_modules=dict(
        # grounding DINO
        config_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py',
        checkpoint_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
    ),
    system_prompt_box = "",
    # story_generation=None,
    story_generation=dict(
        # models_dict = {
        # "Juggernaut":"RunDiffusion/Juggernaut-XL-v8",
        # "RealVision":"/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/RealVisXL_V4.0" ,
        # "SDXL":"/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/stable-diffusion-xl-base-1.0" ,
        # "Unstable": "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/sdxl-unstable-diffusers-y"
        # }
        model_name="Unstable",
    ),
    system_prompt_story="",
    max_context_len=4096,
)
