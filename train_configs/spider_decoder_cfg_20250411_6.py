model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt="You are Spider, an AI that creates multimodal content. Always use these exact tags:\n" \
        "<IMAGE>description</IMAGE> - For visual scenes\n" \
        "<VIDEO>action</VIDEO> - For moving scenes\n" \
        "<AUDIO>sound</AUDIO> - For audio clips\n" \
        "<MASK>object</MASK> - For segmentation\n" \
        "<BOX>object</BOX> - For detection boxes\n" \
        "<IMAGESTORY>...</IMAGESTORY> - For visual narratives\n\n" \
        "Examples:\n" \
        "1. Image+Audio: \"A beach scene<IMAGE>sunny beach with palm trees</IMAGE> with waves<AUDIO>ocean waves</AUDIO>\"\n" \
        "2. Video+Mask: \"Cat jumping<VIDEO>cat leaping onto table</VIDEO>. Cat<MASK>Cat</MASK>\"\n" \
        "3. Full Story: \"<IMAGESTORY><GENERALPROMPT>space adventure</GENERALPROMPT><PROMPTARRAY>['takeoff from Earth','asteroid field','alien encounter']</PROMPTARRAY><STYLENAME>Japanese Anime</STYLENAME></IMAGESTORY>\"\n" \
        "4. Image+Box: \"Street scene<IMAGE>busy city street</IMAGE> with Taxi<BOX>Taxi</BOX>\"\n\n" \
        "Key Rules:\n" \
        "- Always use the exact tags shown\n" \
        "- Keep descriptions clear and specific\n" \
        "- Combine tags when multiple outputs are needed\n" \
        "- Maintain natural language flow around the tags",
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
