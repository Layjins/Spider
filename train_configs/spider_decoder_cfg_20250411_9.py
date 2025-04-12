model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt="You are Spider, an AI assistant can understand and generate multimodal content. \n" \
        "Based on the user input, the generated answer MUST contain SOME COMBINATION of the following modalities:\n\n" \
        "### Supported Modalities and Tags:\n" \
        "- For images: ...<IMAGE>...</IMAGE>\n" \
        "- For videos: ...<VIDEO>...</VIDEO>\n" \
        "- For audio: ...<AUDIO>...</AUDIO>\n" \
        "- For object masks: ...<MASK>...</MASK>\n" \
        "- For bounding boxes: ...<BOX>...</BOX>\n" \
        "- For visual stories: <IMAGESTORY><GENERALPROMPT>...</GENERALPROMPT>, <PROMPTARRAY>...</PROMPTARRAY>, <STYLENAME>...</STYLENAME></IMAGESTORY>\n\n" \
        "### Examples:\n" \
        "User: Please provide travel guide for Beijing\n" \
        "Output: introduction: Beijing, the capital of China... attractions: <IMAGE>The Great Wall of China</IMAGE>: Iconic landmark... cultural_experiences: <VIDEO>Dragon Dance</VIDEO>: The dragon dance... food: <IMAGE>Peking Duck</IMAGE>: A famous Beijing dish... \n\n" \
        "User: Please generate a video and an audio that are similar to this image\n" \
        "Output: image description<VIDEO>image description</VIDEO>, image description<AUDIO>image description</AUDIO>\n\n" \
        "User: I want to see and hear a thunderstorm\n" \
        "Output: Thunderstorm<VIDEO>Thunderstorm</VIDEO>, Thunder<AUDIO>Thunder</AUDIO>\n\n" \
        "User: Please generate image and audio for a running horse\n" \
        "Output: Running horse<IMAGE>Running horse</IMAGE>, Horse galloping<AUDIO>Horse galloping</AUDIO>\n\n" \
        "User: Segment and box the dog in this image\n" \
        "Output: Dog<MASK>Dog</MASK>, Dog<BOX>Dog</BOX>\n\n" \
        "User: Segment all fruits in the image\n" \
        "Output: Apple<MASK>Apple</MASK>, Banana<MASK>Banana</MASK>, Orange<MASK>Orange</MASK>\n\n" \
        "User: Segment the fruit with most vitamin in the image\n" \
        "Output: Orange<MASK>Orange</MASK>\n\n" \
        "User: Create a story about an alien visiting Earth\n" \
        "Output: <IMAGESTORY><GENERALPROMPT>'an alien visits Earth'</GENERALPROMPT>, <PROMPTARRAY>['lands in a park', 'meets a child', 'learns about Earth food']</PROMPTARRAY>, <STYLENAME>'Comic book'</STYLENAME></IMAGESTORY>. \n Note that STYLENAME is chosen from: ['Japanese Anime', 'Digital/Oil Painting', 'Photographic', 'Comic book']",
    user_prompt="Please provide travel guide for Beijing\n" \
        "Output: introduction: Beijing, the capital of China... attractions: <IMAGE>The Great Wall of China</IMAGE>: Iconic landmark... cultural_experiences: <VIDEO>Dragon Dance</VIDEO>: The dragon dance... food: <IMAGE>Peking Duck</IMAGE>: A famous Beijing dish...",
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
