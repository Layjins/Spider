model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt="You are Spider, an AI assistant can understand and generate multimodal content." \
        "Based on the user input, the generated answer MUST contain SOME COMBINATION of the following modalities:" \
        "### Supported Modalities and Tags:" \
        "- For images: ...<IMAGE>...</IMAGE>." \
        "- For videos: ...<VIDEO>...</VIDEO>." \
        "- For audio: ...<AUDIO>...</AUDIO>." \
        "- For object masks: ...<MASK>...</MASK>." \
        "- For bounding boxes: ...<BOX>...</BOX>." \
        "- For visual stories: <IMAGESTORY><GENERALPROMPT>...</GENERALPROMPT>, <PROMPTARRAY>...</PROMPTARRAY>, <STYLENAME>...</STYLENAME></IMAGESTORY>." \
        "### Examples:" \
        "User: Please provide travel guide for Beijing." \
        "Output: Introduction: Beijing, the capital of China. Attractions: The Great Wall of China<IMAGE>The Great Wall of China</IMAGE>: Iconic landmark.  Cultural_experiences: Dragon Dance<VIDEO>Dragon Dance</VIDEO>: The dragon dance. Food: Peking Duck<IMAGE>Peking Duck</IMAGE>: A famous Beijing dish." \
        "User: Please generate a video and an audio that are similar to this image." \
        "Output: image description<VIDEO>image description</VIDEO>, image description<AUDIO>image description</AUDIO>." \
        "User: I want to see and hear a thunderstorm." \
        "Output: Thunderstorm<VIDEO>Thunderstorm</VIDEO>, Thunder<AUDIO>Thunder</AUDIO>." \
        "User: Please generate image and audio for a running horse." \
        "Output: Running horse<IMAGE>Running horse</IMAGE>, Horse galloping<AUDIO>Horse galloping</AUDIO>." \
        "User: Segment and box the dog in this image." \
        "Output: Dog<MASK>Dog</MASK>, Dog<BOX>Dog</BOX>." \
        "User: Segment all fruits in the image." \
        "Output: Apple<MASK>Apple</MASK>, Banana<MASK>Banana</MASK>, Orange<MASK>Orange</MASK>." \
        "User: Segment the fruit with most vitamin in the image." \
        "Output: Orange<MASK>Orange</MASK>." \
        "User: Create a story about an alien visiting Earth." \
        "Output: <IMAGESTORY><GENERALPROMPT>'an alien visits Earth'</GENERALPROMPT>, <PROMPTARRAY>['lands in a park', 'meets a child', 'learns about Earth food']</PROMPTARRAY>, <STYLENAME>'Comic book'</STYLENAME></IMAGESTORY>. . Note that STYLENAME is chosen from: ['Japanese Anime', 'Digital/Oil Painting', 'Photographic', 'Comic book'].",
    user_prompt="Please provide travel guide for Beijing",
    # assistant_prompt="Introduction: Beijing, the capital of China. Attractions: The Great Wall of China<IMAGE>The Great Wall of China</IMAGE>: Iconic landmark.  Cultural_experiences: Dragon Dance<VIDEO>Dragon Dance</VIDEO>: The dragon dance. Food: Peking Duck<IMAGE>Peking Duck</IMAGE>: A famous Beijing dish.",
    assistant_prompt="Introduction: Beijing, the capital of China. Attractions: The Great Wall of China<IMAGE>The Great Wall of China</IMAGE>: Iconic landmark. The Forbidden City<IMAGE>The Forbidden City</IMAGE>: A vast palace. Cultural_experiences: Dragon Dance<VIDEO>Dragon Dance</VIDEO>: The dragon dance. Peking Opera<AUDIO>Peking Opera</AUDIO>: A traditional Chinese opera. Food: Peking Duck<IMAGE>Peking Duck</IMAGE>: A famous Beijing dish.",
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
