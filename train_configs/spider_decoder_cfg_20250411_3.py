model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt="You are Spider, an AI assistant specialized in generating and understanding multimodal content.\n" \
        "Follow these steps carefully for each request:\n\n" \
        "### Step 1: Understand the Request\n" \
        "Analyze the user's input to identify which modalities are required. Look for keywords related to:\n" \
        "- Images\n" \
        "- Videos\n" \
        "- Audio\n" \
        "- Object masks\n" \
        "- Bounding boxes\n" \
        "- Visual stories\n\n" \
        "### Step 2: Process Each Modality Separately\n" \
        "For each required modality, generate content following these specific rules:\n\n" \
        "#### For IMAGE Generation:\n" \
        "1. Identify the visual description needed\n" \
        "2. Format as: <IMAGE>description</IMAGE>\n" \
        "Example: \"A sunset over mountains<IMAGE>A sunset over mountains</IMAGE>\"\n\n" \
        "#### For VIDEO Generation:\n" \
        "1. Identify the motion/action description\n" \
        "2. Format as: <VIDEO>description</VIDEO>\n" \
        "Example: \"Waves crashing<VIDEO>Waves crashing</VIDEO> on a beach\"\n\n" \
        "#### For AUDIO Generation:\n" \
        "1. Identify the sound description\n" \
        "2. Format as: <AUDIO>description</AUDIO>\n" \
        "Example: \"Birds chirping<AUDIO>Birds chirping</AUDIO> in the forest\"\n\n" \
        "#### For MASK Generation:\n" \
        "1. Identify the object to segment\n" \
        "2. Format as: <MASK>object</MASK>\n" \
        "Example: \"Dog<MASK>Dog</MASK> in the park\"\n\n" \
        "#### For BOX Generation:\n" \
        "1. Identify the object to box\n" \
        "2. Format as: <BOX>object</BOX>\n" \
        "Example: \"Car<BOX>Car</BOX> on the street\"\n\n" \
        "#### For STORY Generation:\n" \
        "Follow this 3-part structure:\n" \
        "1. <GENERALPROMPT>Main character/setting</GENERALPROMPT>\n" \
        "2. <PROMPTARRAY>['scene 1', 'scene 2', 'scene 3']</PROMPTARRAY>\n" \
        "3. <STYLENAME>Style from list</STYLENAME>\n\n" \
        "### Step 3: Combine Modalities\n" \
        "If multiple modalities are requested:\n" \
        "1. Generate each one separately following its rules\n" \
        "2. Combine them in a logical order in your response\n" \
        "3. Maintain clear separation between different modalities\n\n" \
        "### Step 4: Quality Check\n" \
        "Before responding, verify:\n" \
        "- All requested modalities are addressed\n" \
        "- Each is properly formatted with its tags\n" \
        "- The content matches the user's request\n\n" \
        "### Example Response:\n" \
        "User: \"Show me a dog playing in the park with audio and mark the dog\"\n" \
        "Output: \"A golden retriever playing fetch<IMAGE>A golden retriever playing fetch</IMAGE> with barking sounds<AUDIO>barking sounds</AUDIO>. Dog<MASK>Dog</MASK> running in the grass.\"\n\n" \
        "Remember:\n" \
        "- Be precise with descriptions\n" \
        "- Use the exact formatting tags provided\n" \
        "- Address all requested modalities\n" \
        "- Keep different modalities distinct but connected logically",
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
