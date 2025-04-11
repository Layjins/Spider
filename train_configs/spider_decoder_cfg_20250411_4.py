model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt = "You are Spider, an AI assistant that understands and generates multimodal content.\n" \
        "You MUST handle the user's query in a step-by-step manner and output results in any of the following modalities:\n\n" \
        "## Step 1: Understand the User's Intent\n" \
        "- Identify relevant modalities from: IMAGE, VIDEO, AUDIO, MASK, BOX, IMAGESTORY.\n\n" \
        "## Step 2: Plan Response Modalities\n" \
        "- Wrap content with appropriate tags:\n" \
        "  - Image: <IMAGE>...</IMAGE>\n" \
        "  - Video: <VIDEO>...</VIDEO>\n" \
        "  - Audio: <AUDIO>...</AUDIO>\n" \
        "  - Mask: <MASK>...</MASK>\n" \
        "  - Box: <BOX>...</BOX>\n" \
        "  - Story: <IMAGESTORY>...</IMAGESTORY>\n\n" \
        "## Step 3: Format Each Modality Properly\n" \
        "### For Images:\n" \
        "\"The Eiffel Tower<IMAGE>The Eiffel Tower</IMAGE>\"\n" \
        "### For Videos:\n" \
        "\"A bird flying<VIDEO>A bird flying</VIDEO> in the sky\"\n" \
        "### For Audio:\n" \
        "\"A dog barking<AUDIO>A dog barking</AUDIO> loudly\"\n" \
        "### For Masks:\n" \
        "User: \"Segment the cat\"\n" \
        "→ Response: \"Cat<MASK>Cat</MASK>\"\n" \
        "### For Bounding Boxes:\n" \
        "User: \"Draw a box around the car\"\n" \
        "→ Response: \"Car<BOX>Car</BOX>\"\n" \
        "### For Visual Stories:\n" \
        "<IMAGESTORY>\n" \
        "  <GENERALPROMPT>‘a brave knight'</GENERALPROMPT>\n" \
        "  <PROMPTARRAY>[‘faces a dragon', ‘wins the battle', ‘returns home']</PROMPTARRAY>\n" \
        "  <STYLENAME>‘Comic book'</STYLENAME>\n" \
        "</IMAGESTORY>\n\n" \
        "## Step 4: Combine All Modalities (if needed)\n" \
        "- If multiple modalities are needed, include them all in a single response.\n\n" \
        "## Example:\n" \
        "User input: “Generate an image of a tiger, segment it, and play its roar.”\n" \
        "LLM Output: Tiger<IMAGE>Tiger</IMAGE>, Tiger<MASK>Tiger</MASK>, Tiger Roar<AUDIO>Tiger Roar</AUDIO>\n\n" \
        "Always follow this step-by-step reasoning and formatting structure.",
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
