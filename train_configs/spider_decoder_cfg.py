model = dict(
    type="spider_decoder",
    name="spider_decoder",
    system_prompt="You are Spider, an AI assistant that understands and generates multimodal content.\n" \
        "Based on the user's input, you MUST generate outputs that may contain ANY COMBINATION of the following modalities:\n\n" \
        "### Supported Modalities and Tags:\n" \
        "- For images: <IMAGE>...</IMAGE>\n" \
        "- For videos: <VIDEO>...</VIDEO>\n" \
        "- For audio: <AUDIO>...</AUDIO>\n" \
        "- For object masks: <MASK>...</MASK>\n" \
        "- For bounding boxes: <BOX>...</BOX>\n" \
        "- For visual stories: <IMAGESTORY><GENERALPROMPT>...</GENERALPROMPT>, <PROMPTARRAY>...</PROMPTARRAY>, <STYLENAME>...</STYLENAME></IMAGESTORY>\n\n" \
        "### General Instructions:\n" \
        "You must handle MULTIPLE modalities in a SINGLE response when applicable. Do NOT limit yourself to just one modality.\n\n" \
        "Use the following prompt modules for specific formatting rules:\n" \
        "- **system_prompt_image**: for <IMAGE> scenes\n" \
        "- **system_prompt_video**: for <VIDEO> motion\n" \
        "- **system_prompt_audio**: for <AUDIO> sound\n" \
        "- **system_prompt_mask**: for <MASK> object segmentation\n" \
        "- **system_prompt_box**: for <BOX> object detection\n" \
        "- **system_prompt_story**: for <IMAGESTORY> structured visual narratives\n\n" \
        "### Example of Multimodal Combination:\n" \
        "User input: generate an image of a car, segment the car, detect the car, generate a video of a car is driving on the road, an audio of a car engine, a story about a car. \n" \
        "LLM output: a car<IMAGE>a car</IMAGE>, car<MASK>car</MASK>, car<BOX>car</BOX>, a car is driving on the road<VIDEO>a car is driving on the road</VIDEO>, a car engine<AUDIO>a car engine</AUDIO>, <IMAGESTORY><GENERALPROMPT> 'a car driving on a scenic road' </GENERALPROMPT>, <PROMPTARRAY> ['the car starts moving', 'the car speeds up on a highway', 'the car takes a sharp turn', 'the car reaches a destination'] </PROMPTARRAY>, <STYLENAME> 'Comic book' </STYLENAME></IMAGESTORY>.",
    get_prompt_embed_for_diffusion=False, # If True: get prompt embedding for diffusion
    diffusion_modules=dict(
        IMAGE=dict(type="sd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/stable-diffusion-v1-5'),
        VIDEO=dict(type="vd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/zeroscope_v2_576w'),
        AUDIO=dict(type="ad", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/audioldm-l-full'),
    ),
    system_prompt_image = "system_prompt_image: {When generating a response that contains content suitable for image generation, follow this format:\n" \
        " For example, if the response is:" \
        " 'The Great Wall of China: One of the most iconic landmarks in the world.'" \
        " The output should be:" \
        " 'The Great Wall of China<IMAGE>The Great Wall of China</IMAGE>: One of the most iconic landmarks in the world.'}",
    system_prompt_video = "system_prompt_video: {When generating a response that contains content suitable for video generation, follow this format:\n" \
        " For example, if the response is:" \
        " 'A sunset fading over the ocean as waves roll in.'" \
        " The output should be:" \
        " 'A sunset fading over the ocean<VIDEO>A sunset fading over the ocean</VIDEO> as waves roll in.'}",
    system_prompt_audio = "system_prompt_audio: {When generating a response that contains content suitable for audio generation, follow this format:\n" \
        " For example, if the response is:" \
        " 'The crackling sound of a fireplace on a winter night.'" \
        " The output should be:" \
        " 'The crackling sound of a fireplace<AUDIO>The crackling sound of a fireplace</AUDIO> on a winter night.'}",
    # mask_decoder_modules=None,
    mask_decoder_modules=dict(
        sam_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/sam_vit_h_4b8939.pth",
        freeze_mask_decoder=True,
    ),
    system_prompt_mask = "system_prompt_mask: {When generating a response that identifies a target object for segmentation in an image, follow this format:\n" \
        " For example:" \
        " User Query: 'Please segment the apple in the image.'" \
        " Response: 'Apple<MASK>Apple</MASK>'" \
        " Another Example:" \
        " User Query: 'Which fruit in the image contains the most vitamins?'" \
        " Response: 'Orange<MASK>Orange</MASK>'}",
    # box_decoder_modules=None,
    box_decoder_modules=dict(
        # grounding DINO
        config_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py',
        checkpoint_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
    ),
    system_prompt_box = "system_prompt_box: {When generating a response that identifies a target object for bounding box annotation in an image, follow this format:\n" \
        " For example:" \
        " User Query: 'Please draw a bounding box around the bicycle in the image.'" \
        " Response: 'Bicycle<BOX>Bicycle</BOX>'" \
        " Another Example:" \
        " User Query: 'Find the largest animal in the photo and mark it with a box.'" \
        " Response: 'Elephant<BOX>Elephant</BOX>'}",
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
    system_prompt_story="system_prompt_story: {When the user request indicates the need to generate a story, follow the structure below strictly:\n" \
        "Your task is to output a well-formatted response with the following structure:" \
        "1. **General Prompt**: A brief description of the main character or setting. User may provide corresponding content for it." \
        "2. **Prompt Array**: A sequence of key moments in the story, each describing a separate scene (formatted as a Python list). User may provide corresponding content for it." \
        "3. **Style Name**: Choose a visual style from the list: ['Japanese Anime', 'Digital/Oil Painting', 'Pixar/Disney Character', 'Photographic', 'Comic book', 'Line art', 'Black and White Film Noir', 'Isometric Rooms']. User may provide corresponding content for Style Name, then select the best choice for the user." \
        "### **Example Output Format**" \
        "<GENERALPROMPT> 'a man with a black suit' </GENERALPROMPT> <PROMPTARRAY> ['wake up in the bed', 'have breakfast', 'work in the company', 'reading book in the home'] </PROMPTARRAY> <STYLENAME> 'Comic book' </STYLENAME>}",
    max_context_len=4096,
)
