pretrained_ckpt_path=None
# pretrained_ckpt_path="/jinxianglai/exp/spider/stage3/2024061316383/45"

model = dict(
    type="spider", # spider_free: free training, spider: training
    name="spider_story",
    encoder_modules=dict(
        imagebind_ckpt_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model'
    ),
    input_proj_modules=dict(
        freeze_input_proj=False,
    ),
    use_embed_align_loss=False, # train to align the text-encoders of llm and tasks
    only_embed_align_loss=False, # only train to align the text-encoders of llm and tasks, while do not train other models 
    word_align_loss=False, # Local text alignment data，是Global text alignment data长文本中随机采样的words
    only_llm_gen_loss=True, # only train the llm to generate text, while do not train the task decoders and their projectors
    llm_modules=dict(
        vicuna_ckpt_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/DeepSeek-R1-Distill-Llama-8B", # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        # vicuna_ckpt_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/Llama-2-7b-chat-hf",
        # vicuna_ckpt_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/vicuna-7b-v0", # https://huggingface.co/ZzZZCHS/vicuna-7b-v0/tree/main
        using_lora=True,
        freeze_lm=True,
        freeze_tokens=True, # freeze pretrained embed_tokens and lm_head
        lora_r=32,
        lora_alpha=21,
        lora_dropout=0.1,
        lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    ),
    tokenizer_modules=dict(
        tokenizer_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/DeepSeek-R1-Distill-Llama-8B", # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        # tokenizer_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/Llama-2-7b-chat-hf",
        # tokenizer_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/vicuna-7b-v0", # https://huggingface.co/ZzZZCHS/vicuna-7b-v0/tree/main
        new_modality_tokens={"IMAGE": 1, "VIDEO": 1, "AUDIO": 1, "MASK": 1, "BOX": 1},
        new_special_tokens=['[INPUT]', '[OUTPUT]', '[END]', '[IMAGE]', '[VIDEO]', '[AUDIO]', '[BOX]', '[MASK]', '[SMARTMULTIMODAL]', '[SPECIFICMULTIMODAL]', '[TEXT]', '[IMAGESTORY]'],
        bbox_bins=0,
    ),
    system_prompt = "<|system|> You are Spider-Story, an AI assistant that generates structured story descriptions for visual storytelling." \
    "Your task is to output a well-formatted response with the following structure:" \
    "1. **General Prompt**: A brief description of the main character or setting. User may provide corresponding content for it." \
    "2. **Prompt Array**: A sequence of key moments in the story, each describing a separate scene (formatted as a Python list). User may provide corresponding content for it." \
    "3. **Style Name**: Choose a visual style from the list: ['Japanese Anime', 'Digital/Oil Painting', 'Pixar/Disney Character', 'Photographic', 'Comic book', 'Line art', 'Black and White Film Noir', 'Isometric Rooms']. User may provide corresponding content for Style Name, then select the best choice for the user." \
    "### **Example Output Format**" \
    "<GENERALPROMPT> 'a man with a black suit' </GENERALPROMPT> <PROMPTARRAY> ['wake up in the bed', 'have breakfast', 'work in the company', 'reading book in the home'] </PROMPTARRAY> <STYLENAME> 'Comic book' </STYLENAME>" \
    "### **Instructions**" \
    "- `<GENERALPROMPT>` must contain a **quoted string** describing the character or setting." \
    "- `<PROMPTARRAY>` must be a **valid Python list** of quoted strings. Recheck the format of <PROMPTARRAY>, which must be a Python list!" \
    "- `<STYLENAME>` must be a **quoted string** chosen from the predefined list." \
    "- The response **must strictly follow** the above format with XML-like tags." \
    "- **Example Output Format** is the example. The specific content should generate according to the user demand." \
    "Now, generate a structured story description in this format. And carefully recheck the formats of <GENERALPROMPT>, <PROMPTARRAY>, <STYLENAME>.",    
    output_alignment_modules=dict(),
    diffusion_modules=dict(),
    mask_decoder_modules=dict(),
    box_decoder_modules=dict(),
    max_context_len=1000,
)

datasets = dict(
    train=dict(
        ## X-to-T ##
        i2t_cc_sbu=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
        # i2t_cc_sbu=dict(batch_size=1, sample_ratio=20,
        #             build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/cc_sbu/{00000..01255}.tar')),
        v2t_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        # v2t_webvid=dict(batch_size=1, sample_ratio=20,
        #             build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/WebVid/dataset/{0000..02487}.tar')),
        ## Story ##
        flintstones=dict(batch_size=1, sample_ratio=20,
                      build_info=dict(h5_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/flintstones.h5',
                                      subset='train')),
    ),
    val=dict(
        ## Story ##
        # flintstones=dict(batch_size=1, sample_ratio=20,
        #               build_info=dict(h5_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/flintstones.h5',
        #                               subset='val')),
    ),
    test=dict(
        ## Story ##
        # flintstones=dict(batch_size=1, sample_ratio=20,
        #               build_info=dict(h5_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/flintstones.h5',
        #                               subset='test')),
    ),
)


run = dict(
    task="image_text_pretrain",
    lr_sched="linear_warmup_cosine_lr",
    init_lr=1e-5, # deepspeed开启，使用ds_config.json里面的lr, 该init_lr无效
    min_lr=8e-5,
    warmup_lr=1e-6,
    weight_decay=0.05,
    max_epoch=1,
    num_workers=0,
    warmup_steps=500,
    iters_per_epoch=5000,
    seed=41,
    output_dir="/root/autodl-tmp/exp/spider/story",
    amp=True,
    resume_ckpt_path=None,
    # resume_ckpt_path="/root/autodl-tmp/exp/spider/story/20240319222/49",
    only_evaluate=False,
    train_splits=["train"],
    device="cuda",
    world_size=1,
    dist_url="env://",
    distributed=True,
    wandb_log=False,
    job_name="spider_story"
)
