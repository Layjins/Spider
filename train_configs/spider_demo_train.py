pretrained_ckpt_path=None
# pretrained_ckpt_path="/jinxianglai/exp/spider/stage3/2024061316383/45"

model = dict(
    type="spider", # spider_free: free training, spider: training
    name="spider",
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
        freeze_lm=False,
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
        new_special_tokens=['[INPUT]', '[OUTPUT]', '[END]', '[IMAGE]', '[VIDEO]', '[AUDIO]', '[BOX]', '[MASK]', '[SMARTMULTIMODAL]', '[SPECIFICMULTIMODAL]', '[TEXT]'],
        bbox_bins=0,
    ),
    system_prompt = None,
    output_alignment_modules=dict(
        IMAGE=dict(alignment_module_mode='moe_transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                 alignment_input_tokens=1, alignment_output_tokens=77, alignment_output_dim=768), # alignment_input_tokens is invalid when alignment_module_mode='transformer'
        VIDEO=dict(alignment_module_mode='moe_transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                 alignment_input_tokens=1, alignment_output_tokens=77, alignment_output_dim=1024),
        AUDIO=dict(alignment_module_mode='moe_transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                 alignment_input_tokens=1, alignment_output_tokens=1, alignment_output_dim=512),
        MASK=dict(alignment_module_mode='moe_transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
                  alignment_input_tokens=1, alignment_output_tokens=1, alignment_output_dim=256),
        # BOX=dict(alignment_module_mode='moe_transformer', reconstruct_loss=False, alignment_layer=[-1], freeze_output_alignment_proj=False,
        #          alignment_input_tokens=1, alignment_output_tokens=1, alignment_output_dim=256),
    ),
    diffusion_modules=dict(
        IMAGE=dict(type="sd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/stable-diffusion-v1-5'),
        VIDEO=dict(type="vd", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/zeroscope_v2_576w'),
        AUDIO=dict(type="ad", ckpt='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/audioldm-l-full'),
    ),
    mask_decoder_modules=dict(
        sam_path="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/sam_vit_h_4b8939.pth",
        freeze_mask_decoder=False,
    ),
    box_decoder_modules=dict(
        # grounding DINO
        config_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py',
        checkpoint_file='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth',
    ),
    max_context_len=500,
)

datasets = dict(
    train=dict(
        #### X-to-X dataset for X-to-X pretraining ####
        it2b_refcoco=dict(batch_size=1, sample_ratio=20,
                      build_info=dict(image_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/COCO2014/train2014',
                                      ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Referring_Expression',
                                      dataset='refcoco',
                                      splitBy='unc')),
        it2m_refcoco=dict(batch_size=1, sample_ratio=20,
                      build_info=dict(image_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/COCO2014/train2014',
                                      ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Referring_Expression',
                                      dataset='refcoco',
                                      splitBy='unc')),
        ## X-to-T ##
        i2t_cc_sbu=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
        v2t_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        a2t_audiocap=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(audio_dir='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/wav_files',
                                    ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/test.json')),
        ## T-to-X ##
        t2i_cc_sbu=dict(batch_size=1, sample_ratio=50,
                    build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
        t2v_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        t2a_audiocap=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(audio_dir='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/wav_files',
                                    ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/test.json')),

        #### T-to-Ts dataset for T-to-Xs pretraining ####
        t2i_ts_cc_sbu=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
        t2v_ts_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        t2a_ts_audiocap=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(audio_dir='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/wav_files',
                                    ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/test.json')),
        t2v_ts_mul_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        t2v_ts_spec_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        travel_guide=dict(batch_size=1, sample_ratio=20,
                      build_info=dict(json_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/datasets/travel_guide.json')),

        #### X-to-Ts dataset for X-to-Xs pretraining ####
        i2t_ts_cc_sbu=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
        v2t_ts_webvid=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(webdataset_path=[
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000000.tar',
                        '/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_webvid/00000001.tar',
                    ])),
        a2t_ts_audiocap=dict(batch_size=1, sample_ratio=20,
                    build_info=dict(audio_dir='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/wav_files',
                                    ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/AudioCaps/test.json')),
        it2b_ts_refcoco=dict(batch_size=1, sample_ratio=20,
                      build_info=dict(image_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/COCO2014/train2014',
                                      ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Referring_Expression',
                                      dataset='refcoco',
                                      splitBy='unc')),
        it2m_ts_refcoco=dict(batch_size=1, sample_ratio=20,
                      build_info=dict(image_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/COCO2014/train2014',
                                      ann_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Referring_Expression',
                                      dataset='refcoco',
                                      splitBy='unc')),
    ),
    val=dict(
        # i2t_cc_sbu=dict(batch_size=1,
        #             build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
    ),
    test=dict(
        # i2t_cc_sbu=dict(batch_size=1,
        #             build_info=dict(webdataset_path='/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/dataset/small_cc_sbu/{00000..00003}.tar')),
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
    iters_per_epoch=50000,
    seed=41,
    output_dir="/jinxianglai/exp/spider/demo",
    amp=True,
    resume_ckpt_path=None,
    # resume_ckpt_path="/jinxianglai/exp/spider/stage4/20240319222/49",
    only_evaluate=False,
    train_splits=["train"],
    device="cuda",
    world_size=1,
    dist_url="env://",
    distributed=True,
    wandb_log=False,
    job_name="spider_stage_pretrain"
)
