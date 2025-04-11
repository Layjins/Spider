# mode
# mode="sleep"
# mode="spider_stage1_pretrain"
# mode="spider_stage2_pretrain"
# mode="spider_stage3_pretrain"
mode="spider_demo_train"
# mode="spider_story"


# sleep
if [ "$mode" == "sleep" ]
then
    python3 sleep.py
fi


# spider_demo_train
if [ "$mode" == "spider_demo_train" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    # 提高gpu利用率: run_me_bkg.py
    # python3 run_me_bkg.py &
    deepspeed --include=localhost:0 --master_port 60000 train.py train_configs/spider_demo_train.py
    # deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 60000 train.py train_configs/spider_demo_train.py
fi


# spider_story
if [ "$mode" == "spider_story" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    # 提高gpu利用率: run_me_bkg.py
    # python3 run_me_bkg.py &
    deepspeed --include=localhost:0 --master_port 60000 train.py train_configs/spider_story.py
    # deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 60000 train.py train_configs/spider_story.py
fi


# spider_stage1_pretrain
if [ "$mode" == "spider_stage1_pretrain" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0,1,2,3 --master_port 60000 train.py train_configs/spider_stage1_pretrain.py
    # deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 60000 train.py train_configs/spider_stage1_pretrain.py
fi

# spider_stage2_pretrain
if [ "$mode" == "spider_stage2_pretrain" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0,1,2,3 --master_port 60000 train.py train_configs/spider_stage2_pretrain.py
    # deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 60000 train.py train_configs/spider_stage2_pretrain.py
fi

# spider_stage3_pretrain
if [ "$mode" == "spider_stage3_pretrain" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0,1,2,3 --master_port 60000 train.py train_configs/spider_stage3_pretrain.py
    # deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 60000 train.py train_configs/spider_stage3_pretrain.py
fi

