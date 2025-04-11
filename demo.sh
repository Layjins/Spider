# mode="spider_stage1_pretrain"
# mode="spider_stage2_pretrain"
# mode="spider_stage3_pretrain"
# mode="grounding_dino_test"
# mode="spider_demo_train"
# mode="spider_story"
mode="spider_story_free_llama3"



if [ "$mode" == "spider_story_free_llama3" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    python3 demo/frontend.py train_configs/spider_story_free_llama3.py
fi

if [ "$mode" == "spider_demo_train" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0 \
            --master_port 60000 \
            demo/frontend.py train_configs/spider_demo_train.py
fi

if [ "$mode" == "spider_story" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0 \
            --master_port 60000 \
            demo/frontend.py train_configs/spider_story.py
fi

if [ "$mode" == "spider_stage1_pretrain" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0 \
            --master_port 60000 \
            demo/frontend.py train_configs/spider_stage1_pretrain.py
fi

if [ "$mode" == "spider_stage2_pretrain" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0 \
            --master_port 60000 \
            demo/frontend.py train_configs/spider_stage2_pretrain.py
fi

if [ "$mode" == "spider_stage3_pretrain" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    deepspeed --include=localhost:0 \
            --master_port 60000 \
            demo/frontend.py train_configs/spider_stage3_pretrain.py
fi


if [ "$mode" == "grounding_dino_test" ]
then
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection:$PYTHONPATH"
    export PYTHONPATH="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet:$PYTHONPATH"

    python3 grounding_dino_test.py
fi

