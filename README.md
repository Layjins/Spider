# Highlight

The [Inference](#inference) section provides various inference demos for multimodal generation using different model architectures. Key capabilities include:

- **Any-to-Many Generation**: Support for cross-modal inputs and outputs (text, image, audio, video, box, mask)  
- **Specialized Story Generation**: Convert multimodal inputs into coherent text-image story  
- **Model Variants**: Implementations using both Qwen2.5-Omni and DeepSeek-Llama architectures

### Supported Features
✅ Gradio web interfaces  
✅ Python API calls  
✅ Autodl cloud deployment  

> **Quick Start Tip**: We recommend starting with [SpiderFree (Qwen2.5-Omni)](#SpiderFree-Qwen) for quick experimentation with multimodal generation capabilities.


### Our Paper
Spider: Any-to-Many Multimodal LLM: https://arxiv.org/pdf/2411.09439
```bibtex
@article{lai2024spider,
  title={Spider: Any-to-Many Multimodal LLM},
  author={Lai, Jinxiang and Zhang, Jie and Liu, Jun and Li, Jian and Lu, Xiaocheng and Guo, Song},
  journal={arXiv preprint arXiv:2411.09439},
  year={2024}
}
```

# Table of Contents

- [Highlight](#highlight)
    - [Supported Features](#supported-features)
    - [Our Paper](#our-paper)
- [Table of Contents](#table-of-contents)
- [Quick start](#quick-start)
  - [Inference](#inference)
      - [1. SpiderFree (Qwen2.5-Omni)](#1-spiderfree-qwen25-omni)
      - [2. SpiderStory free (Qwen2.5-Omni)](#2-spiderstory-free-qwen25-omni)
      - [3. SpiderStory free (DeepSeek-R1-Distill-Llama-8B)](#3-spiderstory-free-deepseek-r1-distill-llama-8b)
- [Environment setting](#environment-setting)
  - [Spider+Llama3+Qwen+Story Environment](#spiderllama3qwenstory-environment)
      - [Qwen Environment setting](#qwen-environment-setting)
      - [modify deepspeed:](#modify-deepspeed)
      - [mmdet Environment](#mmdet-environment)
- [Train](#train)
  - [Train of Spider](#train-of-spider)
- [Inference](#inference-1)
  - [Inference Demo of Spider](#inference-demo-of-spider)
      - [Spider](#spider)
      - [SpiderStory](#spiderstory)
  - [Inference Demo of SpiderFree](#inference-demo-of-spiderfree)
      - [SpiderStory free (DeepSeek-R1-Distill-Llama-8B)](#spiderstory-free-deepseek-r1-distill-llama-8b)
      - [SpiderStory free (Qwen2.5-Omni)](#spiderstory-free-qwen25-omni)
      - [SpiderFree (Qwen2.5-Omni)](#spiderfree-qwen25-omni)
  - [Inference Demo of DeepSeek-R1-Distill-Llama-8B](#inference-demo-of-deepseek-r1-distill-llama-8b)
  - [Inference Demo of StoryDiffusion](#inference-demo-of-storydiffusion)
  - [Inference Demo of SpiderDecoder](#inference-demo-of-spiderdecoder)
  - [Inference Demo of Qwen2.5-Omni](#inference-demo-of-qwen25-omni)
  - [Inference Demo of NextGPT](#inference-demo-of-nextgpt)
- [Code](#code)
  - [Code Structure](#code-structure)
  - [Code Base](#code-base)
      - [Code Base - Spider](#code-base---spider)
      - [Code Base - Story](#code-base---story)
- [Dataset](#dataset)
  - [Dataset - Spider](#dataset---spider)
  - [Dataset - Story](#dataset---story)
- [Citation](#citation)
- [Contact](#contact)


# Quick start
## [Inference](#Inference)
Includes many models. Some models are recommended as below:

#### 1. [SpiderFree (Qwen2.5-Omni)](#SpiderFree-Qwen)
Any-to-Many modalities generation. The generated examples are shown in [visual.md](./visual.md)

#### 2. [SpiderStory free (Qwen2.5-Omni)](#Spider-Story-Free-Qwen)
Any modalities to text-image story generation.

#### 3. [SpiderStory free (DeepSeek-R1-Distill-Llama-8B)](#Spider-Story-Free-Llama3)
Text to text-image story generation.



# Environment setting
(If you need the docker in autodl, please provide your autodl-ID for docker-sharing, and contact Jinxiang Lai: layjins1994@gmail.com)

## Spider+Llama3+Qwen+Story Environment
<a id="QwenEnvironment"></a>
base docker: PyTorch 2.1.0, Python 3.10(ubuntu22.04), CUDA 12.1

docker: spider_qwen

```shell
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
pip3 install -r requirements_spider_llama3.txt
```

#### Qwen Environment setting
```shell
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
pip3 install -r requirements_spider_qwen.txt
```

1. install transformer with Qwen: https://github.com/QwenLM/Qwen2.5-Omni

or offline install transformer with Qwen: 
```shell
# pip3 uninstall transformers
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/transformers/dist
pip3 install transformers-4.50.0.dev0.tar.gz
# pip3 install accelerate
pip3 install qwen-omni-utils[decord]
sudo apt update && sudo apt install ffmpeg -y
```

2. Alternative: Flash-Attention 2 to speed up generation. manually download: https://github.com/Dao-AILab/flash-attention/releases
```shell
# pip3 install -U flash-attn --no-build-isolation
# pip3 install flash-attn==2.7.3 --no-build-isolation
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model
pip3 install flash_attn-2.5.8+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

3. If Error when do story generation.
```shell
File "/root/miniconda3/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py", line 28, in <module>
    from huggingface_hub import cached_download, hf_hub_download, model_info
ImportError: cannot import name 'cached_download' from 'huggingface_hub' (/root/miniconda3/envs/qwen/lib/python3.10/site-packages/huggingface_hub/__init__.py)
```

remove cached_download
```shell
vim /root/miniconda3/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py
# from huggingface_hub import cached_download, hf_hub_download, model_info
from huggingface_hub import hf_hub_download, model_info # remove cached_download
```

4. Error:
```shell
File "/root/miniconda3/lib/python3.10/site-packages/pytorchvideo/transforms/augmentations.py", line 9, in <module>
    import torchvision.transforms.functional.to_tensor as F_t
ModuleNotFoundError: No module named 'torchvision.transforms.functional.to_tensor'
```

Fix:
```shell
vim /root/miniconda3/lib/python3.10/site-packages/pytorchvideo/transforms/augmentations.py
# import torchvision.transforms.functional.to_tensor as F_t
import torchvision.transforms.functional as F_t
```

5. Fix:
```shell
vim /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/runners/runner_base.py
#from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
```

#### modify deepspeed:

1. pip3 install --upgrade deepspeed==0.16.5

2. In python3 environment，find the path of the installed deepspeed：
import deepspeed
print(deepspeed)
<module 'deepspeed' from '/root/miniconda3/lib/python3.10/site-packages/deepspeed/__init__.py'>

3. Replace the installed deepspeed with the corresponding files in /myGPT/myDeepSpeed0.16.5：

deepspeed/inference/engine.py

deepspeed/module_inject/load_checkpoint.py

4. In Spider/demo/inference_api.py
load_ckpt_mode = 'manul'


#### mmdet Environment
mmcv:
```shell
pip3 install -U openmim
mim install mmengine
# mim install mmcv==2.1.0
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
# https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html
```

mmdet
```shell
mim install mmdet
```

error: AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=1.7.2, <2.2.0
```shell
vim /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider/spider/models/mmdetection/mmdet/__init__.py
mmcv_maximum_version = '2.2.1'
vim /root/miniconda3/lib/python3.10/site-packages/mmdet/__init__.py
mmcv_maximum_version = '2.2.1'
```


nltk_data for grounding DINO: https://blog.csdn.net/qq_43140627/article/details/103895811
```shell
mv /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/nltk_data.zip /root
cd /root
unzip nltk_data.zip
```


# Train
## Train of Spider
docker in autodl: spider_qwen
**1. spider_demo_train**

(1) modify start.sh:

mode="spider_demo_train"

(2) related config:

train_configs/spider_demo_train.py

train_configs/ds_config.json (make sure the "train_batch_size" is adjusted correctly according to the GPU numbers)

(3) Finally:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
sh start.sh
```


# Inference
<a id="Inference"></a>

## Inference Demo of Spider
#### Spider
docker in autodl: spider_qwen

Inference Pipeline: Spider/demo/inference_api.py

(1) select the checkpoint (train by train_configs/spider_demo_train.py), by modifying train_configs/demo_config.json:

"checkpoints": "path/to/checkpoint.pt"

(2) modify demo/frontend.py, "server_name" is the IP of the running machine:

demo.launch(share=True, enable_queue=True, server_port=8081, server_name='11.213.119.213')

demo.launch(share=True, server_port=6006) # autodl

(3) gradio in autodl: https://blog.csdn.net/weixin_43976646/article/details/143723135

E:\jinxianglai\code\AutoDL-SSH-Tools\AutoDL.exe

(4) Corresponding setting in Spider/spider/models/spider.py.
```bash
# init Grounding DINO if needed
init_dino_flag = True
```

(5) Finally:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
# mode="spider_demo_train", in demo.sh
sh demo.sh
```

#### SpiderStory
docker in autodl: spider_qwen

Inference Pipeline: Spider/demo/inference_api.py

(1) select the checkpoint (train by train_configs/spider_story.py), by modifying train_configs/demo_config.json:

"checkpoints": "path/to/checkpoint.pt"

(2) Finally:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
# mode="spider_story", in demo.sh
sh demo.sh
```


## Inference Demo of SpiderFree
#### SpiderStory free (DeepSeek-R1-Distill-Llama-8B)
<a id="Spider-Story-Free-Llama3"></a>
docker in autodl: spider_qwen

Inference Pipeline: Spider/demo/inference_api.py

(1) Finally:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
# mode="spider_story_free_llama3", in demo.sh
sh demo.sh
```


#### SpiderStory free (Qwen2.5-Omni)
<a id="Spider-Story-Free-Qwen"></a>
docker in autodl: spider_qwen

Inference Pipeline: Spider/qwen2.5omni_spider_web.py

1. Chatbot in web:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
# MODEL_NAME="spider_story_free_qwen", in qwen2.5omni_spider_web.py
python3 qwen2.5omni_spider_web.py
```

#### SpiderFree (Qwen2.5-Omni)
<a id="SpiderFree-Qwen"></a>
docker in autodl: spider_qwen

Config: Spider/train_configs/spider_decoder_cfg.py (Note: We are still working on designing a better system prompt to let Qwen2.5-Omni output a better formated text.)

Inference Pipeline: Spider/spider_decoder_infer.py

Gradio: Spider/qwen2.5omni_spider_web.py


1. Chatbot in web:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
# MODEL_NAME="spider_free_qwen", in qwen2.5omni_spider_web.py
python3 qwen2.5omni_spider_web.py
```



## Inference Demo of DeepSeek-R1-Distill-Llama-8B
docker in autodl: spider_qwen

1. Text Chatbot in gradio:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 r1_llama3_8B_gradio.py
```

2. Text Chat in python:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 r1_llama3_8B_chat.py
```

3. Text generation in python:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 r1_llama3_8B_infer.py
```


## Inference Demo of StoryDiffusion
docker in autodl: spider_qwen

```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 story_diffusion_infer.py
```

## Inference Demo of SpiderDecoder
docker in autodl: spider_qwen

```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 spider_decoder_infer.py
```


## Inference Demo of Qwen2.5-Omni
docker in autodl: spider_qwen

1. Chatbot in web:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 qwen2.5omni_web.py
```

2. Inference in python:
```bash
cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Spider
python3 qwen2.5omni_infer.py
```



## Inference Demo of NextGPT
docker in autodl: nextgpt

```bash
conda activate nextgpt

cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Opencodes-Multimodal/NExT-GPT/NExT-GPT-old-jinxiang/ckpt/pretrained_ckpt/imagebind_ckpt/huge
ln -s /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/imagebind_huge.pth

cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Opencodes-Multimodal/NExT-GPT/NExT-GPT-old-jinxiang/ckpt/pretrained_ckpt/vicuna_ckpt
ln -s /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/vicuna/7b_v0 7b_v0

cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Opencodes-Multimodal/NExT-GPT/NExT-GPT-old-jinxiang/ckpt/delta_ckpt/nextgpt
rm -rf 7b_tiva_v0
ln -s /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/nextgpt_7b_tiva_v0 7b_tiva_v0

cd /root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/myGPT/Opencodes-Multimodal/NExT-GPT/NExT-GPT-old-jinxiang/code
bash scripts/app.sh
```

# Code
## Code Structure

The code of the model is in spider/models/spider.py

The inference code: Spider/demo/inference_api.py


## Code Base
#### Code Base - Spider
https://github.com/Vision-CAIR/MiniGPT-4

https://github.com/NExT-GPT/NExT-GPT

https://github.com/microsoft/unilm/tree/master/kosmos-2

https://github.com/dvlab-research/LISA

https://github.com/QwenLM/Qwen2.5-Omni


#### Code Base - Story
https://github.com/HVision-NKU/StoryDiffusion

https://github.com/xichenpan/ARLDM

https://github.com/TencentARC/SEED-Story


# Dataset
## Dataset - Spider
https://github.com/Vision-CAIR/MiniGPT-4

https://github.com/NExT-GPT/NExT-GPT

https://huggingface.co/datasets/sailvideo/webvid10m/tree/main

https://huggingface.co/datasets/Olivia714/audiocaps

https://github.com/NExT-GPT/NExT-GPT/blob/main/data/T_X_pair_data/audiocap/prepare.md


## Dataset - Story
https://github.com/xichenpan/ARLDM?tab=readme-ov-file

https://github.com/TencentARC/SEED-Story




# Citation
If you use this code for your research, please cite our paper:
```bibtex
@article{lai2024spider,
  title={Spider: Any-to-Many Multimodal LLM},
  author={Lai, Jinxiang and Zhang, Jie and Liu, Jun and Li, Jian and Lu, Xiaocheng and Guo, Song},
  journal={arXiv preprint arXiv:2411.09439},
  year={2024}
}
```

# Contact
Jinxiang Lai: layjins1994@gmail.com