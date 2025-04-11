import soundfile as sf

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
# qwen_omni = Qwen2_5OmniModel.from_pretrained("/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
qwen_omni = Qwen2_5OmniModel.from_pretrained(
    "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = Qwen2_5OmniProcessor.from_pretrained("/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/Qwen2.5-Omni-7B")


if __name__ == "__main__":
    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs. Only generates text, no audio output.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Who are you?"},
            ],
        },
    ]

    # Conversation with mixed media
    # conversation4 = [
    #     {
    #         "role": "system",
    #         "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text.",
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "/path/to/image.jpg"},
    #             {"type": "video", "video": "/path/to/video.mp4"},
    #             {"type": "audio", "audio": "/path/to/audio.wav"},
    #             {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
    #         ],
    #     }
    # ]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
    inputs = inputs.to(qwen_omni.device).to(qwen_omni.dtype)

    # Inference: Generation of the output text and audio
    text_ids, audio = qwen_omni.generate(**inputs, use_audio_in_video=True)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(text)
    # output audio
    # sf.write(
    #     "output.wav",
    #     audio.reshape(-1).detach().cpu().numpy(),
    #     samplerate=24000,
    # )