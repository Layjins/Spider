# Use a pipeline as a high-level helper
from transformers import pipeline

r1_llama3_8B = pipeline("text-generation", model="/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/DeepSeek-R1-Distill-Llama-8B")

def r1(content):
    messages = [
        {"role": "user", "content": content},
    ]
    return r1_llama3_8B(messages)


if __name__ == "__main__":
    single_run = True
    if single_run:
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        r1_llama3_8B(messages)
    else:
        import pdb
        pdb.set_trace()
        print(r1("Who are you?"))