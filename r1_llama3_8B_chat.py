from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 DeepSeek-R1-Distill-Llama-8B 模型和分词器
model_name = "/root/autodl-tmp/4e5ee6e154984712803fe75176fe7a38/Pretrain_model/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # 自动分配到 GPU

# 定义聊天函数
def chat(user_input):
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试对话
if __name__ == "__main__":
    user_input = "Who are you?"  # 你可以修改这里的 "content"
    reply = chat(user_input)
    print("Model Response:", reply)
    import pdb
    pdb.set_trace()
    # you can run in pdb: chat("Who are you?")
