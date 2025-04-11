import gradio as gr
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
    outputs = model.generate(**inputs, max_new_tokens=4096)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("DeepSeek-R1-Distill-Llama-8B Chatbot")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="User Input", placeholder="Enter your message here...")
            submit_btn = gr.Button("Generate Response")
        with gr.Column():
            output_text = gr.Textbox(label="Model Response", interactive=False)

    submit_btn.click(fn=chat, inputs=user_input, outputs=output_text)

# 运行 Web 界面
if __name__ == "__main__":
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    demo.launch(share=True, server_port=6006) # autodl
