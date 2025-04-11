import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">Spider Demo</h1>""")
    gr.Markdown("""<h3 align="center">Welcome to Our Spider, a multimodal LLM!</h3>""")

    with gr.Row():
        # 左侧
        with gr.Column(scale=2):
            input_modalities = gr.CheckboxGroup(["Image", "Box", "Mask", "Audio", "Video"], label="Select Input Modalities")
            imagebox = gr.Image(type="pil", tool='sketch', brush_radius=20, visible=True)
            audiobox = gr.Audio(type="filepath", visible=False)
            videobox = gr.Video(visible=False)

        # 中间
        with gr.Column(scale=6):
            # 对话框
            with gr.Row():
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Emu Chatbot",
                    visible=True,
                    height=1070,
                )

            with gr.Row():
                # 输入框
                with gr.Column(scale=8):
                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and add to prompt",
                        visible=True,
                        container=False,
                    )

                # 发送按钮
                with gr.Column(scale=1, min_width=60):
                    send_btn = gr.Button(value="Send")

                # 清空历史
                with gr.Column(scale=1, min_width=60):
                    clear_btn = gr.Button(value="Clear")

        # 右侧
        with gr.Column(scale=2):
            output_modalities = gr.CheckboxGroup(["Image", "Audio", "Video", "Box", "Mask"], label="Select Output Modalities")
            image_output = gr.Image(type="pil", tool='sketch', brush_radius=20)
            audio_output = gr.Audio(type="filepath")
            video_output = gr.Video()

    @input_modalities.observe
    def update_input_modalities():
        if "Image" in input_modalities.value:
            audiobox.visible = False
            videobox.visible = False
            if "Audio" in input_modalities.value:
                input_modalities.value.remove("Audio")
            if "Video" in input_modalities.value:
                input_modalities.value.remove("Video")
        elif "Box" in input_modalities.value:
            audiobox.visible = False
            videobox.visible = False
            if "Mask" in input_modalities.value:
                input_modalities.value.remove("Mask")
            if "Image" not in input_modalities.value:
                input_modalities.value.append("Image")
        elif "Mask" in input_modalities.value:
            audiobox.visible = False
            videobox.visible = False
            if "Box" in input_modalities.value:
                input_modalities.value.remove("Box")
            if "Image" not in input_modalities.value:
                input_modalities.value.append("Image")
        elif "Audio" in input_modalities.value:
            imagebox.visible = False
            videobox.visible = False
            if "Image" in input_modalities.value:
                input_modalities.value.remove("Image")
            if "Box" in input_modalities.value:
                input_modalities.value.remove("Box")
            if "Mask" in input_modalities.value:
                input_modalities.value.remove("Mask")
        elif "Video" in input_modalities.value:
            imagebox.visible = False
            audiobox.visible = False
            if "Image" in input_modalities.value:
                input_modalities.value.remove("Image")
            if "Box" in input_modalities.value:
                input_modalities.value.remove("Box")
            if "Mask" in input_modalities.value:
                input_modalities.value.remove("Mask")

demo.launch(share=True, enable_queue=True, server_port=8000, server_name='0.0.0.0')