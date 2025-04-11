import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn


title = """<h1 align="center">Spider Demo</h1>"""
description = """<h3 align="center">Welcome to Our Spider, a multimodal LLM!</h3>"""


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=2):
            gr.CheckboxGroup(["Image", "Audio", "Video", "Box", "Mask"], label="Select Input Modalities")
            image = gr.Image(type="pil", tool='sketch', brush_radius=20)
            audio = gr.Audio(type="filepath")
            video = gr.Video()

        with gr.Column(scale=6):
            with gr.Row():
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Emu Chatbot",
                    visible=True,
                    height=1070,
                )

            with gr.Row():
                with gr.Column(scale=8):
                    textbox = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and add to prompt",
                        visible=True,
                        container=False,
                    )

                with gr.Column(scale=1, min_width=60):
                    add_btn = gr.Button(value="Send")

                with gr.Column(scale=1, min_width=60):
                    add_btn = gr.Button(value="Clear")

        with gr.Column(scale=2):
            gr.CheckboxGroup(["Image", "Audio", "Video", "Box", "Mask"], label="Select Output Modalities")
            image = gr.Image(type="pil", tool='sketch', brush_radius=20)
            audio = gr.Audio(type="filepath")
            video = gr.Video()


demo.launch(share=True, enable_queue=True, server_port=8000, server_name='0.0.0.0')
