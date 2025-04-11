import os
import json
import random
import logging
import warnings

from torch.utils.data import Dataset
from spider.common.registry import registry


class TravelGuideDataset(Dataset):
    def __init__(self, json_path):
        """
        json_path (string): Path to the travel guide JSON file.
        """
        if not os.path.exists(json_path):
            warnings.warn(f"JSON path {json_path} does not exist.")
            return

        # Load the travel guide data from the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['travel_guide']  # Assuming 'travel_guide' is the root key in JSON

        # Define a set of instructions to generate different types of questions
        self.instruction_pool = [
            "Please provide travel guide for {}.",
            "Please provide travel recommendations for {}.",
            "Tell me about the must-see attractions in {}.",
            "Describe the travelling experience in {}.",
            "Give me some tips for visiting {}.",
            "Could you summarize the key highlights of {}?",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data for the specified index
        entry = self.data[index]

        # Extract location and multimodal content (text, images, etc.)
        location = entry['location']
        content = entry['answer_multimodal']  # Assuming this contains sections like 'introduction', 'must_see_attractions', etc.

        # Select a random instruction from the pool and format it with the location
        question = random.choice(self.instruction_pool).format(location)

        # Build the answer by combining different sections of the guide
        answer_parts = []
        for section, details in content.items():
            section_title = section.replace('_', ' ').title()
            if isinstance(details, list):
                section_content = "\n".join([f" - {item}" for item in details])
            else:
                section_content = details
            answer_parts.append(f"{section_title}:\n{section_content}")
        
        # Join the answer parts into a single string
        answer = "\n".join(answer_parts)

        # Return a dictionary containing the processed data
        return {
            "Question": question,
            "TaskPrompt": "[SMARTMULTIMODAL]",
            "Answer": answer,
            "Location": location,
        }


@registry.register_builder("travel_guide")
class TravelGuideBuilder:
    train_dataset_cls = TravelGuideDataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building Travel Guide datasets...")

        build_info = self.config.build_info
        json_path = build_info.json_path

        if not os.path.exists(json_path):
            warnings.warn(f"JSON path {json_path} does not exist.")
            return None

        # Create datasets
        dataset_cls = self.train_dataset_cls
        return dataset_cls(json_path=json_path)
