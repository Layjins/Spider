"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from spider.processors.base_processor import BaseProcessor
from spider.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor,
    BlipCaptionProcessor,
)
from spider.processors.vision_processor import vision_aug_transform, vision_aug_transform_512, vision_aug_transform_1024, vision_tensor_transform, torch_transform
from spider.processors.text_processor import text_processor

from spider.common.registry import registry

# __all__ = [
#     "BaseProcessor",
#     "Blip2ImageTrainProcessor",
#     "Blip2ImageEvalProcessor",
#     "BlipCaptionProcessor",
# ]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
