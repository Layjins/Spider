import os
import logging
import warnings
import random
import imageio.v3 as iio
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np

import torch
from torch.utils.data import Dataset

from spider.models.segment_anything.utils.transforms import ResizeLongestSide
from spider.processors.vision_processor import sam_preprocess
from spider.common.registry import registry
from spider.datasets.utils import REFER
from spider.processors import *


class IT2MTsReferCOCODataset(Dataset):
    def __init__(self, image_path, ann_path, dataset='refcoco', splitBy='unc'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.image_path = image_path

        self.refer = REFER(ann_path, image_path, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split="train")

        self.instruction_pool = [
            "Segment and generate {}",
            "Segment {}, and generate the content of it",
            "where the mask and what is {}",
            "where is the mask of {}, and what is it",
            "give me the mask and the content of {}",
            "give me the mask of {}, and the content of it",
            "from this image, tell me the mask and generate the content of {}",
            "could you tell me the mask and generate the content for {}",
            "could you tell me the mask for {}, and generate the content for it",
            "give me the mask of {}, and generate an <IMAGE> of it",
            "give me the mask of {}, and generate a <VIDEO> of it",
            "give me the mask of {}, and generate an <AUDIO> of it",
            "give me the mask of {}, and generate an <IMAGE> and a <VIDEO> for it",
            "give me the mask of {}, and generate an <IMAGE> and an <AUDIO> for it",
            "give me the mask of {}, and generate a <VIDEO> and an <AUDIO> for it",
            "give me the mask of {}, and generate an <IMAGE>, a <VIDEO>, and an <AUDIO> for it",
        ]

    def __len__(self):
        return len(self.ref_ids)

    def preprocess(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        # image
        image_file = 'COCO_train2014_{:0>12}.jpg'.format(ref["image_id"])
        image_path = os.path.join(self.image_path, image_file)
        image = iio.imread(image_path, pilmode="RGB")

        # import pdb
        # pdb.set_trace()

        # mask
        mask = self.refer.getMask(ref)['mask']

        # llama aug
        image_aug, _ = vision_aug_transform(image=image,
                                                  segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape))
        # mask_aug = mask_aug.get_arr()
        image_aug = vision_tensor_transform(image_aug)

        # sam aug
        sam_transform = ResizeLongestSide(1024)
        image_sam = sam_transform.apply_image(image)
        image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        # sam mask aug
        _, mask_aug = vision_aug_transform_1024(image=image,
                                                  segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape))
        mask_aug = mask_aug.get_arr()

        # bounding box os the mask_aug
        # box_mask_aug = None
        indices = np.argwhere(mask_aug)
        min_x = np.min(indices[:, 1])
        max_x = np.max(indices[:, 1])
        min_y = np.min(indices[:, 0])
        max_y = np.max(indices[:, 0])
        box_mask_aug = torch.tensor([min_x, min_y, max_x, max_y])

        # caption
        caption = random.choice(ref['sentences'])['raw']
        caption = text_processor(caption)

        # import pdb
        # pdb.set_trace()

        meta_info = {}
        meta_info['original_shape'] = torch.tensor(image.shape[0:2])
        meta_info['aug_shape'] = torch.tensor(image_aug.shape[1::])
        meta_info['sam_shape'] = torch.tensor(image_sam.shape[1::])

        # meta_info['original_box'] = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        # meta_info['aug_box'] = [box_aug[0].x1, box_aug[0].y1, box_aug[0].x2, box_aug[0].y2]

        return image_aug, mask_aug, box_mask_aug, image_sam, caption, meta_info

    def __getitem__(self, index):
        image_aug, mask_aug, box_mask_aug, image_sam, caption, meta_info = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(caption)
        instruction_q = instruction

        # Determine which modalities to include based on the instruction
        modalities = {
            "<IMAGE>": "<IMAGE>{}<IMAGE-Placeholder></IMAGE>",
            "<VIDEO>": "<VIDEO>{}<VIDEO-Placeholder></VIDEO>",
            "<AUDIO>": "<AUDIO>{}<AUDIO-Placeholder></AUDIO>",
        }

        # Build the question and answer based on the selected instruction
        answer_components = []
        if "<IMAGE>" in instruction:
            instruction_q = instruction_q.replace("<IMAGE>", "image")
            answer_components.append(modalities["<IMAGE>"].format(caption))
        if "<VIDEO>" in instruction:
            instruction_q = instruction_q.replace("<VIDEO>", "video")
            answer_components.append(modalities["<VIDEO>"].format(caption))
        if "<AUDIO>" in instruction:
            instruction_q = instruction_q.replace("<AUDIO>", "audio")
            answer_components.append(modalities["<AUDIO>"].format(caption))
        if ("<IMAGE>" not in instruction) and ("<VIDEO>" not in instruction) and ("<AUDIO>" not in instruction):
            answer_components.append(modalities["<IMAGE>"].format(caption))
            answer_components.append(modalities["<VIDEO>"].format(caption))
            answer_components.append(modalities["<AUDIO>"].format(caption))

        question = "<IMAGE><IMAGE-Placeholder></IMAGE> {}".format(instruction_q)
        answer = "<MASK>{}MASK-Placeholder></MASK>{}".format(caption, "".join(answer_components))

        return {
            "Question": question,
            "TaskPrompt": "[SMARTMULTIMODAL]",
            "Answer": answer,
            "IMAGE": image_aug,
            "IMAGE_SAM": image_sam,
            "MASK": mask_aug,
            "BOX_of_MASK": box_mask_aug,
            "Meta_info": meta_info,
            "Caption": caption
        }


@registry.register_builder("it2m_ts_refcoco")
class IT2MTsRefCOCOBuilder:
    train_dataset_cls = IT2MTsReferCOCODataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building IT2MTsReferCOCO datasets...")

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        return dataset_cls(
            ann_path=ann_path,
            image_path=image_path,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )
