import os
import logging
import warnings
import random
import imageio.v3 as iio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
from torch.utils.data import Dataset

from spider.models.segment_anything.utils.transforms import ResizeLongestSide
from spider.processors.vision_processor import detr_preprocess
from spider.common.registry import registry
from spider.datasets.utils import REFER
from spider.processors import *


class IT2BReferCOCODataset(Dataset):
    def __init__(self, image_path, ann_path, dataset='refcoco', splitBy='unc'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.image_path = image_path

        self.refer = REFER(ann_path, image_path, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split="train")

        self.instruction_pool = [
            "{}",
            "Detect {}",
            "give me the bounding box of {}",
            "give me the location of {}",
            "where is {}",
            "from this image, tell me the location of {}",
            "the location of {} is",
            "could you tell me the location for {}",
            "where can I locate the {}",
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
        # print(image.shape)

        # box
        box = self.refer.getRefBox(ref['ref_id'])
        # print("ori-ori", box)

        # augmentation
        # import pdb
        # pdb.set_trace()
        # print("ori:", box)
        # print("ori:", image.shape)
        # import pdb
        # pdb.set_trace()
        image_aug, _ = vision_aug_transform(image=image,
                                            bounding_boxes=BoundingBoxesOnImage(
                                                [BoundingBox(x1=int(box[0]),
                                                            y1=int(box[1]),
                                                            x2=int(box[0] + box[2]),
                                                            y2=int(box[1] + box[3]))],
                                                shape=image.shape))

        # print("aug:", box_aug)
        # print("aug:", image_aug.shape)
        # import pdb
        # pdb.set_trace()

        image_aug = vision_tensor_transform(image_aug)

        # import pdb
        # pdb.set_trace()
        # detr aug
        detr_transform = ResizeLongestSide(512)
        image_detr = detr_transform.apply_image(image)
        image_detr = detr_preprocess(torch.from_numpy(image_detr).permute(2, 0, 1).contiguous())

        # box gt
        _, box_aug = vision_aug_transform_512(image=image,
                                            bounding_boxes=BoundingBoxesOnImage(
                                                [BoundingBox(x1=int(box[0]),
                                                            y1=int(box[1]),
                                                            x2=int(box[0] + box[2]),
                                                            y2=int(box[1] + box[3]))],
                                                shape=image.shape))
        box_aug = box_aug.remove_out_of_image().clip_out_of_image()
        box_aug_format = torch.tensor([box_aug[0].x1, box_aug[0].y1, box_aug[0].x2, box_aug[0].y2])

        # import pdb
        # pdb.set_trace()
        # caption
        caption = random.choice(ref['sentences'])['raw']
        caption = text_processor(caption)

        # import pdb
        # pdb.set_trace()

        meta_info = {}
        meta_info['original_shape'] = torch.tensor(image.shape[0:2])
        meta_info['aug_shape'] = torch.tensor(image_aug.shape[1::])
        meta_info['original_box'] = torch.tensor([box[0], box[1], box[0] + box[2], box[1] + box[3]])  # x1y1x2y2
        meta_info['aug_box'] = box_aug_format

        return image_aug, box_aug_format, image_detr, caption, meta_info, image

    def __getitem__(self, index):
        # import torch.distributed as dist
        # import logging
        # if dist.get_rank() == 0:
        #     logging.info(f"++++++++++++++++++++++++++++{str(index)}")
        # elif dist.get_rank() == 1:
        #     logging.info(f"----------------------------{str(index)}")

        image_aug, box_aug, image_detr, caption, meta_info, image_ori_array = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(caption)

        question = "<IMAGE><IMAGE-Placeholder></IMAGE> {} ".format(instruction)

        # answer = "<BOX><BOX-Placeholder></BOX>"
        # answer = "<BOX><BOX-Placeholder></BOX> {} ".format(caption)
        answer = "<BOX>{}<BOX-Placeholder></BOX>".format(caption)

        return {
            "Question": question,
            "TaskPrompt": "[BOX]",
            "Answer": answer,
            "IMAGE": image_aug,
            "IMAGE_DETR": image_detr,
            "BOX": box_aug,
            "Meta_info": meta_info,
            "Image_ori_array": image_ori_array,
            "Caption": caption
        }


@registry.register_builder("it2b_refcoco")
class IT2BRefCOCOBuilder:
    train_dataset_cls = IT2BReferCOCODataset

    def __init__(self, cfg=None):
        self.config = cfg

    def build_datasets(self):
        logging.info("Building IT2BReferCOCO datasets...")

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
