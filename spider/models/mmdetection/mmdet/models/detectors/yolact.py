# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@MODELS.register_module()
class YOLACT(SingleStageInstanceSegmentor):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 mask_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
