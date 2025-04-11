from imgaug import augmenters as iaa

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

vision_aug_transform = iaa.Sequential([
            iaa.Resize({"longer-side": 224, "shorter-side": "keep-aspect-ratio"}),
            # iaa.CropToFixedSize(width=224, height=224),
            iaa.PadToFixedSize(width=224, height=224),
        ])

vision_aug_transform_512 = iaa.Sequential([
            iaa.Resize({"longer-side": 512, "shorter-side": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(width=512, height=512),
        ])

vision_aug_transform_1024 = iaa.Sequential([
            iaa.Resize({"longer-side": 1024, "shorter-side": "keep-aspect-ratio"}),
            iaa.PadToFixedSize(width=1024, height=1024),
        ])

vision_tensor_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
     ]
)


torch_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.2),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
            ]
        )



def sam_preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def detr_preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 512
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x