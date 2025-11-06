# src/training_setup/augmentations/unlabeled_augment.py
from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged,
    Resized, ToTensord, RandGaussianNoised, RandAffined,
    RandFlipd, ConcatItemsd, NormalizeIntensityd,
    RandScaleIntensityd, RandShiftIntensityd, Orientationd
)

IMG_KEYS = ["t2w", "adc", "hbv"]

def get_unlabeled_weak_transforms():
    """
    Weak (light) transforms for unlabeled data.
    - Minimal augmentations, just reorientation, normalization, etc.
    """
    return Compose([
        LoadImaged(keys=IMG_KEYS),
        EnsureChannelFirstd(keys=IMG_KEYS),
        ConcatItemsd(keys=IMG_KEYS, name="img"),
        Resized(keys=["img"], spatial_size=(144, 128, 16)),
        Orientationd(keys=["img"], axcodes="RAS"),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ToTensord(keys=["img"]),
    ])


def get_unlabeled_strong_transforms():
    """
    Strong transforms for unlabeled data.
    - Adds noise, affine jitter, intensity scaling/shifting, and random flips.
    """
    return Compose([
        LoadImaged(keys=IMG_KEYS),
        EnsureChannelFirstd(keys=IMG_KEYS),
        ConcatItemsd(keys=IMG_KEYS, name="img"),
        Resized(keys=["img"], spatial_size=(144, 128, 16)),
        RandAffined(keys=["img"], prob=0.2, translate_range=10.0),
        Orientationd(keys=["img"], axcodes="RAS"),
        RandFlipd(keys=["img"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["img"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["img"], spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys="img", prob=0.4),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="img", factors=0.1, prob=0.4),
        RandShiftIntensityd(keys="img", offsets=0.1, prob=0.4),
        ToTensord(keys=["img"]),
    ])
