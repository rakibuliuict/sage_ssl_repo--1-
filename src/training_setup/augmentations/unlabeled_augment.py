from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, RandGaussianNoised, RandFlipd, RandAffined,
    RandGaussianSmoothd, EnsureTyped, Compose, RandZoomd, RandElasticd
)
def get_unlabeled_weak_transforms():
    keys = ["t2w","adc","hbv"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes='RAS'),
        Spacingd(keys=keys, pixdim=(1.0,1.0,1.0), mode=('bilinear','bilinear','bilinear')),
        ScaleIntensityRanged(keys=keys, a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys),
    ])
def get_unlabeled_strong_transforms():
    keys = ["t2w","adc","hbv"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes='RAS'),
        Spacingd(keys=keys, pixdim=(1.0,1.0,1.0), mode=('bilinear','bilinear','bilinear')),
        ScaleIntensityRanged(keys=keys, a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        RandGaussianNoised(keys=keys, prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=keys, prob=0.3, sigma_x=(0.5,1.2)),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandAffined(keys=keys, prob=0.4, rotate_range=(0.08,0.08,0.08), scale_range=(0.1,0.1,0.0), mode=('bilinear','bilinear','bilinear')),
        RandZoomd(keys=keys, prob=0.3, min_zoom=0.85, max_zoom=1.15, mode=('bilinear','bilinear','bilinear')),
        RandElasticd(keys=keys, prob=0.25, sigma_range=(5,7), magnitude_range=(50,80), mode=('bilinear','bilinear','bilinear')),
        EnsureTyped(keys=keys),
    ])
