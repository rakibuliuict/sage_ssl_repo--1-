from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged, 
    Resized, ToTensord, RandGaussianNoised, RandAffined, 
    RandFlipd, ConcatItemsd, NormalizeIntensityd, 
    RandScaleIntensityd, RandShiftIntensityd, Orientationd
)

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="img"),
        Resized(keys=["img", "seg"], spatial_size=(144, 128, 16)),
        RandAffined(keys=["img", "seg"], prob=0.2, translate_range=10.0),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        RandFlipd(keys=["img", "seg"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys="img", prob=0.4),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="img", factors=0.1, prob=0.4),
        RandShiftIntensityd(keys="img", offsets=0.1, prob=0.4),
        ToTensord(keys=["img", "seg"]),
    ])



# from monai.transforms import (
#     LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
#     ScaleIntensityRanged, RandGaussianNoised, RandFlipd, RandAffined,
#     RandGaussianSmoothd, EnsureTyped, Compose, RandZoomd, RandGridDistortiond
# )
# def get_train_transforms():
#     keys = ["t2w","adc","hbv","seg"]
#     return Compose([
#         LoadImaged(keys=keys),
#         EnsureChannelFirstd(keys=keys),
#         Orientationd(keys=keys, axcodes='RAS'),
#         Spacingd(keys=keys, pixdim=(1.0,1.0,1.0), mode=('bilinear','bilinear','bilinear','nearest')),
#         ScaleIntensityRanged(keys=["t2w","adc","hbv"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
#         RandGaussianNoised(keys=["t2w","adc","hbv"], prob=0.15, mean=0.0, std=0.05),
#         RandGaussianSmoothd(keys=["t2w","adc","hbv"], prob=0.2, sigma_x=(0.5,1.0)),
#         RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
#         RandAffined(keys=keys, prob=0.3, rotate_range=(0.05,0.05,0.05), scale_range=(0.1,0.1,0.0), mode=('bilinear','bilinear','bilinear','nearest')),
#         RandZoomd(keys=keys, prob=0.2, min_zoom=0.9, max_zoom=1.1, mode=('bilinear','bilinear','bilinear','nearest')),
#         # RandGridDistortiond(keys=keys, prob=0.15, sigma_range=(5,7), magnitude_range=(50,80), mode=('bilinear','bilinear','bilinear','nearest')),
#         # RandGridDistortiond(keys=keys, prob=0.25, distort_limit=0.05, mode=("bilinear","bilinear","bilinear")),
#         EnsureTyped(keys=keys),
#     ])
