from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, Compose
)
def get_test_transforms():
    keys = ["t2w","adc","hbv","seg"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes='RAS'),
        Spacingd(keys=keys, pixdim=(1.0,1.0,1.0), mode=('bilinear','bilinear','bilinear','nearest')),
        ScaleIntensityRanged(keys=["t2w","adc","hbv"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=keys),
    ])
