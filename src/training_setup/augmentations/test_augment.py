from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged, 
    Resized, ToTensord, ConcatItemsd, 
    NormalizeIntensityd, Orientationd
)

def get_test_transforms():
    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="img"),
        Resized(keys=["img", "seg"], spatial_size=(144, 128, 16)),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ToTensord(keys=["img", "seg"]),
    ])