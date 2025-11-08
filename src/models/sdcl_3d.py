import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from einops import rearrange

def Norm(C, groups=8):
    g = min(groups, C)
    return nn.GroupNorm(num_groups=g, num_channels=C)

def conv_dw3d(c_in, c_out, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(c_in, c_in, k, s, p, groups=c_in, bias=False),
        nn.Conv3d(c_in, c_out, 1, 1, 0, bias=False),
        Norm(c_out),
        nn.ReLU(inplace=True),
    )

class Encoder3D(nn.Module):
    def __init__(self, chs=(16, 32, 64, 128, 256), in_ch=3):
        super().__init__()
        C = chs
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, C[0], 3, padding=1, bias=False),
            Norm(C[0]), nn.ReLU(inplace=True))
        self.e1 = conv_dw3d(C[0], C[1], s=2)
        self.e2 = conv_dw3d(C[1], C[2], s=2)
        self.e3 = conv_dw3d(C[2], C[3], s=2)
        self.e4 = conv_dw3d(C[3], C[4], s=2)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        return [s0, s1, s2, s3, s4]

class UpBlock3D(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(c_in + c_skip, c_out, 3, padding=1, bias=False),
            Norm(c_out), nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 5, padding=2, bias=False),
            Norm(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Decoder3D(nn.Module):
    def __init__(self, chs=(16, 32, 64, 128, 256)):
        super().__init__()
        C = chs
        self.b4 = nn.Sequential(
            nn.Conv3d(C[4], C[4], 3, padding=2, dilation=2, bias=False),
            Norm(C[4]), nn.ReLU(inplace=True))
        self.u3 = UpBlock3D(C[4], C[3], C[3])
        self.u2 = UpBlock3D(C[3], C[2], C[2])
        self.u1 = UpBlock3D(C[2], C[1], C[1])
        self.u0 = UpBlock3D(C[1], C[0], C[0])

    def forward(self, feats: List[torch.Tensor]):
        s0, s1, s2, s3, s4 = feats
        x = self.b4(s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)
        return x

class SDCL3D(nn.Module):
    """
    SDCL: two students share encoder, each has its own decoder+head.
    """
    def __init__(self, num_classes=2, in_ch=3, chs=(16, 32, 64, 128, 256)):
        super().__init__()
        self.enc = Encoder3D(chs=chs, in_ch=in_ch)
        self.dec1 = Decoder3D(chs=chs); self.head1 = nn.Conv3d(chs[0], num_classes, 1)
        self.dec2 = Decoder3D(chs=chs); self.head2 = nn.Conv3d(chs[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.enc(x)
        d1 = self.dec1(feats); d2 = self.dec2(feats)
        p1 = self.head1(d1); p2 = self.head2(d2)
        return p1, p2
