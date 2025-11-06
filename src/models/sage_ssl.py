from typing import List, Optional
import torch, torch.nn as nn
from einops import rearrange

# ------------------------
# 3D building blocks
# ------------------------

def conv_dw_3d(c_in, c_out, k=3, s=1, p=1):
    """Depthwise-separable 3D conv: depthwise + pointwise."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_in, kernel_size=k, stride=s, padding=p, groups=c_in, bias=False),
        nn.Conv3d(c_in, c_out, kernel_size=1, bias=False),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
    )

class DepthwiseEncoder3D(nn.Module):
    def __init__(self, chs=(32,64,128,256,512), in_ch=3):
        super().__init__()
        C = chs
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, C[0], 3, padding=1, bias=False),
            nn.BatchNorm3d(C[0]),
            nn.ReLU(inplace=True)
        )
        # Downsample by stride=2 in D,H,W each stage
        self.e1 = conv_dw_3d(C[0], C[1], s=2)
        self.e2 = conv_dw_3d(C[1], C[2], s=2)
        self.e3 = conv_dw_3d(C[2], C[3], s=2)
        self.e4 = conv_dw_3d(C[3], C[4], s=2)

    def forward(self, x):
        s0 = self.stem(x)  # C0
        s1 = self.e1(s0)   # C1
        s2 = self.e2(s1)   # C2
        s3 = self.e3(s2)   # C3
        s4 = self.e4(s3)   # C4
        return [s0, s1, s2, s3, s4]

class UpBlock3D(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(c_in + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm3d(c_out), nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm3d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Align shapes if off by 1 voxel
        if x.shape[-3:] != skip.shape[-3:]:
            dz = (x.shape[-3] - skip.shape[-3]) // 2
            dy = (x.shape[-2] - skip.shape[-2]) // 2
            dx = (x.shape[-1] - skip.shape[-1]) // 2
            x = x[..., dz:dz+skip.shape[-3], dy:dy+skip.shape[-2], dx:dx+skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Decoder3D(nn.Module):
    def __init__(self, chs=(32,64,128,256,512)):
        super().__init__()
        C = chs
        self.b4 = nn.Conv3d(C[4], C[4], 1)
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
        return x  # channels = C0

class SKCLite3D(nn.Module):
    """Low-rank cross-attention across labeled/unlabeled bottleneck features in 3D."""
    def __init__(self, C, rank=8, groups=4):
        super().__init__()
        g = max(1, min(groups, C))
        r = max(1, min(rank, max(1, C // 4)))
        self.qL = nn.Conv3d(C, r, 1)
        self.kU = nn.Conv3d(C, r, 1)
        self.vU = nn.Conv3d(C, C, 1, groups=g)

        self.qU = nn.Conv3d(C, r, 1)
        self.kL = nn.Conv3d(C, r, 1)
        self.vL = nn.Conv3d(C, C, 1, groups=g)

        self.beta = nn.Parameter(torch.tensor(0.5))

    def attend(self, q, k, v):
        # q: (B, r, D, H, W), k: (B, r, D, H, W), v: (B, C, D, H, W)
        B, r, D, H, W = q.shape
        q = rearrange(q, 'b r d h w -> b r (d h w)')
        k = rearrange(k, 'b r d h w -> b r (d h w)')
        v = rearrange(v, 'b c d h w -> b c (d h w)')
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)  # (B, DHW, DHW)
        out = torch.bmm(v, attn).view(B, v.shape[1], D, H, W)
        return out

    def forward(self, fL, fU):
        LU = self.attend(self.qL(fL), self.kU(fU), self.vU(fU))
        UL = self.attend(self.qU(fU), self.kL(fL), self.vL(fL))
        fL2 = fL + self.beta * LU
        fU2 = fU + self.beta * UL
        return fL2, fU2

class NCP3D(nn.Module):
    """Small residual denoiser on bottleneck features (training only)."""
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(C, C, 3, padding=1, groups=4), nn.ReLU(inplace=True),
            nn.Conv3d(C, C, 3, padding=1, groups=4), nn.ReLU(inplace=True),
            nn.Conv3d(C, C, 1)
        )

    def forward(self, f_noisy):
        return self.net(f_noisy)

class AuxBalancedHead3D(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(C, max(1, C // 2), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, C // 2), num_classes, 1),
        )

    def forward(self, f):
        return torch.sigmoid(self.gate(f))

# ------------------------
# 3D SAGESSL (fixed)
# ------------------------

class SAGESSL(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_ch: int = 3,
        chs=(24, 48, 96, 192, 384),
        rank: int = 8,
        input_order: str = "HWD",   # "HWD" for MONAI tensors (B,C,H,W,D), "DHW" for PyTorch default (B,C,D,H,W)
        return_order: Optional[str] = None,  # if None, match input_order
    ):
        """
        3D variant.

        - Set input_order="HWD" if you feed MONAI-style tensors (your pipeline does).
        - For inputs sized (B, 3, 144, 128, 16), the encoder/decoder will handle any D/H/W.
        """
        super().__init__()
        self.input_order = input_order.upper()
        self.return_order = (return_order or self.input_order).upper()

        self.enc = DepthwiseEncoder3D(chs=chs, in_ch=in_ch)
        self.dec_L = Decoder3D(chs=chs)
        self.dec_U = Decoder3D(chs=chs)
        self.head_L = nn.Conv3d(chs[0], num_classes, 1)
        self.head_U = nn.Conv3d(chs[0], num_classes, 1)

        # IMPORTANT FIX:
        # ab_feats comes from feats[-2] which has channels chs[3], so build head with C=chs[3]
        self.ab_head = AuxBalancedHead3D(chs[3], num_classes)

        self.skc = SKCLite3D(chs[-1], rank=rank, groups=4)
        self.ncp = NCP3D(chs[-1])

    @staticmethod
    def _to_dhw(x, order: str):
        # Convert (B,C,H,W,D) -> (B,C,D,H,W) if needed
        if order == "HWD":
            return x.permute(0, 1, 4, 2, 3).contiguous()
        return x

    @staticmethod
    def _from_dhw(x, target_order: str):
        # Convert (B,C,D,H,W) -> (B,C,H,W,D) if needed
        if target_order == "HWD":
            return x.permute(0, 1, 3, 4, 2).contiguous()
        return x

    def forward(
        self,
        xL: Optional[torch.Tensor] = None,
        xU: Optional[torch.Tensor] = None,
        train: bool = True
    ):
        """
        xL/xU: either (B, C, H, W, D) with input_order="HWD" (MONAI) or (B, C, D, H, W) with "DHW".
        Returns keys: pL, pU, pU_noisy (train), ab_feats (for aux balancing).
        All outputs are returned in return_order (defaults to match input order).
        """
        out = {}
        featsL = featsU = None

        # Reorder to (B,C,D,H,W) for convs
        if xL is not None:
            xL = self._to_dhw(xL, self.input_order)
            featsL = self.enc(xL)
        if xU is not None:
            xU = self._to_dhw(xU, self.input_order)
            featsU = self.enc(xU)

        # Cross-attend bottlenecks during training
        if train and (featsL is not None and featsU is not None):
            fL, fU = self.skc(featsL[-1], featsU[-1])
            featsL = featsL[:-1] + [fL]
            featsU = featsU[:-1] + [fU]

        if featsL is not None:
            decL = self.dec_L(featsL)            # -> C0
            pL = self.head_L(decL)               # [B, num_classes, D, H, W]
            out["pL"] = self._from_dhw(pL, self.return_order)

            # Use C3 features for aux head (matches ab_head channels)
            ab_feats = featsL[-2]                # channels = chs[3]
            out["ab_feats"] = ab_feats           # keep in DHW (feature for loss/inspection)
            out["ab_gate"] = self.ab_head(ab_feats)  # [B, num_classes, D, H, W]
            # ab_gate kept in DHW; adapt as needed by caller

        if featsU is not None:
            decU = self.dec_U(featsU)
            pU = self.head_U(decU)
            out["pU"] = self._from_dhw(pU, self.return_order)

        if train and featsU is not None:
            f = featsU[-1]
            noise = torch.randn_like(f) * 0.1
            f_noisy = f + noise
            res = self.ncp(f_noisy)
            featsU_noisy = featsU[:-1] + [f_noisy + res]
            pU_noisy = self.head_U(self.dec_U(featsU_noisy))
            out["pU_noisy"] = self._from_dhw(pU_noisy, self.return_order)

        return out
