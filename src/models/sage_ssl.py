from typing import List, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

def conv_dw(c_in, c_out, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False),
        nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )

class DepthwiseEncoder(nn.Module):
    def __init__(self, chs=(32,64,128,256,512), in_ch=3):
        super().__init__()
        C = chs
        self.stem = nn.Sequential(nn.Conv2d(in_ch, C[0], 3, padding=1, bias=False), nn.BatchNorm2d(C[0]), nn.ReLU(inplace=True))
        self.e1 = conv_dw(C[0], C[1], s=2)
        self.e2 = conv_dw(C[1], C[2], s=2)
        self.e3 = conv_dw(C[2], C[3], s=2)
        self.e4 = conv_dw(C[3], C[4], s=2)
    def forward(self, x):
        s0 = self.stem(x); s1 = self.e1(s0); s2 = self.e2(s1); s3 = self.e3(s2); s4 = self.e4(s3)
        return [s0, s1, s2, s3, s4]

class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.up(x); x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, chs=(32,64,128,256,512)):
        super().__init__()
        C = chs
        self.b4 = nn.Conv2d(C[4], C[4], 1)
        self.u3 = UpBlock(C[4], C[3], C[3])
        self.u2 = UpBlock(C[3], C[2], C[2])
        self.u1 = UpBlock(C[2], C[1], C[1])
        self.u0 = UpBlock(C[1], C[0], C[0])
    def forward(self, feats: List[torch.Tensor]):
        s0,s1,s2,s3,s4 = feats
        x = self.b4(s4); x = self.u3(x, s3); x = self.u2(x, s2); x = self.u1(x, s1); x = self.u0(x, s0)
        return x

class SKCLite(nn.Module):
    # Low-rank channel attention between labeled/unlabeled bottleneck features
    def __init__(self, C, rank=8, groups=4):
        super().__init__()
        g = max(1, min(groups, C)); r = max(1, min(rank, C//4))
        self.qL = nn.Conv2d(C, r, 1); self.kU = nn.Conv2d(C, r, 1); self.vU = nn.Conv2d(C, C, 1, groups=g)
        self.qU = nn.Conv2d(C, r, 1); self.kL = nn.Conv2d(C, r, 1); self.vL = nn.Conv2d(C, C, 1, groups=g)
        self.beta = nn.Parameter(torch.tensor(0.5))
    def attend(self, q, k, v):
        B, r, H, W = q.shape
        q = rearrange(q, 'b r h w -> b r (h w)'); k = rearrange(k, 'b r h w -> b r (h w)'); v = rearrange(v, 'b c h w -> b c (h w)')
        attn = torch.softmax(torch.bmm(q.transpose(1,2), k), dim=-1)  # (b, hw, hw)
        out = torch.bmm(v, attn).view(B, v.shape[1], H, W)
        return out
    def forward(self, fL, fU):
        LU = self.attend(self.qL(fL), self.kU(fU), self.vU(fU))
        UL = self.attend(self.qU(fU), self.kL(fL), self.vL(fL))
        fL2 = fL + self.beta * LU; fU2 = fU + self.beta * UL
        return fL2, fU2

class NCP(nn.Module):
    # Tiny residual predictor to denoise noisy bottleneck features (training only)
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=4), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1, groups=4), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1))
    def forward(self, f_noisy): return self.net(f_noisy)

class AuxBalancedHead(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(C, C//2, 1), nn.ReLU(inplace=True), nn.Conv2d(C//2, num_classes, 1))
    def forward(self, f): return torch.sigmoid(self.gate(f))

class SAGESSL(nn.Module):
    def __init__(self, num_classes=2, in_ch=3, chs=(32,64,128,256,512), rank=8):
        super().__init__()
        self.enc = DepthwiseEncoder(chs=chs, in_ch=in_ch)
        self.dec_L = Decoder(chs=chs); self.dec_U = Decoder(chs=chs)
        self.head_L = nn.Conv2d(chs[0], num_classes, 1); self.head_U = nn.Conv2d(chs[0], num_classes, 1)
        self.ab_head = AuxBalancedHead(chs[1], num_classes)
        self.skc = SKCLite(chs[-1], rank=rank, groups=4)
        self.ncp = NCP(chs[-1])
    def forward(self, xL: Optional[torch.Tensor]=None, xU: Optional[torch.Tensor]=None, train=True):
        out = {}; featsL = featsU = None
        if xL is not None: featsL = self.enc(xL)
        if xU is not None: featsU = self.enc(xU)
        if train and (featsL is not None and featsU is not None):
            fL, fU = self.skc(featsL[-1], featsU[-1])
            featsL = featsL[:-1] + [fL]; featsU = featsU[:-1] + [fU]
        if featsL is not None:
            decL = self.dec_L(featsL); out["pL"] = self.head_L(decL); out["ab_feats"] = featsL[-2]
        if featsU is not None:
            decU = self.dec_U(featsU); out["pU"] = self.head_U(decU)
        if train and featsU is not None:
            f = featsU[-1]; noise = torch.randn_like(f) * 0.1; f_noisy = f + noise; res = self.ncp(f_noisy)
            featsU_noisy = featsU[:-1] + [f_noisy + res]; out["pU_noisy"] = self.head_U(self.dec_U(featsU_noisy))
        return out
