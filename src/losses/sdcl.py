import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(targets, C):
    return F.one_hot(targets, num_classes=C).permute(0, 4, 1, 2, 3).float()

def dice_ce_3d(logits, targets, smooth=1e-5, w_ce=1.0, w_dice=1.0):
    C = logits.shape[1]
    if targets.dtype == torch.long:
        ce = F.cross_entropy(logits, targets)
        oh = one_hot(targets, C)
    else:
        oh = targets
        ce = F.cross_entropy(logits, targets.argmax(1))
    dims = (2, 3, 4)
    p = torch.softmax(logits, dim=1)
    inter = (p * oh).sum(dim=dims)
    denom = p.sum(dim=dims) + oh.sum(dim=dims) + smooth
    dice = 1 - (2 * inter + smooth) / denom
    return w_ce * ce + w_dice * dice.mean()

def sym_kl(p_logits, q_logits, mask=None):
    p = torch.softmax(p_logits, dim=1)
    q = torch.softmax(q_logits, dim=1)
    kl_pq = F.kl_div((p + 1e-8).log(), q, reduction="none").sum(dim=1, keepdim=True)
    kl_qp = F.kl_div((q + 1e-8).log(), p, reduction="none").sum(dim=1, keepdim=True)
    kl = kl_pq + kl_qp
    if mask is not None:
        kl = (kl * mask).sum() / (mask.sum() + 1e-8)
    else:
        kl = kl.mean()
    return kl

def entropy(logits):
    p = torch.softmax(logits, dim=1)
    return -(p * (p + 1e-8).log()).sum(dim=1, keepdim=True)

class SDCLUnlabeledLoss(nn.Module):
    def __init__(self, w_agree=1.0, w_disagree=0.5, temp=1.0):
        super().__init__()
        self.w_agree = w_agree
        self.w_disagree = w_disagree
        self.temp = temp

    def forward(self, s1_logits, s2_logits):
        s1 = s1_logits / self.temp
        s2 = s2_logits / self.temp
        hard1 = s1.softmax(1).argmax(1, keepdim=True)
        hard2 = s2.softmax(1).argmax(1, keepdim=True)
        agree = (hard1 == hard2).float()
        disagree = 1.0 - agree
        loss_agree = sym_kl(s1, s2, mask=agree)
        negH = -entropy(s1) - entropy(s2)
        loss_disagree = (negH * disagree).sum() / (disagree.sum() + 1e-8)
        return self.w_agree * loss_agree + self.w_disagree * loss_disagree
