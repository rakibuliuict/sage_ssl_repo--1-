import torch, torch.nn as nn, torch.nn.functional as F
class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5, weight_ce=1.0, weight_dice=1.0):
        super().__init__(); self.smooth=smooth; self.wce=weight_ce; self.wdice=weight_dice
    def forward(self, logits, targets):
        if targets.dtype == torch.long:
            ce = F.cross_entropy(logits, targets)
            one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0,3,1,2).float()
        else:
            one_hot = targets; ce = F.cross_entropy(logits, targets.argmax(1))
        probs = torch.softmax(logits, dim=1)
        intersect = (probs * one_hot).sum(dim=(2,3))
        denom = probs.sum(dim=(2,3)) + one_hot.sum(dim=(2,3)) + self.smooth
        dice = 1 - (2*intersect + self.smooth) / denom
        return self.wce * ce + self.wdice * dice.mean()
