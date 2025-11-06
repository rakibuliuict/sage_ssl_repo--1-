# src/losses/dice_ce.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    """
    Combines Cross Entropy and (soft) Dice.
    - Supports 2D logits [B,C,H,W] and 3D logits [B,C,D,H,W].
    - Supports hard targets [B,H,W] / [B,D,H,W] (int) and soft targets [B,C,...].
    """
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.smooth = smooth

    def _spatial_dims(self, x):
        # returns (2,3) for 2D or (2,3,4) for 3D logits
        return tuple(range(2, x.ndim))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,H,W] or [B,C,D,H,W]
        targets:
           - hard: [B,H,W] or [B,D,H,W] (long)
           - soft: [B,C,H,W] or [B,C,D,H,W] (probabilities/one-hot)
        """
        assert logits.ndim in (4, 5), f"Expected 4D/5D logits, got {logits.shape}"

        num_classes = logits.shape[1]
        dims = self._spatial_dims(logits)
        probs = torch.softmax(logits, dim=1)

        # ---- Cross-Entropy ----
        if targets.ndim == logits.ndim and targets.shape[1] == num_classes:
            # soft targets (e.g., pseudo labels). Use soft CE: -sum q * log p
            q = targets.clamp(min=0, max=1)
            ce_map = -(q * (probs.clamp_min(1e-8)).log()).sum(dim=1)  # [B, *spatial]
            ce = ce_map.mean()
            tgt_for_dice = q
        else:
            # hard targets (integer labels)
            if targets.dtype != torch.long:
                targets = targets.long()
            ce = F.cross_entropy(logits, targets)
            # one-hot for dice
            one_hot = F.one_hot(targets, num_classes=num_classes)
            if logits.ndim == 4:      # [B,H,W,C] -> [B,C,H,W]
                one_hot = one_hot.permute(0, 3, 1, 2).float()
            else:                      # [B,D,H,W,C] -> [B,C,D,H,W]
                one_hot = one_hot.permute(0, 4, 1, 2, 3).float()
            tgt_for_dice = one_hot

        # ---- Dice ----
        intersection = (probs * tgt_for_dice).sum(dim=dims)         # [B,C]
        cardinality = probs.sum(dim=dims) + tgt_for_dice.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_w * ce + self.dice_w * dice_loss
