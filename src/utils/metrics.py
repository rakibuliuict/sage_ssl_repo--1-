import torch
def dice_score(pred_logits, target_onehot, eps=1e-5):
    probs = torch.softmax(pred_logits, dim=1)
    intersect = (probs * target_onehot).sum(dim=(2,3))
    denom = probs.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3)) + eps
    dice = (2*intersect + eps) / denom
    return dice.mean().item()
