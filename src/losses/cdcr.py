import torch, torch.nn as nn, torch.nn.functional as F
class CDCRLoss(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__(); self.tau=tau
    def forward(self, p1_logits, p2_logits):
        p1 = torch.softmax(p1_logits, dim=1); p2 = torch.softmax(p2_logits, dim=1)
        p_avg = (p1 + p2) / 2.0
        p_sharp = (p_avg ** (1.0 / (self.tau + 1e-6))); p_sharp = p_sharp / (p_sharp.sum(dim=1, keepdim=True) + 1e-6)
        loss = F.kl_div((p1+1e-8).log(), p_sharp, reduction='batchmean') + F.kl_div((p2+1e-8).log(), p_sharp, reduction='batchmean')
        ent = -(p_sharp * (p_sharp+1e-8).log()).sum(dim=1).mean()
        return loss + 0.1 * ent
