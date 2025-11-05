import torch, torch.nn.functional as F
def entropy(p): return -(p * (p+1e-8).log()).sum(dim=1, keepdim=True)
def self_confidence(logits):
    p = torch.softmax(logits, dim=1); H = entropy(p)
    Hn = (H - H.min()) / (H.max() - H.min() + 1e-8)
    return 1.0 - Hn.clamp(0,1)
def mutual_agreement(logits_a, logits_b):
    pa = torch.softmax(logits_a, dim=1); pb = torch.softmax(logits_b, dim=1)
    agree = (pa.argmax(1) == pb.argmax(1)).float().unsqueeze(1)
    return agree
def blended_pseudolabel(logits_u, logits_prior, alpha, conf_drop_fraction=0.2):
    pa = torch.softmax(logits_u, dim=1); pb = torch.softmax(logits_prior, dim=1)
    hard_a = pa.argmax(1, keepdim=True); hard_b = pb.argmax(1, keepdim=True)
    blended = torch.where(alpha>0.5, hard_b, hard_a).squeeze(1).long()
    conf = torch.max(pa, dim=1, keepdim=True).values
    thr = torch.quantile(conf.view(conf.shape[0], -1), q=conf_drop_fraction, dim=1).view(-1,1,1,1)
    mask = (conf >= thr)
    return blended, mask
