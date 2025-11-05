class EMA:
    def __init__(self, model, decay=0.99):
        self.shadow = {}; self.decay = decay
        for name, param in model.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    def apply_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad: param.data = self.shadow[name].clone()
