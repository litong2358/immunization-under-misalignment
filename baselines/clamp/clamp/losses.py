import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.long().unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()
