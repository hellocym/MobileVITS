import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # [B, 192, T]
        x = x.transpose(1, -1)
        # [B, T, 192]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        x = x.transpose(1, -1)
        # [B, 192, T]
        return 