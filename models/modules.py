import torch
from torch import nn
from torch.nn import functional as F

from wavenet import WN

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
        return x

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)  # 0 vector, [B]
            return x, logdet
        else:
            return x
        

class ResiduleCouplingLayer(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0.0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels must be even"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        # self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        if mean_only:
            self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        else:
            self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()


    def forward(self, x, x_mask, g=None, reverse=False):
        # x: [B, 192, T]
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)    # [B, 96, T], [B, 96, T]
        h = self.pre(x0) * x_mask   # [B, 192, T]
        h = self.enc(h, x_mask, g=g)   # [B, 192, T]
        stats = self.post(h) * x_mask   # [B, 192, T] for mean_only=False, [B, 96, T] for mean_only=True
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)   # [B, 96, T], [B, 96, T]    
        else:
            m = stats   # [B, 96, T]
            logs = torch.zeros_like(m)  
        
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask   # [B, 96, T]
            x = torch.cat([x0, x1], 1)  # [B, 192, T]
            logdet = torch.sum(logs, dim=[1, 2])  # Log Jacobian Determinant, [B]
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask   # [B, 96, T]
            x = torch.cat([x0, x1], 1)  # [B, 192, T]
            return x
