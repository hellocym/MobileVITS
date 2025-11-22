import torch
from torch import nn
from wavenet import WN

import commons

class PosteriorEncoder(nn.Module):
    def __init__(self, 
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)    # [B, 1, T]
        x = self.pre(x) * x_mask                                # [B, hidden_channels, T]
        x = self.enc(x, x_mask, g=g)                            # [B, hidden_channels, T]
        stats = self.proj(x) * x_mask                           # [B, out_channels*2, T]
        mu, logs = torch.split(stats, self.out_channels, dim=1) # [B, out_channels, T], [B, out_channels, T]
        z = (mu + torch.randn_like(mu) * torch.exp(logs)) * x_mask  # 采样 [B, out_channels, T]
        return z, mu, logs, x_mask


if __name__ == '__main__':
    import json
    json_path = 'configs/chinese_base.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    posterior_encoder = PosteriorEncoder(
        in_channels=config['data']['filter_length'] // 2 + 1,
        out_channels=config['model']['inter_channels'],
        hidden_channels=config['model']['hidden_channels'],
        kernel_size=5,
        dilation_rate=1,
        n_layers=16,
        gin_channels=config['model']['gin_channels'],
    )

    B = 10
    T = 100
    x = torch.randn(B, config['data']['filter_length'] // 2 + 1, T)
    x_lengths = torch.randint(1, T, (B,))
    g = torch.randn(B, config['model']['gin_channels'], 1)
    z, mu, logs, x_mask = posterior_encoder(x, x_lengths, g)
    assert z.shape == (B, config['model']['inter_channels'], T)
    assert mu.shape == (B, config['model']['inter_channels'], T)
    assert logs.shape == (B, config['model']['inter_channels'], T)
    assert x_mask.shape == (B, 1, T)
    print("All tests passed.")