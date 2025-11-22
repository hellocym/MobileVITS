import math

import torch
from torch import nn

import attentions
from commons import sequence_mask

class TextEncoder(nn.Module):
    def __init__(self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, mean=0.0, std=hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(self, x, x_lengths):
        """
        x: [B, T]
        x_lengths: [B] 文本长度
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)                   # [B, T, H]
        x = torch.transpose(x, 1, -1)                                       # [B, H, T]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1)    # [B, 1, T]
        x_mask = x_mask.to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)                                # [B, H, T]
        stats = self.proj(x) * x_mask                                       # [B, 2*H, T]

        m, logs = torch.split(stats, self.out_channels, dim=1)              # [B, H, T], [B, H, T]
        
        return x, m, logs, x_mask

if __name__ == '__main__':
    import json
    json_path = 'configs/chinese_base.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    text_encoder = TextEncoder(
        n_vocab=len(config['symbols']),
        out_channels=config['model']['inter_channels'],
        hidden_channels=config['model']['hidden_channels'],
        filter_channels=config['model']['filter_channels'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        kernel_size=config['model']['kernel_size'],
        p_dropout=config['model']['p_dropout']
    )

    B = 1
    T = 10
    x = torch.randint(0, len(config['symbols']), (B, T))  # [B, T], 随机生成一个文本序列
    assert x.shape == (B, T)
    x_lengths = torch.tensor([T])
    x, m, logs, x_mask = text_encoder(x, x_lengths)
    assert x.shape == (B, config['model']['inter_channels'], T)
    assert m.shape == (B, config['model']['inter_channels'], T)
    assert logs.shape == (B, config['model']['inter_channels'], T)
    assert x_mask.shape == (B, 1, T)
    print("All tests passed.")