import torch
from torch import nn

import modules

class Flow(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        for i in range(n_flows):
            self.flows.append(modules.ResiduleCouplingLayer(
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=gin_channels,
                mean_only=True,
            ))
            self.flows.append(modules.Flip())

        
    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

if __name__ == '__main__':
    import json
    json_path = 'configs/chinese_base.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    f = Flow(
        channels=config['model']['inter_channels'],
        hidden_channels=config['model']['hidden_channels'],
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        gin_channels=config['model']['gin_channels'],
    )

    # print(f)
    
    B = 1
    T = 100
    x = torch.randn(B, config['model']['inter_channels'], T)
    x_mask = torch.ones(B, 1, T).to(x.device)
    g = torch.randn(B, config['model']['gin_channels'], 1)
    a = f(x, x_mask, g)
    
    assert a.shape == (B, config['model']['inter_channels'], T)
    print("All tests passed!")