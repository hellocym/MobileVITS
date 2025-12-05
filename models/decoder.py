from math import prod

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm

import modules
from commons import init_weights


class Decoder(nn.Module):
    def __init__(self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (kernel_size, upsample_rate) in enumerate(zip(upsample_kernel_sizes, upsample_rates)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size,
                        upsample_rate,
                        padding=(kernel_size - upsample_rate) // 2,
                    )
                )
            )
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (kernel_size, dilation_rate) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    resblock(ch, kernel_size, dilation_rate)
                )
            
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        # 若有说话人
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        
    def forward(self, x, g=None):
        # x: [B, C, T]
        x = self.conv_pre(x)    
        if g is not None:
            x = x + self.cond(g)

        # x: [B, upsample_initial_channel, T]
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            # x: channel /= 2, T *= upsample_rate
            

            # resblocks层，多个不同 kernel size 的 resblock
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels + j](x)
            x = xs / self.num_kernels

        # x: [B, ch, T * prod(upsample_rates)]
        x = F.leaky_relu(x)
        x = self.conv_post(x)

        # x: [B, 1, T * prod(upsample_rates)]
        x = torch.tanh(x)

        return x
        
            

if __name__ == '__main__':
    import json
    json_path = 'configs/chinese_base.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    f = Decoder(
        initial_channel=config['model']['inter_channels'],
        resblock=config['model']['resblock'],
        resblock_kernel_sizes=config['model']['resblock_kernel_sizes'],
        resblock_dilation_sizes=config['model']['resblock_dilation_sizes'],
        upsample_rates=config['model']['upsample_rates'],
        upsample_initial_channel=config['model']['upsample_initial_channel'],
        upsample_kernel_sizes=config['model']['upsample_kernel_sizes'],
        gin_channels=config['model']['gin_channels'],
    )

    print(f)
    
    B = 1
    T = 100
    x = torch.randn(B, config['model']['inter_channels'], T)
    g = torch.randn(B, config['model']['gin_channels'], 1)

    w = f(x, g)
    # print(w.shape)
    # print((B, 1, T*prod(config['model']['upsample_rates'])))
    assert w.shape == (B, 1, T*prod(config['model']['upsample_rates']))
    print("All tests passed!")