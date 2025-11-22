import torch
from torch import nn

import commons

class WN(nn.Module):
    def __init__(self, 
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0.0,
    ):
        super().__init__()
        assert(kernel_size % 2 == 1, "kernel_size must be odd")
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        # 条件输入，在VITS中为说话人id
        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            # self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
            self.cond_layer = torch.nn.utils.parametrizations.weight_norm(cond_layer, name="weight")  # migrate to parametrizations.weight_norm
        
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size, dilation=dilation, padding=padding)
            # in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            in_layer = torch.nn.utils.parametrizations.weight_norm(in_layer, name="weight")  # migrate to parametrizations.weight_norm
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            # res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            res_skip_channels = torch.nn.utils.parametrizations.weight_norm(res_skip_layer, name="weight")  # migrate to parametrizations.weight_norm
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)    # [B, hidden_channels, T]
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        # 若有条件输入
        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x) # [B, 2*hidden_channels, T]
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels, :]   # [B, 2*hidden_channels, T]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor,
            )
            acts = self.drop(acts)

            # 残差链接
            res_skip_acts = self.res_skip_layers[i](acts)    # [B, res_skip_channels, T], 其中res_skip_channels = 2*hidden_channels if i < n_layers - 1 else hidden_channels
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask