from typing import List

import torch

def init_weights(m, mean=0.0, std=0.01):
    """
    初始化卷积层的权重为正态分布
    mean: 正态分布的均值
    std: 正态分布的标准差
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size: int, dilation: int = 1):
    """
    计算卷积层的padding大小，确保输出与输入长度相同
    kernel_size: 卷积核大小
    dilation: 膨胀系数
    """
    return int((kernel_size * dilation - dilation) / 2)

def sequence_mask(lengths, max_len=None):
    """
    生成一个形状为[B, T]的mask矩阵，其中小于等于文本长度的位置为True，其他位置为False
    lengths: [B] 文本长度
    max_len: 最大文本长度
    """
    if max_len is None:
        max_len = lengths.max()
    
    x = torch.arange(max_len, dtype=lengths.dtype, device=lengths.device)   # [T]
    return x.unsqueeze(0) <= lengths.unsqueeze(1)

def convert_pad_shape(pad_shape: List[List[int]]):
    """
    为F.pad函数准备pad_size。
    由于F.pad要求pad_size从最后一维开始，且传入pad_size为一维列表，
    因此需要将pad_shape从[[l_d1, r_d1], [l_d2, r_d2], ..., [l_dn, r_dn]]
    转换为[l_dn, r_dn, ..., l_d1, r_d1]
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def fused_add_tanh_sigmoid_multiply(
    input_a,
    input_b,
    n_channels_tensor,
):
    """
    融合了add、tanh、sigmoid和multiply操作的函数。
    输入：
        input_a: [B, 2*C, T]
        input_b: [B, 2*C, T]
        n_channels_tensor: [1] 隐藏层通道数
    输出：
        output: [B, C, T]
    """
    n_channels = n_channels_tensor.item()
    in_act = input_a + input_b
    # split into two parts
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    act = t_act * s_act
    return act
