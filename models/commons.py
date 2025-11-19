from typing import List

import torch

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