"""
Common utility functions for RVC neural network modules.
Contains initialization, padding, and tensor manipulation utilities.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize module weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for same output size."""
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: list) -> list:
    """Convert padding shape from [[a,b],[c,d]] to [c,d,a,b] for F.pad."""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def slice_segments(
    x: torch.Tensor, 
    ids_str: torch.Tensor, 
    segment_size: int = 4
) -> torch.Tensor:
    """
    Slice segments from tensor based on start indices.
    
    Args:
        x: Input tensor [B, C, T]
        ids_str: Start indices [B]
        segment_size: Size of each segment
    
    Returns:
        Sliced segments [B, C, segment_size]
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, 
    x_lengths: Optional[torch.Tensor] = None, 
    segment_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slice segments from tensor.
    
    Args:
        x: Input tensor [B, C, T]
        x_lengths: Sequence lengths [B]
        segment_size: Size of each segment
    
    Returns:
        Tuple of (sliced segments, start indices)
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(
    length: torch.Tensor, 
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Create sequence mask from lengths.
    
    Args:
        length: Sequence lengths [B]
        max_length: Maximum sequence length
    
    Returns:
        Boolean mask [B, max_length]
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, 
    input_b: torch.Tensor, 
    n_channels: int
) -> torch.Tensor:
    """
    Fused gated activation: tanh(a) * sigmoid(b).
    
    Args:
        input_a: First input tensor
        input_b: Second input tensor (added to first)
        n_channels: Number of channels (used for splitting)
    
    Returns:
        Gated activation result
    """
    n_channels_int = n_channels[0] if isinstance(n_channels, (list, tuple)) else n_channels
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


@torch.jit.script
def fused_add_tanh_sigmoid_multiply_jit(
    input_a: torch.Tensor, 
    input_b: torch.Tensor, 
    n_channels: int
) -> torch.Tensor:
    """JIT-compiled version of fused gated activation."""
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts


class LayerNorm(nn.Module):
    """Layer normalization for 1D convolutions (channels first)."""
    
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor [B, C, T]
        
        Returns:
            Normalized tensor [B, C, T]
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
