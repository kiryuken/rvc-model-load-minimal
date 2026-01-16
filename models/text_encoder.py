"""
Text/Content Encoder for RVC.
Encodes HuBERT features into latent representations for voice conversion.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.attentions import Encoder
from models.commons import sequence_mask


class TextEncoder(nn.Module):
    """
    Text/Content encoder for RVC synthesizers.
    Uses 1D convolutions and transformer blocks to encode HuBERT features.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        """
        Initialize text encoder.
        
        Args:
            out_channels: Output channel dimension
            hidden_channels: Hidden layer dimension
            filter_channels: Feed-forward filter dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            kernel_size: Convolution kernel size
            p_dropout: Dropout probability
            f0: Whether to use F0 conditioning
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.f0 = f0

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        
        # Output projections for mean and log variance
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input features.
        
        Args:
            x: Input features [B, C, T]
            x_lengths: Feature lengths [B]
            g: Optional speaker embedding [B, C, 1]
        
        Returns:
            Tuple of (mean, log_std, mask)
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class TextEncoder256(nn.Module):
    """
    Text encoder for v1 models (256-dim HuBERT features from layer 9).
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        """
        Initialize 256-dim text encoder.
        
        Args:
            out_channels: Output dimension (typically 192)
            hidden_channels: Hidden dimension (typically 192)
            filter_channels: FFN filter dimension (typically 768)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            kernel_size: Convolution kernel size
            p_dropout: Dropout probability
            f0: Whether model uses F0
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.f0 = f0

        # Pre-net: project 256-dim HuBERT to hidden_channels
        self.pre = nn.Conv1d(256, hidden_channels, kernel_size=5, padding=2)
        
        # Encoder
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        
        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through 256-dim encoder."""
        # Project from 256 to hidden_channels
        x = self.pre(x)
        
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class TextEncoder768(nn.Module):
    """
    Text encoder for v2 models (768-dim HuBERT features from layer 12).
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ):
        """
        Initialize 768-dim text encoder.
        
        Args:
            out_channels: Output dimension (typically 192)
            hidden_channels: Hidden dimension (typically 192)
            filter_channels: FFN filter dimension (typically 768)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            kernel_size: Convolution kernel size
            p_dropout: Dropout probability
            f0: Whether model uses F0
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.f0 = f0

        # Pre-net: project 768-dim HuBERT to hidden_channels
        self.pre = nn.Conv1d(768, hidden_channels, kernel_size=5, padding=2)
        
        # Encoder
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        
        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through 768-dim encoder."""
        # Project from 768 to hidden_channels
        x = self.pre(x)
        
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """
    Posterior encoder for extracting latent representations from spectrograms.
    Used during training; during inference, the prior from TextEncoder is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        """
        Initialize posterior encoder.
        
        Args:
            in_channels: Input spectrogram channels
            out_channels: Output latent dimension
            hidden_channels: Hidden layer dimension
            kernel_size: Convolution kernel size
            dilation_rate: Dilation rate for WaveNet-style convolutions
            n_layers: Number of WaveNet layers
            gin_channels: Speaker embedding dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNetEncoder(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode spectrogram to latent distribution.
        
        Args:
            x: Input spectrogram [B, C, T]
            x_lengths: Sequence lengths [B]
            g: Speaker embedding [B, gin_channels, 1]
        
        Returns:
            Tuple of (z, mean, log_std, mask)
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class WaveNetEncoder(nn.Module):
    """
    WaveNet-style encoder with dilated convolutions.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        """
        Initialize WaveNet encoder.
        
        Args:
            hidden_channels: Hidden channel dimension
            kernel_size: Convolution kernel size
            dilation_rate: Base dilation rate
            n_layers: Number of layers
            gin_channels: Speaker conditioning dimension
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(in_layer)

            # Last layer only outputs hidden_channels
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through WaveNet encoder.
        
        Args:
            x: Input tensor [B, hidden_channels, T]
            x_mask: Mask [B, 1, T]
            g: Speaker conditioning [B, gin_channels, 1]
        
        Returns:
            Encoded tensor [B, hidden_channels, T]
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self._fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def _fused_add_tanh_sigmoid_multiply(self, input_a, input_b, n_channels):
        n_channels_int = n_channels[0].item()
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels_int, :])
        s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
        acts = t_act * s_act
        return acts
