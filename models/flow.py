"""
Normalizing Flow modules for RVC.
Implements residual coupling blocks for invertible transformations.
"""

from typing import Optional

import torch
from torch import nn

from models.commons import fused_add_tanh_sigmoid_multiply


class ResidualCouplingLayer(nn.Module):
    """
    Single residual coupling layer for normalizing flows.
    Implements an affine coupling transformation.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        """
        Initialize residual coupling layer.
        
        Args:
            channels: Input/output channel dimension
            hidden_channels: Hidden layer dimension
            kernel_size: Convolution kernel size
            dilation_rate: Dilation rate for WaveNet
            n_layers: Number of WaveNet layers
            p_dropout: Dropout probability
            gin_channels: Speaker conditioning dimension
            mean_only: Only predict mean (no scale)
        """
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
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through coupling layer.
        
        Args:
            x: Input tensor [B, channels, T]
            x_mask: Mask [B, 1, T]
            g: Speaker conditioning [B, gin_channels, 1]
            reverse: Whether to run in reverse (for generation)
        
        Returns:
            Transformed tensor [B, channels, T]
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlock(nn.Module):
    """
    Stack of residual coupling layers with alternating splits.
    Forms a complete normalizing flow block.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        """
        Initialize residual coupling block.
        
        Args:
            channels: Input/output channel dimension
            hidden_channels: Hidden layer dimension
            kernel_size: Convolution kernel size
            dilation_rate: Dilation rate for WaveNet
            n_layers: Number of WaveNet layers per flow
            n_flows: Number of flow layers
            gin_channels: Speaker conditioning dimension
        """
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
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through flow block.
        
        Args:
            x: Input tensor [B, channels, T]
            x_mask: Mask [B, 1, T]
            g: Speaker conditioning
            reverse: Run in reverse for generation
        
        Returns:
            Transformed tensor
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Flip(nn.Module):
    """
    Flip layer - reverses channel order.
    Used between coupling layers to ensure all dimensions are transformed.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        reverse: bool = False,
        **kwargs
    ):
        """Flip channels."""
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class WN(nn.Module):
    """
    WaveNet-style network for residual coupling layers.
    Uses dilated convolutions with gated activations.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        """
        Initialize WaveNet.
        
        Args:
            hidden_channels: Hidden channel dimension
            kernel_size: Convolution kernel size
            dilation_rate: Base dilation rate
            n_layers: Number of layers
            gin_channels: Speaker conditioning dimension
            p_dropout: Dropout probability
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")

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
            in_layer = nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through WaveNet.
        
        Args:
            x: Input tensor [B, hidden_channels, T]
            x_mask: Mask [B, 1, T]
            g: Speaker conditioning
        
        Returns:
            Output tensor [B, hidden_channels, T]
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

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        if self.gin_channels != 0:
            nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            nn.utils.remove_weight_norm(l)
