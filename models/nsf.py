"""
Neural Source-Filter (NSF) HiFi-GAN generator for RVC.
Implements source excitation and filter network for high-quality waveform synthesis.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from models.commons import init_weights, get_padding


class SourceModuleHnNSF(nn.Module):
    """
    Source module for Neural Source-Filter model.
    Generates harmonic and noise source signals from F0.
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        """
        Initialize source module.
        
        Args:
            sampling_rate: Audio sampling rate
            harmonic_num: Number of harmonics to generate
            sine_amp: Amplitude of sinusoidal components
            add_noise_std: Standard deviation of additive noise
            voiced_threshold: F0 threshold for voiced detection
        """
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = sampling_rate
        self.voiced_threshold = voiced_threshold

        # Linear layer to merge harmonics and noise
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(
        self,
        f0: torch.Tensor,
        uv: Optional[torch.Tensor] = None,
        upp: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate source signal from F0.
        
        Args:
            f0: Fundamental frequency [B, 1, T]
            uv: Unvoiced mask (optional) [B, 1, T]
            upp: Upsampling factor
        
        Returns:
            Tuple of (source signal, UV mask, noise)
        """
        # Upsample F0 to waveform rate
        f0 = f0.unsqueeze(1) if f0.dim() == 2 else f0
        f0 = F.interpolate(f0, scale_factor=upp, mode="nearest")
        
        # Generate voiced/unvoiced mask
        if uv is None:
            uv = (f0 > self.voiced_threshold).float()
        else:
            uv = F.interpolate(uv.unsqueeze(1).float(), scale_factor=upp, mode="nearest")

        # Generate sine waves for harmonics
        with torch.no_grad():
            # Compute phase
            rad_values = (f0 / self.sampling_rate) % 1
            
            # Cumulative sum for phase
            rad_values = torch.cumsum(rad_values, dim=2)
            rad_values = rad_values * 2 * math.pi
            rad_values = rad_values % (2 * math.pi)
            
            # Generate harmonics
            sine_waves = self.sine_amp * torch.sin(rad_values) * uv
            
            # Add harmonics if specified
            if self.harmonic_num > 0:
                harmonics = []
                for i in range(1, self.harmonic_num + 1):
                    harmonic_f0 = f0 * (i + 1)
                    rad_h = torch.cumsum(harmonic_f0 / self.sampling_rate, dim=2) * 2 * math.pi
                    harmonics.append(self.sine_amp * torch.sin(rad_h % (2 * math.pi)) * uv)
                sine_waves = torch.cat([sine_waves] + harmonics, dim=1)

        # Add noise for unvoiced regions
        noise = torch.randn_like(sine_waves[:, :1, :]) * self.noise_std

        # Combine using linear layer
        if self.harmonic_num > 0:
            sine_merge = self.l_linear(sine_waves.transpose(1, 2)).transpose(1, 2)
        else:
            sine_merge = sine_waves

        sine_merge = self.l_tanh(sine_merge)

        return sine_merge, uv, noise


class ResBlock1(nn.Module):
    """
    Residual block with dilated convolutions (Type 1).
    Uses multiple dilation rates within each block.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
    ):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilation: Tuple of dilation rates
        """
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=get_padding(kernel_size, d)
            ))
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=get_padding(kernel_size, 1)
            ))
            for _ in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    """
    Residual block with dilated convolutions (Type 2).
    Simpler version with single convolution per dilation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3),
    ):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilation: Tuple of dilation rates
        """
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=get_padding(kernel_size, d)
            ))
            for d in dilation
        ])
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for l in self.convs:
            remove_weight_norm(l)


class GeneratorNSF(nn.Module):
    """
    NSF-HiFi-GAN Generator.
    Combines neural source-filter approach with HiFi-GAN architecture.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[Tuple[int, ...]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        gin_channels: int = 256,
        sr: int = 40000,
        is_half: bool = False,
    ):
        """
        Initialize NSF generator.
        
        Args:
            initial_channel: Initial input channels
            resblock: Residual block type ("1" or "2")
            resblock_kernel_sizes: Kernel sizes for residual blocks
            resblock_dilation_sizes: Dilation rates for residual blocks
            upsample_rates: Upsampling rates for each stage
            upsample_initial_channel: Initial channel count for upsampling
            upsample_kernel_sizes: Kernel sizes for upsampling convolutions
            gin_channels: Speaker embedding dimension
            sr: Sample rate
            is_half: Use FP16
        """
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr
        self.is_half = is_half

        # F0 upsampling factor
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates))
        
        # Source module
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr,
            harmonic_num=0,
        )
        
        # Noise convolution
        self.noise_convs = nn.ModuleList()
        
        # Initial convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        # Select residual block type
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        c_cur,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            
            # Noise convolution for source module
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        # Output convolution
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate waveform from latent and F0.
        
        Args:
            x: Latent representation [B, initial_channel, T]
            f0: Fundamental frequency [B, T] or [B, 1, T]
            g: Speaker embedding [B, gin_channels, 1]
        
        Returns:
            Generated waveform [B, 1, T_audio]
        """
        # Prepare F0
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        
        # Upsample F0 to match audio rate
        f0 = self.f0_upsamp(f0)
        
        # Generate source signal
        har_source, _, _ = self.m_source(f0, upp=1)
        har_source = har_source.transpose(1, 2)
        
        # Initial convolution
        x = self.conv_pre(x)
        
        # Add speaker conditioning
        if g is not None:
            x = x + self.cond(g)

        # Upsampling with residual blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Add source signal
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            
            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Output
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Generator(nn.Module):
    """
    Standard HiFi-GAN Generator (without NSF source module).
    For models that don't use F0 conditioning.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[Tuple[int, ...]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        gin_channels: int = 256,
        sr: int = 40000,
        is_half: bool = False,
    ):
        """
        Initialize standard generator.
        
        Args:
            initial_channel: Initial input channels
            resblock: Residual block type ("1" or "2")
            resblock_kernel_sizes: Kernel sizes for residual blocks
            resblock_dilation_sizes: Dilation rates for residual blocks
            upsample_rates: Upsampling rates for each stage
            upsample_initial_channel: Initial channel count for upsampling
            upsample_kernel_sizes: Kernel sizes for upsampling convolutions
            gin_channels: Speaker embedding dimension
            sr: Sample rate
            is_half: Use FP16
        """
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr
        self.is_half = is_half

        # Initial convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )

        # Select residual block type
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        # Output convolution
        ch = upsample_initial_channel // (2 ** len(self.ups))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # Speaker conditioning
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate waveform from latent representation.
        
        Args:
            x: Latent representation [B, initial_channel, T]
            g: Speaker embedding [B, gin_channels, 1]
        
        Returns:
            Generated waveform [B, 1, T_audio]
        """
        x = self.conv_pre(x)
        
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
