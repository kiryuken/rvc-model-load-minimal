"""
RVC v1 Synthesizer Networks.
Implements SynthesizerTrnMs256NSFsid (with F0) and SynthesizerTrnMs256NSFsid_nono (without F0).
These use 256-dim HuBERT features from layer 9.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from models.text_encoder import TextEncoder256, PosteriorEncoder
from models.flow import ResidualCouplingBlock
from models.nsf import GeneratorNSF, Generator


class SynthesizerTrnMs256NSFsid(nn.Module):
    """
    RVC v1 Synthesizer with F0 (pitch) conditioning.
    Uses 256-dim HuBERT features extracted from layer 9.
    
    Architecture:
    - TextEncoder256: Encodes 256-dim HuBERT features to latent
    - ResidualCouplingBlock: Normalizing flow for voice conversion
    - GeneratorNSF: Neural source-filter vocoder with F0 conditioning
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        **kwargs
    ):
        """
        Initialize v1 synthesizer with F0.
        
        Args:
            spec_channels: Spectrogram channels
            segment_size: Training segment size
            inter_channels: Intermediate channels
            hidden_channels: Hidden layer channels
            filter_channels: FFN filter channels
            n_heads: Attention heads
            n_layers: Encoder layers
            kernel_size: Convolution kernel size
            p_dropout: Dropout probability
            resblock: Residual block type
            resblock_kernel_sizes: Kernel sizes for res blocks
            resblock_dilation_sizes: Dilation sizes for res blocks
            upsample_rates: Upsampling rates
            upsample_initial_channel: Initial upsample channels
            upsample_kernel_sizes: Upsample kernel sizes
            spk_embed_dim: Speaker embedding dimension
            gin_channels: Global conditioning channels
            sr: Sample rate
        """
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        # Text/content encoder for 256-dim features
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=True,
        )
        
        # NSF Generator with F0 conditioning
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        
        # Posterior encoder (for training)
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,  # kernel_size
            1,  # dilation_rate
            16,  # n_layers
            gin_channels=gin_channels,
        )
        
        # Flow for voice conversion
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,  # kernel_size
            1,  # dilation_rate
            3,  # n_layers
            gin_channels=gin_channels,
        )
        
        # F0 embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        
        # F0 quantization embedding (256 bins)
        self.f0_emb = nn.Embedding(256, hidden_channels)

    def remove_weight_norm(self):
        """Remove weight normalization from all modules."""
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm() if hasattr(self.enc_q, 'remove_weight_norm') else None

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        f0: torch.Tensor,
        f0_nsf: torch.Tensor,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            phone: HuBERT features [B, 256, T]
            phone_lengths: Feature lengths [B]
            f0: Quantized F0 [B, T]
            f0_nsf: Continuous F0 for NSF [B, T]
            sid: Speaker ID [B]
        
        Returns:
            Generated audio [B, 1, T_audio]
        """
        # Get speaker embedding
        g = self.emb_g(sid).unsqueeze(-1)  # [B, gin_channels, 1]
        
        # Encode content features
        m_p, logs_p, x_mask = self.enc_p(phone, phone_lengths, None)
        
        # Add F0 embedding
        f0_emb = self.f0_emb(f0.long().clamp(0, 255)).transpose(1, 2)
        m_p = m_p + f0_emb
        
        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.66666  # Noise scale
        
        # Flow: transform to decoder space
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        
        # Generate audio
        o = self.dec(z * x_mask, f0_nsf, g=g)
        
        return o

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        f0: torch.Tensor,
        f0_nsf: torch.Tensor,
        sid: torch.Tensor,
        rate: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference with optional speed control.
        
        Args:
            phone: HuBERT features [B, 256, T]
            phone_lengths: Feature lengths [B]
            f0: Quantized F0 [B, T]
            f0_nsf: Continuous F0 for NSF [B, T]
            sid: Speaker ID [B]
            rate: Speed factor (not used in standard inference)
        
        Returns:
            Generated audio [B, 1, T_audio]
        """
        return self.forward(phone, phone_lengths, f0, f0_nsf, sid)


class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    """
    RVC v1 Synthesizer WITHOUT F0 conditioning.
    For models trained without pitch information.
    Uses 256-dim HuBERT features from layer 9.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        **kwargs
    ):
        """
        Initialize v1 synthesizer without F0.
        
        Args:
            spec_channels: Spectrogram channels
            segment_size: Training segment size
            inter_channels: Intermediate channels
            hidden_channels: Hidden layer channels
            filter_channels: FFN filter channels
            n_heads: Attention heads
            n_layers: Encoder layers
            kernel_size: Convolution kernel size
            p_dropout: Dropout probability
            resblock: Residual block type
            resblock_kernel_sizes: Kernel sizes for res blocks
            resblock_dilation_sizes: Dilation sizes for res blocks
            upsample_rates: Upsampling rates
            upsample_initial_channel: Initial upsample channels
            upsample_kernel_sizes: Upsample kernel sizes
            spk_embed_dim: Speaker embedding dimension
            gin_channels: Global conditioning channels
            sr: Sample rate
        """
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        # Text/content encoder for 256-dim features
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=False,
        )
        
        # Standard Generator without F0 (no NSF)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        
        # Posterior encoder (for training)
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        
        # Flow for voice conversion
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        
        # Speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        """Remove weight normalization from all modules."""
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inference (no F0).
        
        Args:
            phone: HuBERT features [B, 256, T]
            phone_lengths: Feature lengths [B]
            sid: Speaker ID [B]
        
        Returns:
            Generated audio [B, 1, T_audio]
        """
        # Get speaker embedding
        g = self.emb_g(sid).unsqueeze(-1)
        
        # Encode content features
        m_p, logs_p, x_mask = self.enc_p(phone, phone_lengths, None)
        
        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.66666
        
        # Flow: transform to decoder space
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        
        # Generate audio (no F0)
        o = self.dec(z * x_mask, g=g)
        
        return o

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
        rate: float = 1.0,
    ) -> torch.Tensor:
        """
        Inference with optional speed control.
        
        Args:
            phone: HuBERT features [B, 256, T]
            phone_lengths: Feature lengths [B]
            sid: Speaker ID [B]
            rate: Speed factor
        
        Returns:
            Generated audio [B, 1, T_audio]
        """
        return self.forward(phone, phone_lengths, sid)
