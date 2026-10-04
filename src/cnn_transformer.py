"""
Hybrid CNN + Transformer Architecture for EEG-Based ADHD Classification
========================================================================

Architecture Overview:
  1. Temporal Convolution (1D FIR Bandpass): Learns frequency-selective wavelets (e.g. theta, alpha, beta).
  2. Spatial Depthwise Convolution: Learns electrode topography across 26 scalp channels.
  3. Temporal Downsampling & Token Projection: Converts continuous EEG into a sequence of temporal feature tokens.
  4. Transformer Encoder: Multi-Head Self-Attention layers model long-range temporal dependencies and cross-frequency interactions.
  5. Attention-Weighted Pooling & Classification Readout: Computes diagnostic ADHD risk probability with calibrated confidence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal or learnable positional encoding for temporal token sequences.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class AttentionPooling(nn.Module):
    """
    Learnable attention-weighted temporal pooling over token sequences.
    Assigns higher clinical relevance to salient burst epochs / microstates.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, seq_len, d_model)
        scores = self.attn(x)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch_size, d_model)
        return pooled, weights.squeeze(-1)


class EEGConvTransformer(nn.Module):
    """
    Hybrid Deep Learning Model: 2D/1D CNN Front-End + Transformer Encoder Backbone.
    
    Parameters:
      n_channels: Number of scalp EEG channels (default 26)
      n_samples: Number of temporal points per epoch (default 1001, ~2.0s @ 500Hz)
      F1: Number of temporal convolutional filters (default 16)
      D: Depth multiplier for spatial convolution (default 2 -> 32 spatial filters)
      F2: Number of pointwise filters (default 32)
      kernel_length: Temporal filter window length (default 64)
      d_model: Transformer hidden dimension (default 64)
      nhead: Number of attention heads (default 4)
      num_layers: Number of Transformer encoder layers (default 2)
      dim_feedforward: Transformer MLP expansion dimension (default 128)
      dropout: Dropout rate across convolutional and attention layers (default 0.25)
    """
    def __init__(
        self,
        n_channels: int = 26,
        n_samples: int = 1001,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
        kernel_length: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.d_model = d_model

        # -------------------------------------------------------------
        # STAGE 1: Convolutional Spatial-Temporal Feature Extractor
        # -------------------------------------------------------------
        # Block 1: Temporal Conv (acts as learnable digital bandpass filter)
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Spatial Depthwise Conv (learns spatial scalp topography)
        self.spatial_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))  # Downsamples time by 4
        self.drop1 = nn.Dropout(dropout)

        # Block 3: Separable Temporal Conv
        self.sep_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))  # Total temporal downsample: 4 * 8 = 32
        self.drop2 = nn.Dropout(dropout)

        # Token Projection: maps F2 channels to d_model for the Transformer
        self.token_proj = nn.Conv1d(
            in_channels=F2,
            out_channels=d_model,
            kernel_size=1
        )
        self.token_norm = nn.LayerNorm(d_model)

        # -------------------------------------------------------------
        # STAGE 2: Transformer Encoder (Self-Attention over Temporal Tokens)
        # -------------------------------------------------------------
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=200)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(d_model)

        # -------------------------------------------------------------
        # STAGE 3: Attention-Weighted Pooling & Classifier Readout
        # -------------------------------------------------------------
        self.pooler = AttentionPooling(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Logit output for binary ADHD classification
        )

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs CNN front-end to produce temporal sequence tokens.
        Input x: (Batch, Channels, Time) -> (Batch, 1, Channels, Time)
        Output: (Batch, Seq_Len, d_model)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        # Block 1: Temporal Conv
        h = self.temporal_conv(x)
        h = self.bn1(h)

        # Block 2: Spatial Depthwise Conv
        h = self.spatial_conv(h)
        h = self.bn2(h)
        h = F.elu(h)
        h = self.pool1(h)
        h = self.drop1(h)

        # Block 3: Separable Temporal Conv
        h = self.sep_depthwise(h)
        h = self.sep_pointwise(h)
        h = self.bn3(h)
        h = F.elu(h)
        h = self.pool2(h)
        h = self.drop2(h)

        # Squeeze electrode dimension (now 1): (B, F2, 1, T_down) -> (B, F2, T_down)
        h = h.squeeze(2)

        # Project to d_model: (B, d_model, T_down) -> (B, T_down, d_model)
        tokens = self.token_proj(h).transpose(1, 2)
        tokens = self.token_norm(tokens)
        return tokens

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
          logits: (Batch, 1) raw logit output
          attn_weights: (Batch, Seq_Len) temporal attention weights
        """
        # Step 1: CNN Feature Tokenizer
        tokens = self.extract_tokens(x)  # (B, Seq_Len, d_model)

        # Step 2: Positional Encoding
        tokens = self.pos_encoding(tokens)

        # Step 3: Transformer Multi-Head Self-Attention
        encoded = self.transformer_encoder(tokens)  # (B, Seq_Len, d_model)
        encoded = self.final_norm(encoded)

        # Step 4: Attention Pooling across temporal tokens
        pooled, attn_weights = self.pooler(encoded)  # (B, d_model), (B, Seq_Len)

        # Step 5: Classifier Readout
        logits = self.classifier(pooled)  # (B, 1)
        return logits, attn_weights

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes ADHD probability in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs.squeeze(-1)
