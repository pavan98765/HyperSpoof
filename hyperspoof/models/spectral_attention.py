"""
Spectral Attention Module.

This module applies attention mechanisms to hyperspectral data to highlight
discriminative spectral features and reduce dimensionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectralAttention(nn.Module):
    """
    Spectral Attention Module for highlighting discriminative spectral features.
    
    Reduces hyperspectral data to a compact 3-channel representation.
    """
    
    def __init__(
        self,
        input_channels: int = 31,
        output_channels: int = 3,
        attention_dim: int = 64,
    ):
        """
        Initialize Spectral Attention module.
        
        Args:
            input_channels: Number of input HSI channels (31)
            output_channels: Number of output channels (3)
            attention_dim: Dimension for attention computation
        """
        super(SpectralAttention, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.attention_dim = attention_dim
        
        # Global average pooling for channel attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(input_channels, attention_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attention_dim, input_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Spectral reduction
        self.spectral_reduction = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(input_channels // 2, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectral attention.
        
        Args:
            x: Input HSI tensor of shape (B, 31, H, W)
            
        Returns:
            Attention-weighted features of shape (B, 3, H, W)
        """
        # Channel attention
        channel_att = self.global_pool(x)  # (B, 31, 1, 1)
        channel_att = channel_att.view(channel_att.size(0), -1)  # (B, 31)
        channel_att = self.channel_attention(channel_att)  # (B, 31)
        channel_att = channel_att.view(channel_att.size(0), -1, 1, 1)  # (B, 31, 1, 1)
        
        # Apply channel attention
        x_att = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_attention(x_att)  # (B, 1, H, W)
        
        # Apply spatial attention
        x_att = x_att * spatial_att
        
        # Spectral reduction
        output = self.spectral_reduction(x_att)  # (B, 3, H, W)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            x: Input HSI tensor
            
        Returns:
            Tuple of (channel_attention, spatial_attention)
        """
        # Channel attention
        channel_att = self.global_pool(x)
        channel_att = channel_att.view(channel_att.size(0), -1)
        channel_att = self.channel_attention(channel_att)
        channel_att = channel_att.view(channel_att.size(0), -1, 1, 1)
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        
        return channel_att, spatial_att
