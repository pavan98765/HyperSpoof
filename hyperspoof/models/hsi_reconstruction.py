"""
Hyperspectral Image (HSI) Reconstruction Module.

This module converts RGB images to hyperspectral data using the MST++ approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HSIReconstruction(nn.Module):
    """
    HSI Reconstruction Module using MST++ approach.
    
    Converts RGB images to 31-channel hyperspectral data.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 31,
        hidden_dim: int = 64,
    ):
        """
        Initialize HSI Reconstruction module.
        
        Args:
            input_channels: Number of input channels (RGB = 3)
            output_channels: Number of output HSI channels (31)
            hidden_dim: Hidden dimension for the network
        """
        super(HSIReconstruction, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
        
        # Residual connection
        self.residual_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HSI reconstruction.
        
        Args:
            x: Input RGB tensor of shape (B, 3, H, W)
            
        Returns:
            Reconstructed HSI tensor of shape (B, 31, H, W)
        """
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(encoded)
        
        # Residual connection
        residual = self.residual_conv(x)
        
        # Combine with residual connection
        output = decoded + residual
        
        return output
    
    def get_spectral_bands(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get individual spectral bands for analysis.
        
        Args:
            x: Input RGB tensor
            
        Returns:
            Tensor of shape (B, 31, H, W) with individual spectral bands
        """
        hsi = self.forward(x)
        return hsi
