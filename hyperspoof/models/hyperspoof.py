"""
Main HyperSpoof model implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .hsi_reconstruction import HSIReconstruction
from .spectral_attention import SpectralAttention
from .classifier import EfficientNetClassifier


class HyperSpoofModel(nn.Module):
    """
    HyperSpoof: A novel framework for face anti-spoofing using hyperspectral reconstruction.
    
    This model consists of three main components:
    1. HSI Reconstruction Module: Converts RGB to hyperspectral data
    2. Spectral Attention Module: Highlights discriminative spectral features
    3. Classification Module: Performs binary classification
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hsi_channels: int = 31,
        attention_channels: int = 3,
        num_classes: int = 2,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize HyperSpoof model.
        
        Args:
            input_channels: Number of input channels (RGB = 3)
            hsi_channels: Number of hyperspectral channels (31)
            attention_channels: Number of output channels from attention (3)
            num_classes: Number of output classes (2 for binary)
            model_name: Name of the classifier backbone
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(HyperSpoofModel, self).__init__()
        
        self.input_channels = input_channels
        self.hsi_channels = hsi_channels
        self.attention_channels = attention_channels
        self.num_classes = num_classes
        
        # HSI Reconstruction Module
        self.hsi_reconstruction = HSIReconstruction(
            input_channels=input_channels,
            output_channels=hsi_channels
        )
        
        # Spectral Attention Module
        self.spectral_attention = SpectralAttention(
            input_channels=hsi_channels,
            output_channels=attention_channels
        )
        
        # Classification Module
        self.classifier = EfficientNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            input_channels=attention_channels
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the HyperSpoof model.
        
        Args:
            x: Input RGB tensor of shape (B, 3, H, W)
            
        Returns:
            Dictionary containing:
                - 'logits': Raw classification logits
                - 'probabilities': Softmax probabilities
                - 'hsi_features': Reconstructed HSI features
                - 'attention_features': Attention-weighted features
        """
        # HSI Reconstruction
        hsi_features = self.hsi_reconstruction(x)  # (B, 31, H, W)
        
        # Spectral Attention
        attention_features = self.spectral_attention(hsi_features)  # (B, 3, H, W)
        
        # Classification
        logits = self.classifier(attention_features)  # (B, num_classes)
        probabilities = torch.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'hsi_features': hsi_features,
            'attention_features': attention_features
        }
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for analysis.
        
        Args:
            x: Input RGB tensor
            
        Returns:
            Dictionary containing intermediate features
        """
        with torch.no_grad():
            hsi_features = self.hsi_reconstruction(x)
            attention_features = self.spectral_attention(hsi_features)
            
        return {
            'hsi_features': hsi_features,
            'attention_features': attention_features
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input tensor.
        
        Args:
            x: Input RGB tensor
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['probabilities']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and parameter count.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'input_channels': self.input_channels,
            'hsi_channels': self.hsi_channels,
            'attention_channels': self.attention_channels,
            'num_classes': self.num_classes,
        }
