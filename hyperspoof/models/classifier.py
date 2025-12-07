"""
Classification Module using EfficientNet.

This module provides the final classification layer for face anti-spoofing.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for face anti-spoofing.
    
    Adapts EfficientNet to work with 3-channel attention features.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        input_channels: int = 3,
    ):
        """
        Initialize EfficientNet classifier.
        
        Args:
            model_name: Name of the EfficientNet model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            input_channels: Number of input channels
        """
        super(EfficientNetClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load EfficientNet model
        if model_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get the number of input features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # If input channels are different from 3, modify the first layer
        if input_channels != 3:
            self._modify_input_layer(input_channels)
    
    def _modify_input_layer(self, input_channels: int):
        """
        Modify the first convolutional layer to accept different input channels.
        
        Args:
            input_channels: Number of input channels
        """
        # Get the original first layer
        original_conv = self.backbone.features[0][0]
        
        # Create new first layer
        new_conv = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights
        if input_channels == 3:
            # Copy pretrained weights
            new_conv.weight.data = original_conv.weight.data
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data
        else:
            # Initialize with pretrained weights (repeat or average)
            with torch.no_grad():
                if input_channels == 1:
                    # Grayscale: average RGB channels
                    new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    # Multiple channels: repeat RGB channels
                    repeat_factor = input_channels // 3
                    remainder = input_channels % 3
                    
                    weight_data = original_conv.weight.data.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        weight_data = torch.cat([
                            weight_data,
                            original_conv.weight.data[:, :remainder, :, :]
                        ], dim=1)
                    
                    new_conv.weight.data = weight_data
                
                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data
        
        # Replace the first layer
        self.backbone.features[0][0] = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor before classification
        """
        # Get features from backbone
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        return features
