"""
Tests for model components.
"""

import pytest
import torch
import numpy as np
from hyperspoof.models import HyperSpoofModel, SpectralAttention, HSIReconstruction, EfficientNetClassifier


class TestHSIReconstruction:
    """Test HSI Reconstruction module."""
    
    def test_hsi_reconstruction_forward(self):
        """Test HSI reconstruction forward pass."""
        model = HSIReconstruction(input_channels=3, output_channels=31)
        
        # Test with batch of images
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 31, 256, 256)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_hsi_reconstruction_spectral_bands(self):
        """Test spectral bands extraction."""
        model = HSIReconstruction(input_channels=3, output_channels=31)
        
        input_tensor = torch.randn(2, 3, 128, 128)
        spectral_bands = model.get_spectral_bands(input_tensor)
        
        assert spectral_bands.shape == (2, 31, 128, 128)


class TestSpectralAttention:
    """Test Spectral Attention module."""
    
    def test_spectral_attention_forward(self):
        """Test spectral attention forward pass."""
        model = SpectralAttention(input_channels=31, output_channels=3)
        
        # Test with batch of HSI data
        batch_size = 4
        input_tensor = torch.randn(batch_size, 31, 256, 256)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 3, 256, 256)
    
    def test_spectral_attention_maps(self):
        """Test attention maps extraction."""
        model = SpectralAttention(input_channels=31, output_channels=3)
        
        input_tensor = torch.randn(2, 31, 128, 128)
        channel_att, spatial_att = model.get_attention_maps(input_tensor)
        
        assert channel_att.shape == (2, 31, 1, 1)
        assert spatial_att.shape == (2, 1, 128, 128)


class TestEfficientNetClassifier:
    """Test EfficientNet Classifier."""
    
    def test_classifier_forward(self):
        """Test classifier forward pass."""
        model = EfficientNetClassifier(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False,
            input_channels=3
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 2)
    
    def test_classifier_features(self):
        """Test feature extraction."""
        model = EfficientNetClassifier(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False,
            input_channels=3
        )
        
        input_tensor = torch.randn(2, 3, 256, 256)
        features = model.get_features(input_tensor)
        
        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] > 0  # Feature dimension


class TestHyperSpoofModel:
    """Test complete HyperSpoof model."""
    
    def test_hyperspoof_forward(self):
        """Test HyperSpoof forward pass."""
        model = HyperSpoofModel(
            input_channels=3,
            hsi_channels=31,
            attention_channels=3,
            num_classes=2,
            model_name="efficientnet_b0",
            pretrained=False
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        outputs = model(input_tensor)
        
        # Check output structure
        assert 'logits' in outputs
        assert 'probabilities' in outputs
        assert 'hsi_features' in outputs
        assert 'attention_features' in outputs
        
        # Check shapes
        assert outputs['logits'].shape == (batch_size, 2)
        assert outputs['probabilities'].shape == (batch_size, 2)
        assert outputs['hsi_features'].shape == (batch_size, 31, 256, 256)
        assert outputs['attention_features'].shape == (batch_size, 3, 256, 256)
        
        # Check probability normalization
        prob_sum = torch.sum(outputs['probabilities'], dim=1)
        assert torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-6)
    
    def test_hyperspoof_features(self):
        """Test feature extraction."""
        model = HyperSpoofModel(
            input_channels=3,
            hsi_channels=31,
            attention_channels=3,
            num_classes=2,
            model_name="efficientnet_b0",
            pretrained=False
        )
        
        input_tensor = torch.randn(2, 3, 256, 256)
        features = model.get_features(input_tensor)
        
        assert 'hsi_features' in features
        assert 'attention_features' in features
        assert features['hsi_features'].shape == (2, 31, 256, 256)
        assert features['attention_features'].shape == (2, 3, 256, 256)
    
    def test_hyperspoof_predict(self):
        """Test prediction method."""
        model = HyperSpoofModel(
            input_channels=3,
            hsi_channels=31,
            attention_channels=3,
            num_classes=2,
            model_name="efficientnet_b0",
            pretrained=False
        )
        
        input_tensor = torch.randn(2, 3, 256, 256)
        predictions = model.predict(input_tensor)
        
        assert predictions.shape == (2, 2)
        prob_sum = torch.sum(predictions, dim=1)
        assert torch.allclose(prob_sum, torch.ones(2), atol=1e-6)
    
    def test_hyperspoof_model_info(self):
        """Test model information."""
        model = HyperSpoofModel(
            input_channels=3,
            hsi_channels=31,
            attention_channels=3,
            num_classes=2,
            model_name="efficientnet_b0",
            pretrained=False
        )
        
        info = model.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['model_size_mb'] > 0


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_model_batch_sizes(batch_size):
    """Test model with different batch sizes."""
    model = HyperSpoofModel(
        input_channels=3,
        hsi_channels=31,
        attention_channels=3,
        num_classes=2,
        model_name="efficientnet_b0",
        pretrained=False
    )
    
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    outputs = model(input_tensor)
    
    assert outputs['logits'].shape[0] == batch_size
    assert outputs['probabilities'].shape[0] == batch_size
    assert outputs['hsi_features'].shape[0] == batch_size
    assert outputs['attention_features'].shape[0] == batch_size


@pytest.mark.parametrize("input_size", [(128, 128), (256, 256), (512, 512)])
def test_model_input_sizes(input_size):
    """Test model with different input sizes."""
    model = HyperSpoofModel(
        input_channels=3,
        hsi_channels=31,
        attention_channels=3,
        num_classes=2,
        model_name="efficientnet_b0",
        pretrained=False
    )
    
    input_tensor = torch.randn(2, 3, *input_size)
    outputs = model(input_tensor)
    
    assert outputs['logits'].shape == (2, 2)
    assert outputs['probabilities'].shape == (2, 2)
    assert outputs['hsi_features'].shape == (2, 31, *input_size)
    assert outputs['attention_features'].shape == (2, 3, *input_size)
