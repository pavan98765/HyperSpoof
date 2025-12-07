"""
Tests for data handling components.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from PIL import Image
from hyperspoof.data import SpoofDataset, CrossDatasetSpoofDataset, get_transforms, get_augmentation_transforms


class TestSpoofDataset:
    """Test SpoofDataset class."""
    
    def create_temp_dataset(self, num_images=10):
        """Create temporary dataset for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create real images
        real_dir = os.path.join(temp_dir, "real")
        os.makedirs(real_dir, exist_ok=True)
        
        for i in range(num_images):
            img = Image.new('RGB', (256, 256), color=(255, 0, 0))
            img.save(os.path.join(real_dir, f"real_{i}.jpg"))
        
        # Create spoof images
        spoof_dir = os.path.join(temp_dir, "spoof")
        os.makedirs(spoof_dir, exist_ok=True)
        
        for i in range(num_images):
            img = Image.new('RGB', (256, 256), color=(0, 255, 0))
            img.save(os.path.join(spoof_dir, f"spoof_{i}.jpg"))
        
        return temp_dir
    
    def test_spoof_dataset_creation(self):
        """Test SpoofDataset creation."""
        temp_dir = self.create_temp_dataset()
        
        # Test real dataset
        real_dataset = SpoofDataset(temp_dir, "real", 0)
        assert len(real_dataset) == 10
        assert real_dataset.label == 0
        
        # Test spoof dataset
        spoof_dataset = SpoofDataset(temp_dir, "spoof", 1)
        assert len(spoof_dataset) == 10
        assert spoof_dataset.label == 1
    
    def test_spoof_dataset_getitem(self):
        """Test SpoofDataset __getitem__ method."""
        temp_dir = self.create_temp_dataset()
        transform = get_transforms()
        
        dataset = SpoofDataset(temp_dir, "real", 0, transform)
        
        # Test getting an item
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)
        assert label == 0
        assert isinstance(label, torch.Tensor)
    
    def test_spoof_dataset_stats(self):
        """Test SpoofDataset statistics."""
        temp_dir = self.create_temp_dataset()
        dataset = SpoofDataset(temp_dir, "real", 0)
        
        stats = dataset.get_stats()
        
        assert stats['num_samples'] == 10
        assert stats['label'] == 0
        assert 'category' in stats
        assert 'root_dir' in stats


class TestCrossDatasetSpoofDataset:
    """Test CrossDatasetSpoofDataset class."""
    
    def create_temp_cross_dataset(self):
        """Create temporary cross-dataset for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create dataset 1
        dataset1_dir = os.path.join(temp_dir, "dataset1")
        os.makedirs(os.path.join(dataset1_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(dataset1_dir, "spoof"), exist_ok=True)
        
        for i in range(5):
            img = Image.new('RGB', (256, 256), color=(255, 0, 0))
            img.save(os.path.join(dataset1_dir, "real", f"real_{i}.jpg"))
        
        for i in range(5):
            img = Image.new('RGB', (256, 256), color=(0, 255, 0))
            img.save(os.path.join(dataset1_dir, "spoof", f"spoof_{i}.jpg"))
        
        # Create dataset 2
        dataset2_dir = os.path.join(temp_dir, "dataset2")
        os.makedirs(os.path.join(dataset2_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(dataset2_dir, "spoof"), exist_ok=True)
        
        for i in range(3):
            img = Image.new('RGB', (256, 256), color=(0, 0, 255))
            img.save(os.path.join(dataset2_dir, "real", f"real_{i}.jpg"))
        
        for i in range(3):
            img = Image.new('RGB', (256, 256), color=(255, 255, 0))
            img.save(os.path.join(dataset2_dir, "spoof", f"spoof_{i}.jpg"))
        
        return temp_dir
    
    def test_cross_dataset_creation(self):
        """Test CrossDatasetSpoofDataset creation."""
        temp_dir = self.create_temp_cross_dataset()
        
        dataset_configs = [
            {
                'root_dir': os.path.join(temp_dir, "dataset1"),
                'category': 'real',
                'label': 0,
                'name': 'dataset1_real'
            },
            {
                'root_dir': os.path.join(temp_dir, "dataset1"),
                'category': 'spoof',
                'label': 1,
                'name': 'dataset1_spoof'
            },
            {
                'root_dir': os.path.join(temp_dir, "dataset2"),
                'category': 'real',
                'label': 0,
                'name': 'dataset2_real'
            },
            {
                'root_dir': os.path.join(temp_dir, "dataset2"),
                'category': 'spoof',
                'label': 1,
                'name': 'dataset2_spoof'
            }
        ]
        
        cross_dataset = CrossDatasetSpoofDataset(dataset_configs)
        
        assert len(cross_dataset) == 16  # 5 + 5 + 3 + 3
        assert len(cross_dataset.datasets) == 4
    
    def test_cross_dataset_getitem(self):
        """Test CrossDatasetSpoofDataset __getitem__ method."""
        temp_dir = self.create_temp_cross_dataset()
        
        dataset_configs = [
            {
                'root_dir': os.path.join(temp_dir, "dataset1"),
                'category': 'real',
                'label': 0,
                'name': 'dataset1_real'
            }
        ]
        
        cross_dataset = CrossDatasetSpoofDataset(dataset_configs)
        
        # Test getting an item
        image, label, dataset_idx = cross_dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)
        assert label == 0
        assert dataset_idx == 0
    
    def test_cross_dataset_stats(self):
        """Test CrossDatasetSpoofDataset statistics."""
        temp_dir = self.create_temp_cross_dataset()
        
        dataset_configs = [
            {
                'root_dir': os.path.join(temp_dir, "dataset1"),
                'category': 'real',
                'label': 0,
                'name': 'dataset1_real'
            }
        ]
        
        cross_dataset = CrossDatasetSpoofDataset(dataset_configs)
        stats = cross_dataset.get_dataset_stats()
        
        assert len(stats) == 1
        assert stats[0]['num_samples'] == 5
        assert stats[0]['label'] == 0


class TestTransforms:
    """Test data transforms."""
    
    def test_get_transforms(self):
        """Test get_transforms function."""
        transform = get_transforms()
        
        assert transform is not None
        assert hasattr(transform, '__call__')
    
    def test_get_augmentation_transforms(self):
        """Test get_augmentation_transforms function."""
        for strength in ['light', 'medium', 'strong']:
            transform = get_augmentation_transforms(augmentation_strength=strength)
            assert transform is not None
            assert hasattr(transform, '__call__')
    
    def test_transform_application(self):
        """Test transform application."""
        transform = get_transforms()
        
        # Create dummy image
        image = Image.new('RGB', (256, 256), color=(255, 0, 0))
        
        # Apply transform
        transformed = transform(image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        assert transformed.dtype == torch.float32


@pytest.mark.parametrize("num_images", [1, 5, 10])
def test_dataset_sizes(num_images):
    """Test dataset with different sizes."""
    temp_dir = tempfile.mkdtemp()
    
    # Create real images
    real_dir = os.path.join(temp_dir, "real")
    os.makedirs(real_dir, exist_ok=True)
    
    for i in range(num_images):
        img = Image.new('RGB', (256, 256), color=(255, 0, 0))
        img.save(os.path.join(real_dir, f"real_{i}.jpg"))
    
    dataset = SpoofDataset(temp_dir, "real", 0)
    assert len(dataset) == num_images
