"""
HyperSpoof: A novel framework for face anti-spoofing using hyperspectral reconstruction.

This package provides tools for:
- Hyperspectral reconstruction from RGB images
- Spectral attention mechanisms
- Face anti-spoofing classification
- Cross-dataset evaluation
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import HyperSpoofModel, SpectralAttention, HSIReconstruction
from .data import SpoofDataset, get_transforms
from .utils import set_seed, get_device, load_config
from .metrics import calculate_metrics, plot_confusion_matrix

__all__ = [
    "HyperSpoofModel",
    "SpectralAttention", 
    "HSIReconstruction",
    "SpoofDataset",
    "get_transforms",
    "set_seed",
    "get_device",
    "load_config",
    "calculate_metrics",
    "plot_confusion_matrix",
]
