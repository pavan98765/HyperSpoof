# HyperSpoof

[![CI/CD Pipeline](https://github.com/yourusername/HyperSpoof/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/HyperSpoof/actions)
[![PyPI version](https://badge.fury.io/py/hyperspoof.svg)](https://badge.fury.io/py/hyperspoof)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

HyperSpoof is a novel framework for face anti-spoofing that leverages hyperspectral reconstruction and attention mechanisms. By reconstructing hyperspectral information from standard RGB inputs and utilizing spectral attention layers, HyperSpoof enhances detection accuracy and robustness against sophisticated spoofing attacks, such as high-quality prints, video replays, and 3D masks.

## 🚀 Features

- **Hyperspectral Reconstruction**: Converts RGB images into 31-channel hyperspectral data using the MST++ model
- **Spectral Attention Mechanism**: Highlights discriminative features, reducing hyperspectral data into a compact 3-channel representation
- **Efficient Classification**: Employs an EfficientNetB0 classifier for accurate binary predictions
- **Cross-Dataset Generalization**: Demonstrates robustness across diverse datasets and spoofing types
- **Real-Time Efficiency**: Optimized for high performance and low computational cost
- **Comprehensive Evaluation**: Supports multiple metrics including ACER, HTER, and EER
- **Easy-to-Use API**: Simple Python interface for training, evaluation, and inference

## 📊 Results

### Cross-Dataset Evaluation
HyperSpoof achieves superior generalization compared to baseline models, with higher accuracy and lower error rates across unseen datasets.

| Model          | Train Accuracy (%) | Test Accuracy (%) | ACER  | HTER  | EER   |
|----------------|--------------------|-------------------|-------|-------|-------|
| ResNet50       | 100.00            | 66.70            | 0.323 | 0.323 | 0.6437 |
| EfficientNetB0 | 100.00            | 68.49            | 0.314 | 0.314 | 0.628  |
| **HyperSpoof** | **100.00**        | **81.30**        | **0.181** | **0.181** | **0.360** |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA-enabled GPU (recommended)

### Quick Install
```bash
pip install hyperspoof
```

### Development Install
```bash
git clone https://github.com/yourusername/HyperSpoof.git
cd HyperSpoof
pip install -e .
```

### From Source
```bash
git clone https://github.com/yourusername/HyperSpoof.git
cd HyperSpoof
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from hyperspoof import HyperSpoofModel
from hyperspoof.data import get_transforms
from PIL import Image

# Load model
model = HyperSpoofModel(pretrained=True)
model.eval()

# Load and preprocess image
transform = get_transforms()
image = Image.open("path/to/image.jpg")
image_tensor = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    outputs = model(image_tensor)
    prediction = torch.argmax(outputs['logits'], dim=1)
    confidence = torch.max(outputs['probabilities'], dim=1)[0]

print(f"Prediction: {'Real' if prediction.item() == 0 else 'Spoof'}")
print(f"Confidence: {confidence.item():.3f}")
```

### Training

```python
from hyperspoof import HyperSpoofModel
from hyperspoof.data import create_cross_dataset_dataloaders
from hyperspoof.training import Trainer
from hyperspoof.utils import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Create model
model = HyperSpoofModel(**config['model'])

# Create data loaders
train_loader, val_loader, test_loader = create_cross_dataset_dataloaders(
    train_configs=config['data']['train_configs'],
    test_configs=config['data']['test_configs']
)

# Create trainer
trainer = Trainer(model=model, **config['training'])

# Train model
history = trainer.train(train_loader, val_loader)
```

### Evaluation

```python
from hyperspoof.metrics import calculate_metrics, plot_confusion_matrix

# Evaluate model
test_metrics = trainer.evaluate(test_loader)

# Create visualizations
plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png")
```

## 📁 Project Structure

```
HyperSpoof/
├── hyperspoof/                 # Main package
│   ├── models/                # Model definitions
│   │   ├── hyperspoof.py      # Main HyperSpoof model
│   │   ├── hsi_reconstruction.py
│   │   ├── spectral_attention.py
│   │   └── classifier.py
│   ├── data/                  # Data handling
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── dataloader.py
│   ├── training/              # Training utilities
│   │   ├── trainer.py
│   │   └── early_stopping.py
│   ├── metrics/               # Evaluation metrics
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── utils/                 # Utility functions
│   │   ├── config.py
│   │   ├── device.py
│   │   ├── logging.py
│   │   └── checkpoint.py
│   └── cli/                   # Command-line interface
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
├── configs/                   # Configuration files
├── examples/                  # Example scripts
├── tests/                     # Test suite
├── docs/                      # Documentation
└── README.md
```

## 🔧 Configuration

HyperSpoof uses YAML configuration files for easy experiment management:

```yaml
# configs/default.yaml
model:
  name: "hyperspoof"
  input_channels: 3
  hsi_channels: 31
  attention_channels: 3
  num_classes: 2
  classifier: "efficientnet_b0"
  pretrained: true
  dropout_rate: 0.2

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4
  scheduler: "cosine"
  early_stopping_patience: 10

data:
  batch_size: 32
  image_size: [256, 256]
  augmentation: true
  augmentation_strength: "medium"
```

## 📊 Datasets

HyperSpoof has been tested on multiple datasets:

- **RECOD-MPAD**: Mobile-based presentation attack dataset
- **CelebA-Spoof**: Large-scale dataset with rich annotations
- **LCC FASD**: Focused on face anti-spoofing
- **CASIA-SURF**: Multi-modal benchmark dataset

### Dataset Links
- [ReCodMpad](https://zenodo.org/records/3749309)
- [CASIA-SURF](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020)
- [CelebA-Spoof](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html)
- [LCC FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd)

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hyperspoof --cov-report=html

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_metrics.py -v
```

## 📈 Command Line Interface

### Training
```bash
hyperspoof-train --config configs/default.yaml --data_root /path/to/data
```

### Evaluation
```bash
hyperspoof-eval --config configs/default.yaml --checkpoint checkpoints/best_model.pth
```

### Prediction
```bash
hyperspoof-predict --config configs/default.yaml --checkpoint checkpoints/best_model.pth --input /path/to/image.jpg
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use HyperSpoof in your research, please cite our paper:

```bibtex
@article{hyperspoof2024,
  title={HyperSpoof: A Novel Framework for Face Anti-Spoofing Using Hyperspectral Reconstruction},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 🙏 Acknowledgments

- The authors would like to thank the contributors to the open-source face anti-spoofing community
- Special thanks to the dataset providers for making their data publicly available
- This work was supported by [Funding Agency/Institution]

## 📞 Contact

- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/HyperSpoof/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/HyperSpoof/discussions)

## 🔗 Related Projects

- [Face Anti-Spoofing Survey](https://github.com/example/face-anti-spoofing-survey)
- [Hyperspectral Image Processing](https://github.com/example/hyperspectral-processing)
- [Deep Learning for Biometrics](https://github.com/example/deep-biometrics)
