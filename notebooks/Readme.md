# HyperSpoof

HyperSpoof is a novel framework for face anti-spoofing that leverages hyperspectral reconstruction and attention mechanisms. By reconstructing hyperspectral information from standard RGB inputs and utilizing spectral attention layers, HyperSpoof enhances detection accuracy and robustness against sophisticated spoofing attacks, such as high-quality prints, video replays, and 3D masks.

## Features

- **Hyperspectral Reconstruction**: Converts RGB images into 31-channel hyperspectral data using the MST++ model.
- **Spectral Attention Mechanism**: Highlights discriminative features, reducing hyperspectral data into a compact 3-channel representation.
- **Efficient Classification**: Employs an EfficientNetB0 classifier for accurate binary predictions.
- **Cross-Dataset Generalization**: Demonstrates robustness across diverse datasets and spoofing types.
- **Real-Time Efficiency**: Optimized for high performance and low computational cost.

## Methodology

The HyperSpoof framework consists of three main components:
1. **HSI Reconstruction Module**: Uses MST++ to reconstruct detailed hyperspectral images from RGB inputs.
2. **Spectral Attention Module**: Refines hyperspectral data, emphasizing relevant spectral bands.
3. **Classification Module**: Utilizes EfficientNetB0 for binary classification into real or spoofed faces.

![Pipeline](assets/hyperspoof_pipeline.png)

## Datasets

HyperSpoof has been tested on multiple datasets to validate its effectiveness:
- **RECOD-MPAD**: Mobile-based presentation attack dataset.
- **CelebA-Spoof**: Large-scale dataset with rich annotations.
- **LCC FASD**: Focused on face anti-spoofing.
- **CASIA-SURF**: Multi-modal benchmark dataset.

## Results

### Cross-Dataset Evaluation
HyperSpoof achieves superior generalization compared to baseline models, with higher accuracy and lower error rates across unseen datasets.

| Model          | Train Accuracy (%) | Test Accuracy (%) | ACER  | HTER  | EER   |
|----------------|--------------------|-------------------|-------|-------|-------|
| ResNet50       | 100.00            | 66.70            | 0.323 | 0.323 | 0.6437 |
| EfficientNetB0 | 100.00            | 68.49            | 0.314 | 0.314 | 0.628  |
| HyperSpoof     | 100.00            | 81.30            | 0.181 | 0.181 | 0.360  |

### Cross-Attack Evaluation
HyperSpoof demonstrates robust performance across diverse attack modalities, significantly outperforming existing models.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-enabled GPU

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/pavan98765/HyperSpoof.git
   cd HyperSpoof

## Installation

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model using the specified configuration files.
2. Evaluate the model for performance on various datasets.
3. Use pretrained models available in the `models/` directory for inference
