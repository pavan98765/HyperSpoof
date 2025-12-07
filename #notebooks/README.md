# 📓 Training & Experiment Notebooks

This directory contains the **original training notebooks** used for all experiments reported in the HyperSpoof paper. These Jupyter notebooks demonstrate the complete pipeline from data loading to model evaluation and were used to generate the results published in our work.

## 🔬 Experimental Notebooks

All experiments were conducted using these notebooks on an **NVIDIA RTX A6000 GPU (48GB VRAM)**.

### 1. Cross-Attack Evaluation (RECOD-MPAD)

#### **`Crossdataset_RecodMpad_p1+hp_p2+cce_mstpp_our_pipeline.ipynb`** ⭐
**Our Main Pipeline - HyperSpoof**

- **Training Set**: RECOD-MPAD Print1 + HP attacks
- **Test Set**: RECOD-MPAD Print2 + CCE attacks (unseen attacks)
- **Architecture**: MST++ → Spectral Attention → EfficientNetB0
- **Key Results**:
  - Test Accuracy: **81.30%**
  - ACER: **0.181**
  - HTER: **0.181**
  - EER: **0.360**
- **Purpose**: Demonstrates robustness against **unseen attack types** within same dataset

#### **`Crossdataset_RecodMpad_p1+hp_p2+cce_simple_resnet50.ipynb`**
**Baseline Comparison - ResNet50**

- Same train/test split as above
- **Architecture**: Simple ResNet50 (no hyperspectral reconstruction)
- **Key Results**:
  - Test Accuracy: 66.70%
  - ACER: 0.323
  - HTER: 0.323
  - EER: 0.644
- **Purpose**: Baseline to demonstrate improvement from hyperspectral approach

---

### 2. Cross-Dataset Evaluation (Leave-One-Out Protocol)

#### **`LOO_train_val_LCC_FASD+CASIA_test_RECOD_our_pipeline.ipynb`** 🌐
**Cross-Domain Generalization Test #1**

- **Training Set**: LCC FASD + CASIA-SURF (combined)
- **Validation Set**: Split from training datasets
- **Test Set**: RECOD-MPAD (completely unseen dataset)
- **Architecture**: Full HyperSpoof pipeline
- **Key Results**:
  - Test Accuracy: **80.52%**
  - ACER: **0.229**
  - HTER: **0.195**
  - EER: **0.286**
- **Purpose**: Evaluates **cross-dataset generalization** - can model detect spoofs from entirely different dataset?

#### **`LOO_train_val_LCC_FASD+Recod_test_casia_our_pipeline.ipynb`** 🌐
**Cross-Domain Generalization Test #2**

- **Training Set**: LCC FASD + RECOD-MPAD (combined)
- **Validation Set**: Split from training datasets
- **Test Set**: CASIA-SURF (completely unseen dataset)
- **Architecture**: Full HyperSpoof pipeline
- **Key Results**: [Cross-domain performance metrics]
- **Purpose**: Alternative cross-dataset evaluation to validate generalization capability

---

## 🎯 Experiment Summary

| Notebook | Experiment Type | Train Datasets | Test Dataset | Best Metric |
|----------|----------------|----------------|--------------|-------------|
| **HyperSpoof Pipeline** | Cross-Attack | RECOD (P1+HP) | RECOD (P2+CCE) | **81.30% Acc** |
| ResNet50 Baseline | Cross-Attack | RECOD (P1+HP) | RECOD (P2+CCE) | 66.70% Acc |
| **Cross-Domain #1** | Leave-One-Out | LCC+CASIA | RECOD | **80.52% Acc** |
| **Cross-Domain #2** | Leave-One-Out | LCC+RECOD | CASIA | - |

## 📊 What These Results Demonstrate

1. **Cross-Attack Robustness** (+14.6% over ResNet50)
   - HyperSpoof detects **unseen attack types** significantly better than RGB-only baseline
   - Hyperspectral reconstruction captures material properties invisible to RGB

2. **Cross-Dataset Generalization** (80.52% on unseen dataset)
   - Model trained on LCC+CASIA successfully detects spoofs on RECOD
   - Minimal performance drop compared to single-dataset evaluation
   - Spectral features generalize better than appearance-based features

3. **Real-World Applicability**
   - Leave-one-out protocol simulates real deployment scenarios
   - Model can handle different cameras, lighting, and attack materials

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
pip install -r ../requirements.txt

# Download datasets (see links in main README)
# - RECOD-MPAD: https://zenodo.org/records/3749309
# - LCC FASD: https://www.kaggle.com/datasets/faber24/lcc-fasd
# - CASIA-SURF: https://sites.google.com/view/face-anti-spoofing-challenge
```

### Running Experiments

1. **Update Dataset Paths**: Open each notebook and modify the dataset paths in the first few cells:
   ```python
   # Example:
   RECOD_PATH = "/path/to/RECOD-MPAD"
   LCC_PATH = "/path/to/LCC_FASD"
   CASIA_PATH = "/path/to/CASIA-SURF"
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Run Cells Sequentially**: Each notebook is organized as:
   - Data loading and preprocessing
   - Model initialization (MST++ weights loaded if available)
   - Training loop with metrics tracking
   - Evaluation on test set
   - Results visualization and metric calculation

### GPU Requirements

- **Minimum**: 12GB VRAM (reduce batch size if needed)
- **Recommended**: 24GB+ VRAM (as used in paper: RTX A6000 48GB)
- **CPU-only**: Possible but very slow (not recommended)

## 📝 Notes

- **MST++ Weights**: The HSI reconstruction module uses pretrained MST++ weights from ARAD-1K dataset
  - Download from: [MST++ repository](https://github.com/caiyuanhao1998/MST-plus-plus)
  - Place in: `checkpoints/mst_plus_plus.pth`

- **Training Time**:
  - Cross-attack experiments: ~2-3 hours
  - Cross-dataset experiments: ~4-6 hours (larger combined datasets)

- **Reproducibility**: Random seeds are set in notebooks for reproducibility, but minor variations may occur due to GPU non-determinism

## 🔧 Troubleshooting

**Out of Memory Errors**:
```python
# Reduce batch size in the notebook
BATCH_SIZE = 16  # Default: 32
```

**Dataset Loading Issues**:
- Ensure dataset paths are correct
- Check file structure matches expected format (see dataset README files)

**Missing MST++ Weights**:
- Model will train from scratch if weights not found (slower convergence)
- Download pretrained weights for best results


---

**For the production-ready package implementation, see the `hyperspoof/` directory.**
