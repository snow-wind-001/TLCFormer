# TLCFormer: Synergizing Temporal Motion and Local Contrast for Robust Infrared Video Small Object Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task: IRSTD](https://img.shields.io/badge/Task-Infrared%20Small%20Target%20Detection-blue.svg)](https://github.com/topics/infrared-small-target-detection)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

[**中文版本 (Chinese Version)**](README_CN.md)

> **Official PyTorch implementation** of the paper *"TLCFormer: Synergizing Temporal Motion and Local Contrast for Robust Infrared Video Small Object Detection"*

## 📖 Introduction

**TLCFormer** (Temporal-Local-Contrast Transformer) is a single-stage Transformer model specifically designed for **infrared video small object detection under complex backgrounds**.

Existing methods like OSFormer suffer from detection failures under strong background noise (cloud edges, ground clutter) and small target features being overwhelmed in deep networks. We propose three physics-prior-guided mechanisms:

1. **MADA (Motion-Aware Difference Attention)**: Leverages multi-frame differencing to capture motion priors and suppress static backgrounds
2. **DLCM (Deep Local Contrast Module)**: Enhances high-frequency small target features based on local contrast principles
3. **Hybrid Energy-Preserving Mixer**: Replaces traditional AvgPool to prevent small target energy loss during downsampling

## 🏗️ Architecture

The TLCFormer pipeline consists of:
- **Cube Encoding**: RGBT multimodal fusion with temporal sampling
- **MADA**: Motion-aware background suppression
- **DLCM**: Local contrast enhancement
- **VPA Encoder**: Multi-scale feature extraction with Hybrid Mixer
- **Detection Head**: Sequence regression for bounding box prediction
- **TemporalFeatureExtractor**: Lightweight per-frame feature extraction from Cube for cross-frame offset prediction

## 🚀 Key Innovations

### 1. Motion-Aware Difference Attention (MADA)

Unlike OSFormer's FFT-based Doppler filtering (susceptible to high-frequency noise), MADA explicitly utilizes **temporal frame differencing**:

```
D_pre = |I_t - I_{t-1}|
D_next = |I_{t+1} - I_t|
M_raw = D_pre ⊙ D_next  (Hadamard product)
A_motion = σ(F_motion(M_raw))
I'_t = I_t · (1 + α · A_motion)
```

This generates dynamic motion masks that force the network to focus on moving regions, effectively filtering static clouds and ground textures.

### 2. Deep Local Contrast Module (DLCM)

Introduces the classic **Local Contrast** theory from infrared physics:

```
μ_bg = AvgPool_{K_out}(X)     # Background estimation
L_max = MaxPool_{K_in}(X)      # Target intensity
C = L_max² / (μ_bg + ε)        # Contrast response
X_out = X + β · C              # Adaptive fusion
```

Enhances target-background contrast at early feature extraction stages, preventing weak targets from being lost before entering the backbone.

### 3. Hybrid Energy-Preserving Mixer

Reconstructs the Token Mixer in VPA (Varied-Size Patch Attention):

| Version | Pooling | Issue |
|---------|---------|-------|
| **OSFormer** | AvgPool only | Point targets averaged out by surrounding background |
| **TLCFormer** | Max-Mean Hybrid | MaxPool preserves target extrema; AvgPool maintains texture |

```python
P_max = MaxPool2d(X, k, s)
P_avg = AvgPool2d(X, k, s)
P_hybrid = Concat(P_max, P_avg)
X_mixed = GELU(Conv1x1(P_hybrid))
X_out = X + X_mixed  # Residual connection
```

This significantly improves small target **Recall** by preserving peak energy through network layers.

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (recommended)

```bash
# 1. Clone repository
git clone https://github.com/snow-wind-001/TLCFormer.git
cd TLCFormer

# 2. Create virtual environment
conda create -n tlcformer python=3.8
conda activate tlcformer

# 3. Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
TLCFormer/
├── models/
│   ├── osformer.py      # Main TLCFormer model
│   ├── mada.py          # Motion-Aware Difference Attention
│   ├── dlcm.py          # Deep Local Contrast Module
│   ├── vpa.py           # VPA with Hybrid Mixer
│   ├── cube_encoding.py # Multimodal cube encoding
│   ├── neck.py          # Feature refinement neck
│   └── seq_head.py      # Sequence head with TemporalFeatureExtractor + OffsetPredictor
├── utils/
│   └── loss.py          # Loss functions (Focal, CIoU, etc.)
├── verify_model.py      # Model verification script
├── requirements.txt     # Dependencies
├── CHANGELOG.md         # Version history
└── README.md            # This file
```

## 🚂 Usage

### Model Initialization

```python
from models import TLCFormer, TLCFormerConfig

# Create configuration
config = TLCFormerConfig(
    num_frames=3,
    embed_dim=256,
    use_mada=True,      # Enable MADA
    use_dlcm=True,      # Enable DLCM
    mada_alpha=0.5,     # MADA scaling factor
    dlcm_beta=0.3,      # DLCM enhancement weight
)

# Initialize model
model = TLCFormer(config)
model.cuda()
```

### Forward Pass

```python
import torch

# Input: RGB and Thermal video frames
rgb_frames = torch.randn(2, 3, 3, 512, 512).cuda()      # [B, T, C, H, W]
thermal_frames = torch.randn(2, 3, 1, 512, 512).cuda()  # [B, T, 1, H, W]

# Forward
outputs = model(rgb_frames, thermal_frames)

# Outputs: classification, bbox, centerness for each scale
for key, value in outputs.items():
    print(f"{key}: {value.shape}")
```

### Model Verification

```bash
# Run comprehensive model tests
python verify_model.py
```

This script validates:
- Individual module functionality (MADA, DLCM, HybridMixer)
- Full model forward/backward passes
- Training loop with mock data

## 📊 Loss Functions

TLCFormer uses a multi-task loss:

```python
L_total = λ_cls · L_focal + λ_bbox · L_ciou + λ_center · L_bce
```

| Loss | Purpose | Weight |
|------|---------|--------|
| Weighted Focal Loss | Classification with class imbalance handling | 1.0 |
| CIoU Loss | Bounding box regression | 2.0 |
| BCE Loss | Centerness prediction | 1.0 |

## 📄 Paper

The paper *"TLCFormer: Synergizing Temporal Motion and Local Contrast for Robust Infrared Video Small Object Detection"* (ICML 2026 format) is not included in this repository. Key sections include detailed derivations for MADA, DLCM, Hybrid Mixer, and experimental analysis.

## 📈 Results

*Coming soon: Benchmark results on NUDT-SIRST, IRSTD-1k, and custom datasets*

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 3 | Number of temporal frames |
| `embed_dim` | 256 | Feature embedding dimension |
| `use_mada` | True | Enable MADA module |
| `use_dlcm` | True | Enable DLCM module |
| `mada_alpha` | 0.5 | MADA attention scaling |
| `dlcm_beta` | 0.3 | DLCM contrast weight |
| `use_doppler` | False | Legacy Doppler filter (disabled when MADA active) |

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tlcformer2026,
  title={TLCFormer: Synergizing Temporal Motion and Local Contrast for Robust Infrared Video Small Object Detection},
  author={Anonymous},
  booktitle={Proceedings of ICML},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon insights from [OSFormer](https://github.com/xxx/OSFormer)
- Inspired by classical infrared small target detection methods (LCM, MPCM)
- Thanks to the PyTorch team for the excellent deep learning framework

## 📬 Contact

For questions or collaborations, please open an issue or contact via email.

---

**Star ⭐ this repo if you find it helpful!**
