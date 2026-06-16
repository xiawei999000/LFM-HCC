# A CT Lesion Foundation Model for Non-invasive Assessment of Hepatocellular Carcinoma Aggressiveness and Prediction of Recurrence Risk

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Overview

This repository provides the complete pipeline for:

| Task | Method |
|------|--------|
| **Self-supervised pretraining** | DINOv2 (ViT-S) on CT lesion patches |
| **MVI & pathological grade evaluation** | Cross-phase attention + attention-based multiple instance learning |
| **Robustness evaluation** | Input perturbation + limited training data tests |
| **3D lesion foundation model baseline** | FMCIB: 3D lesion volume + modified SimCLR |
| **Radiomics baseline** | PyRadiomics + mRMR + LASSO |
| **RFS survival prediction** | Elastic-net Cox regression on deep features |
| **Genomic association** | TCGA-LIHC CT imaging data matched transcriptomic data analysis |

## Installation

```bash
git clone https://github.com/xiawei999000/LFM-HCC.git
cd LFM-HCC
pip install -r requirements.txt
```

> **Note:** The FMCIB comparison method requires the external `fmcib` package. See [FMCIB](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker) for installation.

## Data & Model Weights

- **Pretraining data:** [![DOI](https://img.shields.io/badge/Zenodo-https://zenodo.org/records/20710784-red)](https://zenodo.org/records/20710784) DeepLesion and HCC-specific CT patches.
- **Model weights:** [![DOI](https://img.shields.io/badge/Zenodo-https://zenodo.org/records/20710784-blue)](https://zenodo.org/records/20710784) Pretrained and fine-tuned model weights.

## Models

| Name | Description |
|------|-------------|
| **LFM-Base** | Official DINOv2 weights |
| **LFM-DL** | Pretrained on DeepLesion CT patches |
| **LFM-Mix** | Pretrained on DeepLesion + HCC patches |
| **LFM-Seq** | Sequential pretrained: DeepLesion -> HCC patches |


## Project Structure

```
├── config.py              # Global paths and hyperparameters
├── preprocessing/         # CT registration, lesion extraction, normalization
├── pretraining/           # DINOv2 self-supervised pretraining
├── finetuning/            # Multi-task MIL fine-tuning (MVI + grade)
├── fmcib/                 # FMCIB baseline method
├── radiomics/             # PyRadiomics feature extraction + mRMR + LASSO
├── inference/             # Model inference utilities
├── robustness/            # Perturbation tests, stratified sampling
├── visualization/         # Occlusion heatmap tools
├── survival/              # Elastic-net Cox regression for RFS
├── genomic_association/   # TCGA radiogenomic analysis pipeline
└── README.md
```

## Quick Start

### 1. Configuration

Edit `config.py` to set your data and output paths:

```python
DATA_DIR = "/path/to/your/data"
MODEL_DIR = "/path/to/your/models"
OUTPUT_DIR = "/path/to/your/results"
```

### 2. Preprocessing

```bash
python preprocessing/register_ct_phases.py --data_path <CT_dir>
python preprocessing/extract_lesion_volumes.py --input_dir <dir> --output_dir <out> --phase AP
```

### 3. Pretraining

```bash
# DeepLesion only (LFM-DL)
python pretraining/pretrain_dinov2.py --data_dir <dl_patches> --output_dir <ckpt_dl>

# Combined pretraining (LFM-Mix)
python pretraining/pretrain_dinov2.py --data_dir <mix_patches> --output_dir <ckpt_mix>

# Sequential pretraining (LFM-Seq)
python pretraining/pretrain_dinov2.py --data_dir <dl_patches> --output_dir <ckpt_dl>
python pretraining/pretrain_dinov2.py --data_dir <hcc_patches> --output_dir <ckpt_seq> --resume <ckpt_dl/best.pth>
```

### 4. Fine-tuning

```bash
python finetuning/finetune_dinov2.py --lr_list 1e-5 5e-5 1e-4 --epochs 50
```

### 5. Robustness Tests

```bash
python robustness/perturbation_dinov2.py
python robustness/stratified_sampling.py
```

### 6. Genomic Association Analysis

Biological interpretation of imaging-derived risk scores using TCGA-LIHC transcriptomic data.

```bash
# Requires R (≥4.0) with packages: DESeq2, clusterProfiler, igraph, survival, etc.
cd genomic_association
Rscript radiogenomic_analysis.R
```

See `genomic_association/README.md` for details.



## License

MIT License. See [LICENSE](LICENSE) for details.
