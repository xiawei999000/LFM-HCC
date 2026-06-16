# HCC-LFM: Liver Foundation Model for Non-invasive HCC Assessment

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

A vision foundation model pipeline for non-invasive assessment of hepatocellular carcinoma (HCC) aggressiveness and recurrence-free survival (RFS) prediction from contrast-enhanced CT.

## Overview

This repository provides the complete pipeline for:

| Task | Method |
|------|--------|
| **Self-supervised pretraining** | DINOv2 (ViT-S) on CT lesion patches |
| **MVI & grade prediction** | Cross-phase attention + attention-based MIL |
| **RFS survival prediction** | Elastic-net Cox regression on deep features |
| **Robustness evaluation** | Input perturbation + limited training data tests |
| **Radiomics baseline** | PyRadiomics + mRMR + LASSO |
| **Genomic association** | TCGA-LIHC transcriptomic + immune deconvolution |

## Installation

```bash
git clone https://github.com/xxx/HCC-LFM.git
cd HCC-LFM
pip install -r requirements.txt
```

> **Note:** The FMCIB comparison method requires the external `fmcib` package. See [FMCIB](https://github.com/xxx/fmcib) for installation.

## Data & Model Weights

- **Pretraining data:** DeepLesion (public) + HCC cohort CT patches
- **Model weights:** [![DOI](https://img.shields.io/badge/Zenodo-link_to_be_added-blue)](https://zenodo.org)

Clinical data and fine-tuned model weights are available on Zenodo (link to be added).

## Project Structure

```
├── config.py              # Global paths and hyperparameters
├── preprocessing/         # CT registration, lesion extraction, normalization
├── pretraining/           # DINOv2 self-supervised pretraining
├── finetuning/            # Multi-task MIL fine-tuning (MVI + grade)
├── fmcib/                 # FMCIB baseline method
├── radiomics/             # PyRadiomics feature extraction + LASSO
├── survival/              # Elastic-net Cox regression for RFS
├── inference/             # Model inference utilities
├── evaluation/            # DeLong test, Cox regression, subgroup analysis
├── robustness/            # Perturbation tests, stratified sampling
├── visualization/         # Occlusion heatmap tools
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

### 5. Evaluation

```bash
# Model comparison (DeLong test)
python evaluation/compare_models.py --clinical <path> --predictions <path>

# Cox regression
python evaluation/cox_regression.py --clinical <path> --risk_scores <path>

# Subgroup analysis
python evaluation/subgroup_analysis.py --clinical <path> --predictions <path>
```

### 6. Robustness Tests

```bash
python robustness/perturbation_dinov2.py
python robustness/stratified_sampling.py
```

### 7. Genomic Association Analysis

Biological interpretation of imaging-derived risk scores using TCGA-LIHC transcriptomic data.

```bash
# Requires R (≥4.0) with packages: DESeq2, clusterProfiler, igraph, survival, etc.
cd genomic_association
Rscript radiogenomic_analysis.R
```

| Analysis | Method |
|----------|--------|
| Differential expression | DESeq2 (Wald test, apeglm shrinkage) |
| GO enrichment | clusterProfiler (hypergeometric test, BH correction) |
| PPI network + hub genes | STRING v12.0 + Maximal Clique Centrality |
| Immune deconvolution | CIBERSORT (ν-SVR, LM22 signature) |

**Input**: Imaging risk scores, TCGA-LIHC RNA-seq (STAR-Counts)  
**Output**: Combined figure (transcriptome + immune microenvironment), hub genes, CIBERSORT fractions  

See `genomic_association/README.md` for details.

## Models

| Name | Description |
|------|-------------|
| **LFM-Base** | Official DINOv2 weights (ImageNet pretrained) |
| **LFM-DL** | Pretrained on DeepLesion CT patches |
| **LFM-Mix** | Pretrained on DeepLesion + HCC patches |
| **LFM-Seq** | Sequential: DeepLesion -> HCC patches |
| **FMCIB** | 3D medical imaging foundation model (baseline) |
| **Radiomics** | Traditional handcrafted features + LASSO (baseline) |
| **Clinical model** | Clinical variables + elastic-net Cox (baseline) |

## Citation

```bibtex
@article{xxx,
  title={xxx},
  author={xxx},
  journal={xxx},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
