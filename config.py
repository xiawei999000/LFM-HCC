"""
Global configuration for HCC LFM project.
All paths and hyperparameters are centralized here.
Update data_dir, model_dir, and output_dir before running.
"""

import os

# ── Root directories (user must set these) ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# ── Model naming ──────────────────────────────────────────────
MVI_GRADE_MODELS = [
    "Radiomics", "FMCIB", "LFM-Base",
    "LFM-DL", "LFM-Mix", "LFM-Seq",
]
RFS_MODELS = [
    "Clinical model", "Radiomics", "FMCIB",
    "LFM-Base", "LFM-DL", "LFM-Mix", "LFM-Seq",
]

MODEL_COLORS = {
    "Radiomics": "#D62728", "FMCIB": "#9467BD",
    "LFM-Base": "#7F7F7F", "LFM-DL": "#2CA02C",
    "LFM-Mix": "#1F77B4", "LFM-Seq": "#FF7F0E",
    "Clinical model": "#999999",
}
MODEL_LINESTYLES = {
    "Radiomics": "--", "FMCIB": "--",
    "LFM-Base": ":", "LFM-DL": "-",
    "LFM-Mix": "-", "LFM-Seq": "-",
    "Clinical model": "-",
}
MODEL_LINEWIDTHS = {
    "Radiomics": 1.2, "FMCIB": 1.2,
    "LFM-Base": 0.9, "LFM-DL": 1.5,
    "LFM-Mix": 1.5, "LFM-Seq": 1.8,
    "Clinical model": 1.5,
}

# ── Radiomics ─────────────────────────────────────────────────
RADIOMICS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "radiomics")
RADIOMICS_FEATURES_PATH = os.path.join(OUTPUT_DIR, "radiomics", "radiomics_features_all.xlsx")

# Radiomics cohort paths (set by user)
COHORT_A_BASE = os.path.join(DATA_DIR, "CohortA")
COHORT_B_BASE = os.path.join(DATA_DIR, "CohortB")
COHORT_A_MASK_NAME = "mask.nii.gz"
COHORT_B_MASK_NAME = "mask.nii.gz"
AP_NAME = "arterial-phase.nii.gz"
PV_NAME = "portal-venous_reg.nii.gz"
CLINICAL_COHORT_A = os.path.join(DATA_DIR, "clinical_cohort_a.xlsx")
CLINICAL_COHORT_B = os.path.join(DATA_DIR, "clinical_cohort_b.xlsx")
FEATURES_COHORT_A = os.path.join(OUTPUT_DIR, "radiomics", "features_cohort_a.csv")
FEATURES_COHORT_B = os.path.join(OUTPUT_DIR, "radiomics", "features_cohort_b.csv")

PYRADIOMICS_PARAMS = os.path.join(PROJECT_ROOT, "radiomics", "params.yaml")

# ── RFS ───────────────────────────────────────────────────────
RFS_MODEL_DIR = os.path.join(MODEL_DIR, "RFS_model")

# ── Evaluation ────────────────────────────────────────────────
CLINICAL_INFO_PATH = os.path.join(DATA_DIR, "clinical_info_all_filtered.xlsx")
PRED_MVI_GRADE_PATH = os.path.join(OUTPUT_DIR, "predictions", "all_models_predictions_MVI_grade.xlsx")
PRED_RFS_PATH = os.path.join(OUTPUT_DIR, "predictions", "all_model_rfs_risk_scores.xlsx")

# Feature table paths for survival modeling (set by user)
FEATURE_TABLES = {
    "LFM-Seq": "features_lfm_seq.xlsx",
    "LFM-Mix": "features_lfm_mix.xlsx",
    "LFM-DL": "features_lfm_dl.xlsx",
    "LFM-Base": "features_lfm_base.xlsx",
    "FMCIB": "features_fmcib.xlsx",
    "Radiomics": "features_radiomics.xlsx",
}

# ── Pretraining ───────────────────────────────────────────────
PRETRAIN_IMG_SIZE = 128
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_LR = 1e-3
PRETRAIN_EPOCHS = 100

# ── Finetuning ────────────────────────────────────────────────
FINETUNE_LR_RANGE = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 32


