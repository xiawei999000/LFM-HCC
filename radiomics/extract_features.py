"""
PyRadiomics feature extraction with wavelet transforms.
Extracts features from AP and PV phases, combines them per patient.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

from config import (
    OUTPUT_DIR, FEATURES_COHORT_A, FEATURES_COHORT_B,
    COHORT_A_BASE, COHORT_A_MASK_NAME,
    COHORT_B_BASE, COHORT_B_MASK_NAME,
    AP_NAME, PV_NAME,
    CLINICAL_COHORT_A, CLINICAL_COHORT_B, PYRADIOMICS_PARAMS,
)

PYRADIOMICS_YAML = """
imageType:
  Original: {}
  Wavelet: {}
featureClass:
  firstorder: []
  shape: []
  glcm: []
  glrlm: []
  glszm: []
  gldm: []
  ngtdm: []
setting:
  normalize: false
  force2D: false
  geometryTolerance: 1e-5
  resegmentRange: null
"""


def write_params():
    with open(PYRADIOMICS_PARAMS, "w") as f:
        f.write(PYRADIOMICS_YAML)


def load_mask_sitk(mask_path):
    """Load mask and binarize (non-zero → 1) to handle label=255 masks."""
    img = sitk.ReadImage(mask_path)
    arr = sitk.GetArrayFromImage(img)
    arr = (arr > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def get_patient_ids(base_dir):
    """Get sorted list of patient IDs from folder names."""
    ids = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            try:
                ids.append(int(name))
            except ValueError:
                continue
    return sorted(ids)


def extract_features_for_cohort(base_dir, mask_name, ids, cohort_name):
    extractor = featureextractor.RadiomicsFeatureExtractor(PYRADIOMICS_PARAMS)

    all_features = {}
    skipped = []

    for pid in tqdm(ids, desc=f"Extracting {cohort_name}"):
        patient_dir = os.path.join(base_dir, str(pid))
        ap_path = os.path.join(patient_dir, AP_NAME)
        pv_path = os.path.join(patient_dir, PV_NAME)
        mask_path = os.path.join(patient_dir, mask_name)

        if not all(os.path.exists(p) for p in [ap_path, pv_path, mask_path]):
            skipped.append(pid)
            continue

        mask_img = load_mask_sitk(mask_path)

        try:
            ap_feat = extractor.execute(ap_path, mask_img)
        except Exception as e:
            print(f"  AP error for {pid}: {e}")
            skipped.append(pid)
            continue

        try:
            pv_feat = extractor.execute(pv_path, mask_img)
        except Exception as e:
            print(f"  PV error for {pid}: {e}")
            skipped.append(pid)
            continue

        patient_features = {"ID": pid}

        for key, val in ap_feat.items():
            if key.startswith("diagnostics_") or key.startswith("general_"):
                continue
            patient_features[f"ap_{key}"] = val

        for key, val in pv_feat.items():
            if key.startswith("diagnostics_") or key.startswith("general_"):
                continue
            patient_features[f"pv_{key}"] = val

        all_features[pid] = patient_features

    df = pd.DataFrame.from_dict(all_features, orient="index")
    df = df.sort_index()
    print(f"  {cohort_name}: extracted {len(df)} patients, skipped {len(skipped)}")

    if skipped:
        skip_file = os.path.join(OUTPUT_DIR, f"skipped_{cohort_name}.txt")
        with open(skip_file, "w") as f:
            f.write("\n".join(map(str, skipped)))
        print(f"  Skipped IDs saved to {skip_file}")

    return df


def main():
    write_params()

    # --- Cohort A ---
    cohort_a_ids = get_patient_ids(COHORT_A_BASE)
    print(f"Cohort A: {len(cohort_a_ids)} patient folders found")

    df_cohort_a = extract_features_for_cohort(COHORT_A_BASE, COHORT_A_MASK_NAME, cohort_a_ids, "Cohort A")
    df_cohort_a.to_csv(FEATURES_COHORT_A, index=False)
    print(f"Saved: {FEATURES_COHORT_A} ({df_cohort_a.shape[0]} rows x {df_cohort_a.shape[1]} cols)")

    # --- Cohort B ---
    cohort_b_ids = get_patient_ids(COHORT_B_BASE)
    print(f"Cohort B: {len(cohort_b_ids)} patient folders found")

    df_cohort_b = extract_features_for_cohort(COHORT_B_BASE, COHORT_B_MASK_NAME, cohort_b_ids, "Cohort B")
    df_cohort_b.to_csv(FEATURES_COHORT_B, index=False)
    print(f"Saved: {FEATURES_COHORT_B} ({df_cohort_b.shape[0]} rows x {df_cohort_b.shape[1]} cols)")


if __name__ == "__main__":
    main()
