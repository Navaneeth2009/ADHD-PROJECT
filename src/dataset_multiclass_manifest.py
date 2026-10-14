"""
Build Multi-Class Psychiatric Cohort Manifest
Module: src/dataset_multiclass_manifest.py

Indexes all available adult recordings (ages 18-65) across 5 clinical diagnostic groups:
- HEALTHY (Neurotypical Controls)
- ADHD (Attention Deficit Hyperactivity Disorder)
- MDD (Major Depressive Disorder)
- OCD (Obsessive Compulsive Disorder)
- SMC (Subjective Memory Complaints / Pre-dementia)

Outputs: data/metadata/cohort_multiclass_manifest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import DATASET_DIR, METADATA_DIR, AGE_MIN, AGE_MAX

TARGET_DIAGNOSES = ["HEALTHY", "ADHD", "MDD", "OCD", "SMC"]
TENSORS_DIR = BASE_DIR / "data" / "processed" / "tensors_restEC"

def build_multiclass_manifest():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = METADATA_DIR / "TDBRAIN_participants_V3.xlsx"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

    print(f">> Loading participant metadata from {meta_path}...")
    df = pd.read_excel(meta_path)

    # Standardize columns
    df["indication"] = df["indication"].astype(str).str.strip()
    df["TDBRAIN_ID"] = df["TDBRAIN_ID"].astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")  # 0=Female, 1=Male or vice versa in TDBRAIN

    # Filter for targeted diagnoses and adult age range
    cohort = df[
        (df["indication"].isin(TARGET_DIAGNOSES)) &
        (df["age"] >= AGE_MIN) &
        (df["age"] <= AGE_MAX)
    ].copy()

    print(f"\n>> Filtered Adult Cohort (Ages {AGE_MIN} - {AGE_MAX}):")
    print(cohort["indication"].value_counts())

    # Locate BDF recordings and check if tensor already preprocessed
    records = []
    for _, row in cohort.iterrows():
        sub_id = row["TDBRAIN_ID"]
        sub_dir = DATASET_DIR / sub_id
        if not sub_dir.exists():
            continue

        bdf_ec = sub_dir / "ses-1" / "eeg" / f"{sub_id}_ses-1_task-restEC_eeg.bdf"
        bdf_eo = sub_dir / "ses-1" / "eeg" / f"{sub_id}_ses-1_task-restEO_eeg.bdf"
        tensor_file = TENSORS_DIR / f"{sub_id}_restEC_epochs.npy"

        has_ec = bdf_ec.exists()
        has_eo = bdf_eo.exists()
        has_tensor = tensor_file.exists()

        if has_ec:
            records.append({
                "subject_id": sub_id,
                "indication": row["indication"],
                "age": row["age"],
                "gender": row["gender"],
                "has_restEC": has_ec,
                "has_restEO": has_eo,
                "has_preprocessed_tensor": has_tensor,
                "bdf_path_restEC": str(bdf_ec),
                "tensor_path_restEC": str(tensor_file) if has_tensor else ""
            })

    manifest_df = pd.DataFrame(records)
    out_csv = METADATA_DIR / "cohort_multiclass_manifest.csv"
    manifest_df.to_csv(out_csv, index=False)
    print(f"\n>> Saved multiclass cohort manifest ({len(manifest_df)} subjects) to:")
    print(f"   {out_csv}")

    # Summary table
    print("\n" + "="*70)
    print("PSYCHIATRIC MULTI-CLASS COHORT SUMMARY:")
    print("="*70)
    summary = manifest_df.groupby("indication").agg(
        Total_Subjects=("subject_id", "count"),
        Preprocessed_Tensors=("has_preprocessed_tensor", "sum"),
        Needing_Preprocessing=("has_preprocessed_tensor", lambda x: (~x).sum()),
        Mean_Age=("age", "mean"),
        Min_Age=("age", "min"),
        Max_Age=("age", "max"),
    ).reset_index()
    print(summary.to_string(index=False))
    print("="*70)

    return manifest_df

if __name__ == "__main__":
    build_multiclass_manifest()
