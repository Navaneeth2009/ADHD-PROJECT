"""
Multi-Class Batch EEG Preprocessor
Module: src/preprocess_multiclass.py

Extracts continuous .bdf resting-state EEG files for MDD, OCD, SMC (and remaining ADHD/Healthy)
and saves:
1. 3D Epoch Tensors (N_epochs, 26, 1001) in data/processed/tensors_restEC/
2. Feature vectors in data/processed/features_restEC_multiclass.csv
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
# Remove tqdm dependency
# from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import PROCESSED_DIR, METADATA_DIR
from src.preprocess import preprocess_eeg_subject
from src.features import compute_band_powers, export_epochs_tensor

MANIFEST_CSV = METADATA_DIR / "cohort_multiclass_manifest.csv"
TENSORS_DIR = PROCESSED_DIR / "tensors_restEC"
OUT_FEATURES_CSV = PROCESSED_DIR / "features_restEC_multiclass.csv"


def run_multiclass_preprocessing(indication: str | None = None, limit: int | None = None, skip_existing: bool = True):
    if not MANIFEST_CSV.exists():
        from src.dataset_multiclass_manifest import build_multiclass_manifest
        manifest_df = build_multiclass_manifest()
    else:
        manifest_df = pd.read_csv(MANIFEST_CSV)

    TENSORS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_subset = manifest_df.copy()
    if indication:
        df_subset = df_subset[df_subset["indication"] == indication.upper()]

    # Filter out already preprocessed if skip_existing
    if skip_existing:
        to_process = []
        for _, row in df_subset.iterrows():
            sub_id = row["subject_id"]
            tensor_file = TENSORS_DIR / f"{sub_id}_restEC_epochs.npy"
            if not tensor_file.exists():
                to_process.append(row)
        df_subset = pd.DataFrame(to_process)

    if limit is not None and len(df_subset) > limit:
        df_subset = df_subset.head(limit)

    total_count = len(df_subset)
    print(f"\n>> Ready to preprocess {total_count} subjects...")
    if indication:
        print(f"   Target indication: {indication.upper()}")
    if total_count == 0:
        print("   All target subjects already have preprocessed tensors!")
        return

    # Load existing multiclass features if exists
    if OUT_FEATURES_CSV.exists():
        features_df_existing = pd.read_csv(OUT_FEATURES_CSV)
        all_features = features_df_existing.to_dict("records")
        processed_ids = set(features_df_existing["subject_id"].tolist())
    else:
        all_features = []
        processed_ids = set()

    success_count = 0
    fail_count = 0

    for idx, (_, row) in enumerate(df_subset.iterrows(), 1):
        sub_id = row["subject_id"]
        bdf_path = row["bdf_path_restEC"]
        tensor_file = TENSORS_DIR / f"{sub_id}_restEC_epochs.npy"

        print(f"[{idx}/{total_count}] Processing {sub_id} ({row['indication']})...", flush=True)

        try:
            epochs, stats = preprocess_eeg_subject(bdf_path, apply_ica=True)
            tensor = export_epochs_tensor(epochs)

            # Save numpy tensor
            np.save(tensor_file, tensor)

            # Compute tabular features (band powers, TBR, FAA)
            feats = compute_band_powers(epochs)
            feats["subject_id"] = sub_id
            feats["indication"] = row["indication"]
            feats["age"] = row["age"]
            feats["gender"] = row["gender"]
            feats["n_epochs"] = stats["n_epochs_retained"]

            all_features.append(feats)
            processed_ids.add(sub_id)
            success_count += 1

            # Periodically write features CSV every 5 subjects
            if success_count % 5 == 0:
                pd.DataFrame(all_features).to_csv(OUT_FEATURES_CSV, index=False)

        except Exception as e:
            print(f"\n[Warning] Failed to preprocess {sub_id}: {e}")
            fail_count += 1

    # Final save
    if all_features:
        pd.DataFrame(all_features).to_csv(OUT_FEATURES_CSV, index=False)

    print(f"\n=======================================================")
    print(f">> Batch Complete: {success_count} succeeded, {fail_count} failed.")
    print(f">> Tensors stored at: {TENSORS_DIR}")
    print(f">> Features stored at: {OUT_FEATURES_CSV}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-class EEG Preprocessing")
    parser.add_argument("--indication", type=str, default=None, help="Filter for specific indication (MDD, OCD, SMC, ADHD, HEALTHY)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of subjects to process in this run")
    parser.add_argument("--force", action="store_true", help="Overwrite existing tensors")
    args = parser.parse_args()

    run_multiclass_preprocessing(
        indication=args.indication,
        limit=args.limit,
        skip_existing=not args.force
    )
