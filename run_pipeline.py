import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from src.config import PROCESSED_DIR, METADATA_DIR
from src.dataset_manifest import build_cohort_manifest
from src.preprocess import preprocess_eeg_subject
from src.features import compute_band_powers, export_epochs_tensor

def run_dry_run():
    """
    Runs a test on 1 ADHD and 1 Healthy subject to verify quality,
    output shapes, and feature integrity.
    """
    manifest_csv = METADATA_DIR / "cohort_manifest.csv"
    if not manifest_csv.exists():
        manifest_df = build_cohort_manifest()
    else:
        manifest_df = pd.read_csv(manifest_csv)
        
    ec_manifest = manifest_df[manifest_df["task"] == "restEC"]
    
    adhd_sample = ec_manifest[ec_manifest["label"] == 1].iloc[0]
    healthy_sample = ec_manifest[ec_manifest["label"] == 0].iloc[0]
    
    samples = [("ADHD", adhd_sample), ("HEALTHY", healthy_sample)]
    
    print("\n==================================================")
    print("STARTING PIPELINE DRY RUN (1 ADHD + 1 HEALTHY)")
    print("==================================================")
    
    for name, sample in samples:
        print(f"\nProcessing {name} sample: {sample['subject_id']} (Age: {sample['age']}, Sex: {sample['gender']})")
        print(f"File: {sample['bdf_path']}")
        
        epochs, stats = preprocess_eeg_subject(sample["bdf_path"], apply_ica=True)
        print(f"Retained Epochs: {stats['n_epochs_retained']}, ICA Bad Components Excluded: {stats['ica_bad_components']}")
        
        # Test feature extraction
        feats = compute_band_powers(epochs)
        print(f"Extracted {len(feats)} features. Key biomarkers:")
        print(f"  - Theta/Beta Ratio at Fz (TBR_Fz): {feats.get('TBR_Fz', 'N/A'):.3f}")
        print(f"  - Theta/Beta Ratio at Cz (TBR_Cz): {feats.get('TBR_Cz', 'N/A'):.3f}")
        print(f"  - Frontal Alpha Asymmetry (FAA):   {feats.get('FAA_F4_F3', 'N/A'):.3f}")
        
        # Test tensor export
        tensor = export_epochs_tensor(epochs)
        print(f"Epoch Tensor Shape: {tensor.shape} (Epochs, Channels, Timepoints)")
        
    print("\n==================================================")
    print("DRY RUN COMPLETED SUCCESSFULLY!")
    print("==================================================")

def run_batch_preprocessing(task="restEC", limit=None, export_tensors=False):
    """
    Processes the cohort for the chosen task and saves extracted features
    and optional deep learning epoch tensors.
    """
    manifest_csv = METADATA_DIR / "cohort_manifest.csv"
    if not manifest_csv.exists():
        manifest_df = build_cohort_manifest()
    else:
        manifest_df = pd.read_csv(manifest_csv)
        
    task_df = manifest_df[manifest_df["task"] == task].copy()
    if limit is not None:
        task_df = task_df.head(limit)
        
    print(f"\nProcessing {len(task_df)} subjects for task '{task}'...")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tensors_dir = PROCESSED_DIR / f"tensors_{task}"
    if export_tensors:
        tensors_dir.mkdir(parents=True, exist_ok=True)
        
    all_features = []
    out_features_csv = PROCESSED_DIR / f"features_{task}_optionA.csv"
    if out_features_csv.exists():
        existing_df = pd.read_csv(out_features_csv)
        all_features = existing_df.to_dict("records")
        processed_ids = set(existing_df["subject_id"].tolist())
    else:
        processed_ids = set()
    
    for idx, row in tqdm(task_df.iterrows(), total=len(task_df), desc=f"Preprocessing {task}"):
        sub_id = row["subject_id"]
        tensor_file = tensors_dir / f"{sub_id}_{task}_epochs.npy"
        
        # Check if already processed
        if sub_id in processed_ids and (not export_tensors or tensor_file.exists()):
            continue
            
        try:
            epochs, stats = preprocess_eeg_subject(row["bdf_path"], apply_ica=True)
            feats = compute_band_powers(epochs)
            
            # Append metadata
            feats["subject_id"] = sub_id
            feats["task"] = task
            feats["label"] = row["label"]
            feats["indication"] = row["indication"]
            feats["age"] = row["age"]
            feats["gender"] = row["gender"]
            feats["n_epochs"] = stats["n_epochs_retained"]
            all_features.append(feats)
            processed_ids.add(sub_id)
            
            if export_tensors:
                tensor = export_epochs_tensor(epochs)
                np.save(tensor_file, tensor)
                
            # Periodically save features CSV every 10 subjects
            if len(all_features) % 10 == 0:
                pd.DataFrame(all_features).to_csv(out_features_csv, index=False)
                
        except Exception as e:
            print(f"Error processing {sub_id}: {e}")
            
    features_df = pd.DataFrame(all_features)
    out_features_csv = PROCESSED_DIR / f"features_{task}_optionA.csv"
    features_df.to_csv(out_features_csv, index=False)
    print(f"\nFeatures saved to {out_features_csv}!")
    print(f"Shape: {features_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDBRAIN Preprocessing Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 ADHD and 1 Healthy sample test")
    parser.add_argument("--task", type=str, default="restEC", choices=["restEC", "restEO"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subjects for partial run")
    parser.add_argument("--export-tensors", action="store_true", help="Save 3D epoch numpy tensors for DL")
    args = parser.parse_args()
    
    if args.dry_run:
        run_dry_run()
    else:
        run_batch_preprocessing(task=args.task, limit=args.limit, export_tensors=args.export_tensors)
