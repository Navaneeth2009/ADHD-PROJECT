import sys
from pathlib import Path
import pandas as pd
import shutil

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATASET_DIR, METADATA_SOURCE, METADATA_DIR,
    AGE_MIN, AGE_MAX
)

def build_cohort_manifest():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy or verify metadata source in data/metadata
    local_meta_path = METADATA_DIR / "TDBRAIN_participants_V3.xlsx"
    if not local_meta_path.exists() and METADATA_SOURCE.exists():
        print(f"Copying metadata from {METADATA_SOURCE} to {local_meta_path}...")
        shutil.copy(METADATA_SOURCE, local_meta_path)
    
    meta_path = local_meta_path if local_meta_path.exists() else METADATA_SOURCE
    print(f"Loading metadata from {meta_path}...")
    df = pd.read_excel(meta_path)
    
    # Standardize column names
    df["indication"] = df["indication"].astype(str).str.strip()
    df["TDBRAIN_ID"] = df["TDBRAIN_ID"].astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
    
    # 2. Filter for Option A: ADHD vs HEALTHY, adults within age bounds
    cohort = df[
        (df["indication"].isin(["ADHD", "HEALTHY"])) &
        (df["age"] >= AGE_MIN) &
        (df["age"] <= AGE_MAX)
    ].copy()
    
    print(f"Filtered adult cohort (Age {AGE_MIN}-{AGE_MAX}):")
    print(cohort["indication"].value_counts())
    
    # 3. Locate available BDF files for each subject
    records = []
    for _, row in cohort.iterrows():
        sub_id = row["TDBRAIN_ID"]
        sub_dir = DATASET_DIR / sub_id
        if not sub_dir.exists():
            continue
        
        # Check sessions (primarily ses-1)
        for ses_dir in sub_dir.glob("ses-*"):
            ses_name = ses_dir.name
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue
            
            for task in ["restEC", "restEO"]:
                bdf_file = eeg_dir / f"{sub_id}_{ses_name}_task-{task}_eeg.bdf"
                if bdf_file.exists():
                    records.append({
                        "subject_id": sub_id,
                        "session": ses_name,
                        "task": task,
                        "indication": row["indication"],
                        "label": 1 if row["indication"] == "ADHD" else 0,
                        "age": row["age"],
                        "gender": row["gender"],
                        "bdf_path": str(bdf_file)
                    })
                    
    manifest_df = pd.DataFrame(records)
    out_csv = METADATA_DIR / "cohort_manifest.csv"
    manifest_df.to_csv(out_csv, index=False)
    print(f"\nManifest successfully created at {out_csv}!")
    print(f"Total entries: {len(manifest_df)}")
    print("\nCounts by task and indication:")
    print(manifest_df.groupby(["task", "indication"]).size())
    return manifest_df

if __name__ == "__main__":
    build_cohort_manifest()
