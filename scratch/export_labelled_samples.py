import pandas as pd
import json
from pathlib import Path
from src.explain import ExplainabilityEngine

df = pd.read_csv("data/processed/features_restEC_optionA.csv")
adhd_df = df[df["label"] == 1]
healthy_df = df[df["label"] == 0]

engine = ExplainabilityEngine()

print("--- ADHD Patients ---")
for _, row in adhd_df.head(5).iterrows():
    sid = row["subject_id"]
    d = row.to_dict()
    res = engine.explain_prediction(d)
    out_path = Path(f"sample_patients/ADHD_{sid}.json")
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Saved {out_path.name}: Prob={res['prediction']['probability']}, Status={res['prediction']['status']}, TBR_Fz={res['key_biomarkers']['TBR_Fz']}")

print("\n--- Healthy Controls ---")
for _, row in healthy_df.head(3).iterrows():
    sid = row["subject_id"]
    d = row.to_dict()
    res = engine.explain_prediction(d)
    out_path = Path(f"sample_patients/HEALTHY_{sid}.json")
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Saved {out_path.name}: Prob={res['prediction']['probability']}, Status={res['prediction']['status']}, TBR_Fz={res['key_biomarkers']['TBR_Fz']}")
