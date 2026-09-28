import pandas as pd
import json
from pathlib import Path

df = pd.read_csv("data/processed/features_restEC_optionA.csv")
out_dir = Path("sample_patients")
out_dir.mkdir(exist_ok=True)

for _, row in df.head(10).iterrows():
    sid = str(row["subject_id"])
    d = row.to_dict()
    with open(out_dir / f"{sid}.json", "w") as f:
        json.dump(d, f, indent=2)

print(f"Generated 10 ready-to-upload patient files in {out_dir.resolve()}")
