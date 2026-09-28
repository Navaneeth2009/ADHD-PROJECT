import numpy as np
import pandas as pd
from pathlib import Path
import json

MODEL_DIR = Path("d:/ADHD_Project/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("d:/ADHD_Project/data/processed/features_restEC_optionA.csv")

class CostEffectiveADHDClassifier:
    """
    Stand-alone, sub-millisecond inference ADHD classifier built purely on
    standardized 26-channel EEG resting-state spectral features.
    No demographic data, cognitive tests, or GPU required.
    """
    def __init__(self, alpha=20.0):
        self.alpha = alpha
        self.weights = None
        self.mean = None
        self.scale = None
        self.feature_names = None

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        # Compute standardization parameters
        self.mean = np.nanmean(X, axis=0)
        self.scale = np.nanstd(X, axis=0)
        self.scale[self.scale == 0] = 1.0

        X_s = (X - self.mean) / self.scale

        # Solve regularized ridge system
        n_samples, n_features = X_s.shape
        X_bias = np.hstack([np.ones((n_samples, 1)), X_s])
        y_target = np.where(y == 1, 1.0, -1.0)
        I = np.eye(n_features + 1)
        I[0, 0] = 0.0

        self.weights = np.linalg.solve(X_bias.T @ X_bias + self.alpha * I, X_bias.T @ y_target)

    def predict_single(self, feature_dict):
        """
        Takes a raw dictionary of extracted EEG features, standardizes them,
        and outputs the diagnostic prediction and probability score.
        """
        vector = np.array([feature_dict.get(feat, 0.0) for feat in self.feature_names])
        vector_s = (vector - self.mean) / self.scale
        vector_bias = np.insert(vector_s, 0, 1.0)

        score = np.dot(vector_bias, self.weights)
        prob_adhd = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
        pred_label = "ADHD" if prob_adhd >= 0.5 else "HEALTHY"

        return {
            "prediction": pred_label,
            "adhd_probability": float(prob_adhd),
            "confidence": float(prob_adhd if prob_adhd >= 0.5 else (1.0 - prob_adhd))
        }

    def save_model(self, path=MODEL_DIR / "adhd_eeg_prototype_model.json"):
        model_payload = {
            "model_type": "PureEEG_RidgeClassifier",
            "alpha": self.alpha,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist()
        }
        with open(path, "w") as f:
            json.dump(model_payload, f, indent=2)
        print(f"Exported production model artifact to: {path}")

def train_and_export():
    df = pd.read_csv(DATA_PATH)
    meta_cols = ["subject_id", "task", "label", "indication", "age", "gender", "n_epochs"]
    eeg_cols = [c for c in df.columns if c not in meta_cols]

    X = df[eeg_cols].values
    y = df["label"].values.astype(int)

    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    clf = CostEffectiveADHDClassifier(alpha=20.0)
    clf.fit(X, y, eeg_cols)
    clf.save_model()

    # Test sample inference
    sample_feat = df.iloc[0].to_dict()
    res = clf.predict_single(sample_feat)
    print(f"\nTest Inference on {df.iloc[0]['subject_id']} (Actual: {df.iloc[0]['indication']}):")
    print(f"  Result: {res['prediction']} (Probability: {res['adhd_probability']:.3f}, Confidence: {res['confidence']:.3f})")

if __name__ == "__main__":
    train_and_export()
