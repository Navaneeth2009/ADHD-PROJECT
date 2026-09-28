import numpy as np
import pandas as pd
from pathlib import Path

# Paths
DATA_PATH = Path("d:/ADHD_Project/data/processed/features_restEC_optionA.csv")
REPORTS_DIR = Path("d:/ADHD_Project/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class PureEEGStandardScaler:
    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PureEEGLogisticRegression:
    """
    Cost-effective L2-regularized Logistic Regression classifier
    built on vectorized NumPy.
    """
    def __init__(self, lr=0.08, reg=0.02, max_iter=500):
        self.lr = lr
        self.reg = reg
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y)) + self.reg * self.weights
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class PureEEGKNearestNeighbors:
    """
    K-Nearest Neighbors Classifier on Euclidean distance in standardized spectral space.
    """
    def __init__(self, n_neighbors=7):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):
        probs = []
        for x in X:
            dists = np.sum((self.X_train - x) ** 2, axis=1)
            nn_idx = np.argsort(dists)[:self.k]
            probs.append(np.mean(self.y_train[nn_idx]))
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

class PureEEGRidgeClassifier:
    """
    Regularized Ridge Regression / Linear Least Squares Classifier.
    Closed-form solution via matrix inversion: fast and optimal for high dimensions.
    """
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        # Centered targets: -1 for class 0, +1 for class 1
        y_target = np.where(y == 1, 1.0, -1.0)
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Do not penalize bias
        self.weights = np.linalg.solve(X_bias.T @ X_bias + self.alpha * I, X_bias.T @ y_target)

    def predict_proba(self, X):
        X_bias = np.hstack([np.ones((len(X), 1)), X])
        scores = X_bias @ self.weights
        # Map scores to [0, 1] probability via sigmoid
        return 1.0 / (1.0 + np.exp(-scores))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

def compute_metrics(y_true, y_pred, y_prob):
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        auc = np.mean(pos_probs[:, None] > neg_probs[None, :]) + 0.5 * np.mean(pos_probs[:, None] == neg_probs[None, :])
    else:
        auc = 0.5

    return {
        "Accuracy": acc,
        "Sensitivity (Recall)": recall,
        "Specificity": specificity,
        "Precision": precision,
        "F1-Score": f1,
        "ROC-AUC": auc
    }

def run_cross_validation(X, y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, n_splits)
    neg_folds = np.array_split(neg_idx, n_splits)

    models = {
        "Logistic Regression (L2 / Gradient Descent)": lambda: PureEEGLogisticRegression(lr=0.08, reg=0.02, max_iter=400),
        "Ridge Linear Classifier (Closed-Form L2)": lambda: PureEEGRidgeClassifier(alpha=20.0),
        "K-Nearest Neighbors (k=7 Spectral)": lambda: PureEEGKNearestNeighbors(n_neighbors=7)
    }

    results = {m: [] for m in models}

    for fold in range(n_splits):
        val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        scaler = PureEEGStandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        for name, model_fn in models.items():
            model = model_fn()
            model.fit(X_train_s, y_train)
            prob = model.predict_proba(X_val_s)
            pred = (prob >= 0.5).astype(int)
            metrics = compute_metrics(y_val, pred, prob)
            results[name].append(metrics)

    summary = {}
    for name, fold_list in results.items():
        summary[name] = {}
        for k in fold_list[0].keys():
            vals = [m[k] for m in fold_list]
            summary[name][k] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    return pd.DataFrame(summary).T

def main():
    print("=" * 65)
    print("PURE EEG-BASED ADHD CLASSIFICATION PROTOTYPE BENCHMARK")
    print("=" * 65)
    print(f"Loading preprocessed dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    meta_cols = ["subject_id", "task", "label", "indication", "age", "gender", "n_epochs"]
    eeg_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Total Subjects Evaluated: {len(df)} (ADHD: {sum(df['label']==1)}, Healthy: {sum(df['label']==0)})")
    print(f"Pure Electrophysiological Biomarkers: {len(eeg_cols)}")

    X = df[eeg_cols].values
    y = df["label"].values.astype(int)

    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # 5-Fold Stratified Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation across pure EEG models...")
    eval_df = run_cross_validation(X, y, n_splits=5)
    print("\n--- Model Performance Comparison ---")
    print(eval_df.to_string())

    # Top feature importances
    corrs = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    top_indices = np.argsort(corrs)[::-1][:15]
    print("\nTop 15 Most Discriminative Pure EEG Biomarkers:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {eeg_cols[idx]:20s} (Pearson |r| = {corrs[idx]:.3f})")

    # Fit and save final production model parameters
    scaler = PureEEGStandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = PureEEGRidgeClassifier(alpha=20.0)
    best_model.fit(X_scaled, y)
    probs = best_model.predict_proba(X_scaled)
    preds = (probs >= 0.5).astype(int)
    final_metrics = compute_metrics(y, preds, probs)

    print("\nFinal Production Model Performance:")
    for k, v in final_metrics.items():
        print(f"  {k:22s}: {v:.3f}")

    # Export report
    summary_path = REPORTS_DIR / "eeg_model_evaluation_report.csv"
    eval_df.to_csv(summary_path)
    print(f"\nSaved cross-validation evaluation report to: {summary_path}")

if __name__ == "__main__":
    main()
