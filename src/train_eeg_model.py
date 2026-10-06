import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR

# Suppress convergence or runtime warnings for cleaner output
warnings.filterwarnings("ignore")

def load_pure_eeg_dataset(features_csv=None):
    """
    Loads extracted features and filters STRICTLY to pure EEG spectral features
    (ignoring demographic columns like age, gender, etc.) for cost-effective prototype evaluation.
    """
    if features_csv is None:
        features_csv = PROCESSED_DIR / "features_restEC_optionA.csv"
        
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} subjects from {features_csv}")
    print(f"Class distribution: ADHD={sum(df['label'] == 1)}, Healthy={sum(df['label'] == 0)}")
    
    # Metadata columns to exclude from feature set
    meta_cols = ["subject_id", "task", "label", "indication", "age", "gender", "n_epochs"]
    
    # Select strictly EEG spectral features (channel band powers, TBR, FAA)
    eeg_feature_cols = [col for col in df.columns if col not in meta_cols]
    print(f"Number of pure EEG spectral features: {len(eeg_feature_cols)}")
    
    X = df[eeg_feature_cols].values
    y = df["label"].values
    
    # Handle any potential NaNs with column median
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        
    return X, y, eeg_feature_cols, df

def evaluate_eeg_models(X, y, n_splits=5, random_state=42):
    """
    Evaluates standard, cost-effective prototype ML classifiers using 5-fold Stratified Cross-Validation.
    Models evaluated:
      1. Logistic Regression (L2 regularized)
      2. Support Vector Classifier (RBF kernel)
      3. Random Forest (100 estimators)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    models = {
        "Logistic Regression (Elastic/L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=random_state))
        ]),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state))
        ]),
        "Random Forest Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state))
        ])
    }
    
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall"
    }
    
    results = {}
    print("\n==================================================")
    print("EVALUATING PURE EEG-BASED CLASSIFIERS (5-Fold CV)")
    print("==================================================")
    
    for name, pipeline in models.items():
        cv_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        
        acc_mean = cv_res["test_accuracy"].mean()
        acc_std = cv_res["test_accuracy"].std()
        auc_mean = cv_res["test_roc_auc"].mean()
        auc_std = cv_res["test_roc_auc"].std()
        f1_mean = cv_res["test_f1"].mean()
        rec_mean = cv_res["test_recall"].mean()
        prec_mean = cv_res["test_precision"].mean()
        
        results[name] = {
            "Accuracy": f"{acc_mean:.3f} ± {acc_std:.3f}",
            "ROC-AUC": f"{auc_mean:.3f} ± {auc_std:.3f}",
            "F1-Score": f"{f1_mean:.3f}",
            "Sensitivity (Recall)": f"{rec_mean:.3f}",
            "Precision": f"{prec_mean:.3f}",
            "Raw_AUC": auc_mean
        }
        
        print(f"\nModel: {name}")
        print(f"  Accuracy:    {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"  ROC-AUC:     {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"  F1-Score:    {f1_mean:.3f}")
        print(f"  Sensitivity: {rec_mean:.3f}")
        print(f"  Precision:   {prec_mean:.3f}")
        
    results_df = pd.DataFrame(results).T.drop(columns=["Raw_AUC"])
    return results_df

if __name__ == "__main__":
    X, y, feature_names, df = load_pure_eeg_dataset()
    res_table = evaluate_eeg_models(X, y)
    print("\nFinal Model Comparison Table:")
    print(res_table)
