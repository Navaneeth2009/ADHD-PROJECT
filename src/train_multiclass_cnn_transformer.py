"""
Multi-Class Psychiatric EEGConvTransformer Training Pipeline
Module: src/train_multiclass_cnn_transformer.py

Trains and benchmarks the hybrid CNN + Transformer on multi-class psychiatric
cohorts (e.g., HEALTHY vs ADHD vs MDD vs OCD vs SMC).

Key Features:
- Dynamically discovers all available classes from data/processed/tensors_restEC/
- Subject-Stratified Train / Val / Test splitting (zero data leakage)
- Class-weighted Cross-Entropy Loss to handle sample size imbalance
- Dual epoch-level and patient-level aggregated evaluation
- Multi-class confusion matrix, macro F1, and per-class sensitivity/specificity
- Checkpointing best model weights to models/multiclass_cnn_transformer.pt
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from cnn_transformer import EEGConvTransformer

TENSORS_DIR = BASE_DIR / "data" / "processed" / "tensors_restEC"
MANIFEST_PATH = BASE_DIR / "data" / "metadata" / "cohort_multiclass_manifest.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiClassEEGDataset(Dataset):
    """
    In-memory dataset of (26, 1001) EEG epoch slices with class labels.
    """
    def __init__(self, epochs: np.ndarray, labels: np.ndarray, subject_ids: list[str]):
        self.epochs = torch.from_numpy(epochs).float()
        self.labels = torch.from_numpy(labels).long()
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.epochs[idx], self.labels[idx], self.subject_ids[idx]


def standardize_epoch(epoch: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently for numerical stability."""
    if np.max(np.abs(epoch)) < 1e-2:
        epoch = epoch * 1e6  # Volts to microvolts if needed
    mean = np.mean(epoch, axis=-1, keepdims=True)
    std = np.std(epoch, axis=-1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (epoch - mean) / std


def load_multiclass_data(max_epochs_per_sub: int = 40, target_classes: list[str] | None = None):
    """
    Scans available tensors and compiles subjects matching target_classes.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    tensor_files = list(TENSORS_DIR.glob("*_epochs.npy"))
    available_sids = {f.name.split("_")[0]: f for f in tensor_files}

    matched = []
    for _, row in manifest_df.iterrows():
        sid = str(row["subject_id"])
        ind = str(row["indication"]).strip().upper()
        if sid in available_sids:
            if target_classes is None or ind in target_classes:
                matched.append({
                    "subject_id": sid,
                    "indication": ind,
                    "tensor_path": available_sids[sid],
                    "age": row.get("age", 0),
                    "gender": row.get("gender", 0)
                })

    df = pd.DataFrame(matched)
    if len(df) == 0:
        raise ValueError("No tensor files match the requested cohort.")

    # Unique classes and mapping
    classes = sorted(df["indication"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    print(f">> Multi-Class Cohort: {len(df)} subjects across {len(classes)} classes:")
    for c in classes:
        cnt = (df["indication"] == c).sum()
        print(f"   [{class_to_idx[c]}] {c:<12}: {cnt} subjects")

    return df, class_to_idx, idx_to_class


def partition_subjects(df: pd.DataFrame, class_to_idx: dict[str, int], train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Stratified split at the subject level.
    """
    rng = np.random.RandomState(seed)
    train_sids, val_sids, test_sids = [], [], []

    for ind, group in df.groupby("indication"):
        sids = group["subject_id"].tolist()
        rng.shuffle(sids)
        n = len(sids)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n and n > 2:
            n_train = n - 2
            n_val = 1

        train_sids.extend(sids[:n_train])
        val_sids.extend(sids[n_train:n_train + n_val])
        test_sids.extend(sids[n_train + n_val:])

    train_df = df[df["subject_id"].isin(train_sids)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_sids)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_sids)].reset_index(drop=True)

    print(f"\n>> Partition Breakdown: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def extract_epochs_from_df(sub_df: pd.DataFrame, class_to_idx: dict[str, int], max_epochs: int = 40):
    all_epochs = []
    all_labels = []
    all_sub_ids = []

    for _, row in sub_df.iterrows():
        sid = row["subject_id"]
        label = class_to_idx[row["indication"]]
        tensor = np.load(row["tensor_path"])

        if len(tensor) > max_epochs:
            step = len(tensor) / max_epochs
            indices = [int(i * step) for i in range(max_epochs)]
            tensor = tensor[indices]

        for ep in tensor:
            all_epochs.append(standardize_epoch(ep))
            all_labels.append(label)
            all_sub_ids.append(sid)

    return np.array(all_epochs, dtype=np.float32), np.array(all_labels, dtype=np.int64), all_sub_ids


def evaluate_multiclass(model: nn.Module, loader: DataLoader, device: torch.device, idx_to_class: dict[int, str]):
    model.eval()
    num_classes = len(idx_to_class)

    epoch_preds, epoch_targets = [], []
    sub_probs = {}
    sub_targets = {}

    with torch.no_grad():
        for x, y, sids in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            epoch_preds.extend(preds)
            epoch_targets.extend(y.cpu().numpy())

            for i, sid in enumerate(sids):
                if sid not in sub_probs:
                    sub_probs[sid] = []
                    sub_targets[sid] = int(y[i].cpu().item())
                sub_probs[sid].append(probs[i])

    epoch_preds = np.array(epoch_preds)
    epoch_targets = np.array(epoch_targets)
    epoch_acc = float(np.mean(epoch_preds == epoch_targets))

    # Patient-level aggregation: average epoch probabilities
    sub_preds = []
    sub_true = []
    for sid, p_list in sub_probs.items():
        avg_prob = np.mean(p_list, axis=0)
        pred_c = int(np.argmax(avg_prob))
        sub_preds.append(pred_c)
        sub_true.append(sub_targets[sid])

    sub_preds = np.array(sub_preds)
    sub_true = np.array(sub_true)
    sub_acc = float(np.mean(sub_preds == sub_true))

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(sub_true, sub_preds):
        cm[t, p] += 1

    # Per-class sensitivity & specificity
    per_class = {}
    sensitivities = []
    for c_idx, c_name in idx_to_class.items():
        tp = cm[c_idx, c_idx]
        fn = np.sum(cm[c_idx, :]) - tp
        fp = np.sum(cm[:, c_idx]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        per_class[c_name] = {
            "sensitivity_recall": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "support": int(np.sum(cm[c_idx, :]))
        }
        sensitivities.append(sens)

    balanced_acc = float(np.mean(sensitivities))
    macro_f1 = float(np.mean([m["f1_score"] for m in per_class.values()]))

    return {
        "epoch_accuracy": round(epoch_acc, 4),
        "subject_accuracy": round(sub_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }


def train_multiclass_pipeline(
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 5e-4,
    max_epochs_per_sub: int = 40,
    target_classes: list[str] | None = None
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*65)
    print("STARTING MULTI-CLASS CNN + TRANSFORMER TRAINING PIPELINE")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print("="*65)

    df, class_to_idx, idx_to_class = load_multiclass_data(max_epochs_per_sub, target_classes)
    num_classes = len(class_to_idx)

    train_df, val_df, test_df = partition_subjects(df, class_to_idx, seed=42)

    X_train, y_train, sids_train = extract_epochs_from_df(train_df, class_to_idx, max_epochs_per_sub)
    X_val, y_val, sids_val = extract_epochs_from_df(val_df, class_to_idx, max_epochs_per_sub)
    X_test, y_test, sids_test = extract_epochs_from_df(test_df, class_to_idx, max_epochs_per_sub)

    train_loader = DataLoader(MultiClassEEGDataset(X_train, y_train, sids_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MultiClassEEGDataset(X_val, y_val, sids_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MultiClassEEGDataset(X_test, y_test, sids_test), batch_size=batch_size, shuffle=False)

    # Class-weighted loss
    class_counts = [int(np.sum(y_train == c)) for c in range(num_classes)]
    total_samples = len(y_train)
    class_weights = torch.tensor([total_samples / (num_classes * max(cnt, 1)) for cnt in class_counts], dtype=torch.float32).to(device)
    print(f">> Class Weights for Loss: {[round(float(w), 3) for w in class_weights]}")

    model = EEGConvTransformer(
        n_channels=26,
        n_samples=1001,
        F1=16,
        D=2,
        F2=32,
        kernel_length=64,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.25,
        num_classes=num_classes
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> Trainable Parameters: {params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_epoch = 0
    best_checkpoint = MODEL_DIR / "multiclass_cnn_transformer.pt"

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_b = 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = train_loss / max(n_b, 1)

        val_metrics = evaluate_multiclass(model, val_loader, device, idx_to_class)
        cur_f1 = val_metrics["macro_f1"]

        is_best = cur_f1 > best_val_f1
        if is_best:
            best_val_f1 = cur_f1
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "idx_to_class": idx_to_class,
                "class_to_idx": class_to_idx,
                "hyperparams": {"n_channels": 26, "n_samples": 1001, "num_classes": num_classes}
            }, best_checkpoint)

        flag = " [BEST]" if is_best else ""
        print(f"Epoch [{ep:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Val Acc: {val_metrics['subject_accuracy']*100:.1f}% | Bal Acc: {val_metrics['balanced_accuracy']*100:.1f}% | Macro F1: {cur_f1:.4f}{flag}")

    duration = time.time() - start_time
    print(f"\n>> Training Finished in {duration:.1f}s | Best Epoch: {best_epoch} (Val F1: {best_val_f1:.4f})")

    # Final evaluation on unseen test set
    if best_checkpoint.exists():
        ckpt = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_multiclass(model, test_loader, device, idx_to_class)
    print("\n" + "="*65)
    print("FINAL TEST SET RESULTS (Subject-Level Multi-Class Evaluation)")
    print("="*65)
    print(f"  Subject Accuracy:         {test_metrics['subject_accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy:        {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Macro-Averaged F1:        {test_metrics['macro_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    for c_name, m in test_metrics["per_class"].items():
        print(f"  - {c_name:<10}: Sens={m['sensitivity_recall']*100:5.1f}%, Spec={m['specificity']*100:5.1f}%, F1={m['f1_score']:.4f} (N={m['support']})")
    print(f"\nConfusion Matrix (Rows=True, Cols=Pred):\n{np.array(test_metrics['confusion_matrix'])}")
    print("="*65 + "\n")

    # Export metrics JSON
    metrics_file = MODEL_DIR / "multiclass_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "model": "MultiClass_EEGConvTransformer",
            "classes": idx_to_class,
            "best_epoch": best_epoch,
            "training_duration": round(duration, 2),
            "test_metrics": test_metrics
        }, f, indent=2)

    return test_metrics


if __name__ == "__main__":
    train_multiclass_pipeline(epochs=20, batch_size=32, lr=5e-4)
