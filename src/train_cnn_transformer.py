"""
Training and Evaluation Pipeline for Hybrid CNN + Transformer (EEG-Conformer)
=============================================================================

Key Highlights:
  - Patient-Stratified Train / Val / Test Partitioning: Prevents cross-epoch data leakage.
  - Per-Epoch Microvolt Scaling and Z-Score Standardisation: Robust against channel baseline shift.
  - Class-Weighted BCE Loss: Accounts for Healthy (174) vs ADHD (93) cohort distribution.
  - Dual Evaluation:
      1. Epoch-level diagnostic prediction
      2. Patient-level aggregated diagnosis across 2-second resting-state windows.
  - Model Checkpointing: Saves best weights (.pt) and full metrics dictionary (.json).
"""

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

# Workspace paths
BASE_DIR = Path("d:/ADHD_Project")
TENSORS_DIR = BASE_DIR / "data" / "processed" / "tensors_restEC"
MANIFEST_PATH = BASE_DIR / "data" / "metadata" / "cohort_manifest.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Add src to sys.path
sys.path.insert(0, str(BASE_DIR / "src"))
from cnn_transformer import EEGConvTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGEpochDataset(Dataset):
    """
    PyTorch Dataset storing pre-processed EEG epochs and subject IDs.
    """
    def __init__(self, epochs: np.ndarray, labels: np.ndarray, subject_indices: list, augment: bool = False):
        """
        epochs: (N, 26, 1001) float32
        labels: (N,) float32 (0.0 or 1.0)
        subject_indices: list of subject identifier strings or ints corresponding to each epoch
        """
        self.epochs = torch.from_numpy(epochs).float()
        self.labels = torch.from_numpy(labels).float()
        self.subject_indices = subject_indices
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.epochs[idx]  # (26, 1001)
        y = self.labels[idx]

        if self.augment:
            # Subtle data augmentation for EEG:
            # 1. Mild Gaussian noise (simulates sensor micro-fluctuations)
            if random.random() < 0.5:
                noise = torch.randn_like(x) * 0.03
                x = x + noise
            # 2. Random temporal circular roll (simulates arbitrary window onset)
            if random.random() < 0.5:
                shift = random.randint(-50, 50)
                x = torch.roll(x, shifts=shift, dims=-1)

        return x, y, self.subject_indices[idx]


def standardize_epoch(epoch: np.ndarray) -> np.ndarray:
    """
    Standardize an epoch (26, 1001) per channel via z-score.
    Converts raw EEG voltages into normalized units.
    """
    # epoch is shape (26, 1001)
    # Scale from Volts to microvolts if max < 1e-2
    if np.max(np.abs(epoch)) < 1e-2:
        epoch = epoch * 1e6

    mean = np.mean(epoch, axis=-1, keepdims=True)
    std = np.std(epoch, axis=-1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (epoch - mean) / std


def load_dataset_by_subjects(val_split: float = 0.15, test_split: float = 0.20, seed: int = 42, max_epochs_per_sub: int = 60):
    """
    Loads EEG epochs grouped strictly by subject to prevent data leakage.
    Stratified split ensures balanced ADHD / Healthy representation across splits.
    """
    manifest_df = pd.read_csv(MANIFEST_PATH)
    restec_manifest = manifest_df[manifest_df["task"] == "restEC"].copy()
    
    # Map available files on disk
    tensor_files = {f.name.replace("_restEC_epochs.npy", ""): f for f in TENSORS_DIR.glob("*.npy")}
    
    available_subjects = []
    for _, row in restec_manifest.iterrows():
        sid = str(row["subject_id"])
        if sid in tensor_files:
            available_subjects.append({
                "subject_id": sid,
                "label": int(row["label"]),
                "indication": str(row["indication"]),
                "file_path": tensor_files[sid]
            })

    subjects_df = pd.DataFrame(available_subjects).drop_duplicates(subset=["subject_id"])
    print(f"[Dataset] Discovered {len(subjects_df)} valid subjects with restEC epoch tensors.")
    print(f"          ADHD subjects: {sum(subjects_df['label'] == 1)} | Healthy subjects: {sum(subjects_df['label'] == 0)}")

    # Stratified split by subject
    adhd_subs = subjects_df[subjects_df["label"] == 1].sample(frac=1.0, random_state=seed).to_dict("records")
    healthy_subs = subjects_df[subjects_df["label"] == 0].sample(frac=1.0, random_state=seed).to_dict("records")

    def split_list(lst, test_ratio, val_ratio):
        n = len(lst)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        test_part = lst[:n_test]
        val_part = lst[n_test:n_test + n_val]
        train_part = lst[n_test + n_val:]
        return train_part, val_part, test_part

    train_adhd, val_adhd, test_adhd = split_list(adhd_subs, test_split, val_split)
    train_healthy, val_healthy, test_healthy = split_list(healthy_subs, test_split, val_split)

    train_subjects = train_adhd + train_healthy
    val_subjects = val_adhd + val_healthy
    test_subjects = test_adhd + test_healthy

    print(f"[Split] Train Subjects: {len(train_subjects)} (ADHD: {len(train_adhd)}, Healthy: {len(train_healthy)})")
    print(f"[Split] Val Subjects:   {len(val_subjects)} (ADHD: {len(val_adhd)}, Healthy: {len(val_healthy)})")
    print(f"[Split] Test Subjects:  {len(test_subjects)} (ADHD: {len(test_adhd)}, Healthy: {len(test_healthy)})")

    def load_split_epochs(subs_list, is_train=False):
        all_epochs = []
        all_labels = []
        all_sids = []

        for sub in subs_list:
            raw_tensor = np.load(sub["file_path"]) # (n_epochs, 26, 1001)
            # Limit epochs per subject to maintain balance and fast convergence
            n_avail = len(raw_tensor)
            n_take = min(n_avail, max_epochs_per_sub)
            
            selected = raw_tensor[:n_take]
            for ep in selected:
                norm_ep = standardize_epoch(ep)
                all_epochs.append(norm_ep)
                all_labels.append(sub["label"])
                all_sids.append(sub["subject_id"])

        all_epochs = np.array(all_epochs, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.float32)
        return all_epochs, all_labels, all_sids

    print("[Loading] Processing and standardising epoch tensors...")
    X_train, y_train, sids_train = load_split_epochs(train_subjects, is_train=True)
    X_val, y_val, sids_val = load_split_epochs(val_subjects, is_train=False)
    X_test, y_test, sids_test = load_split_epochs(test_subjects, is_train=False)

    print(f"[Loaded] Train Epochs: {len(X_train)} | Val Epochs: {len(X_val)} | Test Epochs: {len(X_test)}")
    return (X_train, y_train, sids_train, train_subjects), (X_val, y_val, sids_val, val_subjects), (X_test, y_test, sids_test, test_subjects)


def compute_binary_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5):
    """
    Computes diagnostic classification performance metrics.
    """
    y_true = np.array(y_true).astype(int)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / max(1, (tp + fn))  # Recall / Sensitivity
    spec = tn / max(1, (tn + fp))  # Specificity
    bal_acc = (sens + spec) / 2.0
    prec = tp / max(1, (tp + fp))
    f1 = 2 * prec * sens / max(1e-6, (prec + sens))

    # ROC AUC calculation (Wilcoxon-Mann-Whitney rank sum)
    pos_probs = y_probs[y_true == 1]
    neg_probs = y_probs[y_true == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        ranks = []
        for p in pos_probs:
            ranks.append(np.sum(p > neg_probs) + 0.5 * np.sum(p == neg_probs))
        auc = float(np.sum(ranks) / (len(pos_probs) * len(neg_probs)))
    else:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluates both epoch-level and subject-level aggregated performance.
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_sids = []

    with torch.no_grad():
        for x, y, sids in dataloader:
            x = x.to(device)
            logits, _ = model(x)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(y.numpy().tolist())
            all_sids.extend(sids)

    # 1. Epoch-level metrics
    epoch_metrics = compute_binary_metrics(all_targets, all_probs)

    # 2. Patient-level metrics (averaging probabilities across all epochs of a patient)
    subject_dict = {}
    for sid, prob, target in zip(all_sids, all_probs, all_targets):
        if sid not in subject_dict:
            subject_dict[sid] = {"probs": [], "target": target}
        subject_dict[sid]["probs"].append(prob)

    sub_targets = []
    sub_mean_probs = []
    for sid, data in subject_dict.items():
        sub_targets.append(data["target"])
        sub_mean_probs.append(float(np.mean(data["probs"])))

    subject_metrics = compute_binary_metrics(sub_targets, sub_mean_probs)

    return epoch_metrics, subject_metrics, subject_dict


def train_cnn_transformer(
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    max_epochs_per_sub: int = 50
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Starting Training: CNN + Transformer for EEG ADHD Detection")
    print(f"Device: {device} | PyTorch version: {torch.__version__}")
    print(f"=======================================================\n")

    # Load splits
    (X_train, y_train, sids_train, train_subs), \
    (X_val, y_val, sids_val, val_subs), \
    (X_test, y_test, sids_test, test_subs) = load_dataset_by_subjects(max_epochs_per_sub=max_epochs_per_sub)

    # Compute class weight for loss function
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    pos_weight_val = n_neg / max(1, n_pos)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[Training Setup] Class weight (pos_weight for ADHD): {pos_weight_val:.2f}")

    # Datasets and Loaders
    train_dataset = EEGEpochDataset(X_train, y_train, sids_train, augment=True)
    val_dataset = EEGEpochDataset(X_val, y_val, sids_val, augment=False)
    test_dataset = EEGEpochDataset(X_test, y_test, sids_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Model
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
        dropout=0.25
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Architecture] Trainable Parameters: {total_params:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop tracking
    best_val_auc = 0.0
    best_epoch = 0
    history = []
    best_model_path = MODEL_DIR / "cnn_transformer_adhd.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_batches)

        # Evaluate on validation set
        val_ep_metrics, val_sub_metrics, _ = evaluate_model(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_epoch_acc": round(val_ep_metrics["accuracy"] * 100, 2),
            "val_epoch_auc": round(val_ep_metrics["roc_auc"], 4),
            "val_subject_acc": round(val_sub_metrics["accuracy"] * 100, 2),
            "val_subject_bal_acc": round(val_sub_metrics["balanced_accuracy"] * 100, 2),
            "val_subject_auc": round(val_sub_metrics["roc_auc"], 4),
            "val_subject_sens": round(val_sub_metrics["sensitivity"] * 100, 2),
            "val_subject_spec": round(val_sub_metrics["specificity"] * 100, 2),
            "lr": round(optimizer.param_groups[0]["lr"], 6)
        })

        # Track best subject AUC
        cur_val_metric = val_sub_metrics["roc_auc"]
        is_best = cur_val_metric > best_val_auc
        if is_best:
            best_val_auc = cur_val_metric
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_sub_metrics": val_sub_metrics,
                "architecture": "EEGConvTransformer",
                "hyperparams": {
                    "n_channels": 26,
                    "n_samples": 1001,
                    "F1": 16,
                    "D": 2,
                    "F2": 32,
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2
                }
            }, best_model_path)

        flag = " * [BEST]" if is_best else ""
        print(f"Epoch [{epoch:2d}/{epochs}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Epoch Acc: {val_ep_metrics['accuracy']*100:4.1f}% | "
              f"Val Subj Acc: {val_sub_metrics['accuracy']*100:4.1f}% (Sens: {val_sub_metrics['sensitivity']*100:.1f}%, Spec: {val_sub_metrics['specificity']*100:.1f}%) | "
              f"Val Subj AUC: {val_sub_metrics['roc_auc']:.4f}{flag}")

    train_duration = time.time() - start_time
    print(f"\n[Training Complete] Elapsed time: {train_duration:.1f}s")
    print(f"[Best Checkpoint] Epoch {best_epoch} with Val Subject AUC = {best_val_auc:.4f}")

    # Load best checkpoint for final independent test set evaluation
    print("\n-------------------------------------------------------")
    print("Evaluating Best Checkpoint on Unseen Test Cohort...")
    print("-------------------------------------------------------")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_ep_metrics, test_sub_metrics, test_sub_details = evaluate_model(model, test_loader, device)

    print("\n========== FINAL TEST SET RESULTS (Subject-Level) ==========")
    print(f"  Subject Accuracy:         {test_sub_metrics['accuracy']*100:.2f}%")
    print(f"  Subject Balanced Accuracy: {test_sub_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Subject Sensitivity (ADHD): {test_sub_metrics['sensitivity']*100:.2f}%")
    print(f"  Subject Specificity (HC):  {test_sub_metrics['specificity']*100:.2f}%")
    print(f"  Subject Precision:         {test_sub_metrics['precision']*100:.2f}%")
    print(f"  Subject F1-Score:          {test_sub_metrics['f1_score']:.4f}")
    print(f"  Subject ROC-AUC:           {test_sub_metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:          TP={test_sub_metrics['tp']}, TN={test_sub_metrics['tn']}, FP={test_sub_metrics['fp']}, FN={test_sub_metrics['fn']}")
    print("============================================================\n")

    # Export metrics JSON
    summary_report = {
        "model_name": "CNN_Transformer_EEG_ADHD",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_duration_seconds": round(train_duration, 2),
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "parameters_count": total_params,
        "test_subject_metrics": test_sub_metrics,
        "test_epoch_metrics": test_ep_metrics,
        "validation_history": history,
        "test_patient_predictions": {
            sid: {
                "ground_truth": int(data["target"]),
                "diagnosis": "ADHD" if data["target"] == 1 else "HEALTHY",
                "predicted_prob": round(float(np.mean(data["probs"])), 4),
                "predicted_class": "ADHD" if np.mean(data["probs"]) >= 0.5 else "HEALTHY",
                "n_epochs": len(data["probs"])
            }
            for sid, data in test_sub_details.items()
        }
    }

    metrics_out = MODEL_DIR / "cnn_transformer_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Artifact] Detailed test metrics saved to: {metrics_out}")

    return summary_report


if __name__ == "__main__":
    train_cnn_transformer(epochs=20, batch_size=32, lr=5e-4, max_epochs_per_sub=40)
