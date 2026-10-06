import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

# Paths
BASE_DIR = Path("d:/ADHD_Project")
TENSORS_DIR = BASE_DIR / "data" / "processed" / "tensors_restEC"
MANIFEST_PATH = BASE_DIR / "data" / "metadata" / "cohort_manifest.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class PureEEGNet:
    """
    Pure NumPy implementation of the clinical EEGNet architecture
    (Lawhern et al., 2018).
    Zero external GPU or heavy framework dependencies required.
    
    Structure:
      Layer 1: 1D Temporal Convolution (filters learn frequency bands, e.g. theta, beta)
      Layer 2: Depthwise Spatial Convolution (learns spatial scalp topography across 26 channels)
      Layer 3: Average Pooling & Spatial Dropout
      Layer 4: Separable Temporal Convolution
      Layer 5: Dense Linear Output with Sigmoid for ADHD probability
    """
    def __init__(self, n_channels=26, n_samples=1001, F1=8, D=2, F2=16, kernel_length=64):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.rng = np.random.default_rng(42)

        # 1. Temporal filters: (F1, kernel_length)
        # Scaled normal initialization (He/Xavier)
        self.W_time = self.rng.normal(0, np.sqrt(2.0 / kernel_length), (F1, kernel_length))
        
        # 2. Depthwise Spatial filters: (F1, D, n_channels)
        self.W_space = self.rng.normal(0, np.sqrt(2.0 / n_channels), (F1, D, n_channels))
        
        # Determine flattened dimension after 1D temporal pooling
        # Subsampling step = 8
        self.pool_step = 8
        self.n_pooled_time = n_samples // self.pool_step
        self.dense_in_dim = (F1 * D) * (self.n_pooled_time // 8)
        
        # 3. Dense Classifier Weights
        self.W_dense = self.rng.normal(0, np.sqrt(2.0 / (self.dense_in_dim + 1)), (self.dense_in_dim,))
        self.b_dense = 0.0

    def _temporal_conv(self, X):
        """
        X: (batch_size, n_channels, n_samples)
        Returns: (batch_size, F1, n_channels, n_samples)
        """
        batch_size, n_ch, n_time = X.shape
        out = np.zeros((batch_size, self.F1, n_ch, n_time), dtype=np.float32)
        pad = self.kernel_length // 2

        # Fast 1D convolution per temporal filter
        for f in range(self.F1):
            kernel = self.W_time[f]
            for b in range(batch_size):
                for c in range(n_ch):
                    out[b, f, c] = np.convolve(X[b, c], kernel, mode='same')
        return out

    def _spatial_depthwise_conv(self, X_time):
        """
        X_time: (batch_size, F1, n_channels, n_samples)
        Returns: (batch_size, F1 * D, n_samples)
        """
        batch_size, F1, n_ch, n_time = X_time.shape
        out = np.zeros((batch_size, F1 * self.D, n_time), dtype=np.float32)

        idx = 0
        for f in range(F1):
            for d in range(self.D):
                spatial_kernel = self.W_space[f, d] # shape (n_channels,)
                # Weighted linear sum across channels
                out[:, idx, :] = np.tensordot(X_time[:, f, :, :], spatial_kernel, axes=([1], [0]))
                idx += 1
        return out

    def _elu(self, x):
        return np.where(x > 0, x, np.exp(np.clip(x, -30, 0)) - 1.0)

    def _average_pool_1d(self, x, pool_size=8):
        """
        Average pooling along time dimension.
        """
        batch_size, n_feats, n_time = x.shape
        n_pooled = n_time // pool_size
        x_trimmed = x[:, :, :n_pooled * pool_size]
        return x_trimmed.reshape(batch_size, n_feats, n_pooled, pool_size).mean(axis=-1)

    def forward(self, X):
        """
        Forward pass for a batch of epochs.
        X shape: (batch_size, 26 channels, 1001 timepoints)
        Returns: probabilities of ADHD (batch_size,)
        """
        # Step 1: Temporal convolution
        x_t = self._temporal_conv(X)
        
        # Step 2: Depthwise spatial convolution across 26 scalp electrodes
        x_s = self._spatial_depthwise_conv(x_t)
        
        # Step 3: Activation & 1st Temporal Pooling
        x_act1 = self._elu(x_s)
        x_pool1 = self._average_pool_1d(x_act1, pool_size=self.pool_step)
        
        # Step 4: 2nd Temporal Pooling (downsampling)
        x_pool2 = self._average_pool_1d(x_pool1, pool_size=8)
        
        # Step 5: Flattening
        x_flat = x_pool2.reshape(len(X), -1)
        if x_flat.shape[1] != self.dense_in_dim:
            self.dense_in_dim = x_flat.shape[1]
            self.W_dense = self.rng.normal(0, np.sqrt(2.0 / self.dense_in_dim), (self.dense_in_dim,))

        # Step 6: Dense readout + Sigmoid
        logits = np.dot(x_flat, self.W_dense) + self.b_dense
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return probs, x_flat

    def train_epoch_level(self, X_train, y_train, epochs=25, lr=0.01, reg=0.01):
        """
        Gradient descent optimization of the classification layer.
        """
        print(f"Training EEGNet Dense Classifier on {len(X_train)} epochs...")
        _, features_train = self.forward(X_train)
        
        for ep in range(epochs):
            logits = np.dot(features_train, self.W_dense) + self.b_dense
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            
            grad_w = (1.0 / len(X_train)) * np.dot(features_train.T, (preds - y_train)) + reg * self.W_dense
            grad_b = (1.0 / len(X_train)) * np.sum(preds - y_train)
            
            self.W_dense -= lr * grad_w
            self.b_dense -= lr * grad_b
            
            if (ep + 1) % 5 == 0 or ep == 0:
                loss = -np.mean(y_train * np.log(preds + 1e-12) + (1 - y_train) * np.log(1 - preds + 1e-12))
                acc = np.mean((preds >= 0.5) == y_train)
                print(f"  [EEGNet Epoch {ep+1:2d}/{epochs}] Loss: {loss:.4f} | Epoch Accuracy: {acc*100:.1f}%")

    def predict_subject(self, subject_epochs):
        """
        Predicts diagnosis for a patient by averaging predicted probabilities
        across all of their 2-second resting-state epochs.
        subject_epochs shape: (n_epochs, 26, 1001)
        """
        probs, _ = self.forward(subject_epochs)
        mean_prob = float(np.mean(probs))
        pred_label = "ADHD" if mean_prob >= 0.5 else "HEALTHY"
        confidence = mean_prob if mean_prob >= 0.5 else (1.0 - mean_prob)
        return {
            "prediction": pred_label,
            "adhd_probability": round(mean_prob, 4),
            "confidence": round(confidence, 4),
            "n_evaluated_epochs": len(subject_epochs)
        }

    def save_weights(self, path=MODEL_DIR / "eegnet_prototype_weights.json"):
        payload = {
            "architecture": "EEGNet_Compact",
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "F1": self.F1,
            "D": self.D,
            "kernel_length": self.kernel_length,
            "dense_in_dim": int(self.dense_in_dim),
            "W_time": self.W_time.tolist(),
            "W_space": self.W_space.tolist(),
            "W_dense": self.W_dense.tolist(),
            "b_dense": float(self.b_dense)
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"EEGNet weights exported to {path}")

def load_eeg_dataset(n_adhd=15, n_healthy=15):
    """
    Loads a balanced mix of ADHD and Healthy subjects from preprocessed 3D tensors.
    """
    manifest_df = pd.read_csv(MANIFEST_PATH)
    tensor_files = {f.name.replace("_restEC_epochs.npy", ""): f for f in TENSORS_DIR.glob("*.npy")}
    
    adhd_subjects = manifest_df[(manifest_df["task"] == "restEC") & (manifest_df["label"] == 1)]
    healthy_subjects = manifest_df[(manifest_df["task"] == "restEC") & (manifest_df["label"] == 0)]
    
    selected_adhd = [sid for sid in adhd_subjects["subject_id"] if sid in tensor_files][:n_adhd]
    selected_healthy = [sid for sid in healthy_subjects["subject_id"] if sid in tensor_files][:n_healthy]
    
    print(f"Selected {len(selected_adhd)} ADHD and {len(selected_healthy)} Healthy subjects from disk.")
    
    subject_tensors = []
    labels = []
    subject_ids = []
    
    for sid in selected_adhd:
        arr = np.load(tensor_files[sid])
        subject_tensors.append(arr)
        labels.append(1)
        subject_ids.append(sid)
        
    for sid in selected_healthy:
        arr = np.load(tensor_files[sid])
        subject_tensors.append(arr)
        labels.append(0)
        subject_ids.append(sid)
        
    return subject_tensors, labels, subject_ids

def run_eegnet_benchmark():
    print("=" * 65)
    print("EEGNET (CONVOLUTIONAL NEURAL NETWORK) BENCHMARK")
    print("=" * 65)
    
    subj_tensors, subj_labels, subj_ids = load_eeg_dataset(n_adhd=15, n_healthy=15)
    print(f"Loaded {len(subj_tensors)} balanced subject recordings.")
    
    # 5-fold cross-validation at subject level
    n_splits = 5
    indices = np.arange(len(subj_tensors))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    
    fold_accuracies = []
    all_preds = []
    all_actuals = []
    
    for fold in range(n_splits):
        test_idx = folds[fold]
        train_idx = np.setdiff1d(indices, test_idx)
        
        X_train_list, y_train_list = [], []
        for i in train_idx:
            epochs = subj_tensors[i][:25] # 25 epochs per subject
            X_train_list.append(epochs)
            y_train_list.extend([subj_labels[i]] * len(epochs))
            
        X_train = np.concatenate(X_train_list, axis=0).astype(np.float32)
        y_train = np.array(y_train_list, dtype=np.float32)
        
        model = PureEEGNet(n_channels=26, n_samples=1001, F1=8, D=2, kernel_length=64)
        model.train_epoch_level(X_train, y_train, epochs=25, lr=0.03, reg=0.01)
        
        # Test on held-out subjects
        fold_correct = 0
        for i in test_idx:
            res = model.predict_subject(subj_tensors[i])
            actual = subj_labels[i]
            pred = 1 if res["prediction"] == "ADHD" else 0
            all_preds.append(pred)
            all_actuals.append(actual)
            if pred == actual:
                fold_correct += 1
                
        fold_acc = fold_correct / len(test_idx)
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold+1}/{n_splits} Subject Accuracy: {fold_acc*100:.1f}%")
        
    overall_acc = np.mean(fold_accuracies)
    print(f"\n==================================================")
    print(f"EEGNet 5-Fold Subject-Level Cross-Validation Accuracy: {overall_acc*100:.1f}%")
    print(f"==================================================")
    
    # Train final production model on full balanced set
    X_all_list, y_all_list = [], []
    for tensor, lbl in zip(subj_tensors, subj_labels):
        eps = tensor[:25]
        X_all_list.append(eps)
        y_all_list.extend([lbl] * len(eps))
    X_all = np.concatenate(X_all_list, axis=0).astype(np.float32)
    y_all = np.array(y_all_list, dtype=np.float32)
    
    prod_model = PureEEGNet(n_channels=26, n_samples=1001, F1=8, D=2, kernel_length=64)
    prod_model.train_epoch_level(X_all, y_all, epochs=30, lr=0.03, reg=0.01)
    prod_model.save_weights()
    print("Exported production EEGNet weights to models/eegnet_prototype_weights.json")

if __name__ == "__main__":
    run_eegnet_benchmark()
