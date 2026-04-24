# train_brainbert.py
# Training script for BrainBERT model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import editdistance
import argparse

from model_brainbert import BrainBERT, BrainBERTLite
from preprocess_brainbert import BrainBERTPreprocessor
from ID2phoneme import id2phoneme

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# =========================================================
# Configuration
# =========================================================
# Input dimension: 128 or 256
# 128 = first 128 channels only
# 256 = all 256 channels (full)
INPUT_DIM = 128  # <-- Change this to 128 or 256

STACK_K = 5
MODEL_INPUT_DIM = INPUT_DIM * STACK_K  # After time stacking


# =========================================================
# Dataset
# =========================================================
class SpikeDataset(Dataset):
    def __init__(self, tfrecord_paths, max_len=500, input_dim=256):
        self.paths = tfrecord_paths
        self.max_len = max_len
        self.input_dim = input_dim  # How many channels to use

        self.raw = tf.data.TFRecordDataset(tfrecord_paths)

        self.feature_desc = {
            "inputFeatures": tf.io.FixedLenSequenceFeature([256], tf.float32, allow_missing=True),
            "seqClassIDs": tf.io.FixedLenFeature((max_len,), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
        }

        self.samples = list(self.raw)
        self._compute_stats()

    def _compute_stats(self):
        allX = []
        for ex in self.samples:
            e = tf.io.parse_single_example(ex, self.feature_desc)
            # Only use first input_dim channels
            allX.append(e["inputFeatures"].numpy()[:, :self.input_dim])

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std = Xcat.std(axis=0) + 1e-5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        # Only use first input_dim channels
        X = e["inputFeatures"].numpy()[:, :self.input_dim]
        y = e["seqClassIDs"].numpy()
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]
        return X.astype(np.float32), y, T, L, self.mean, self.std


# =========================================================
# Collate
# =========================================================
def make_collate_fn(input_dim):
    def collate_fn(batch):
        Xs, ys, Ts, Ls, means, stds = zip(*batch)
        B = len(batch)
        T_max = max(Ts)

        X_pad = np.zeros((B, T_max, input_dim), dtype=np.float32)
        for i in range(B):
            X_pad[i, :Ts[i]] = Xs[i]

        # Concat labels for CTC
        ycat = []
        for y in ys:
            ycat.extend(y.tolist())
        ycat = torch.tensor(ycat, dtype=torch.long)

        return (
            torch.tensor(X_pad, dtype=torch.float32),
            ycat,
            torch.tensor(Ls, dtype=torch.long),
            torch.tensor(Ts, dtype=torch.long),
            np.stack(means),
            np.stack(stds),
        )
    return collate_fn


# =========================================================
# CTC decode
# =========================================================
def greedy_decode(logits):
    """
    Greedy CTC decoding.
    logits: (T, C) - log probabilities for one sample
    """
    pred = logits.argmax(dim=-1).cpu().tolist()

    seq = []
    prev = -1
    for p in pred:
        if p != 0 and p != prev:  # 0 is CTC blank
            seq.append(p)
        prev = p
    return seq


# =========================================================
# Evaluate
# =========================================================
def evaluate(model, loader, preproc, max_samples=50):
    model.eval()

    total_pref = 0
    total_edit = 0
    total_cnt = 0

    for (X, ycat, Ls, Ts, means, stds) in loader:
        if total_cnt >= max_samples:
            break

        X = X.numpy()
        B = X.shape[0]

        # Preprocess
        X_new, T_new = [], []
        for i in range(B):
            Xi = X[i, :Ts[i]]
            Xi, Tnew = preproc(Xi, means[i], stds[i])
            X_new.append(Xi)
            T_new.append(Tnew)

        Tmax = max(T_new)
        feat_dim = X_new[0].shape[1]
        X_pad = np.zeros((B, Tmax, feat_dim), dtype=np.float32)
        for i in range(B):
            X_pad[i, :T_new[i]] = X_new[i]

        X_pad = torch.tensor(X_pad, dtype=torch.float32).to(DEVICE)
        T_new_tensor = torch.tensor(T_new, dtype=torch.long).to(DEVICE)

        # Forward
        with torch.no_grad():
            logits = model(X_pad, T_new_tensor)
            log0 = logits[:, 0, :]  # First sample in batch

        pred = greedy_decode(log0)
        true = ycat[:Ls[0]].tolist()

        # Prefix accuracy
        Lmin = min(len(pred), len(true))
        if Lmin > 0:
            pref = sum(pred[i] == true[i] for i in range(Lmin)) / Lmin
        else:
            pref = 0

        # Edit distance (PER)
        edit = editdistance.eval(pred, true) / max(len(true), 1)

        total_pref += pref
        total_edit += edit
        total_cnt += 1

    return total_pref / max(total_cnt, 1), total_edit / max(total_cnt, 1)


# =========================================================
# Show one sample
# =========================================================
def show_one(model, loader, preproc):
    model.eval()
    
    X, ycat, Ls, Ts, means, stds = next(iter(loader))
    X = X.numpy()

    Xi = X[0, :Ts[0]]
    Xi, Tnew = preproc(Xi, means[0], stds[0])

    Xi = torch.tensor(Xi[None, :, :], dtype=torch.float32).to(DEVICE)
    Tnew = torch.tensor([Tnew], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(Xi, Tnew)
        log0 = logits[:, 0, :]

    pred = greedy_decode(log0)
    true = ycat[:Ls[0]].tolist()

    print(f"\n===== BrainBERT Sample Decode (INPUT_DIM={INPUT_DIM}) =====")
    print(f"Pred IDs: {pred[:30]}...")
    print(f"True IDs: {true[:30]}...")
    print(f"Pred PH: {[id2phoneme.get(p, '?') for p in pred[:20]]}...")
    print(f"True PH: {[id2phoneme.get(p, '?') for p in true[:20]]}...")
    print(f"Edit distance: {editdistance.eval(pred, true)}")
    print("=" * 50 + "\n")


# =========================================================
# Train Loop
# =========================================================
def train():
    BASE = r"D:\DeepLearning\BCI\Dataset\derived\tfRecords"
    # All available dates (24 sessions)
    DATES = [
        # April
        "t12.2022.04.28",
        # May
        "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
        "t12.2022.05.24", "t12.2022.05.26",
        # June
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
        "t12.2022.06.28",
        # July
        "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
        "t12.2022.07.27", "t12.2022.07.29",
        # August
        "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
        "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
    ]

    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
    test_paths = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in DATES]

    print(f"\n{'='*50}")
    print(f"Training BrainBERT with INPUT_DIM = {INPUT_DIM}")
    print(f"Model input dim after stacking = {MODEL_INPUT_DIM}")
    print(f"{'='*50}\n")

    print("Loading datasets...")
    train_ds = SpikeDataset(train_paths, input_dim=INPUT_DIM)
    test_ds = SpikeDataset(test_paths, input_dim=INPUT_DIM)

    collate_fn = make_collate_fn(INPUT_DIM)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    preproc = BrainBERTPreprocessor(
        smooth_sigma=1.5,
        stack_k=STACK_K,
        stack_stride=2,
        subsample_factor=3,
    )

    # =======================
    # Model Configuration
    # =======================
    USE_LITE = True  # Use lightweight version for faster training
    
    if USE_LITE:
        print("Using BrainBERTLite model...")
        model = BrainBERTLite(
            input_dim=MODEL_INPUT_DIM,  # INPUT_DIM * STACK_K
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            num_classes=41
        ).to(DEVICE)
    else:
        print("Using full BrainBERT model...")
        model = BrainBERT(
            input_dim=MODEL_INPUT_DIM,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            num_classes=41
        ).to(DEVICE)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_edit = 999
    NUM_EPOCHS = 50

    # Model save path based on input dim
    save_path = f"best_brainbert_{INPUT_DIM}dim.pt"

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Model will be saved to: {save_path}\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for X, ycat, Ls, Ts, means, stds in pbar:
            X = X.numpy()
            B = X.shape[0]

            # Preprocess
            X_new, T_new = [], []
            for i in range(B):
                Xi = X[i, :Ts[i]]
                Xi, Tt = preproc(Xi, means[i], stds[i])
                X_new.append(Xi)
                T_new.append(Tt)

            Tmax = max(T_new)
            feat_dim = X_new[0].shape[1]

            X_pad = np.zeros((B, Tmax, feat_dim), dtype=np.float32)
            for i in range(B):
                X_pad[i, :T_new[i]] = X_new[i]

            X_pad = torch.tensor(X_pad, dtype=torch.float32).to(DEVICE)
            T_new_tensor = torch.tensor(T_new, dtype=torch.long).to(DEVICE)
            ycat = ycat.to(DEVICE)
            Ls = Ls.to(DEVICE)

            # Forward
            logits = model(X_pad, T_new_tensor)
            log_probs = logits.log_softmax(dim=-1)

            # CTC Loss
            loss = criterion(log_probs, ycat, T_new_tensor.cpu(), Ls.cpu())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        # Evaluate
        train_pref, train_edit = evaluate(model, train_loader, preproc, max_samples=30)
        test_pref, test_edit = evaluate(model, test_loader, preproc, max_samples=50)

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n[Epoch {epoch}] Loss={avg_loss:.3f}")
        print(f"  Train: Prefix={train_pref:.3f} | PER={train_edit:.3f}")
        print(f"  Test:  Prefix={test_pref:.3f} | PER={test_edit:.3f}")

        # Show sample
        show_one(model, test_loader, preproc)

        # Save best
        if test_edit < best_edit:
            best_edit = test_edit
            # Save model with config
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': INPUT_DIM,
                'model_input_dim': MODEL_INPUT_DIM,
                'best_per': best_edit,
            }, save_path)
            print(f">>> Saved BEST model to {save_path} (PER={test_edit:.3f})\n")

    print(f"\nTraining complete! Best PER: {best_edit:.3f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train()
