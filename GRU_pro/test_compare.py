# test_compare.py
# Compare GRU models with different input dimensions (128 vs 256)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import editdistance
import os

from preprocess_gru import GRUPreprocessor
from ID2phoneme import id2phoneme

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models", "gru")


# =========================================================
# Dataset
# =========================================================
class SpikeDataset(Dataset):
    def __init__(self, tfrecord_paths, max_len=500, input_dim=256):
        self.paths = tfrecord_paths
        self.max_len = max_len
        self.input_dim = input_dim

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
            allX.append(e["inputFeatures"].numpy()[:, :self.input_dim])

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std = Xcat.std(axis=0) + 1e-5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        X = e["inputFeatures"].numpy()[:, :self.input_dim]
        y = e["seqClassIDs"].numpy()
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]
        return X.astype(np.float32), y, T, L, self.mean, self.std


def make_collate_fn(input_dim):
    def collate_fn(batch):
        Xs, ys, Ts, Ls, means, stds = zip(*batch)
        B = len(batch)
        T_max = max(Ts)

        X_pad = np.zeros((B, T_max, input_dim), dtype=np.float32)
        for i in range(B):
            X_pad[i, :Ts[i]] = Xs[i]

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
# GRU Model
# =========================================================
class GRU6(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)
        return logits.permute(1, 0, 2)


def greedy_decode(logits):
    pred = logits.argmax(dim=-1).cpu().tolist()
    seq, prev = [], -1
    for p in pred:
        if p != 0 and p != prev:
            seq.append(p)
        prev = p
    return seq


def evaluate_model(model, loader, preproc, num_samples=200):
    """Full evaluation on test set."""
    model.eval()

    total_pref = 0
    total_edit = 0
    total_cnt = 0
    all_pred_lens = []
    all_true_lens = []

    with torch.no_grad():
        for (X, ycat, Ls, Ts, means, stds) in tqdm(loader, desc="Evaluating"):
            if total_cnt >= num_samples:
                break

            X = X.numpy()
            B = X.shape[0]

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

            logits = model(X_pad, T_new_tensor)

            offset = 0
            for i in range(B):
                if total_cnt >= num_samples:
                    break

                log_i = logits[:T_new[i], i, :]
                pred = greedy_decode(log_i)
                true = ycat[offset:offset + Ls[i]].tolist()
                offset += Ls[i]

                Lmin = min(len(pred), len(true))
                if Lmin > 0:
                    pref = sum(pred[j] == true[j] for j in range(Lmin)) / Lmin
                else:
                    pref = 0

                edit = editdistance.eval(pred, true) / max(len(true), 1)

                total_pref += pref
                total_edit += edit
                total_cnt += 1
                all_pred_lens.append(len(pred))
                all_true_lens.append(len(true))

    avg_pref = total_pref / max(total_cnt, 1)
    avg_edit = total_edit / max(total_cnt, 1)
    avg_pred_len = np.mean(all_pred_lens)
    avg_true_len = np.mean(all_true_lens)

    return {
        'prefix_acc': avg_pref,
        'per': avg_edit,
        'avg_pred_len': avg_pred_len,
        'avg_true_len': avg_true_len,
        'num_samples': total_cnt,
    }


def load_model(checkpoint_path, input_dim, stack_k=5):
    """Load a trained model from checkpoint."""
    model_input_dim = input_dim * stack_k

    model = GRU6(
        input_dim=model_input_dim,
        hidden=256,
        num_layers=5,
        num_classes=41,
        dropout=0.1
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        print(f"  - Saved input_dim: {checkpoint.get('input_dim', 'unknown')}")
        print(f"  - Saved best_per: {checkpoint.get('best_per', 'unknown'):.3f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path} (old format)")

    return model


def compare_models():
    """Compare 128-dim and 256-dim models."""

    BASE = r"D:\DeepLearning\BCI\Dataset\derived\tfRecords"
    DATES = [
        "t12.2022.04.28",
        "t12.2022.05.05", "t12.2022.05.17", "t12.2022.05.19",
        "t12.2022.05.24", "t12.2022.05.26",
        "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
        "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
        "t12.2022.06.28",
        "t12.2022.07.05", "t12.2022.07.14", "t12.2022.07.21",
        "t12.2022.07.27", "t12.2022.07.29",
        "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
        "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25",
    ]

    test_paths = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in DATES]

    model_128_path = os.path.join(MODEL_DIR, "best_gru_128dim.pt")
    model_256_path = os.path.join(MODEL_DIR, "best_gru_256dim.pt")

    results = {}

    print("\n" + "=" * 60)
    print("GRU Model Comparison: 128-dim vs 256-dim")
    print("=" * 60)

    # Test 128-dim model
    if os.path.exists(model_128_path):
        print(f"\n>>> Loading 128-dim model from {model_128_path}")

        test_ds_128 = SpikeDataset(test_paths, input_dim=128)
        collate_fn_128 = make_collate_fn(128)
        test_loader_128 = DataLoader(test_ds_128, batch_size=8, shuffle=False, collate_fn=collate_fn_128)

        preproc_128 = GRUPreprocessor(
            smooth_sigma=1.5,
            stack_k=5,
            stack_stride=2,
            subsample_factor=3,
        )

        model_128 = load_model(model_128_path, input_dim=128)
        results['128-dim'] = evaluate_model(model_128, test_loader_128, preproc_128, num_samples=200)
    else:
        print(f"\n[SKIP] 128-dim model not found at {model_128_path}")
        print("       Run: python train_gru_pro.py with INPUT_DIM=128")

    # Test 256-dim model
    if os.path.exists(model_256_path):
        print(f"\n>>> Loading 256-dim model from {model_256_path}")

        test_ds_256 = SpikeDataset(test_paths, input_dim=256)
        collate_fn_256 = make_collate_fn(256)
        test_loader_256 = DataLoader(test_ds_256, batch_size=8, shuffle=False, collate_fn=collate_fn_256)

        preproc_256 = GRUPreprocessor(
            smooth_sigma=1.5,
            stack_k=5,
            stack_stride=2,
            subsample_factor=3,
        )

        model_256 = load_model(model_256_path, input_dim=256)
        results['256-dim'] = evaluate_model(model_256, test_loader_256, preproc_256, num_samples=200)
    else:
        print(f"\n[SKIP] 256-dim model not found at {model_256_path}")
        print("       Run: python train_gru_pro.py with INPUT_DIM=256")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if len(results) == 0:
        print("\nNo models found! Please train models first:")
        print("  1. Set INPUT_DIM=128 in train_gru_pro.py, then run it")
        print("  2. Set INPUT_DIM=256 in train_gru_pro.py, then run it")
        return

    print(f"\n{'Model':<15} {'PER':>10} {'Prefix Acc':>12} {'Pred Len':>10} {'True Len':>10}")
    print("-" * 60)

    for name, res in results.items():
        print(f"{name:<15} {res['per']*100:>9.2f}% {res['prefix_acc']*100:>11.2f}% {res['avg_pred_len']:>10.1f} {res['avg_true_len']:>10.1f}")

    print("-" * 60)

    if '128-dim' in results and '256-dim' in results:
        per_diff = results['128-dim']['per'] - results['256-dim']['per']
        print(f"\n256-dim vs 128-dim PER difference: {per_diff*100:+.2f}%")
        if per_diff > 0:
            print(f"  → 256-dim is BETTER by {per_diff*100:.2f}% absolute PER")
        else:
            print(f"  → 128-dim is BETTER by {-per_diff*100:.2f}% absolute PER")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    compare_models()

