# ===========================================================
# train_KF.py
# Kalman Filter + Logistic Regression baseline (one-shot)
# ===========================================================

import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import sys

from preprocess_kf import KalmanPreprocessor
from ID2phoneme import id2phoneme
import editdistance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log(msg):
    """Print with flush to ensure output is visible."""
    print(msg, flush=True)

# ===========================================================
# Configuration
# ===========================================================
INPUT_DIM = 256  # <-- Change this to 128 or 256


# ===========================================================
#  Dataset
# ===========================================================
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
            # Only use first input_dim dimensions
            allX.append(e["inputFeatures"].numpy()[:, :self.input_dim])

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std = Xcat.std(axis=0) + 1e-5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        # Only use first input_dim dimensions
        X = e["inputFeatures"].numpy()[:, :self.input_dim].astype(np.float32)
        y = e["seqClassIDs"].numpy().reshape(-1).astype(int)
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]
        return X, y, T, L, self.mean, self.std


# ===========================================================
# 1. Fit LDS (A, C, Q, R) - One-shot estimation
# ===========================================================
from sklearn.decomposition import TruncatedSVD

def fit_lds(Xs, latent_dim=40, max_samples=None):
    """Fit LDS parameters using PCA + least squares (one-shot)."""
    log(f"fit_lds called with {len(Xs)} sequences, latent_dim={latent_dim}")
    
    # Limit samples to avoid memory issues (None = use all)
    if max_samples is not None and len(Xs) > max_samples:
        log(f"Subsampling to {max_samples} sequences...")
        indices = np.random.choice(len(Xs), max_samples, replace=False)
        Xs_sub = [Xs[i] for i in indices]
    else:
        log("Using ALL sequences for LDS fitting...")
        Xs_sub = Xs
    
    log("Concatenating data...")
    allX = np.concatenate(Xs_sub, axis=0)
    log(f"Total frames for PCA: {len(allX)}, shape: {allX.shape}")

    log("Running TruncatedSVD for initialization...")
    allX_centered = allX - allX.mean(0, keepdims=True)
    
    # Use TruncatedSVD (much faster than full SVD)
    svd = TruncatedSVD(n_components=latent_dim, random_state=42)
    svd.fit(allX_centered)
    C = svd.components_.T      # (INPUT_DIM, K)
    log(f"C shape: {C.shape}")
    pinvC = np.linalg.pinv(C)  # (K, INPUT_DIM)
    log(f"pinvC shape: {pinvC.shape}")

    # Project ALL sequences to latent (not just subsampled)
    log("Projecting all sequences to latent space...")
    X_latents = [X @ pinvC.T for X in Xs]
    log(f"Projected {len(X_latents)} sequences")

    log("Estimating A...")
    K = latent_dim
    A_num = np.zeros((K, K))
    A_den = np.zeros((K, K))

    for Z in X_latents:
        if len(Z) < 2:
            continue
        Zt = Z[:-1]
        Zt1 = Z[1:]
        A_num += Zt1.T @ Zt
        A_den += Zt.T @ Zt

    A = A_num @ np.linalg.pinv(A_den)
    log(f"A shape: {A.shape}")

    log("Estimating Q...")
    Qs = []
    for Z in X_latents:
        if len(Z) < 2:
            continue
        err = Z[1:] - Z[:-1] @ A.T
        Qs.append(err.T @ err / len(err))
    Q = np.mean(Qs, axis=0)
    log(f"Q shape: {Q.shape}")

    log("Estimating R...")
    Rs = []
    for X, Z in zip(Xs, X_latents):
        err = X - Z @ C.T
        Rs.append(err.T @ err / len(err))
    R = np.mean(Rs, axis=0)
    log(f"R shape: {R.shape}")

    log("LDS fitting complete!")
    return A, C, Q, R, X_latents


# ===========================================================
# 2. Kalman Filter forward (standard version with diagonal R optimization)
# ===========================================================
def kalman_filter(Y, A, C, Q, R):
    """
    Standard Kalman Filter with diagonal R optimization.
    
    State space model:
        x_t = A @ x_{t-1} + w_t,  w_t ~ N(0, Q)  (state transition)
        y_t = C @ x_t + v_t,      v_t ~ N(0, R)  (observation)
    
    Args:
        Y: (T, D) observations
        A: (K, K) state transition matrix
        C: (D, K) observation matrix
        Q: (K, K) process noise covariance
        R: (D, D) observation noise covariance (use diagonal for speed)
    
    Returns:
        Z: (T, K) filtered latent states
    """
    T, D = Y.shape
    K = A.shape[0]
    
    # Use diagonal of R for faster computation
    R_diag = np.diag(R) + 1e-6  # (D,)
    
    # Initialize
    x = np.zeros(K)
    P = np.eye(K) * 0.1
    
    out = np.zeros((T, K))
    
    for t in range(T):
        # === Predict ===
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        
        # === Update ===
        y = Y[t]
        
        # With diagonal R, S = C @ P_pred @ C.T + R is still dense
        # But we can use Woodbury identity for faster inversion
        # Or compute K_gain directly: K = P_pred @ C.T @ inv(S)
        
        # Simplified: use diagonal approximation of S
        # S_diag ≈ diag(C @ P_pred @ C.T) + R_diag
        CP = C @ P_pred  # (D, K)
        S_diag = np.sum(CP * C, axis=1) + R_diag  # (D,)
        
        # K_gain ≈ P_pred @ C.T @ diag(1/S_diag)
        K_gain = P_pred @ C.T / S_diag  # (K, D)
        
        # Innovation
        innovation = y - C @ x_pred  # (D,)
        
        # State update
        x = x_pred + K_gain @ innovation
        
        # Covariance update (simplified)
        P = P_pred - K_gain @ C @ P_pred
        
        out[t] = x
    
    return out


# ===========================================================
# 3. Create frame-level labels (uniform alignment)
# ===========================================================
def create_frame_labels(T_frames, phoneme_seq):
    """Uniformly align phoneme sequence to frame sequence."""
    L = len(phoneme_seq)
    if L == 0:
        return np.zeros(T_frames, dtype=int)
    
    frame_labels = np.zeros(T_frames, dtype=int)
    frames_per_phoneme = T_frames / L
    
    for i in range(T_frames):
        phoneme_idx = min(int(i / frames_per_phoneme), L - 1)
        frame_labels[i] = phoneme_seq[phoneme_idx]
    
    return frame_labels


# ===========================================================
# 4. Train classifier (Logistic Regression)
# ===========================================================
from sklearn.linear_model import LogisticRegression

def train_classifier(latent_list, label_list, max_frames=None):
    """Train classifier with proper frame-level alignment."""
    log("Preparing data for classifier training...")
    X_all = []
    y_all = []

    for Z, y in zip(latent_list, label_list):
        T_frames = len(Z)
        y = np.asarray(y).reshape(-1).astype(int)
        frame_labels = create_frame_labels(T_frames, y)
        X_all.append(Z)
        y_all.append(frame_labels)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    log(f"Total frames: {len(X_all)}")
    
    # Subsample if too many frames (None = use all)
    if max_frames is not None and len(X_all) > max_frames:
        log(f"Subsampling to {max_frames} frames for classifier...")
        indices = np.random.choice(len(X_all), max_frames, replace=False)
        X_all = X_all[indices]
        y_all = y_all[indices]
    else:
        log("Using ALL frames for classifier training...")
    
    log(f"Training classifier on {len(X_all)} frames...")
    log(f"Label distribution (first 10): {np.bincount(y_all)[:10]}...")
    
    clf = LogisticRegression(max_iter=300, class_weight='balanced', n_jobs=-1, verbose=1)
    clf.fit(X_all, y_all)
    log("Classifier training complete!")
    return clf


# ===========================================================
# 5. Decode sequence with smoothing
# ===========================================================
from scipy.stats import mode

def smooth_predictions(pred, window_size=5):
    """Apply majority voting smoothing."""
    T = len(pred)
    smoothed = np.zeros(T, dtype=int)
    half_w = window_size // 2
    
    for i in range(T):
        start = max(0, i - half_w)
        end = min(T, i + half_w + 1)
        window = pred[start:end]
        smoothed[i] = mode(window, keepdims=False)[0]
    
    return smoothed


def remove_short_segments(pred, min_len=3):
    """Remove segments shorter than min_len frames."""
    result = pred.copy()
    T = len(pred)
    
    i = 0
    while i < T:
        j = i
        while j < T and pred[j] == pred[i]:
            j += 1
        seg_len = j - i
        
        if seg_len < min_len:
            if i > 0:
                result[i:j] = result[i-1]
            elif j < T:
                result[i:j] = pred[j]
        
        i = j
    
    return result


def decode_sequence(Y, A, C, Q, R, clf, smooth=True):
    """Decode with optional smoothing."""
    Z = kalman_filter(Y, A, C, Q, R)
    pred = clf.predict(Z)
    
    if smooth:
        pred = smooth_predictions(pred, window_size=7)
        pred = remove_short_segments(pred, min_len=5)
    
    return pred


def collapse_repeats(seq):
    """Collapse consecutive repeated phonemes."""
    if len(seq) == 0:
        return []
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != result[-1]:
            result.append(seq[i])
    return result


# ===========================================================
# 6. Evaluation
# ===========================================================
def evaluate_kf(test_loader, preproc, A, C, Q, R, clf, num_samples=20):
    total_pref_frame = 0
    total_edit_seq = 0
    count = 0

    log(f"Evaluating {num_samples} samples...")

    for i, (X, y, T, L, mean, std) in enumerate(test_loader):
        if count >= num_samples:
            break

        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)
        L_val = int(L.numpy())
        yi = yi[:L_val]

        Xi, T_proc = preproc(Xi, mean.numpy(), std.numpy())
        pred_frames = decode_sequence(Xi, A, C, Q, R, clf)
        
        yi_frames = create_frame_labels(len(pred_frames), yi)
        pref_frame = np.mean(pred_frames == yi_frames)
        
        pred_seq = collapse_repeats(pred_frames.tolist())
        edit_seq = editdistance.eval(pred_seq, yi.tolist()) / max(len(yi), 1)

        total_pref_frame += pref_frame
        total_edit_seq += edit_seq
        count += 1

    return total_pref_frame / count, total_edit_seq / count


def show_one(test_loader, preproc, A, C, Q, R, clf):
    X, y, T, L, mean, std = next(iter(test_loader))

    Xi = X[0, :T].numpy()
    yi = y.numpy().reshape(-1).astype(int)
    L_val = int(L.numpy())
    yi = yi[:L_val]

    Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
    pred_frames = decode_sequence(Xi, A, C, Q, R, clf)
    pred_seq = collapse_repeats(pred_frames.tolist())

    log("\n===== KF Sample Decode =====")
    log(f"Frame predictions (first 50): {pred_frames[:50].tolist()}")
    log(f"Pred Seq IDs ({len(pred_seq)}): {pred_seq[:30]}...")
    log(f"True Seq IDs ({len(yi)}): {yi.tolist()[:30]}...")
    log(f"Pred PH: {[id2phoneme.get(p, '?') for p in pred_seq[:20]]}...")
    log(f"True PH: {[id2phoneme.get(p, '?') for p in yi[:20]]}...")
    log(f"Edit distance: {editdistance.eval(pred_seq, yi.tolist())}")
    log("================================\n")


# ===========================================================
# 7. Model save/load
# ===========================================================
def save_model(A, C, Q, R, clf, filepath):
    """Save KF model parameters and classifier."""
    model_data = {
        'A': A,
        'C': C,
        'Q': Q,
        'R': R,
        'clf': clf,
        'input_dim': INPUT_DIM
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    log(f"Model saved to {filepath}")


def load_model(filepath):
    """Load KF model parameters and classifier."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['A'], model_data['C'], model_data['Q'], model_data['R'], model_data['clf']


# ===========================================================
# 8. Main training entry
# ===========================================================
def train_kf():
    try:
        BASE = r"D:\DeepLearning\BCI\Dataset\derived\tfRecords"
        
        # Complete dataset dates
        DATES = [
            "t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17",
            "t12.2022.05.19", "t12.2022.05.24", "t12.2022.05.26",
            "t12.2022.06.02", "t12.2022.06.07", "t12.2022.06.14",
            "t12.2022.06.16", "t12.2022.06.21", "t12.2022.06.23",
            "t12.2022.06.28", "t12.2022.07.05", "t12.2022.07.14",
            "t12.2022.07.21", "t12.2022.07.27", "t12.2022.07.29",
            "t12.2022.08.02", "t12.2022.08.11", "t12.2022.08.13",
            "t12.2022.08.18", "t12.2022.08.23", "t12.2022.08.25"
        ]

        log(f"\n{'='*50}")
        log(f"Training KF with INPUT_DIM = {INPUT_DIM}")
        log(f"{'='*50}\n")

        train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
        test_paths = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in DATES]

        log("Loading datasets...")
        train_ds = SpikeDataset(train_paths, input_dim=INPUT_DIM)
        test_ds = SpikeDataset(test_paths, input_dim=INPUT_DIM)
        log(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        preproc = KalmanPreprocessor(sigma=1.5)

        # =======================
        # Load + preprocess train
        # =======================
        log("Loading & preprocessing training data...")
        Xs = []
        Ys = []

        for X, y, T, L, mean, std in tqdm(train_loader):
            Xi = X[0, :T].numpy()
            yi = y.numpy().reshape(-1).astype(int)

            Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
            Xs.append(Xi)
            Ys.append(yi)

        log(f"Loaded {len(Xs)} sequences")

        # =======================
        # Fit LDS (one-shot)
        # =======================
        log("\nFitting LDS (one-shot)...")
        A, C, Q, R, _ = fit_lds(Xs, latent_dim=40)
        
        # =======================
        # Re-compute latents using full Kalman Filter
        # (so training and evaluation are consistent)
        # =======================
        log("\nComputing latents with Kalman Filter...")
        latents = []
        for i, X in enumerate(Xs):
            Z = kalman_filter(X, A, C, Q, R)
            latents.append(Z)
            if (i + 1) % 2000 == 0:
                log(f"  Processed {i+1}/{len(Xs)} sequences")
        log(f"  Done: {len(latents)} sequences")

        # =======================
        # Fit classifier
        # =======================
        log("\nTraining classifier...")
        clf = train_classifier(latents, Ys)

        # =======================
        # Save model
        # =======================
        model_path = f"kf_model_{INPUT_DIM}dim.pkl"
        save_model(A, C, Q, R, clf, model_path)

        # =======================
        # Evaluate
        # =======================
        frame_acc, seq_per = evaluate_kf(test_loader, preproc, A, C, Q, R, clf, num_samples=50)
        log(f"\nKF Results (INPUT_DIM={INPUT_DIM}):")
        log(f"  Frame-level Accuracy: {frame_acc:.3f}")
        log(f"  Sequence PER (after collapse): {seq_per:.3f}")

        # =======================
        # Show one sample
        # =======================
        show_one(test_loader, preproc, A, C, Q, R, clf)

        return frame_acc, seq_per
    
    except Exception as e:
        import traceback
        log(f"\nERROR: {e}")
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    train_kf()

