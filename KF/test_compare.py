# ===========================================================
# test_compare.py
# Compare KF models trained with different input dimensions
# ===========================================================

import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import pickle
import editdistance

from preprocess_kf import KalmanPreprocessor
from ID2phoneme import id2phoneme


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
            allX.append(e["inputFeatures"].numpy()[:, :self.input_dim])

        Xcat = np.concatenate(allX, axis=0)
        self.mean = Xcat.mean(axis=0)
        self.std = Xcat.std(axis=0) + 1e-5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = tf.io.parse_single_example(self.samples[idx], self.feature_desc)

        X = e["inputFeatures"].numpy()[:, :self.input_dim].astype(np.float32)
        y = e["seqClassIDs"].numpy().reshape(-1).astype(int)
        L = int(e["nSeqElements"].numpy())
        T = int(e["nTimeSteps"].numpy())

        y = y[:L]
        return X, y, T, L, self.mean, self.std


# ===========================================================
# Helper functions
# ===========================================================
def kalman_filter_fast(Y, A, C, Q, R):
    """
    Standard Kalman Filter with diagonal R optimization.
    """
    T, D = Y.shape
    K = A.shape[0]
    
    # Use diagonal of R for faster computation
    R_diag = np.diag(R) + 1e-6
    
    x = np.zeros(K)
    P = np.eye(K) * 0.1
    out = np.zeros((T, K))
    
    for t in range(T):
        # Predict
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        
        # Update with diagonal S approximation
        y = Y[t]
        CP = C @ P_pred
        S_diag = np.sum(CP * C, axis=1) + R_diag
        K_gain = P_pred @ C.T / S_diag
        
        innovation = y - C @ x_pred
        x = x_pred + K_gain @ innovation
        P = P_pred - K_gain @ C @ P_pred
        
        out[t] = x
    
    return out


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
    Z = kalman_filter_fast(Y, A, C, Q, R)
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


def load_model(filepath):
    """Load KF model parameters and classifier."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return (model_data['A'], model_data['C'], model_data['Q'], 
            model_data['R'], model_data['clf'], model_data['input_dim'])


# ===========================================================
# Evaluation
# ===========================================================
def evaluate_model(test_loader, preproc, A, C, Q, R, clf, num_samples=50):
    """Evaluate a single model."""
    total_frame_acc = 0
    total_seq_per = 0
    count = 0

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
        frame_acc = np.mean(pred_frames == yi_frames)
        
        pred_seq = collapse_repeats(pred_frames.tolist())
        seq_per = editdistance.eval(pred_seq, yi.tolist()) / max(len(yi), 1)

        total_frame_acc += frame_acc
        total_seq_per += seq_per
        count += 1

    return total_frame_acc / count, total_seq_per / count


def show_sample(test_loader, preproc, A, C, Q, R, clf, model_name):
    """Show one decode sample."""
    X, y, T, L, mean, std = next(iter(test_loader))

    Xi = X[0, :T].numpy()
    yi = y.numpy().reshape(-1).astype(int)
    L_val = int(L.numpy())
    yi = yi[:L_val]

    Xi, _ = preproc(Xi, mean.numpy(), std.numpy())
    pred_frames = decode_sequence(Xi, A, C, Q, R, clf)
    pred_seq = collapse_repeats(pred_frames.tolist())

    print(f"\n----- {model_name} Sample Decode -----")
    print(f"Pred Seq ({len(pred_seq)}): {[id2phoneme.get(p, '?') for p in pred_seq[:15]]}...")
    print(f"True Seq ({len(yi)}): {[id2phoneme.get(p, '?') for p in yi[:15]]}...")
    print(f"Edit distance: {editdistance.eval(pred_seq, yi.tolist())}")


# ===========================================================
# Main comparison
# ===========================================================
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models", "kf")

def compare_models():
    BASE = r"D:\DeepLearning\BCI\Dataset\derived\tfRecords"
    
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

    test_paths = [f"{BASE}/{d}/test/chunk_0.tfrecord" for d in DATES]

    preproc = KalmanPreprocessor(sigma=1.5)

    print("="*60)
    print("Comparing KF Models: 128-dim vs 256-dim")
    print("="*60)

    results = {}

    # --- 128-dim model ---
    print("\n[1] Loading 128-dim model...")
    model_path_128 = os.path.join(MODEL_DIR, "kf_model_128dim.pkl")
    try:
        A_128, C_128, Q_128, R_128, clf_128, input_dim_128 = load_model(model_path_128)
        print(f"    Model input_dim: {input_dim_128}")
        
        test_ds_128 = SpikeDataset(test_paths, input_dim=128)
        test_loader_128 = DataLoader(test_ds_128, batch_size=1, shuffle=False)
        
        print("    Evaluating 128-dim model...")
        frame_acc_128, seq_per_128 = evaluate_model(
            test_loader_128, preproc, A_128, C_128, Q_128, R_128, clf_128, num_samples=50
        )
        results['128-dim'] = {'frame_acc': frame_acc_128, 'seq_per': seq_per_128}
        print(f"    Frame Accuracy: {frame_acc_128:.4f}")
        print(f"    Sequence PER: {seq_per_128:.4f}")
        
        show_sample(test_loader_128, preproc, A_128, C_128, Q_128, R_128, clf_128, "128-dim")
    except FileNotFoundError:
        print(f"    Model file not found: {model_path_128}")
        results['128-dim'] = None

    # --- 256-dim model ---
    print("\n[2] Loading 256-dim model...")
    model_path_256 = os.path.join(MODEL_DIR, "kf_model_256dim.pkl")
    try:
        A_256, C_256, Q_256, R_256, clf_256, input_dim_256 = load_model(model_path_256)
        print(f"    Model input_dim: {input_dim_256}")
        
        test_ds_256 = SpikeDataset(test_paths, input_dim=256)
        test_loader_256 = DataLoader(test_ds_256, batch_size=1, shuffle=False)
        
        print("    Evaluating 256-dim model...")
        frame_acc_256, seq_per_256 = evaluate_model(
            test_loader_256, preproc, A_256, C_256, Q_256, R_256, clf_256, num_samples=50
        )
        results['256-dim'] = {'frame_acc': frame_acc_256, 'seq_per': seq_per_256}
        print(f"    Frame Accuracy: {frame_acc_256:.4f}")
        print(f"    Sequence PER: {seq_per_256:.4f}")
        
        show_sample(test_loader_256, preproc, A_256, C_256, Q_256, R_256, clf_256, "256-dim")
    except FileNotFoundError:
        print(f"    Model file not found: {model_path_256}")
        results['256-dim'] = None

    # --- Summary ---
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Frame Acc':<15} {'Seq PER':<15}")
    print("-"*45)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<15} {metrics['frame_acc']:<15.4f} {metrics['seq_per']:<15.4f}")
        else:
            print(f"{model_name:<15} {'N/A':<15} {'N/A':<15}")
    
    print("="*60)
    
    # Analysis
    if results.get('128-dim') and results.get('256-dim'):
        per_diff = results['256-dim']['seq_per'] - results['128-dim']['seq_per']
        acc_diff = results['256-dim']['frame_acc'] - results['128-dim']['frame_acc']
        
        print("\nAnalysis:")
        print(f"  Frame Accuracy difference (256 - 128): {acc_diff:+.4f}")
        print(f"  Sequence PER difference (256 - 128): {per_diff:+.4f}")
        
        if per_diff < 0:
            print(f"  -> 256-dim has LOWER PER (better)")
        else:
            print(f"  -> 128-dim has LOWER PER (better)")


if __name__ == "__main__":
    compare_models()

