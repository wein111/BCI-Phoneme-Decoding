# ===========================================================
# show_best_samples.py
# Find and display the best decoding examples from the Training Set
# for presentation slides (256-dim models).
# ===========================================================

import sys
import os
import pickle
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import editdistance
from scipy.ndimage import gaussian_filter1d

# Add paths to import original models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GRU_pro'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BrainBert'))

from train_gru_pro import GRU6
from model_brainbert import BrainBERTLite
from ID2phoneme import id2phoneme

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

NUM_CLASSES = 41
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ===========================================================
# Dataset & Preprocessors (Same as compare_all.py)
# ===========================================================
class SpikeDataset(Dataset):
    def __init__(self, tfrecord_paths, max_len=500, input_dim=256):
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


class KFPreprocessor:
    def __init__(self, sigma=1.5):
        self.sigma = sigma
    def __call__(self, X, mean, std):
        X = (X - mean) / (std + 1e-6)
        if self.sigma:
            X = gaussian_filter1d(X, sigma=self.sigma, axis=0)
        return X.astype(np.float32), X.shape[0]


class NeuralPreprocessor:
    def __init__(self, sigma=1.5, stack_k=5, stack_stride=2, subsample=3):
        self.sigma = sigma
        self.stack_k = stack_k
        self.stack_stride = stack_stride
        self.subsample = subsample

    def __call__(self, X, mean, std):
        X = (X - mean) / (std + 1e-6)
        if self.sigma:
            X = gaussian_filter1d(X, sigma=self.sigma, axis=0)
        T, D = X.shape
        stacked = []
        for i in range(0, T - self.stack_k + 1, self.stack_stride):
            window = X[i : i + self.stack_k].flatten()
            stacked.append(window)
        if not stacked:
            return np.zeros((1, D * self.stack_k), dtype=np.float32), 1
        X_new = np.array(stacked)
        if self.subsample > 1:
            X_new = X_new[::self.subsample]
        return X_new.astype(np.float32), X_new.shape[0]

# ===========================================================
# Helper Functions
# ===========================================================
def collapse_repeats(seq):
    if len(seq) == 0: return []
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != result[-1]:
            result.append(seq[i])
    return result

def create_frame_labels(T_frames, phoneme_seq):
    L = len(phoneme_seq)
    if L == 0: return np.zeros(T_frames, dtype=int)
    frame_labels = np.zeros(T_frames, dtype=int)
    frames_per_phoneme = T_frames / L
    for i in range(T_frames):
        idx = min(int(i / frames_per_phoneme), L - 1)
        frame_labels[i] = phoneme_seq[idx]
    return frame_labels

def greedy_decode_ctc(logits):
    pred = logits.argmax(dim=-1).cpu().tolist()
    seq, prev = [], -1
    for p in pred:
        if p != 0 and p != prev:
            seq.append(p)
        prev = p
    return seq, pred

# ===========================================================
# KF Model Wrapper
# ===========================================================
def kalman_filter_fast(Y, A, C, Q, R):
    T, D = Y.shape
    K = A.shape[0]
    R_diag = np.diag(R) + 1e-6
    x = np.zeros(K)
    P = np.eye(K) * 0.1
    out = np.zeros((T, K))
    for t in range(T):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        y = Y[t]
        CP = C @ P_pred
        S_diag = np.sum(CP * C, axis=1) + R_diag
        K_gain = P_pred @ C.T / S_diag
        x = x_pred + K_gain @ (y - C @ x_pred)
        P = P_pred - K_gain @ C @ P_pred
        out[t] = x
    return out

class KFModel:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.A, self.C = data['A'], data['C']
        self.Q, self.R = data['Q'], data['R']
        self.clf = data['clf']
    
    def predict(self, X):
        Z = kalman_filter_fast(X, self.A, self.C, self.Q, self.R)
        pred = self.clf.predict(Z)
        from scipy.stats import mode
        smoothed = np.zeros_like(pred)
        w = 3 # Window size
        for i in range(len(pred)):
            s, e = max(0, i-w), min(len(pred), i+w+1)
            smoothed[i] = mode(pred[s:e], keepdims=False)[0]
        return smoothed

# ===========================================================
# Find Good Sample (longer sentence, with 1-2 small errors)
# ===========================================================
def find_good_sample(model, model_type, loader, preproc, num_samples=150):
    """
    Find a sample that:
    1. Has longer phoneme sequence (>15 phonemes)
    2. Has small errors (PER between 5%-25%) - not perfect, more realistic
    """
    candidates = []
    
    print(f"Scanning {num_samples} samples for good presentation example...")
    
    for i, (X, y, T, L, mean, std) in enumerate(loader):
        if i >= num_samples: break
        
        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)[:int(L)]
        
        # Skip short sentences
        if len(yi) < 15:
            continue
            
        Xi_proc, T_proc = preproc(Xi, mean.numpy(), std.numpy())
        
        pred_seq = []
        if model_type == 'kf':
            pred_frames = model.predict(Xi_proc)
            pred_seq = collapse_repeats(pred_frames.tolist())
        else:
            with torch.no_grad():
                Xi_tensor = torch.tensor(Xi_proc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lengths = torch.tensor([T_proc], dtype=torch.long).to(DEVICE)
                logits = model(Xi_tensor, lengths)
                pred_seq, _ = greedy_decode_ctc(logits[:, 0, :])
        
        dist = editdistance.eval(pred_seq, yi.tolist())
        per = dist / len(yi)
        
        # Collect candidates with small errors (5%-25% PER for neural, higher for KF)
        if model_type == 'kf':
            # KF is weaker, accept higher PER range
            if 0.3 <= per <= 0.8:
                candidates.append({
                    'per': per,
                    'true': yi.tolist(),
                    'pred': pred_seq,
                    'id': i,
                    'len': len(yi)
                })
        else:
            # Neural models: prefer samples with 1-3 errors
            if 0.05 <= per <= 0.25:
                candidates.append({
                    'per': per,
                    'true': yi.tolist(),
                    'pred': pred_seq,
                    'id': i,
                    'len': len(yi)
                })
    
    if not candidates:
        # Fallback: just find best if no ideal candidate
        print("  No ideal candidate found, falling back to best available...")
        return find_best_fallback(model, model_type, loader, preproc, num_samples)
    
    # Sort by length (prefer longer), then by lower PER
    candidates.sort(key=lambda x: (-x['len'], x['per']))
    return candidates[0]


def find_best_fallback(model, model_type, loader, preproc, num_samples):
    """Fallback: find best PER sample"""
    best_per = 1000.0
    best_sample = None
    
    for i, (X, y, T, L, mean, std) in enumerate(loader):
        if i >= num_samples: break
        
        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)[:int(L)]
        if len(yi) < 10: continue
            
        Xi_proc, T_proc = preproc(Xi, mean.numpy(), std.numpy())
        
        if model_type == 'kf':
            pred_frames = model.predict(Xi_proc)
            pred_seq = collapse_repeats(pred_frames.tolist())
        else:
            with torch.no_grad():
                Xi_tensor = torch.tensor(Xi_proc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lengths = torch.tensor([T_proc], dtype=torch.long).to(DEVICE)
                logits = model(Xi_tensor, lengths)
                pred_seq, _ = greedy_decode_ctc(logits[:, 0, :])
        
        dist = editdistance.eval(pred_seq, yi.tolist())
        per = dist / len(yi)
        
        if per < best_per:
            best_per = per
            best_sample = {'per': per, 'true': yi.tolist(), 'pred': pred_seq, 'id': i, 'len': len(yi)}
            
    return best_sample

# ===========================================================
# Main
# ===========================================================
def main():
    # 1. Setup Paths
    BASE = r"D:\DeepLearning\BCI\Dataset\derived\tfRecords"
    DATES = ["t12.2022.04.28", "t12.2022.05.05", "t12.2022.05.17"] # Use first few dates
    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
    
    # 2. Models Config (256-dim only)
    models = [
        {
            'name': 'KF-256', 'type': 'kf', 
            'path': os.path.join(MODEL_DIR, "kf", "kf_model_256dim.pkl"),
            'preproc': KFPreprocessor(sigma=1.5)
        },
        {
            'name': 'GRU-256', 'type': 'gru',
            'path': os.path.join(MODEL_DIR, "gru", "best_gru_256dim.pt"),
            'preproc': NeuralPreprocessor(sigma=1.5, stack_k=5, stack_stride=2, subsample=3)
        },
        {
            'name': 'BrainBERT-256', 'type': 'brainbert',
            'path': os.path.join(MODEL_DIR, "brainbert", "best_brainbert_256dim.pt"),
            'preproc': NeuralPreprocessor(sigma=1.5, stack_k=5, stack_stride=2, subsample=3)
        }
    ]
    
    # 3. Load Dataset
    ds = SpikeDataset(train_paths, input_dim=256)
    loader = DataLoader(ds, batch_size=1, shuffle=True) # Random shuffle to find good ones
    
    # 4. Process Each Model
    for cfg in models:
        print(f"\n{'='*60}")
        print(f"Finding best sample for: {cfg['name']}")
        print(f"{'='*60}")
        
        # Load Model
        if cfg['type'] == 'kf':
            model = KFModel(cfg['path'])
        elif cfg['type'] == 'gru':
            model = GRU6(input_dim=256*5, hidden=256, num_layers=5, num_classes=NUM_CLASSES).to(DEVICE)
            ckpt = torch.load(cfg['path'], map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            model.eval()
        elif cfg['type'] == 'brainbert':
            model = BrainBERTLite(input_dim=256*5, d_model=128, nhead=4, num_layers=4, dim_feedforward=512, num_classes=NUM_CLASSES).to(DEVICE)
            ckpt = torch.load(cfg['path'], map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            model.eval()
            
        # Find Good Sample (longer, with small errors)
        res = find_good_sample(model, cfg['type'], loader, cfg['preproc'], num_samples=150)
        
        # Display
        print(f"\n>>> Best Result (Sample ID: {res['id']})")
        print(f"PER: {res['per']*100:.1f}%")
        
        # Convert to Phonemes
        true_ph = [id2phoneme.get(p, '?') for p in res['true']]
        pred_ph = [id2phoneme.get(p, '?') for p in res['pred']]
        
        print("-" * 60)
        print(f"TRUE ({len(true_ph)}): {' '.join(true_ph)}")
        print(f"PRED ({len(pred_ph)}): {' '.join(pred_ph)}")
        print("-" * 60)
        print("\n")

if __name__ == "__main__":
    main()
