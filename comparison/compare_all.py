# ===========================================================
# compare_all.py
# Unified comparison of all models: KF, GRU, BrainBERT
# Using ORIGINAL model definitions for consistent evaluation
# ===========================================================

import sys
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import editdistance
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # D:\DeepLearning\BCI
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Add paths to import original models
sys.path.insert(0, os.path.join(BASE_DIR, 'GRU_pro'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BrainBert'))

NUM_CLASSES = 41

# ===========================================================
# Dataset
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


# ===========================================================
# Preprocessors
# ===========================================================
class KFPreprocessor:
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def __call__(self, X, mean, std):
        X = (X - mean) / (std + 1e-6)
        if self.sigma:
            X = gaussian_filter1d(X, sigma=self.sigma, axis=0)
        return X.astype(np.float32), X.shape[0]


class NeuralPreprocessor:
    def __init__(self, sigma=2.0, stack_k=5, stack_stride=2, subsample=3):
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
            stacked.append(X[i:i + self.stack_k].flatten())
        X = np.array(stacked)
        X = X[::self.subsample]
        return X.astype(np.float32), X.shape[0]


# ===========================================================
# Helper functions
# ===========================================================
def collapse_repeats(seq):
    if len(seq) == 0:
        return []
    result = [seq[0]]
    for i in range(1, len(seq)):
        if seq[i] != result[-1]:
            result.append(seq[i])
    return result


def create_frame_labels(T_frames, phoneme_seq):
    L = len(phoneme_seq)
    if L == 0:
        return np.zeros(T_frames, dtype=int)
    frame_labels = np.zeros(T_frames, dtype=int)
    frames_per_phoneme = T_frames / L
    for i in range(T_frames):
        phoneme_idx = min(int(i / frames_per_phoneme), L - 1)
        frame_labels[i] = phoneme_seq[phoneme_idx]
    return frame_labels


def greedy_decode_ctc(logits, blank_id=0):
    preds = np.argmax(logits, axis=-1)
    decoded = []
    prev = -1
    for p in preds:
        if p != blank_id and p != prev:
            decoded.append(p)
        prev = p
    return decoded


# ===========================================================
# KF Model
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
        innovation = y - C @ x_pred
        x = x_pred + K_gain @ innovation
        P = P_pred - K_gain @ C @ P_pred
        out[t] = x
    return out


def smooth_predictions(pred, window_size=7):
    from scipy.stats import mode
    T = len(pred)
    smoothed = np.zeros(T, dtype=int)
    half_w = window_size // 2
    for i in range(T):
        start = max(0, i - half_w)
        end = min(T, i + half_w + 1)
        window = pred[start:end]
        smoothed[i] = mode(window, keepdims=False)[0]
    return smoothed


class KFModel:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.A = data['A']
        self.C = data['C']
        self.Q = data['Q']
        self.R = data['R']
        self.clf = data['clf']
        self.input_dim = data['input_dim']
    
    def predict(self, X):
        Z = kalman_filter_fast(X, self.A, self.C, self.Q, self.R)
        pred = self.clf.predict(Z)
        pred = smooth_predictions(pred)
        return pred


# ===========================================================
# Import ORIGINAL model classes (for consistent evaluation)
# ===========================================================
# Import GRU model from GRU_pro
from train_gru_pro import GRU6

# Import BrainBERT model from BrainBERT
from model_brainbert import BrainBERTLite


# ===========================================================
# Model Loader (using ORIGINAL model classes)
# ===========================================================
def load_model(model_type, model_path, input_dim):
    if model_type == 'kf':
        return KFModel(model_path)
    
    elif model_type == 'gru':
        stack_k = 5
        # Use ORIGINAL GRU6 from train_gru_pro.py
        model = GRU6(input_dim=input_dim * stack_k, hidden=256, num_layers=5, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model
    
    elif model_type == 'brainbert':
        stack_k = 5
        # Use ORIGINAL BrainBERTLite from model_brainbert.py
        model = BrainBERTLite(input_dim=input_dim * stack_k, d_model=128, nhead=4, 
                              num_layers=4, dim_feedforward=512, num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ===========================================================
# Evaluation (using SAME logic as original training scripts)
# ===========================================================
def evaluate_model(model, model_type, test_loader, preproc, num_samples=50):
    results = {
        'per': [],
        'frame_acc': [],
        'token_acc': []
    }
    
    total_phonemes = 0
    total_time = 0
    
    for i, (X, y, T, L, mean, std) in enumerate(test_loader):
        if i >= num_samples:
            break
        
        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1).astype(int)
        L_val = int(L.numpy())
        yi = yi[:L_val]
        
        Xi_proc, T_proc = preproc(Xi, mean.numpy(), std.numpy())
        
        start_time = time.time()
        
        if model_type == 'kf':
            pred_frames = model.predict(Xi_proc)
            pred_seq = collapse_repeats(pred_frames.tolist())
        else:
            # GRU and BrainBERT: Use SAME inference as original training
            # Original models need lengths parameter and return (T, B, C)
            with torch.no_grad():
                Xi_tensor = torch.tensor(Xi_proc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                lengths = torch.tensor([T_proc], dtype=torch.long).to(DEVICE)
                
                # Model forward returns (T, B, C) for CTC
                logits = model(Xi_tensor, lengths)  # (T, B, C)
                
                # Get logits for first (only) sample: (T, C)
                log0 = logits[:, 0, :]  # (T, C)
                
                # Greedy decode (same as training)
                pred = log0.argmax(dim=-1).cpu().tolist()  # List of T predictions
                
                # CTC collapse: remove blanks and consecutive duplicates
                pred_seq = []
                prev = -1
                for p in pred:
                    if p != 0 and p != prev:  # 0 is blank
                        pred_seq.append(p)
                    prev = p
                
                pred_frames = np.array(pred)
        
        elapsed = time.time() - start_time
        
        yi_list = yi.tolist()
        
        # PER (Phoneme Error Rate) - same as training
        edit_dist = editdistance.eval(pred_seq, yi_list)
        per = edit_dist / max(len(yi_list), 1)
        results['per'].append(per)
        
        total_phonemes += len(yi_list)
        total_time += elapsed
        
        # Frame-level accuracy (uniform alignment, same as training)
        yi_frames = create_frame_labels(len(pred_frames), yi)
        frame_acc = np.mean(pred_frames == yi_frames)
        results['frame_acc'].append(frame_acc)
        
        # Token-level accuracy
        min_len = min(len(pred_seq), len(yi_list))
        if min_len > 0:
            correct = sum(1 for j in range(min_len) if pred_seq[j] == yi_list[j])
            token_acc = correct / min_len
        else:
            token_acc = 0
        results['token_acc'].append(token_acc)
    
    avg_results = {
        'Frame Acc': np.mean(results['frame_acc']),
        'PER': np.mean(results['per']),
        'Phonemes/sec': total_phonemes / total_time if total_time > 0 else 0,
        'Token Acc': np.mean(results['token_acc'])
    }
    
    return avg_results


# ===========================================================
# Visualization
# ===========================================================
def plot_comparison(train_results, test_results, save_dir):
    """Create three separate comparison charts with train/test comparison."""
    valid_train = [r for r in train_results if isinstance(r.get('PER'), float)]
    valid_test = [r for r in test_results if isinstance(r.get('PER'), float)]
    
    if len(valid_test) == 0:
        print("No valid results to plot")
        return
    
    models = [r['Model'] for r in valid_test]
    
    # Test metrics
    test_frame_acc = [r['Frame Acc'] * 100 for r in valid_test]
    test_per = [r['PER'] * 100 for r in valid_test]
    test_speed = [r['Phonemes/sec'] for r in valid_test]
    test_token_acc = [r['Token Acc'] * 100 for r in valid_test]
    
    # Train metrics
    train_frame_acc = [r['Frame Acc'] * 100 for r in valid_train]
    train_per = [r['PER'] * 100 for r in valid_train]
    
    # Modern color palette
    colors = {
        'KF': '#EF5350',      # Red
        'GRU': '#66BB6A',     # Green  
        'BERT': '#42A5F5'     # Blue
    }
    
    def get_color(model_name):
        if 'KF' in model_name:
            return colors['KF']
        elif 'GRU' in model_name:
            return colors['GRU']
        else:
            return colors['BERT']
    
    bar_colors = [get_color(m) for m in models]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # ==========================================
    # Chart 1: Train/Test Accuracy & PER (2x2)
    # Layout: Top row = Train, Bottom row = Test
    # ==========================================
    fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    x = np.arange(len(models))
    width = 0.55
    
    # Top-left: Train Frame Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, train_frame_acc, width, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Train - Frame Accuracy', fontsize=13, pad=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax1.set_ylim(0, max(train_frame_acc + test_frame_acc) * 1.3)
    for bar, val in zip(bars1, train_frame_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Top-right: Train PER
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, train_per, width, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax2.set_ylabel('PER (%)', fontsize=11)
    ax2.set_title('Train - Phoneme Error Rate', fontsize=13, pad=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax2.set_ylim(0, max(train_per + test_per) * 1.15)
    ax2.axhline(y=100, color='#BDBDBD', linestyle='--', linewidth=1.2)
    for bar, val in zip(bars2, train_per):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Bottom-left: Test Frame Accuracy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, test_frame_acc, width, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Test - Frame Accuracy', fontsize=13, pad=10, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax3.set_ylim(0, max(train_frame_acc + test_frame_acc) * 1.3)
    for bar, val in zip(bars3, test_frame_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Bottom-right: Test PER
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, test_per, width, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax4.set_ylabel('PER (%)', fontsize=11)
    ax4.set_title('Test - Phoneme Error Rate', fontsize=13, pad=10, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax4.set_ylim(0, max(train_per + test_per) * 1.15)
    ax4.axhline(y=100, color='#BDBDBD', linestyle='--', linewidth=1.2)
    for bar, val in zip(bars4, test_per):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout(pad=2.5)
    path1 = os.path.join(save_dir, "chart1_train_test_comparison.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Chart 1 saved: {path1}")
    plt.close()
    
    # ==========================================
    # Chart 2: Decoding Speed
    # ==========================================
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    
    bars = ax.bar(x, test_speed, width, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.set_ylabel('Phonemes / sec', fontsize=11)
    ax.set_title('Decoding Speed', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylim(100, max(test_speed) * 2.5)
    for bar, val in zip(bars, test_speed):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.25, 
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path2 = os.path.join(save_dir, "chart2_speed.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Chart 2 saved: {path2}")
    plt.close()
    
    # ==========================================
    # Chart 3: Token-level Accuracy
    # ==========================================
    fig3, ax = plt.subplots(figsize=(9, 5.5))
    
    bars = ax.bar(x, test_token_acc, width, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.set_ylabel('Token Accuracy (%)', fontsize=11)
    ax.set_title('Token-level Accuracy', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, max(test_token_acc) * 1.25)
    for bar, val in zip(bars, test_token_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    path3 = os.path.join(save_dir, "chart3_token_acc.png")
    plt.savefig(path3, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Chart 3 saved: {path3}")
    plt.close()


# ===========================================================
# Main
# ===========================================================
def main():
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
    
    models_config = [
        {
            'name': 'KF-128',
            'type': 'kf',
            'path': os.path.join(MODEL_DIR, "kf", "kf_model_128dim.pkl"),
            'input_dim': 128,
            'preproc': KFPreprocessor(sigma=1.5)
        },
        {
            'name': 'KF-256',
            'type': 'kf',
            'path': os.path.join(MODEL_DIR, "kf", "kf_model_256dim.pkl"),
            'input_dim': 256,
            'preproc': KFPreprocessor(sigma=1.5)
        },
        {
            'name': 'GRU-128',
            'type': 'gru',
            'path': os.path.join(MODEL_DIR, "gru", "best_gru_128dim.pt"),
            'input_dim': 128,
            'preproc': NeuralPreprocessor(sigma=1.5, stack_k=5, stack_stride=2, subsample=3)
        },
        {
            'name': 'GRU-256',
            'type': 'gru',
            'path': os.path.join(MODEL_DIR, "gru", "best_gru_256dim.pt"),
            'input_dim': 256,
            'preproc': NeuralPreprocessor(sigma=1.5, stack_k=5, stack_stride=2, subsample=3)
        },
        {
            'name': 'BrainBERT-128',
            'type': 'brainbert',
            'path': os.path.join(MODEL_DIR, "brainbert", "best_brainbert_128dim.pt"),
            'input_dim': 128,
            'preproc': NeuralPreprocessor(sigma=1.5, stack_k=5, stack_stride=2, subsample=3)
        },
        {
            'name': 'BrainBERT-256',
            'type': 'brainbert',
            'path': os.path.join(MODEL_DIR, "brainbert", "best_brainbert_256dim.pt"),
            'input_dim': 256,
            'preproc': NeuralPreprocessor(sigma=1.5, stack_k=5, stack_stride=2, subsample=3)
        },
    ]
    
    train_paths = [f"{BASE}/{d}/train/chunk_0.tfrecord" for d in DATES]
    
    print("=" * 70)
    print("Model Comparison: KF vs GRU vs BrainBERT")
    print("=" * 70)
    
    train_results = []
    test_results = []
    
    for cfg in models_config:
        print(f"\n[Evaluating] {cfg['name']}...")
        
        if not os.path.exists(cfg['path']):
            print(f"  Model not found: {cfg['path']}")
            empty_result = {
                'Model': cfg['name'],
                'Frame Acc': 'N/A',
                'PER': 'N/A',
                'Phonemes/sec': 'N/A',
                'Token Acc': 'N/A'
            }
            train_results.append(empty_result.copy())
            test_results.append(empty_result.copy())
            continue
        
        try:
            model = load_model(cfg['type'], cfg['path'], cfg['input_dim'])
            
            # Evaluate on TRAIN set (random samples, more samples)
            print("  [Train]...")
            train_ds = SpikeDataset(train_paths, input_dim=cfg['input_dim'])
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
            train_res = evaluate_model(model, cfg['type'], train_loader, 
                                       cfg['preproc'], num_samples=100)
            train_res['Model'] = cfg['name']
            train_results.append(train_res)
            print(f"    Frame Acc: {train_res['Frame Acc']*100:.1f}%, PER: {train_res['PER']*100:.1f}%")
            
            # Evaluate on TEST set (random samples, more samples)
            print("  [Test]...")
            test_ds = SpikeDataset(test_paths, input_dim=cfg['input_dim'])
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
            test_res = evaluate_model(model, cfg['type'], test_loader, 
                                      cfg['preproc'], num_samples=100)
            test_res['Model'] = cfg['name']
            test_results.append(test_res)
            print(f"    Frame Acc: {test_res['Frame Acc']*100:.1f}%, PER: {test_res['PER']*100:.1f}%")
            print(f"    Speed: {test_res['Phonemes/sec']:.0f} ph/s, Token Acc: {test_res['Token Acc']*100:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            error_result = {
                'Model': cfg['name'],
                'Frame Acc': 'Error',
                'PER': 'Error',
                'Phonemes/sec': 'Error',
                'Token Acc': 'Error'
            }
            train_results.append(error_result.copy())
            test_results.append(error_result.copy())
    
    # Print summary table
    print("\n" + "=" * 85)
    print("TEST SET SUMMARY")
    print("=" * 85)
    
    print(f"{'Model':<15} {'Frame Acc':<12} {'PER':<12} {'Speed':<15} {'Token Acc':<12}")
    print("-" * 85)
    
    for r in test_results:
        model = r['Model']
        fa = f"{r['Frame Acc']*100:.1f}%" if isinstance(r.get('Frame Acc'), float) else r.get('Frame Acc', 'N/A')
        per = f"{r['PER']*100:.1f}%" if isinstance(r.get('PER'), float) else r.get('PER', 'N/A')
        spd = f"{r['Phonemes/sec']:.0f}" if isinstance(r.get('Phonemes/sec'), float) else r.get('Phonemes/sec', 'N/A')
        ta = f"{r['Token Acc']*100:.1f}%" if isinstance(r.get('Token Acc'), float) else r.get('Token Acc', 'N/A')
        print(f"{model:<15} {fa:<12} {per:<12} {spd:<15} {ta:<12}")
    
    print("=" * 85)
    
    # Find best models
    valid_results = [r for r in test_results if isinstance(r.get('PER'), float)]
    if valid_results:
        best_per = min(valid_results, key=lambda x: x['PER'])
        best_frame = max(valid_results, key=lambda x: x['Frame Acc'])
        best_speed = max(valid_results, key=lambda x: x['Phonemes/sec'])
        
        print(f"\nBest Frame Acc: {best_frame['Model']} ({best_frame['Frame Acc']*100:.1f}%)")
        print(f"Best PER (lowest): {best_per['Model']} ({best_per['PER']*100:.1f}%)")
        print(f"Best Speed: {best_speed['Model']} ({best_speed['Phonemes/sec']:.0f} phonemes/sec)")
    
    # Plot comparison charts (3 separate figures)
    plot_comparison(train_results, test_results, SCRIPT_DIR)


if __name__ == "__main__":
    main()
