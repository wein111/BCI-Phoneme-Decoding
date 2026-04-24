import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import editdistance
import time
from scipy.ndimage import gaussian_filter1d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# Positional Encoding (for BrainBERT)
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ============================================
# Preprocessing (GRU & BrainBERT)
# ============================================
class GRUPreprocessor:
    def __init__(self, smooth_sigma=1.5, stack_k=5, stack_stride=2, subsample_factor=3):
        self.sigma = smooth_sigma
        self.k = stack_k
        self.stride = stack_stride
        self.sub = subsample_factor

    def __call__(self, X, mean, std):
        X = gaussian_filter1d(X, sigma=self.sigma, axis=0)
        X = (X - mean) / (std + 1e-6)
        T, C = X.shape
        T_new = (T - self.k) // self.stride + 1
        X = np.array([X[t:t+self.k].reshape(-1) for t in range(0, T-self.k+1, self.stride)])
        X = X[::self.sub]
        return X, X.shape[0]


# ============================================
# GRU Model
# ============================================
class GRU6(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers, dropout=dropout,
                         batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(out).permute(1, 0, 2)


# ============================================
# BrainBERT Model
# ============================================
class BrainBERTLite(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_proj(x).permute(1, 0, 2)


# ============================================
# Kalman Filter
# ============================================
def kalman_filter(Y, A, C, Q, R):
    T, D = Y.shape
    K = A.shape[0]
    R_diag = np.diag(R) + 1e-6
    x = np.zeros(K)
    P = np.eye(K) * 0.1
    out = np.zeros((T, K))
    
    for t in range(T):
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        CP = C @ P_pred
        S_diag = np.sum(CP * C, axis=1) + R_diag
        K_gain = P_pred @ C.T / S_diag
        x = x_pred + K_gain @ (Y[t] - C @ x_pred)
        P = P_pred - K_gain @ C @ P_pred
        out[t] = x
    return out


# ============================================
# Training Loop (GRU & BrainBERT)
# ============================================
def train(model, train_loader, test_loader, preproc, num_epochs=50):
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    best_per = 999

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X, ycat, Ls, Ts, means, stds in train_loader:
            X_processed, T_new = [], []
            for i in range(len(X)):
                Xi, Ti = preproc(X[i, :Ts[i]], means[i], stds[i])
                X_processed.append(Xi)
                T_new.append(Ti)
            
            X_pad = pad_sequences(X_processed)
            X_pad = torch.tensor(X_pad).to(DEVICE)
            T_new = torch.tensor(T_new).to(DEVICE)
            
            logits = model(X_pad, T_new)
            log_probs = logits.log_softmax(dim=-1)
            loss = criterion(log_probs, ycat, T_new.cpu(), Ls.cpu())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        test_per = evaluate(model, test_loader, preproc)
        
        if test_per < best_per:
            best_per = test_per
            torch.save(model.state_dict(), "best_model.pt")


def pad_sequences(sequences):
    max_len = max(s.shape[0] for s in sequences)
    feat_dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(sequences):
        padded[i, :s.shape[0]] = s
    return padded


def evaluate(model, loader, preproc):
    model.eval()
    total_per = 0
    count = 0
    with torch.no_grad():
        for X, ycat, Ls, Ts, means, stds in loader:
            Xi, Ti = preproc(X[0, :Ts[0]], means[0], stds[0])
            Xi = torch.tensor(Xi[None]).to(DEVICE)
            logits = model(Xi, torch.tensor([Ti]).to(DEVICE))
            pred = greedy_decode(logits[:, 0, :])
            true = ycat[:Ls[0]].tolist()
            per = editdistance.eval(pred, true) / max(len(true), 1)
            total_per += per
            count += 1
    return total_per / count


# ============================================
# Greedy Decoding (CTC)
# ============================================
def greedy_decode(logits):
    pred = logits.argmax(dim=-1).cpu().tolist()
    seq, prev = [], -1
    for p in pred:
        if p != 0 and p != prev:
            seq.append(p)
        prev = p
    return seq


# ============================================
# KF: Fit LDS Parameters
# ============================================
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

def fit_lds(Xs, latent_dim=40):
    allX = np.concatenate(Xs, axis=0)
    allX_centered = allX - allX.mean(0, keepdims=True)
    
    svd = TruncatedSVD(n_components=latent_dim, random_state=42)
    svd.fit(allX_centered)
    C = svd.components_.T
    pinvC = np.linalg.pinv(C)
    
    X_latents = [X @ pinvC.T for X in Xs]
    
    K = latent_dim
    A_num = np.zeros((K, K))
    A_den = np.zeros((K, K))
    for Z in X_latents:
        if len(Z) < 2:
            continue
        A_num += Z[1:].T @ Z[:-1]
        A_den += Z[:-1].T @ Z[:-1]
    A = A_num @ np.linalg.pinv(A_den)
    
    Qs = [((Z[1:] - Z[:-1] @ A.T).T @ (Z[1:] - Z[:-1] @ A.T)) / len(Z) for Z in X_latents if len(Z) > 1]
    Q = np.mean(Qs, axis=0)
    
    Rs = [((X - Z @ C.T).T @ (X - Z @ C.T)) / len(X) for X, Z in zip(Xs, X_latents)]
    R = np.mean(Rs, axis=0)
    
    return A, C, Q, R, X_latents


# ============================================
# KF: Train Classifier
# ============================================
def train_kf_classifier(latents, labels):
    X_all, y_all = [], []
    for Z, y in zip(latents, labels):
        T_frames = len(Z)
        frames_per_phoneme = T_frames / len(y)
        frame_labels = [y[min(int(i / frames_per_phoneme), len(y)-1)] for i in range(T_frames)]
        X_all.append(Z)
        y_all.append(frame_labels)
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_all, y_all)
    return clf


# ============================================
# Model Comparison: Evaluation
# ============================================
def evaluate_all_models(model, model_type, test_loader, preproc, num_samples=100):
    results = {'per': [], 'frame_acc': [], 'token_acc': []}
    total_phonemes, total_time = 0, 0
    
    for i, (X, y, T, L, mean, std) in enumerate(test_loader):
        if i >= num_samples:
            break
        
        Xi = X[0, :T].numpy()
        yi = y.numpy().reshape(-1)[:int(L)]
        Xi_proc, T_proc = preproc(Xi, mean.numpy(), std.numpy())
        
        start_time = time.time()
        
        if model_type == 'kf':
            pred_frames = model.predict(Xi_proc)
            pred_seq = collapse_repeats(pred_frames.tolist())
        else:
            with torch.no_grad():
                Xi_t = torch.tensor(Xi_proc[None]).to(DEVICE)
                logits = model(Xi_t, torch.tensor([T_proc]).to(DEVICE))
                pred_seq = greedy_decode(logits[:, 0, :])
                pred_frames = logits[:, 0, :].argmax(dim=-1).cpu().numpy()
        
        elapsed = time.time() - start_time
        
        per = editdistance.eval(pred_seq, yi.tolist()) / max(len(yi), 1)
        results['per'].append(per)
        
        yi_frames = create_frame_labels(len(pred_frames), yi)
        results['frame_acc'].append(np.mean(pred_frames == yi_frames))
        
        min_len = min(len(pred_seq), len(yi))
        token_acc = sum(pred_seq[j] == yi[j] for j in range(min_len)) / min_len if min_len > 0 else 0
        results['token_acc'].append(token_acc)
        
        total_phonemes += len(yi)
        total_time += elapsed
    
    return {
        'PER': np.mean(results['per']),
        'Frame Acc': np.mean(results['frame_acc']),
        'Token Acc': np.mean(results['token_acc']),
        'Phonemes/sec': total_phonemes / total_time
    }


def create_frame_labels(T_frames, phoneme_seq):
    L = len(phoneme_seq)
    frames_per_phoneme = T_frames / L
    return np.array([phoneme_seq[min(int(i / frames_per_phoneme), L-1)] for i in range(T_frames)])


def collapse_repeats(seq):
    result, prev = [], -1
    for p in seq:
        if p != 0 and p != prev:
            result.append(p)
        prev = p
    return result