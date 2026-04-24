# preprocess_brainbert.py
# Preprocessing for BrainBERT (similar to GRU preprocessing)

import numpy as np
from scipy.ndimage import gaussian_filter1d


class BrainBERTPreprocessor:
    """
    Preprocessing for BrainBERT model.
    Similar to original NeuralDecoder preprocessing.
    """

    def __init__(
        self,
        smooth_sigma=1.5,
        stack_k=5,
        stack_stride=2,
        subsample_factor=3
    ):
        self.sigma = smooth_sigma
        self.k = stack_k
        self.stride = stack_stride
        self.sub = subsample_factor

    def smooth(self, X):
        """Gaussian smoothing along time to denoise."""
        return gaussian_filter1d(X, sigma=self.sigma, axis=0)

    def normalize(self, X, mean, std):
        """Session-level normalization."""
        return (X - mean) / (std + 1e-6)

    def time_stack(self, X):
        """
        Stack k consecutive frames with stride s.
        This expands feature dim from C → C*k and reduces T.
        """
        T, C = X.shape
        k, s = self.k, self.stride

        T_new = (T - k) // s + 1
        out = np.zeros((T_new, C * k), dtype=np.float32)

        idx = 0
        for t in range(0, T - k + 1, s):
            out[idx] = X[t:t+k].reshape(-1)
            idx += 1
        return out

    def subsample(self, X):
        """Reduce time by subsampling factor."""
        return X[::self.sub]

    def __call__(self, X, mean, std):
        """
        Apply: smooth → normalize → time stack → subsample
        Returns (X_new, T_new)
        """
        X = self.smooth(X)
        X = self.normalize(X, mean, std)
        X = self.time_stack(X)
        X = self.subsample(X)
        return X.astype(np.float32), X.shape[0]

