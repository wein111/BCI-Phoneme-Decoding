# preprocess_glm.py
import numpy as np
from scipy.ndimage import gaussian_filter1d

class GLMPreprocessor:
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def __call__(self, X, mean, std):
        # z-score
        X = (X - mean) / (std + 1e-6)

        # smoothing
        if self.sigma:
            X = gaussian_filter1d(X, sigma=self.sigma, axis=0)

        return X.astype(np.float32), X.shape[0]













