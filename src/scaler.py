# src/scaler.py
import numpy as np

class StandardScalerFromScratch:
    """
    Standardization: z = (x - mean) / std
    Implements fit, transform, inverse_transform.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.with_mean:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        if self.with_std:
            # use ddof=0 to match sklearn's StandardScaler default (population std)
            self.scale_ = X.std(axis=0, ddof=0)
            # avoid division by zero
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        Xs = np.asarray(X_scaled, dtype=float)
        return Xs * self.scale_ + self.mean_
