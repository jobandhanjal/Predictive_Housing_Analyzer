# src/metrics.py
import numpy as np

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = y_true.shape[0]
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # handle corner case
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
