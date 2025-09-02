import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.scaler import StandardScalerFromScratch
from sklearn.preprocessing import StandardScaler

def test_scaler_equivalence():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    my_scaler = StandardScalerFromScratch().fit(X)
    sk_scaler = StandardScaler().fit(X)

    X_my = my_scaler.transform(X)
    X_sk = sk_scaler.transform(X)

    # Outputs should be nearly identical
    assert np.allclose(X_my, X_sk, atol=1e-8)

def test_inverse_transform():
    X = np.array([[10, 20], [30, 40]])
    my_scaler = StandardScalerFromScratch().fit(X)
    X_scaled = my_scaler.transform(X)
    X_inv = my_scaler.inverse_transform(X_scaled)

    # Inverse should recover original values
    assert np.allclose(X, X_inv, atol=1e-8)
