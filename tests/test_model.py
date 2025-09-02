import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.model import MyLinearRegression
from src.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def test_linear_regression_simple():
    # Simple y = 2x + 3 dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = 2 * X.ravel() + 3

    model = MyLinearRegression(alpha=0.1, n_iters=1000, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Check R2 close to 1
    assert r2_score(y, y_pred) > 0.99

def test_against_sklearn():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = 4*X[:,0] - 2*X[:,1] + 5

    model = MyLinearRegression(alpha=0.05, n_iters=2000, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    sk = LinearRegression().fit(X, y)
    y_pred_sk = sk.predict(X)

    # Predictions should be very close
    assert mean_squared_error(y_pred, y_pred_sk) < 1e-2
