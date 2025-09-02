# src/model.py
import numpy as np

class MyLinearRegression:
    """
    Linear Regression implemented from scratch using vectorized gradient descent.
    Minimizes J(w,b) = (1/(2m)) * sum((Xw + b - y)^2)
    """

    def __init__(self, alpha=0.01, n_iters=1000, fit_intercept=True, verbose=False, batch_size=None, random_state=None):
        self.alpha = float(alpha)
        self.n_iters = int(n_iters)
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.batch_size = batch_size  # None means full-batch GD
        self.random_state = random_state
        # learned parameters:
        self.w_ = None  # shape (n_features,)
        self.b_ = 0.0
        self.cost_history_ = []

    def _initialize(self, n_features):
        rng = np.random.default_rng(self.random_state)
        # small random init improves symmetry breaking, but zeros also work for linear regression
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = 0.0

    def _compute_cost(self, X, y):
        m = X.shape[0]
        preds = X.dot(self.w_) + self.b_
        errors = preds - y
        return (1.0 / (2 * m)) * np.sum(errors ** 2)

    def _gradient(self, X, y):
        m = X.shape[0]
        preds = X.dot(self.w_) + self.b_
        errors = preds - y
        dw = (1.0 / m) * (X.T.dot(errors))   # shape (n_features,)
        db = (1.0 / m) * np.sum(errors)
        return dw, db

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X: numpy array shape (m, n)
        y: numpy array shape (m,)
        Optionally pass validation data for verbose monitoring.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        m, n_features = X.shape
        self._initialize(n_features)

        batch_size = self.batch_size or m

        for it in range(self.n_iters):
            # Optional: mini-batches
            if batch_size >= m:
                dw, db = self._gradient(X, y)
            else:
                # simple random minibatch
                idx = np.random.choice(m, batch_size, replace=False)
                dw, db = self._gradient(X[idx], y[idx])

            # update
            self.w_ = self.w_ - self.alpha * dw
            self.b_ = self.b_ - self.alpha * db

            cost = self._compute_cost(X, y)
            self.cost_history_.append(cost)

            if self.verbose and (it % max(1, self.n_iters // 10) == 0 or it == self.n_iters -1):
                if X_val is None:
                    print(f"iter={it:5d} cost={cost:.6f}")
                else:
                    val_cost = (1.0 / (2 * X_val.shape[0])) * np.sum((X_val.dot(self.w_) + self.b_ - y_val) ** 2)
                    print(f"iter={it:5d} train_cost={cost:.6f} val_cost={val_cost:.6f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.w_) + self.b_
