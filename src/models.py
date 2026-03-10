"""
Three linear models with different optimization objectives:
1. AccuracyModel  - standard logistic regression (maximizes classification accuracy)
2. ProfitModel    - logistic regression with sample weights proportional to
                    |excess return| to tilt toward profit-relevant predictions
3. SharpeModel    - Ridge regression predicting excess returns, with thresholds
                    tuned to maximize Sharpe ratio of a long-short portfolio
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Accuracy-optimized: multinomial logistic regression
# ---------------------------------------------------------------------------

class AccuracyModel:
    """Standard logistic regression optimizing cross-entropy."""

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.scaler = None
        self.clf = None

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            solver="lbfgs", class_weight="balanced",
        )
        self.clf.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)


# ---------------------------------------------------------------------------
# 2. Profit-optimized: cost-sensitive logistic regression
# ---------------------------------------------------------------------------

class ProfitModel:
    """
    Logistic regression with sample weights proportional to |excess_ret|.
    Getting high-excess-return stocks right matters more for profit.
    """

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.scaler = None
        self.clf = None

    def fit(self, X, y, excess_ret=None):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            solver="lbfgs",
        )
        if excess_ret is not None:
            weights = np.abs(excess_ret) + 1e-6
            weights = weights / weights.mean()
            self.clf.fit(X_scaled, y, sample_weight=weights)
        else:
            self.clf.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)


# ---------------------------------------------------------------------------
# 3. Sharpe-optimized: Ridge regression + threshold tuning
# ---------------------------------------------------------------------------

class SharpeModel:
    """
    Ridge regression to predict excess returns, then classification
    thresholds are optimized to maximize the realized Sharpe ratio
    of a long-short portfolio on training data.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.scaler = None
        self.ridge = None
        self.long_thresh = 0.0
        self.short_thresh = 0.0

    def fit(self, X, y, excess_ret=None, qtr_labels=None):
        if excess_ret is None:
            raise ValueError("SharpeModel requires excess_ret for training")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, excess_ret)
        scores = self.ridge.predict(X_scaled)

        if qtr_labels is not None:
            self._optimize_thresholds(scores, excess_ret, qtr_labels)
        else:
            std = np.std(scores)
            self.long_thresh = 0.5 * std
            self.short_thresh = -0.5 * std

        return self

    def _optimize_thresholds(self, scores, excess_ret, qtr_labels):
        df = pd.DataFrame({
            "score": scores,
            "excess_ret": excess_ret,
            "qtr": qtr_labels,
        })

        def neg_sharpe(params):
            long_t, short_t = params[0], -abs(params[1])
            port_rets = []
            for _, grp in df.groupby("qtr"):
                longs = grp[grp["score"] > long_t]["excess_ret"]
                shorts = grp[grp["score"] < short_t]["excess_ret"]
                ret = 0.0
                n = 0
                if len(longs) > 0:
                    ret += longs.mean()
                    n += 1
                if len(shorts) > 0:
                    ret -= shorts.mean()
                    n += 1
                if n > 0:
                    port_rets.append(ret / n)
            if len(port_rets) < 2:
                return 0.0
            port_rets = np.array(port_rets)
            sharpe = port_rets.mean() / (port_rets.std() + 1e-8)
            return -sharpe

        std = np.std(scores)
        result = minimize(
            neg_sharpe,
            x0=[0.5 * std, 0.5 * std],
            method="Nelder-Mead",
            options={"maxiter": 500},
        )
        self.long_thresh = result.x[0]
        self.short_thresh = -abs(result.x[1])

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.ridge.predict(X_scaled)
        preds = np.zeros(len(scores), dtype=int)
        preds[scores > self.long_thresh] = 1
        preds[scores < self.short_thresh] = -1
        return preds
