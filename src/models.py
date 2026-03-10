"""
Three linear models with different optimization objectives:
1. AccuracyModel  – standard logistic regression (maximizes classification accuracy)
2. ProfitModel    – linear model trained to maximize portfolio profit via
                    custom loss weighting
3. SharpeModel    – linear model trained to maximize risk-adjusted returns (Sharpe)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Accuracy-optimized: multinomial logistic regression
# ---------------------------------------------------------------------------

class AccuracyModel:
    """Standard logistic regression optimizing cross-entropy (≈ accuracy)."""

    def __init__(self, C=1.0, max_iter=1000):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C, max_iter=max_iter,
                solver="lbfgs", class_weight="balanced",
            )),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


# ---------------------------------------------------------------------------
# 2. Profit-optimized: cost-sensitive logistic regression
# ---------------------------------------------------------------------------

class ProfitModel:
    """
    Logistic regression with sample weights proportional to |excess_ret|.
    Getting high-excess-return stocks right matters more than marginal ones.
    This directly tilts the model toward profit-relevant predictions.
    """

    def __init__(self, C=1.0, max_iter=1000):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C, max_iter=max_iter,
                solver="lbfgs",
            )),
        ])

    def fit(self, X, y, excess_ret=None):
        if excess_ret is not None:
            # Weight samples by absolute excess return — correctly
            # predicting stocks with large moves matters more for profit
            weights = np.abs(excess_ret) + 1e-6
            weights = weights / weights.mean()  # normalize
            self.pipeline.fit(X, y, lr__sample_weight=weights)
        else:
            self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


# ---------------------------------------------------------------------------
# 3. Sharpe-optimized: linear scoring model tuned for Sharpe ratio
# ---------------------------------------------------------------------------

class SharpeModel:
    """
    Learns a linear scoring function whose long-short portfolio
    (long predicted +1, short predicted -1) maximizes the Sharpe ratio
    on the training data.

    Approach: Use Ridge regression to predict excess_ret, then convert
    predictions to classes. The Ridge objective is a good proxy because
    higher predicted excess returns align with higher Sharpe when the
    model is well-calibrated. We then fine-tune the classification
    thresholds to maximize realized Sharpe on training data.
    """

    def __init__(self, alpha=1.0):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        self.long_thresh = 0.0
        self.short_thresh = 0.0

    def fit(self, X, y, excess_ret=None, qtr_labels=None):
        """
        X: features
        y: class labels (unused, kept for API consistency)
        excess_ret: the actual quarterly excess returns
        qtr_labels: quarter identifiers for grouping into portfolio returns
        """
        if excess_ret is None:
            raise ValueError("SharpeModel requires excess_ret for training")

        # Stage 1: Ridge regression to predict excess return
        self.pipeline.fit(X, excess_ret)
        scores = self.pipeline.predict(X)

        # Stage 2: Optimize thresholds to maximize Sharpe
        if qtr_labels is not None:
            self._optimize_thresholds(scores, excess_ret, qtr_labels)
        else:
            # Fallback: symmetric thresholds at +/- 0.5 std of scores
            std = np.std(scores)
            self.long_thresh = 0.5 * std
            self.short_thresh = -0.5 * std

        return self

    def _optimize_thresholds(self, scores, excess_ret, qtr_labels):
        """Find thresholds that maximize the realized Sharpe of a long-short portfolio."""
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
                # Long-short return for quarter
                ret = 0.0
                n = 0
                if len(longs) > 0:
                    ret += longs.mean()
                    n += 1
                if len(shorts) > 0:
                    ret -= shorts.mean()  # short the losers
                    n += 1
                if n > 0:
                    port_rets.append(ret / n)
            if len(port_rets) < 2:
                return 0.0
            port_rets = np.array(port_rets)
            sharpe = port_rets.mean() / (port_rets.std() + 1e-8)
            return -sharpe  # minimize negative Sharpe

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
        scores = self.pipeline.predict(X)
        preds = np.zeros(len(scores), dtype=int)
        preds[scores > self.long_thresh] = 1
        preds[scores < self.short_thresh] = -1
        return preds

    def score_raw(self, X):
        """Return raw predicted excess return scores."""
        return self.pipeline.predict(X)
