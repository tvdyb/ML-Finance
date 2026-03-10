"""
Evaluation utilities:
- Long-short portfolio: (mean_long_excess - mean_short_excess) / 2
- Portfolio tracking from quarterly returns
- Sharpe ratio from quarterly portfolio returns
- Classification accuracy
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def accuracy_metrics(y_true, y_pred):
    """Standard classification metrics."""
    report = classification_report(
        y_true, y_pred,
        target_names=["-1 (miss)", "0 (neutral)", "+1 (beat)"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    return {"report": report, "confusion_matrix": cm}


def compute_portfolio_value(quarterly_results):
    """
    Compute cumulative portfolio value using standard long-short construction:
        qtr_return = (mean_long_excess - mean_short_excess) / 2
        x[i+1] = x[i] * (1 + qtr_return)

    Each side (long/short) gets equal capital. Per-position edge matters,
    not number of positions. Neutral predictions are ignored.

    quarterly_results: list of dicts with keys 'preds', 'excess_ret', 'qtr'
    Returns list of portfolio values starting at 1.0
    """
    # Winsorize excess returns at 1st/99th percentile
    all_excess = np.concatenate([qr["excess_ret"] for qr in quarterly_results])
    lo, hi = np.percentile(all_excess, 1), np.percentile(all_excess, 99)

    x = [1.0]
    qtr_profits = []
    for qr in quarterly_results:
        preds = qr["preds"]
        excess_ret = np.clip(qr["excess_ret"], lo, hi)

        long_mask = preds == 1
        short_mask = preds == -1

        # Long side: mean excess return of stocks predicted +1
        long_ret = excess_ret[long_mask].mean() if long_mask.sum() > 0 else 0.0
        # Short side: we profit from the negative of shorts' excess return
        short_ret = -excess_ret[short_mask].mean() if short_mask.sum() > 0 else 0.0

        # Equal capital each side
        n_sides = (long_mask.sum() > 0) + (short_mask.sum() > 0)
        qtr_return = (long_ret + short_ret) / max(n_sides, 1)

        qtr_profits.append({
            "qtr": qr["qtr"],
            "qtr_return": qtr_return,
            "n_long": int(long_mask.sum()),
            "n_short": int(short_mask.sum()),
        })

        x.append(x[-1] * (1 + qtr_return))

    return x, pd.DataFrame(qtr_profits)


def sharpe_ratio(qtr_returns, annualize=True):
    """Compute Sharpe ratio from per-quarter strategy returns."""
    r = np.array(qtr_returns)
    if len(r) < 2:
        return 0.0
    sharpe = r.mean() / (r.std() + 1e-8)
    if annualize:
        sharpe *= np.sqrt(4)  # quarterly -> annual
    return sharpe
