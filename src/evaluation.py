"""
Evaluation utilities matching the course methodology:
- Profit: sum of (prediction * next_period_return) per quarter
- Portfolio tracking: x[i+1] = x[i] + (x[i] / num_stocks) * profit_i
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
    Compute cumulative portfolio value from quarterly predictions,
    matching the course formula:
        profit_i = (preds * excess_ret).sum()
        x[i+1] = x[i] + (x[i] / num_stocks) * profit_i

    quarterly_results: list of dicts with keys 'preds', 'excess_ret', 'qtr'
    Returns list of portfolio values starting at 1.0
    """
    x = [1.0]
    qtr_profits = []
    for qr in quarterly_results:
        preds = qr["preds"]
        excess_ret = qr["excess_ret"]
        num_stocks = np.count_nonzero(preds) or len(preds)

        profit_i = (preds * excess_ret).sum()
        qtr_return = profit_i / num_stocks
        qtr_profits.append({"qtr": qr["qtr"], "profit": profit_i, "num_stocks": num_stocks, "qtr_return": qtr_return})

        x.append(x[-1] + (x[-1] / num_stocks) * profit_i)

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
