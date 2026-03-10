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
        profit_i = (preds * next_period_return).sum()
        x[i+1] = x[i] + (x[i] / num_stocks) * profit_i

    quarterly_results: list of dicts with keys 'preds', 'returns', 'qtr'
    Returns list of portfolio values starting at 1.0
    """
    x = [1.0]
    qtr_profits = []
    for qr in quarterly_results:
        preds = qr["preds"]
        returns = qr["returns"]
        num_stocks = len(preds)

        profit_i = (preds * returns).sum()
        qtr_profits.append({"qtr": qr["qtr"], "profit": profit_i, "num_stocks": num_stocks})

        x.append(x[-1] + (x[-1] / num_stocks) * profit_i)

    return x, pd.DataFrame(qtr_profits)


def sharpe_ratio(portfolio_values, annualize=True):
    """Compute Sharpe ratio from portfolio value series."""
    x = np.array(portfolio_values)
    if len(x) < 3:
        return 0.0
    returns = np.diff(x) / x[:-1]
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / (std_ret + 1e-8)
    if annualize:
        sharpe *= np.sqrt(4)  # quarterly -> annual
    return sharpe
