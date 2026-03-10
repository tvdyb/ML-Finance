"""
Evaluation utilities: accuracy, profit, Sharpe ratio, and comparison reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def accuracy_metrics(y_true, y_pred):
    """Standard classification metrics."""
    report = classification_report(y_true, y_pred, target_names=["-1 (miss)", "0 (neutral)", "+1 (beat)"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    return {"report": report, "confusion_matrix": cm}


def portfolio_profit(y_pred, excess_ret, qtr_labels):
    """
    Simulate a long-short portfolio:
    - Go long stocks predicted +1
    - Go short stocks predicted -1
    - Ignore stocks predicted 0
    Returns quarterly returns and total compounded return.
    """
    df = pd.DataFrame({
        "pred": y_pred,
        "excess_ret": excess_ret,
        "qtr": qtr_labels,
    })

    qtr_rets = []
    for qtr, grp in df.groupby("qtr"):
        longs = grp[grp["pred"] == 1]["excess_ret"]
        shorts = grp[grp["pred"] == -1]["excess_ret"]

        ret = 0.0
        n = 0
        if len(longs) > 0:
            ret += longs.mean()
            n += 1
        if len(shorts) > 0:
            ret -= shorts.mean()
            n += 1
        if n > 0:
            qtr_rets.append({"qtr": qtr, "return": ret / n})

    qtr_df = pd.DataFrame(qtr_rets).sort_values("qtr")
    total_return = (1 + qtr_df["return"]).prod() - 1
    return {"quarterly_returns": qtr_df, "total_return": total_return}


def sharpe_ratio(quarterly_returns: pd.Series, annualize: bool = True) -> float:
    """Compute Sharpe ratio from a series of quarterly returns."""
    if len(quarterly_returns) < 2:
        return 0.0
    mean = quarterly_returns.mean()
    std = quarterly_returns.std()
    sharpe = mean / (std + 1e-8)
    if annualize:
        sharpe *= np.sqrt(4)  # annualize from quarterly
    return sharpe


def full_evaluation(model_name, y_true, y_pred, excess_ret, qtr_labels):
    """Run all evaluations and return a summary dict."""
    acc = accuracy_metrics(y_true, y_pred)
    profit = portfolio_profit(y_pred, excess_ret, qtr_labels)
    sharpe = sharpe_ratio(profit["quarterly_returns"]["return"])

    overall_acc = (y_true == y_pred).mean()

    return {
        "model": model_name,
        "accuracy": overall_acc,
        "total_profit": profit["total_return"],
        "sharpe_ratio": sharpe,
        "classification_report": acc["report"],
        "confusion_matrix": acc["confusion_matrix"],
        "quarterly_returns": profit["quarterly_returns"],
    }
