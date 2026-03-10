"""
Main pipeline using rolling window train/test splits (course methodology):
- Train on 20 quarters (5 years), test on the next quarter
- Retrain each quarter, slide forward
- Per-quarter hyperparameter tuning via expanding-window temporal CV
- Track portfolio value: x[i+1] = x[i] + (x[i] / num_stocks) * profit_i
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_prep import FEATURE_COLS, prepare_dataset, rolling_window_splits
from models import AccuracyModel, ProfitModel, SharpeModel
from evaluation import compute_portfolio_value, sharpe_ratio, accuracy_metrics
from tuning import tune_and_fit


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TRAIN_QUARTERS = 20  # 5 years


def run_rolling_strategy(quarterly, model_class, model_name):
    """
    Run a rolling window strategy with per-quarter hyperparameter tuning:
    - For each window, tune regularization via expanding-window temporal CV
    - Fit final model on full training window with best param
    - Predict on the test quarter
    """
    all_preds = []
    all_labels = []
    quarterly_results = []

    for train_df, test_df in rolling_window_splits(quarterly, TRAIN_QUARTERS):
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df["label"].values
        test_qtr = test_df["yq"].iloc[0]

        # Tune hyperparameters and fit final model
        model, best_param, param_name = tune_and_fit(model_class, train_df)

        preds = model.predict(X_test)
        n_long = (preds == 1).sum()
        n_short = (preds == -1).sum()
        n_total = len(preds)
        print(f"    {test_qtr}: {n_long} long, {n_short} short, "
              f"{n_long + n_short} active / {n_total} total  "
              f"(best {param_name}={best_param})")
        all_preds.extend(preds)
        all_labels.extend(y_test)

        quarterly_results.append({
            "qtr": test_qtr,
            "preds": preds,
            "excess_ret": test_df["excess_ret"].values,
            "labels": y_test,
        })

    return np.array(all_preds), np.array(all_labels), quarterly_results


def main(data_path=None):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and preparing data...")
    quarterly = prepare_dataset(data_path)

    print(f"Total stock-quarters: {len(quarterly)}")
    print(f"Label distribution:\n{quarterly['label'].value_counts().sort_index()}\n")
    print(f"Rolling window: train on {TRAIN_QUARTERS} quarters, test on next quarter\n")

    # Define models (regularization is tuned per quarter via expanding-window CV)
    model_configs = [
        ("Accuracy-Optimized", AccuracyModel),
        ("Profit-Optimized", ProfitModel),
        ("Sharpe-Optimized", SharpeModel),
    ]

    all_results = {}

    for name, model_class in model_configs:
        print(f"Running rolling strategy: {name}...")
        preds, labels, qtr_results = run_rolling_strategy(
            quarterly, model_class, name
        )
        portfolio_values, qtr_profits = compute_portfolio_value(qtr_results)
        sharpe = sharpe_ratio(qtr_profits["qtr_return"])
        accuracy = (preds == labels).mean()
        total_return = portfolio_values[-1] - 1.0

        all_results[name] = {
            "preds": preds,
            "labels": labels,
            "portfolio_values": portfolio_values,
            "qtr_profits": qtr_profits,
            "sharpe": sharpe,
            "accuracy": accuracy,
            "total_return": total_return,
            "qtr_results": qtr_results,
        }

        print(f"  Accuracy:       {accuracy:.4f}")
        print(f"  Total Return:   {total_return:.4f} ({total_return*100:.1f}%)")
        print(f"  Final Value:    {portfolio_values[-1]:.4f} (starting from 1.0)")
        print(f"  Sharpe Ratio:   {sharpe:.4f}")
        print()

    # --- Summary table ---
    summary = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": r["accuracy"],
            "Total Return": r["total_return"],
            "Final Portfolio Value": r["portfolio_values"][-1],
            "Sharpe Ratio (Annualized)": r["sharpe"],
        }
        for name, r in all_results.items()
    ])
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    # --- Plots ---
    _plot_portfolio_values(all_results)
    _plot_confusion_matrices(all_results)
    _plot_summary_bars(summary)

    print(f"\nResults saved to {RESULTS_DIR}/")


def _plot_portfolio_values(all_results):
    """Plot cumulative portfolio value for each model."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in all_results.items():
        vals = res["portfolio_values"]
        ax.plot(range(len(vals)), vals, label=name, marker="o", markersize=3)

    # X-axis labels from the last model's quarters
    last_res = list(all_results.values())[-1]
    qtrs = ["Start"] + last_res["qtr_profits"]["qtr"].tolist()
    step = max(1, len(qtrs) // 15)
    ax.set_xticks(range(0, len(qtrs), step))
    ax.set_xticklabels([qtrs[i] for i in range(0, len(qtrs), step)],
                       rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("Portfolio Value")
    ax.set_title("Rolling Window Strategy: Portfolio Value Over Time")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "portfolio_values.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(all_results):
    """Plot confusion matrices side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, all_results.items()):
        cm = confusion_matrix(res["labels"], res["preds"], labels=[-1, 0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["-1", "0", "+1"], yticklabels=["-1", "0", "+1"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def _plot_summary_bars(summary):
    """Bar chart comparing the three metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["Accuracy", "Total Return", "Sharpe Ratio (Annualized)"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(summary["Model"], summary[metric], color=color, alpha=0.8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metric_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_path)
