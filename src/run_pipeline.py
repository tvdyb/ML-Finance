"""
Main pipeline: prepare data, train all three models, evaluate, and save results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import FEATURE_COLS, prepare_dataset, time_split
from models import AccuracyModel, ProfitModel, SharpeModel
from evaluation import full_evaluation


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(data_path=None):
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Data preparation ---
    print("Loading and preparing data...")
    quarterly = prepare_dataset(data_path)
    train, test = time_split(quarterly, test_start_year=2020)

    print(f"Train: {len(train)} stock-quarters | Test: {len(test)} stock-quarters")
    print(f"Label distribution (train):\n{train['label'].value_counts().sort_index()}\n")
    print(f"Label distribution (test):\n{test['label'].value_counts().sort_index()}\n")

    X_train = train[FEATURE_COLS].values
    y_train = train["label"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["label"].values

    # --- Train models ---
    print("Training Accuracy Model (Logistic Regression)...")
    acc_model = AccuracyModel(C=0.1)
    acc_model.fit(X_train, y_train)

    print("Training Profit Model (Cost-sensitive LR)...")
    profit_model = ProfitModel(C=0.1)
    profit_model.fit(X_train, y_train, excess_ret=train["excess_ret"].values)

    print("Training Sharpe Model (Ridge → Sharpe-optimized thresholds)...")
    sharpe_model = SharpeModel(alpha=1.0)
    sharpe_model.fit(
        X_train, y_train,
        excess_ret=train["excess_ret"].values,
        qtr_labels=train["yq"].values,
    )

    # --- Evaluate ---
    models = {
        "Accuracy-Optimized": acc_model,
        "Profit-Optimized": profit_model,
        "Sharpe-Optimized": sharpe_model,
    }

    results = []
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        preds = model.predict(X_test)
        res = full_evaluation(name, y_test, preds, test["excess_ret"].values, test["yq"].values)
        results.append(res)

        print(f"  Accuracy:     {res['accuracy']:.4f}")
        print(f"  Total Profit: {res['total_profit']:.4f}")
        print(f"  Sharpe Ratio: {res['sharpe_ratio']:.4f}")

    # --- Summary table ---
    summary = pd.DataFrame([
        {"Model": r["model"], "Accuracy": r["accuracy"],
         "Total Profit (Long-Short)": r["total_profit"],
         "Sharpe Ratio (Annualized)": r["sharpe_ratio"]}
        for r in results
    ])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    # --- Plots ---
    _plot_quarterly_returns(results)
    _plot_confusion_matrices(results)
    _plot_summary_bars(summary)

    print(f"\nResults saved to {RESULTS_DIR}/")


def _plot_quarterly_returns(results):
    """Plot cumulative quarterly returns for each model."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for res in results:
        qr = res["quarterly_returns"].sort_values("qtr")
        cum = (1 + qr["return"]).cumprod()
        ax.plot(range(len(cum)), cum.values, label=res["model"], marker="o", markersize=3)
        ax.set_xticks(range(len(cum)))
        ax.set_xticklabels(qr["qtr"].values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cumulative Return (Long-Short)")
    ax.set_title("Cumulative Long-Short Portfolio Returns by Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cumulative_returns.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(results):
    """Plot confusion matrices side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, res in zip(axes, results):
        sns.heatmap(
            res["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["-1", "0", "+1"], yticklabels=["-1", "0", "+1"],
            ax=ax,
        )
        ax.set_title(res["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def _plot_summary_bars(summary):
    """Bar chart comparing the three metrics across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["Accuracy", "Total Profit (Long-Short)", "Sharpe Ratio (Annualized)"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(summary["Model"], summary[metric], color=color, alpha=0.8)
        ax.set_title(metric)
        ax.set_xticklabels(summary["Model"], rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metric_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_path)
