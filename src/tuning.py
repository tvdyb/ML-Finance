"""
Per-window hyperparameter tuning using expanding-window temporal CV.

CV scheme (given 20 training quarters, min_train=12):
  Fold 1: train on quarters 1-12, validate on quarter 13
  Fold 2: train on quarters 1-13, validate on quarter 14
  ...
  Fold 8: train on quarters 1-19, validate on quarter 20

Each model type uses a different objective metric:
  AccuracyModel: classification accuracy
  ProfitModel:   realized profit = (preds * excess_ret).sum()
  SharpeModel:   mean(fold_profits) / std(fold_profits) across all folds
"""

import numpy as np
from data_prep import FEATURE_COLS
from models import AccuracyModel, ProfitModel, SharpeModel


C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

MIN_TRAIN_QUARTERS = 12


def _fit_model(model_class, kwargs, train_df):
    """Instantiate and fit a model on a training DataFrame."""
    X = train_df[FEATURE_COLS].values
    y = train_df["label"].values
    model = model_class(**kwargs)

    if isinstance(model, ProfitModel):
        model.fit(X, y, excess_ret=train_df["excess_ret"].values)
    elif isinstance(model, SharpeModel):
        model.fit(
            X, y,
            excess_ret=train_df["excess_ret"].values,
            qtr_labels=train_df["yq"].values,
        )
    else:
        model.fit(X, y)
    return model


def _fold_score(model, val_df):
    """
    Compute the per-fold score for a model on a validation quarter.
    Returns (metric_value, profit_value).
    metric_value is what gets used for AccuracyModel/ProfitModel selection.
    profit_value is the raw profit, collected across folds for SharpeModel.
    """
    X_val = val_df[FEATURE_COLS].values
    preds = model.predict(X_val)
    excess = val_df["excess_ret"].values
    profit = (preds * excess).sum()

    if isinstance(model, AccuracyModel):
        accuracy = (preds == val_df["label"].values).mean()
        return accuracy, profit
    else:
        # ProfitModel and SharpeModel both use profit per fold
        return profit, profit


def _cv_score(model_class, param_name, param_value, train_df, sorted_qtrs):
    """
    Run expanding-window CV for a single hyperparameter value.
    Returns a scalar score (higher is better).
    """
    kwargs = {param_name: param_value}
    fold_metrics = []
    fold_profits = []

    for i in range(MIN_TRAIN_QUARTERS, len(sorted_qtrs)):
        fold_train_qtrs = set(sorted_qtrs[:i])
        fold_val_qtr = sorted_qtrs[i]

        fold_train_df = train_df[train_df["yq"].isin(fold_train_qtrs)]
        fold_val_df = train_df[train_df["yq"] == fold_val_qtr]

        if len(fold_train_df) == 0 or len(fold_val_df) == 0:
            continue

        model = _fit_model(model_class, kwargs, fold_train_df)
        metric, profit = _fold_score(model, fold_val_df)
        fold_metrics.append(metric)
        fold_profits.append(profit)

    if len(fold_metrics) == 0:
        return -np.inf

    if issubclass(model_class, SharpeModel):
        # Sharpe proxy: mean / std of per-fold profits
        profits = np.array(fold_profits)
        return profits.mean() / (profits.std() + 1e-8)
    else:
        # AccuracyModel: average accuracy; ProfitModel: average profit
        return np.mean(fold_metrics)


def tune_and_fit(model_class, train_df):
    """
    Tune the regularization hyperparameter via expanding-window temporal CV,
    then fit the final model on the full training data with the best param.

    Returns (fitted_model, best_param_value, param_name).
    """
    if issubclass(model_class, SharpeModel):
        param_name, grid = "alpha", ALPHA_GRID
    else:
        param_name, grid = "C", C_GRID

    sorted_qtrs = sorted(train_df["yq"].unique())

    # Grid search over CV
    best_score = -np.inf
    best_param = grid[0]
    for param_value in grid:
        score = _cv_score(model_class, param_name, param_value,
                          train_df, sorted_qtrs)
        if score > best_score:
            best_score = score
            best_param = param_value

    # Fit final model on full training data with best param
    best_kwargs = {param_name: best_param}
    model = _fit_model(model_class, best_kwargs, train_df)

    return model, best_param, param_name
