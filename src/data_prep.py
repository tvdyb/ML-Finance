"""
Data preparation: load panel data, compute quarterly S&P-relative labels,
and produce rolling window train/test splits (matching course methodology).
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "final_master_panel_large_caps.csv"

FEATURE_COLS = [
    "CAPEI", "bm", "evm", "pe_op_basic", "pe_op_dil", "pe_exi", "pe_inc",
    "ps", "pcf", "dpr", "npm", "opmbd", "opmad", "gpm", "ptpm",
    "roa", "roe", "roce", "efftax", "aftret_eq", "aftret_invcapx",
    "aftret_equity", "pretret_noa", "pretret_earnat", "GProf",
    "debt_at", "debt_assets", "debt_capital", "de_ratio",
    "intcov", "intcov_ratio", "cash_ratio", "quick_ratio", "curr_ratio",
    "cash_conversion", "inv_turn", "at_turn", "rect_turn", "pay_turn",
    "sale_invcap", "sale_equity", "sale_nwc", "ptb", "PEG_trailing", "divyield",
]


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load raw CSV and parse dates."""
    path = path or DATA_PATH
    df = pd.read_csv(path, parse_dates=["adate", "qdate", "public_date"])
    return df


def compute_quarterly_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly next_month_ret into a quarterly compounded return
    per stock. Quarters are defined by public_date grouped into calendar
    quarters (Q1=Jan-Mar, Q2=Apr-Jun, etc.).
    """
    df = df.copy()
    df["year"] = df["public_date"].dt.year
    df["quarter"] = df["public_date"].dt.quarter
    df["yq"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)

    def compound(rets):
        return (1 + rets).prod() - 1

    quarterly = (
        df.groupby(["permno", "yq", "year", "quarter"])
        .agg(
            qtr_ret=("next_month_ret", compound),
            public_date_first=("public_date", "first"),
            **{col: (col, "last") for col in FEATURE_COLS},
        )
        .reset_index()
    )
    return quarterly


def compute_sp_benchmark(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quarterly S&P benchmark as the cross-sectional mean return
    of all stocks in each quarter (equal-weighted proxy).
    """
    sp_bench = quarterly.groupby("yq")["qtr_ret"].mean().rename("sp_ret")
    quarterly = quarterly.merge(sp_bench, on="yq")
    return quarterly


def assign_labels(quarterly: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    Classify each stock-quarter:
      -1 : misses S&P by > threshold (relative return < -threshold)
       0 : within [-threshold, +threshold] of S&P
      +1 : beats S&P by > threshold
    """
    quarterly = quarterly.copy()
    quarterly["excess_ret"] = quarterly["qtr_ret"] - quarterly["sp_ret"]
    quarterly["label"] = 0
    quarterly.loc[quarterly["excess_ret"] > threshold, "label"] = 1
    quarterly.loc[quarterly["excess_ret"] < -threshold, "label"] = -1
    return quarterly


def prepare_dataset(path: str | Path | None = None, threshold: float = 0.02):
    """Full pipeline: load -> quarterly aggregation -> labels -> clean features."""
    df = load_data(path)
    quarterly = compute_quarterly_return(df)
    quarterly = compute_sp_benchmark(quarterly)
    quarterly = assign_labels(quarterly, threshold)

    # Replace infinities then drop rows with missing features
    quarterly[FEATURE_COLS] = quarterly[FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    )
    quarterly = quarterly.dropna(subset=FEATURE_COLS)
    quarterly = quarterly.fillna(0)

    # Sort by time for rolling window
    quarterly = quarterly.sort_values("public_date_first").reset_index(drop=True)

    return quarterly


def get_sorted_quarters(quarterly: pd.DataFrame) -> list:
    """Return sorted list of unique quarter labels."""
    qtr_dates = quarterly.groupby("yq")["public_date_first"].first().sort_values()
    return list(qtr_dates.index)


def rolling_window_splits(quarterly: pd.DataFrame, train_quarters: int = 20,
                          embargo: int = 1):
    """
    Generate rolling window train/test splits:
    - Train on `train_quarters` quarters
    - Skip `embargo` quarters (default 1) to prevent information leakage
      from trailing features (e.g. trailing P/E, rolling averages)
    - Test on the next quarter after the embargo gap
    - Slide forward by 1 quarter and repeat

    Yields (train_df, test_df) tuples.
    """
    quarters = get_sorted_quarters(quarterly)

    for i in range(train_quarters + embargo, len(quarters)):
        train_qtrs = quarters[i - train_quarters - embargo:i - embargo]
        test_qtr = quarters[i]

        train_df = quarterly[quarterly["yq"].isin(train_qtrs)]
        test_df = quarterly[quarterly["yq"] == test_qtr]

        if len(train_df) > 0 and len(test_df) > 0:
            yield train_df, test_df
