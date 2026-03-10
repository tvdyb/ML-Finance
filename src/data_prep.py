"""
Data preparation: load panel data, compute quarterly S&P-relative labels,
and produce train/test splits respecting time ordering.
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

    # Compound monthly returns within each stock-quarter
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
    """Full pipeline: load → quarterly aggregation → labels → clean features."""
    df = load_data(path)
    quarterly = compute_quarterly_return(df)
    quarterly = compute_sp_benchmark(quarterly)
    quarterly = assign_labels(quarterly, threshold)

    # Drop rows with missing features
    quarterly = quarterly.dropna(subset=FEATURE_COLS)

    # Replace infinities
    quarterly[FEATURE_COLS] = quarterly[FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    )
    quarterly = quarterly.dropna(subset=FEATURE_COLS)

    return quarterly


def time_split(quarterly: pd.DataFrame, test_start_year: int = 2020):
    """
    Split into train/test respecting time: everything before test_start_year
    is training, the rest is test.
    """
    train = quarterly[quarterly["year"] < test_start_year].copy()
    test = quarterly[quarterly["year"] >= test_start_year].copy()
    return train, test
