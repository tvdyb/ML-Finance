# ML-Finance: Stock Classification with Linear Models

Quarterly stock classification relative to the S&P 500 using three linear models, each optimizing a different objective. Uses a rolling window train/test methodology.

## Classification Labels

Each stock-quarter is labeled based on its excess return over the S&P 500 (proxied by the equal-weighted cross-sectional mean):

| Label | Condition |
|-------|-----------|
| **+1** (Beat) | Excess return > +2% |
| **0** (Neutral) | Excess return between -2% and +2% |
| **-1** (Miss) | Excess return < -2% |

## Rolling Window Strategy

Following the course methodology:
- **Training window**: 20 quarters (5 years)
- **Test window**: the next quarter
- The model is **retrained each quarter** with a fresh scaler
- Slide forward by 1 quarter and repeat

Portfolio value is tracked as:
```
profit_i = (predictions * stock_returns).sum()
x[i+1] = x[i] + (x[i] / num_stocks) * profit_i
```

## Models

### 1. Accuracy-Optimized (Logistic Regression)
Standard multinomial logistic regression with balanced class weights. Maximizes classification accuracy via cross-entropy loss.

### 2. Profit-Optimized (Cost-Sensitive LR)
Logistic regression with sample weights proportional to |excess return|. Stocks with larger moves are weighted more heavily, tilting the model toward profit-relevant predictions.

### 3. Sharpe-Optimized (Ridge + Threshold Tuning)
Two-stage model: Ridge regression predicts excess returns, then classification thresholds are optimized via Nelder-Mead to maximize the realized Sharpe ratio of a long-short portfolio on training data.

## Data

Panel of ~5,000 large-cap US stocks with 45 fundamental/valuation features (2012-2022). Monthly data is compounded to quarterly returns.

**Features include:** valuation ratios (P/E, P/B, P/S, P/CF), profitability (ROA, ROE, margins), leverage (debt ratios, interest coverage), liquidity (current/quick ratio), and efficiency (asset/inventory turnover).

Place `final_master_panel_large_caps.csv` in `data/`.

## Usage

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

Results (summary CSV + plots) are saved to `results/`.

## Project Structure

```
├── data/                  # CSV data (git-ignored)
├── src/
│   ├── data_prep.py       # Loading, quarterly aggregation, labeling, rolling splits
│   ├── models.py          # Three linear model classes
│   ├── evaluation.py      # Portfolio tracking, Sharpe, accuracy
│   └── run_pipeline.py    # Main entry point (rolling window strategy)
├── results/               # Output plots and summary (git-ignored)
├── requirements.txt
└── README.md
```
