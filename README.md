# PnL Attribution & Alpha Decay Framework

> **Strategy forensics for the FinBERT sentiment signal on AAPL.**  
> Decomposes where a signal's edge comes from and how fast it decays.

## Quick start

```bash
git clone https://github.com/maanitmehta/pnl-attribution.git
cd pnl-attribution
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:8050** in your browser.

---

## What it does

The dashboard loads a trade log for AAPL — either **synthetic** (2022–2024, 200 signals
with engineered IC decay) or **real FinBERT signals** from the `sentiment_signals` SQLite
database (2025–2026, 29 signals).  Select the dataset with the radio buttons at the top.

### Panel 1 — Strategy Overview
Simulates a portfolio that acts on each FinBERT signal at a 1-day holding period
(full capital re-investment, no leverage).  Compares against a buy-and-hold baseline.

| Metric | Plain-English meaning |
|---|---|
| **Sharpe Ratio** | Risk-adjusted return: higher is better.  Annualised as mean(trade PnL) / std(trade PnL) × √252. |
| **Hit Rate** | Fraction of active trades where the signal predicted the correct direction. |
| **Max Drawdown** | Worst peak-to-trough loss in the equity curve — a measure of tail risk. |
| **Avg PnL / Trade** | Mean return per trade in basis points (1 bps = 0.01%). |

### Panel 2 — PnL Attribution Waterfall
Breaks total PnL into three components:

| Component | What it measures |
|---|---|
| **Gross Signal Alpha** | What the signal would have earned if you could execute at the signal-day *close* price — the pure predictive edge. |
| **Timing Slippage** | The gap between next-day *open* (actual fill) and signal-day close (theoretical fill).  Overnight moves and bid-ask crossings create this cost. |
| **Execution Drag** | Fixed cost of 7 bps per trade (5 bps bid-ask spread + 2 bps market impact). |
| **Net PnL** | Gross Alpha + Timing Slippage + Execution Drag. |

The scatter plot shows raw signal score vs forward return at the chosen horizon,
with an OLS trend line (slope ≈ IC × volatility scaling).

### Panel 3 — Alpha Decay Curve *(the key chart)*
Plots the **Information Coefficient (IC)** at each holding horizon: 1, 2, 3, 5, 10, 20 days.

**IC** = Spearman rank-correlation between `signal_score` and forward return.

- IC = **1.0** → perfect ranking (every strong-positive score is a winner)
- IC = **0.0** → no predictive power
- IC = **−1.0** → perfectly contrarian

We use Spearman (rank-based) rather than Pearson because equity returns are fat-tailed
and rank ordering is what a portfolio manager actually cares about.

The **95% confidence bands** use the asymptotic standard error √((1−IC²)/(N−2)).
The red dashed line marks the first horizon where IC drops below statistical significance
(p > 0.05) — this is the **maximum useful rebalancing frequency** for this signal.

Green nodes = statistically significant.  Red nodes = not significant.

### Panel 4 — Rolling Signal Quality
Rolling IC computed over a sliding window of trades (default: 60).

- **Green shading** = IC > 0 (signal is adding value in this window)
- **Red shading** = IC < 0 (signal is actively harmful — regime change alert)
- **Amber dashed line** = rolling hit rate (50% = coin flip)

Use the window slider to control sensitivity.  A narrow window (20 trades) shows
short-term IC volatility; a wide window (100 trades) reveals structural decay.

---

## Project layout

```
pnl_attribution/
├── app.py             Dash app — layout + callbacks
├── data_pipeline.py   Data loading, IC computation, PnL decomposition
├── cache/             yfinance price CSVs (auto-created, gitignored)
└── README.md
```

### Key functions in `data_pipeline.py`

| Function | What it does |
|---|---|
| `fetch_prices()` | yfinance OHLCV download, cached to `cache/` |
| `load_real_signals()` | Reads FinBERT signals from `sentiment_signals/data/sentiment.db` |
| `generate_synthetic_trades()` | Generates signals with target IC ≈ 0.18 at 1d, natural √-decay at longer horizons |
| `build_trade_log()` | Main entry: tries real, falls back to synthetic |
| `compute_ic_table()` | IC + t-stat + p-value at each horizon |
| `compute_rolling_ic()` | Rolling 60-trade IC for Panel 4 |
| `compute_pnl_attribution()` | Gross / slippage / drag waterfall decomposition |

---

## How the synthetic IC decay works

The synthetic signal is constructed in rank-normal space:

```
z_ret  = Φ⁻¹( rank(ret_1d) / (N+1) )          # rank-normalise 1d return
signal = ρ · z_ret  +  √(1−ρ²) · ε             # blend with noise at target IC ρ
score  = tanh(signal / 2)                        # squash to (−1, +1)
```

This gives IC ≈ ρ at the 1-day horizon.  At longer horizons, because
multi-day returns accumulate independent daily noise,
IC decays as **IC_h ≈ IC_1 / √h** — a classical result in quantitative finance.
The decay crosses statistical significance at around 3–5 days for a realistic
IC of 0.18, which is the critical insight the Keyrock alpha decay chart captures.

---

*Built as a quant research portfolio piece — demonstrates PnL attribution,
IC analysis, and signal forensics methodology used in systematic trading.*
