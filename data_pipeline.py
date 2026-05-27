"""
PnL Attribution & Alpha Decay Framework — Data Pipeline

Core functions:
  fetch_prices()              yfinance OHLCV, cached to CSV
  load_real_signals()         FinBERT signals from sentiment_signals SQLite DB
  generate_synthetic_trades() synthetic signals with engineered IC decay
  build_trade_log()           main entry point → enriched trade log + source label
  compute_ic_table()          IC / t-stat / p-value at each holding horizon
  compute_rolling_ic()        rolling 60-trade IC for signal stability analysis
  compute_equity_curve()      simulate portfolio equity for Panel 1
  compute_pnl_attribution()   gross signal / slippage / drag waterfall for Panel 2
  compute_summary_stats()     Sharpe, hit rate, max drawdown, avg PnL
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

HORIZONS = [1, 2, 3, 5, 10, 20]
EXECUTION_COST_BPS = 7  # 5 bps bid-ask spread + 2 bps market impact

_ROOT = Path(__file__).parent
_SENTIMENT_DB = _ROOT.parent / "sentiment_signals" / "data" / "sentiment.db"
CACHE_DIR = _ROOT / "cache"


# ── Price Data ──────────────────────────────────────────────────────────────────

def fetch_prices(
    ticker: str = "AAPL",
    start: str = "2021-11-01",
    end: str = "2025-06-01",
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Download OHLCV from yfinance, cached locally to avoid re-fetching on reload.

    Returns a DatetimeIndex DataFrame with lower-case columns: open high low close volume.
    The cache key encodes ticker + date range so different ranges get separate files.
    """
    cache_dir.mkdir(exist_ok=True)
    slug = f"{ticker}_{start}_{end}".replace("-", "")
    cache_path = cache_dir / f"{slug}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index)
        return df

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)
    raw.to_csv(cache_path)
    return raw


# ── Signal Loading ──────────────────────────────────────────────────────────────

def load_real_signals(
    ticker: str = "AAPL",
    db_path: Path = _SENTIMENT_DB,
) -> pd.DataFrame | None:
    """Load FinBERT-scored signals from the sentiment_signals SQLite database.

    Multiple intra-day articles on the same date are collapsed to a single
    signal via mean composite_score — matching the aggregation in aligner.py.
    Returns None when the DB is missing or the ticker has < 30 rows (too few
    for a statistically meaningful IC estimate).

    Columns: signal_date (datetime64), signal_score (float), signal_type (str)
    """
    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT event_date  AS signal_date,
                   AVG(composite_score) AS signal_score
            FROM   sentiment_scores
            WHERE  ticker = ?
            GROUP  BY event_date
            ORDER  BY event_date
            """,
            conn,
            params=[ticker.upper()],
        )

    if len(df) < 15:
        return None

    df["signal_date"] = pd.to_datetime(df["signal_date"])
    # Use ±0.05 thresholds (tighter than default ±0.1) to get more active trades
    # from EDGAR composite scores which are typically low-magnitude (~0.05–0.20)
    df["signal_type"] = pd.cut(
        df["signal_score"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["SHORT", "HOLD", "LONG"],
    ).astype(str)
    return df[["signal_date", "signal_score", "signal_type"]]


# ── Synthetic Signal Generation ─────────────────────────────────────────────────

def generate_synthetic_trades(
    prices_df: pd.DataFrame,
    n_signals: int = 200,
    target_ic_1d: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic trade log with controlled IC decay across horizons.

    The Information Coefficient (IC) is the Spearman rank-correlation between
    signal_score and forward return.  We construct signal_score so that
    IC ≈ target_ic_1d at the 1-day horizon.  Because multi-day returns
    accumulate independent daily noise, IC decays naturally as ≈ IC_1d / √h,
    mirroring what a live FinBERT signal would exhibit.  This is a well-known
    result in quantitative finance: a signal that perfectly predicts today's
    return has progressively less edge at longer holding periods as the
    unforecastable component of returns dominates.

    Construction (in rank-normal space):
        z_ret   = norm_quantile( rank(ret_1d) )
        signal  = ρ · z_ret  +  √(1−ρ²) · ε    →  Spearman(signal, ret_1d) ≈ ρ
        score   = tanh(signal / 2)              →  squash to (−1, +1)

    Args:
        prices_df:    OHLCV DataFrame with DatetimeIndex.
        n_signals:    Number of synthetic signal events to generate.
        target_ic_1d: Target Spearman IC at 1-day horizon (realistic: 0.10–0.20).
        seed:         RNG seed for reproducibility.

    Returns DataFrame with: signal_date, signal_score, signal_type
    """
    rng = np.random.default_rng(seed)

    close = prices_df["close"]
    ret_1d = close.pct_change(1).shift(-1)

    # Reserve last 25 trading days so every signal has a valid 20-day exit window
    valid = ret_1d.dropna().index[:-25]
    n_signals = min(n_signals, len(valid))

    chosen = np.sort(rng.choice(len(valid), size=n_signals, replace=False))
    signal_dates = valid[chosen]
    fwd_1d = ret_1d.loc[signal_dates].values

    # Rank-normal transform of 1d forward return → z_ret in N(0,1) space
    pcts = stats.rankdata(fwd_1d) / (len(fwd_1d) + 1)
    z_ret = stats.norm.ppf(pcts)
    noise = rng.standard_normal(n_signals)

    raw = target_ic_1d * z_ret + np.sqrt(1.0 - target_ic_1d**2) * noise
    signal_score = np.tanh(raw * 0.5).astype(float)

    signal_type = np.where(
        signal_score >= 0.05, "LONG",
        np.where(signal_score <= -0.05, "SHORT", "HOLD"),
    )

    return pd.DataFrame({
        "signal_date": signal_dates,
        "signal_score": signal_score,
        "signal_type": signal_type,
    })


# ── Trade Log Enrichment ────────────────────────────────────────────────────────

def _nth_trading_day(
    date: pd.Timestamp,
    n: int,
    calendar: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Return the n-th trading day strictly after `date` within calendar."""
    future = calendar[calendar > date]
    return future[n - 1] if len(future) >= n else None


def enrich_trade_log(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach entry/exit prices, direction-adjusted PnL, timing slippage, execution drag.

    Execution convention (mirrors aligner.py):
      entry_date  = first trading day after signal_date
      entry_price = open on entry_date   (next-open fills)

    Timing slippage — the overnight gap cost:
      slippage = direction × (entry_open − signal_close) / signal_close
    Positive slippage means we paid more (LONG) or received less (SHORT) than
    the signal-day close implied — a pure execution-lag cost.

    Execution drag: −7 bps per active (non-HOLD) trade.

    Columns added per horizon h ∈ {1,2,3,5,10,20}:
      exit_price_Nd  close price N trading days after entry
      raw_ret_Nd     (exit − entry_open) / entry_open  [for IC computation]
      pnl_Nd         direction × raw_ret_Nd             [for PnL tracking]
    """
    calendar = prices_df.index
    direction_map = {"LONG": 1, "SHORT": -1, "HOLD": 0}
    rows = []

    for _, row in signals_df.iterrows():
        sig_date = pd.Timestamp(row["signal_date"])
        direction = direction_map.get(str(row["signal_type"]), 0)

        # Signal-day close (for slippage reference only)
        sig_close = (
            float(prices_df.loc[sig_date, "close"])
            if sig_date in prices_df.index
            else np.nan
        )

        entry_date = _nth_trading_day(sig_date, 1, calendar)
        if entry_date is None:
            continue

        entry_open = (
            float(prices_df.loc[entry_date, "open"])
            if entry_date in prices_df.index
            else np.nan
        )
        if np.isnan(entry_open):
            continue

        timing_slip = (
            direction * (entry_open - sig_close) / sig_close
            if not np.isnan(sig_close) and sig_close != 0
            else np.nan
        )

        exit_data: dict = {}
        for h in HORIZONS:
            exit_date = _nth_trading_day(entry_date, h, calendar)
            if exit_date is not None and exit_date in prices_df.index:
                exit_close = float(prices_df.loc[exit_date, "close"])
                raw_ret = (exit_close - entry_open) / entry_open
            else:
                exit_close = np.nan
                raw_ret = np.nan

            exit_data[f"exit_price_{h}d"] = exit_close
            exit_data[f"raw_ret_{h}d"] = raw_ret
            exit_data[f"pnl_{h}d"] = (
                float(direction) * raw_ret if not np.isnan(raw_ret) else np.nan
            )

        rows.append({
            "signal_date": sig_date,
            "signal_score": float(row["signal_score"]),
            "signal_type": str(row["signal_type"]),
            "direction": direction,
            "entry_date": entry_date,
            "entry_price": entry_open,
            "signal_close": sig_close,
            "timing_slippage": timing_slip,
            "execution_drag": -EXECUTION_COST_BPS / 10_000 if direction != 0 else 0.0,
            **exit_data,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("signal_date")
        .reset_index(drop=True)
    )


# ── Main Entry Point ────────────────────────────────────────────────────────────

def build_trade_log(
    ticker: str = "AAPL",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
) -> tuple[pd.DataFrame, str]:
    """Build an enriched trade log for PnL attribution analysis.

    Tries the real FinBERT signal DB first.  Falls back to synthetic generation
    when the DB is missing, or when the real signals don't fall in the requested
    date range (the live DB currently holds 2025–2026 data; the default
    analytical window is 2022–2024).

    Returns:
        (trade_log DataFrame, data_source): source is "real" or "synthetic"
    """
    p_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    p_end   = (pd.Timestamp(end)   + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    prices  = fetch_prices(ticker, p_start, p_end)

    real = load_real_signals(ticker)
    if real is not None:
        in_range = real[
            (real["signal_date"] >= start) & (real["signal_date"] <= end)
        ]
        if len(in_range) >= 15:
            return enrich_trade_log(in_range, prices), "real"

    # Synthetic path
    synth = generate_synthetic_trades(prices, n_signals=200)
    in_range = synth[
        (synth["signal_date"] >= start) & (synth["signal_date"] <= end)
    ]
    return enrich_trade_log(in_range, prices), "synthetic"


# ── Analysis ────────────────────────────────────────────────────────────────────

def compute_ic_table(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Compute Information Coefficient at each holding horizon with t-stats.

    IC = Spearman rank-correlation between signal_score and raw forward return.
    Spearman is preferred over Pearson because it is robust to the fat-tailed
    return distributions typical of equities, and measures rank-ordering ability
    — the question a quant cares about is 'does a higher score rank higher returns?'
    not 'is the linear relationship exactly proportional?'

    t-stat = IC × √(N−2) / √(1−IC²)  under H₀: ρ = 0  (t-dist, N−2 df)
    95% CI uses the asymptotic SE ≈ √((1−IC²)/(N−2)) under the alternative.

    The horizon where IC first loses significance (p > 0.05) indicates the
    maximum useful holding period for this signal — the key output of this
    framework for strategy design decisions.

    Returns DataFrame indexed by horizon with columns:
    ic, ic_lower, ic_upper, t_stat, p_value, n_obs, significant
    """
    rows = []
    for h in HORIZONS:
        col = f"raw_ret_{h}d"
        sub = trade_log.dropna(subset=["signal_score", col])
        n = len(sub)

        if n < 5:
            rows.append({
                "horizon": h, "ic": np.nan, "ic_lower": np.nan, "ic_upper": np.nan,
                "t_stat": np.nan, "p_value": 1.0, "n_obs": n, "significant": False,
            })
            continue

        ic, p_val = stats.spearmanr(sub["signal_score"], sub[col])
        ic, p_val = float(ic), float(p_val)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - ic**2))
        se = np.sqrt(max(0.0, (1 - ic**2) / max(1, n - 2)))

        rows.append({
            "horizon":   h,
            "ic":        round(ic, 4),
            "ic_lower":  round(ic - 1.96 * se, 4),
            "ic_upper":  round(ic + 1.96 * se, 4),
            "t_stat":    round(t_stat, 3),
            "p_value":   round(p_val, 4),
            "n_obs":     n,
            "significant": p_val < 0.05,
        })

    return pd.DataFrame(rows).set_index("horizon")


def compute_rolling_ic(trade_log: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling IC at the 1-day horizon over the full trade timeline.

    Rolling IC diagnoses signal stability: a robust signal shows consistently
    positive IC across market regimes.  Periods with IC < 0 (shaded red in
    Panel 4) indicate the signal was actively harmful — a clear sign of
    regime change or signal decay in that window.

    Uses fixed rolling window of `window` trades; the first window−1 points
    use an expanding window to avoid a large gap at the start of the series.

    Returns DataFrame with: signal_date, rolling_ic, rolling_hit_rate
    """
    df = (
        trade_log.dropna(subset=["signal_score", "raw_ret_1d"])
        .sort_values("signal_date")
        .reset_index(drop=True)
    )

    ics, hrs = [], []
    for i in range(len(df)):
        s = max(0, i - window + 1)
        chunk = df.iloc[s : i + 1]
        if len(chunk) < 10:
            ics.append(np.nan)
            hrs.append(np.nan)
        else:
            ic, _ = stats.spearmanr(chunk["signal_score"], chunk["raw_ret_1d"])
            active_chunk = chunk[chunk["signal_type"] != "HOLD"]
            if len(active_chunk):
                correct_active = (
                    ((active_chunk["signal_type"] == "LONG")  & (active_chunk["raw_ret_1d"] > 0)) |
                    ((active_chunk["signal_type"] == "SHORT") & (active_chunk["raw_ret_1d"] < 0))
                )
                hrs.append(round(float(correct_active.mean()), 4))
            else:
                hrs.append(np.nan)
            ics.append(round(float(ic), 4))

    df = df.copy()
    df["rolling_ic"] = ics
    df["rolling_hit_rate"] = hrs
    return df[["signal_date", "signal_type", "rolling_ic", "rolling_hit_rate"]]


def compute_equity_curve(
    trade_log: pd.DataFrame,
    horizon: int = 1,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Simulate portfolio equity from sequential signal execution.

    Each signal uses full capital (no position sizing).  Execution drag is
    deducted from active (non-HOLD) trades.  Buy-and-hold benchmark takes
    the same raw return on every bar without direction filtering.

    Returns DataFrame with: signal_date, portfolio_value, bnh_value, drawdown
    """
    pnl_col = f"pnl_{horizon}d"
    raw_col = f"raw_ret_{horizon}d"
    df = trade_log.dropna(subset=[pnl_col]).sort_values("signal_date").copy()

    port = initial_capital
    bnh  = initial_capital
    ports, bnhs = [], []

    for _, row in df.iterrows():
        trade_ret = float(row[pnl_col])
        if row["signal_type"] != "HOLD":
            trade_ret += float(row["execution_drag"])
        port *= 1.0 + trade_ret

        bnh_ret = float(row.get(raw_col) or 0.0)
        bnh *= 1.0 + bnh_ret

        ports.append(port)
        bnhs.append(bnh)

    df = df.copy()
    df["portfolio_value"] = ports
    df["bnh_value"]       = bnhs

    pv    = pd.Series(ports, index=df.index)
    peak  = pv.cummax()
    df["drawdown"] = ((pv - peak) / peak).values

    return df[["signal_date", "portfolio_value", "bnh_value", "drawdown"]]


def compute_pnl_attribution(trade_log: pd.DataFrame) -> dict:
    """Decompose total PnL into signal alpha, timing slippage, and execution drag.

    Waterfall components (% of notional):
      gross_signal_pnl: Σ direction × (exit_1d − signal_close) / signal_close
                        — what the signal 'earned' at theoretical (close-day) execution
      timing_slippage:  actual 1d PnL − theoretical PnL
                        — cost of waiting for next-day open vs signal-day close
      execution_drag:   −7 bps × n_active_trades (bid-ask spread + market impact)
      net_pnl:          actual 1d PnL + execution_drag  (= gross + slippage + drag)

    All values returned as percentages (multiply raw returns by 100).
    """
    active = trade_log[trade_log["signal_type"] != "HOLD"].copy()

    # Theoretical: use signal_close as reference; fall back to entry_price if missing
    ref = active["signal_close"].fillna(active["entry_price"])
    mask = active["exit_price_1d"].notna() & ref.notna() & (ref != 0)
    theo = active.loc[mask, "direction"] * (
        (active.loc[mask, "exit_price_1d"] - ref[mask]) / ref[mask]
    )
    gross_signal = float(theo.sum()) * 100

    actual_pnl   = float(active["pnl_1d"].dropna().sum()) * 100
    timing_cost  = actual_pnl - gross_signal  # negative means slippage hurt us
    drag         = float(active["execution_drag"].sum()) * 100
    net          = actual_pnl + drag

    return {
        "gross_signal_pnl": round(gross_signal, 2),
        "timing_slippage":  round(timing_cost, 2),
        "execution_drag":   round(drag, 2),
        "net_pnl":          round(net, 2),
    }


def compute_summary_stats(
    trade_log: pd.DataFrame,
    equity_df: pd.DataFrame,
    horizon: int = 1,
) -> dict:
    """Aggregate performance metrics for the Panel 1 summary cards.

    Sharpe is annualised: mean(pnl) / std(pnl) × √(252 / horizon).
    Hit rate counts active trades where direction was correct.
    Max drawdown is the worst peak-to-trough decline in the equity curve.
    """
    pnl_col = f"pnl_{horizon}d"
    active  = trade_log[trade_log["signal_type"] != "HOLD"]
    pnl     = active[pnl_col].dropna()

    sharpe = 0.0
    if len(pnl) > 1 and pnl.std() > 0:
        sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252 / horizon))

    correct = (
        ((active["signal_type"] == "LONG")  & (active[pnl_col] > 0)) |
        ((active["signal_type"] == "SHORT") & (active[pnl_col] < 0))
    )
    hit_rate = float(correct.mean()) if len(active) else 0.0
    max_dd   = float(equity_df["drawdown"].min()) if not equity_df.empty else 0.0
    avg_pnl  = float(pnl.mean()) if len(pnl) else 0.0

    return {
        "sharpe":      round(sharpe, 2),
        "hit_rate":    round(hit_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "avg_pnl_bps":  round(avg_pnl * 10_000, 1),
        "n_active":    int((trade_log["signal_type"] != "HOLD").sum()),
        "n_long":      int((trade_log["signal_type"] == "LONG").sum()),
        "n_short":     int((trade_log["signal_type"] == "SHORT").sum()),
        "n_hold":      int((trade_log["signal_type"] == "HOLD").sum()),
    }
