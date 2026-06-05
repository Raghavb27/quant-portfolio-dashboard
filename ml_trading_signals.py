"""
╔══════════════════════════════════════════════════════════════════╗
║         ML Trading Signal Generator  ·  Educational Edition      ║
╠══════════════════════════════════════════════════════════════════╣
║  Pipeline:  OHLCV data  →  Feature engineering  →  ML models    ║
║             Logistic Regression  &  Random Forest (scikit-learn) ║
║  Backtest:  Walk-forward (expanding window) + transaction costs  ║
╚══════════════════════════════════════════════════════════════════╝

⚠️  DISCLAIMER: For educational purposes only.  Past backtest
    performance does not guarantee future results.  Real trading
    requires risk management, live data, and regulatory compliance.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


# ══════════════════════════════════════════════════════════════════
# 1 ·  SYNTHETIC DATA  (Geometric Brownian Motion + realistic noise)
# ══════════════════════════════════════════════════════════════════

def generate_ohlcv(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data using a GBM close series.
    High/Low/Open are derived from the close with intraday randomness.
    Volume is correlated with absolute daily return (realistic).
    """
    rng = np.random.default_rng(seed)

    # --- Close: Geometric Brownian Motion ----------------------------
    mu    = 0.00025   # daily drift   ≈ +6 % annualised
    sigma = 0.015     # daily vol     ≈ 24 % annualised
    S0    = 100.0

    log_ret = rng.normal((mu - 0.5 * sigma ** 2), sigma, n_days)
    close   = S0 * np.exp(np.cumsum(log_ret))

    # --- OHLC from close --------------------------------------------
    ir    = np.abs(rng.normal(0, sigma * 0.6, n_days)) * close
    high  = close + ir * rng.uniform(0.3, 1.0, n_days)
    low   = close - ir * rng.uniform(0.3, 1.0, n_days)
    open_ = close * np.exp(rng.normal(0, sigma * 0.35, n_days))

    # --- Volume: higher on big moves --------------------------------
    volume = (1_000_000
              * (1 + 4 * np.abs(log_ret))
              * rng.lognormal(0, 0.3, n_days)).astype(int)

    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": volume},
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════
# 2 ·  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
#
#  Group A — Momentum  : price rate-of-change at multiple horizons,
#                        RSI, MACD histogram
#  Group B — Volatility: rolling std of returns, ATR, Bollinger bands
#  Group C — Volume    : volume ratio vs rolling mean, OBV momentum
#  Group D — Micro     : high-low range, open-close body
#
#  Target              : binary next-day direction  (1 = up, 0 = down)

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d   = df.copy()
    c, v = d["close"], d["volume"]
    ret  = c.pct_change()

    # ── A · Momentum ─────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        d[f"mom_{w}"]  = c / c.shift(w) - 1

    d["rsi_14"]    = _rsi(c, 14)
    macd           = _ema(c, 12) - _ema(c, 26)
    d["macd_hist"] = macd - _ema(macd, 9)

    # ── B · Volatility ───────────────────────────────────────────
    for w in [5, 10, 20]:
        d[f"vol_{w}"]  = ret.rolling(w).std()

    tr          = pd.concat([
        d["high"] - d["low"],
        (d["high"] - c.shift()).abs(),
        (d["low"]  - c.shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr_14"] = tr.rolling(14).mean() / c           # normalised ATR

    ma20        = c.rolling(20).mean()
    std20       = c.rolling(20).std()
    d["bb_width"] = (2 * std20) / ma20                # band width
    d["bb_pos"]   = (c - ma20) / (2 * std20 + 1e-9)  # position in band

    # ── C · Volume ───────────────────────────────────────────────
    d["vol_ratio"] = v / v.rolling(20).mean()
    d["obv_mom"]   = _obv(c, v).diff(10) / c          # OBV momentum, normalised

    # ── D · Micro-structure ──────────────────────────────────────
    d["hl_ratio"]  = (d["high"] - d["low"]) / c
    d["co_ratio"]  = (d["close"] - d["open"]) / d["open"]

    # ── Target ───────────────────────────────────────────────────
    d["fwd_ret"] = c.shift(-1) / c - 1
    d["target"]  = (d["fwd_ret"] > 0).astype(int)

    return d


FEATURES = [
    "mom_5", "mom_10", "mom_20", "mom_60",
    "rsi_14", "macd_hist",
    "vol_5", "vol_10", "vol_20", "atr_14", "bb_width", "bb_pos",
    "vol_ratio", "obv_mom",
    "hl_ratio", "co_ratio",
]


# ══════════════════════════════════════════════════════════════════
# 3 ·  WALK-FORWARD BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════
#
#  Train window expands from `train_size` → full history (expanding).
#  Each fold re-trains the model on all available history, then
#  predicts on the next `test_size` days.  This mimics live trading
#  where you periodically re-train as new data arrives.
#
#  Signal mapping:  predicted UP  → +1 (long)
#                   predicted DOWN→ -1 (short)
#  Costs: one-way transaction cost deducted whenever the signal flips.

def walk_forward_backtest(
    df         : pd.DataFrame,
    model_fn,                      # callable → fresh unfitted model
    features   : list,
    train_size : int   = 500,      # initial training window (days)
    test_size  : int   = 63,       # out-of-sample window (~1 quarter)
    cost_bps   : float = 10,       # transaction cost in basis points
) -> dict:
    txn_cost = cost_bps / 10_000
    data     = df[features + ["target", "fwd_ret"]].dropna()

    folds, trade_dfs = [], []
    pos = train_size

    while pos + test_size <= len(data):
        train = data.iloc[:pos]
        test  = data.iloc[pos : pos + test_size]

        model = model_fn()
        model.fit(train[features], train["target"])

        prob   = model.predict_proba(test[features])[:, 1]
        pred   = (prob > 0.5).astype(int)
        signal = pred * 2 - 1                          # +1 / -1

        # Costs applied whenever the signal changes
        signal_change = np.abs(np.diff(signal, prepend=signal[0])) > 0
        raw_ret = signal * test["fwd_ret"].values
        net_ret = raw_ret - txn_cost * signal_change.astype(float)

        td = test.copy()
        td["signal"], td["prob"]    = signal, prob
        td["raw_ret"], td["net_ret"]= raw_ret, net_ret
        td["correct"] = (pred == test["target"].values).astype(int)
        trade_dfs.append(td)

        folds.append({
            "fold"      : len(folds) + 1,
            "train_end" : train.index[-1].strftime("%Y-%m-%d"),
            "test_start": test.index[0].strftime("%Y-%m-%d"),
            "test_end"  : test.index[-1].strftime("%Y-%m-%d"),
            "accuracy"  : accuracy_score(test["target"], pred),
            "auc"       : roc_auc_score(test["target"], prob),
            "gross_ret" : float(raw_ret.sum()),
            "net_ret"   : float(net_ret.sum()),
            "n_signals" : int(signal_change.sum()),
        })
        pos += test_size

    trades = pd.concat(trade_dfs)
    trades["cum_net"] = (1 + trades["net_ret"]).cumprod()
    trades["cum_bh"]  = (1 + trades["fwd_ret"]).cumprod()

    return {"folds": pd.DataFrame(folds), "trades": trades}


# ══════════════════════════════════════════════════════════════════
# 4 ·  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════

def performance_table(result: dict, label: str) -> dict:
    r   = result["trades"]["net_ret"]
    cum = result["trades"]["cum_net"]
    bh  = result["trades"]["cum_bh"]

    total_ret = float(cum.iloc[-1] - 1)
    bh_ret    = float(bh.iloc[-1] - 1)
    n         = len(r)
    ann_ret   = (1 + total_ret) ** (252 / n) - 1
    ann_vol   = r.std() * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol else 0
    dd        = (cum / cum.cummax() - 1)
    max_dd    = float(dd.min())
    calmar    = ann_ret / abs(max_dd) if max_dd else 0
    win_rate  = float((r > 0).mean())
    avg_acc   = result["folds"]["accuracy"].mean()
    avg_auc   = result["folds"]["auc"].mean()

    return {
        "Model"          : label,
        "Total Return"   : f"{total_ret:+.1%}",
        "B&H Return"     : f"{bh_ret:+.1%}",
        "Ann. Return"    : f"{ann_ret:+.1%}",
        "Ann. Volatility": f"{ann_vol:.1%}",
        "Sharpe Ratio"   : f"{sharpe:.2f}",
        "Max Drawdown"   : f"{max_dd:.1%}",
        "Calmar Ratio"   : f"{calmar:.2f}",
        "Day Win Rate"   : f"{win_rate:.1%}",
        "Avg OOS Acc."   : f"{avg_acc:.1%}",
        "Avg OOS AUC"    : f"{avg_auc:.3f}",
    }


# ══════════════════════════════════════════════════════════════════
# 5 ·  DASHBOARD PLOT
# ══════════════════════════════════════════════════════════════════

_C = {
    "bg"    : "#0a0e14",
    "panel" : "#12181f",
    "border": "#1e2832",
    "text"  : "#cdd9e5",
    "dim"   : "#8b949e",
    "lr"    : "#38bdf8",   # sky-blue   – Logistic Regression
    "rf"    : "#fb923c",   # orange     – Random Forest
    "bh"    : "#6b7280",   # grey       – Buy & Hold
    "green" : "#34d399",
    "red"   : "#f87171",
}


def _ax_style(ax, title=""):
    ax.set_facecolor(_C["panel"])
    ax.tick_params(colors=_C["dim"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_C["border"])
    ax.grid(True, color=_C["border"], lw=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=_C["text"], fontsize=9.5,
                     fontweight="bold", pad=7)


def plot_dashboard(results: dict, output_path: str):
    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor(_C["bg"])
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.52, wspace=0.38,
        top=0.93, bottom=0.06, left=0.07, right=0.97,
    )

    # ── Panel 1: Equity curves (full width) ──────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _ax_style(ax1, "Portfolio Equity  ·  Walk-Forward Out-of-Sample")
    for key, col, lbl in [("lr", _C["lr"], "Logistic Regression"),
                           ("rf", _C["rf"], "Random Forest")]:
        t = results[key]["trades"]
        ax1.plot(t.index, t["cum_net"], color=col, lw=1.6, label=lbl, zorder=3)
    bh = results["lr"]["trades"]
    ax1.plot(bh.index, bh["cum_bh"], color=_C["bh"], lw=1,
             ls="--", label="Buy & Hold", alpha=0.6)
    ax1.set_ylabel("Equity (×1)", color=_C["text"], fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}×"))
    ax1.legend(facecolor=_C["panel"], edgecolor=_C["border"],
               labelcolor=_C["text"], fontsize=9, loc="upper left")
    ax1.tick_params(axis="both", colors=_C["dim"])

    # ── Panel 2: Drawdown ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _ax_style(ax2, "Drawdown")
    for key, col in [("lr", _C["lr"]), ("rf", _C["rf"])]:
        cum = results[key]["trades"]["cum_net"]
        dd  = cum / cum.cummax() - 1
        ax2.fill_between(dd.index, dd, 0, color=col, alpha=0.20)
        ax2.plot(dd.index, dd, color=col, lw=0.9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.set_ylabel("Drawdown", color=_C["text"], fontsize=8)

    # ── Panel 3: Rolling 20-day accuracy ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _ax_style(ax3, "Rolling 20-Day Directional Accuracy")
    for key, col, lbl in [("lr", _C["lr"], "LR"),
                           ("rf", _C["rf"], "RF")]:
        acc = results[key]["trades"]["correct"].rolling(20).mean()
        ax3.plot(acc.index, acc, color=col, lw=1.2, label=lbl)
    ax3.axhline(0.5, color=_C["dim"], ls="--", lw=0.8, label="50% chance")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.set_ylim(0.2, 0.85)
    ax3.set_ylabel("Accuracy", color=_C["text"], fontsize=8)
    ax3.legend(facecolor=_C["panel"], edgecolor=_C["border"],
               labelcolor=_C["text"], fontsize=8)

    # ── Panel 4: Per-fold net return ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    _ax_style(ax4, "Per-Fold Net Return")
    fl = results["lr"]["folds"]
    fr = results["rf"]["folds"]
    x  = np.arange(len(fl))
    w  = 0.38
    ax4.bar(x - w / 2, fl["net_ret"], w,
            color=_C["lr"], alpha=0.85, label="LR")
    ax4.bar(x + w / 2, fr["net_ret"], w,
            color=_C["rf"], alpha=0.85, label="RF")
    ax4.axhline(0, color=_C["border"], lw=0.8)
    ax4.set_xlabel("Fold", color=_C["dim"], fontsize=8)
    ax4.set_ylabel("Return", color=_C["text"], fontsize=8)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax4.legend(facecolor=_C["panel"], edgecolor=_C["border"],
               labelcolor=_C["text"], fontsize=8)

    # ── Panel 5: RF feature importance ───────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    _ax_style(ax5, "Feature Importance  (Random Forest, last trained fold)")
    fi = pd.Series(
        results["rf"]["last_model"].named_steps["clf"].feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=True)
    colors = [_C["green"] if v >= fi.median() else _C["red"] for v in fi]
    bars = ax5.barh(fi.index, fi.values, color=colors, height=0.65)
    ax5.set_xlabel("Mean Decrease in Impurity", color=_C["dim"], fontsize=8)
    ax5.tick_params(axis="y", labelsize=8)
    # Annotate category groups
    groups = {"Momentum": (0, 5), "Volatility": (6, 11), "Volume": (12, 13), "Micro": (14, 15)}

    # ── Panel 6: Return distribution ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    _ax_style(ax6, "Daily Net Return Distribution")
    for key, col, lbl in [("lr", _C["lr"], "LR"),
                           ("rf", _C["rf"], "RF")]:
        r = results[key]["trades"]["net_ret"]
        ax6.hist(r, bins=40, color=col, alpha=0.45, label=lbl, density=True)
    ax6.axvline(0, color=_C["dim"], lw=0.8)
    ax6.set_xlabel("Daily Return", color=_C["dim"], fontsize=8)
    ax6.set_ylabel("Density", color=_C["text"], fontsize=8)
    ax6.legend(facecolor=_C["panel"], edgecolor=_C["border"],
               labelcolor=_C["text"], fontsize=8)

    # ── Title ─────────────────────────────────────────────────────
    fig.suptitle(
        "ML Trading Signal  ·  Walk-Forward Backtest Dashboard",
        color=_C["text"], fontsize=14, fontweight="bold",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Dashboard saved → {output_path}")


# ══════════════════════════════════════════════════════════════════
# 6 ·  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    BANNER = "═" * 62
    print(f"\n{BANNER}")
    print("   ML Trading Signal Generator")
    print(f"{BANNER}")

    # ── 1 · Data ─────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic OHLCV data …")
    raw  = generate_ohlcv(n_days=1500)
    feat = add_features(raw)
    data = feat.dropna()
    print(f"  ✓ {len(raw):,} trading days  →  {len(data):,} rows after feature NaN drop")
    print(f"  ✓ Date range: {data.index[0].date()} → {data.index[-1].date()}")

    # ── 2 · Describe features ────────────────────────────────────
    print("\n[2/5] Feature summary (first 5 of 16)")
    print(data[FEATURES[:5]].describe().round(4).to_string())
    up_pct = data["target"].mean()
    print(f"\n  Target class balance: {up_pct:.1%} UP  /  {1-up_pct:.1%} DOWN")

    # ── 3 · Model factories ──────────────────────────────────────
    print("\n[3/5] Defining model pipelines …")

    def make_lr():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=0.05, max_iter=1000,
                class_weight="balanced", random_state=42,
            )),
        ])

    def make_rf():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=4,
                min_samples_leaf=25, max_features="sqrt",
                class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ])

    print("  ✓ Logistic Regression  (C=0.05, balanced, L2 penalty)")
    print("  ✓ Random Forest        (300 trees, max_depth=4, min_leaf=25)")

    # ── 4 · Walk-forward backtests ───────────────────────────────
    print("\n[4/5] Running walk-forward backtests (train=500d, test=63d) …")
    results = {}
    for name, factory in [("lr", make_lr), ("rf", make_rf)]:
        print(f"  Running {name.upper()} ", end="", flush=True)
        res = walk_forward_backtest(
            data, factory, FEATURES,
            train_size=500, test_size=63, cost_bps=10,
        )
        # Fit a final model on the first 800 rows for feature importance
        last = factory()
        last.fit(data.iloc[:800][FEATURES], data.iloc[:800]["target"])
        res["last_model"] = last
        results[name] = res
        n_folds = len(res["folds"])
        avg_acc = res["folds"]["accuracy"].mean()
        print(f"→ {n_folds} folds  |  avg OOS accuracy {avg_acc:.1%}")

    # ── 5 · Report ───────────────────────────────────────────────
    print(f"\n[5/5] Performance summary\n{'─'*62}")
    rows = [performance_table(results["lr"], "Logistic Regression"),
            performance_table(results["rf"], "Random Forest")]
    summary = pd.DataFrame(rows).set_index("Model").T
    print(summary.to_string())

    print(f"\n{'─'*62}")
    print("  Fold-level detail (LR):")
    print(results["lr"]["folds"].to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Generating dashboard …")
    plot_dashboard(results, "/mnt/user-data/outputs/backtest_dashboard.png")

    # ── Quant notes ──────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("  QUANTITATIVE NOTES & COMMON PITFALLS")
    print(f"{'═'*62}")
    notes = [
        ("Look-ahead bias",
         "Features use only past data; target is t+1 close."),
        ("Walk-forward validation",
         "Never test on data used in training — every fold's\n"
         "   test period strictly follows its training period."),
        ("Transaction costs",
         "10 bps per trade deducted.  Hidden costs (slippage,\n"
         "   market impact) can be 3–10× larger in live trading."),
        ("Overfitting risk",
         "Random Forest can memorise noise.  Constrain depth &\n"
         "   min_leaf; monitor train vs OOS accuracy gap."),
        ("Class imbalance",
         "class_weight='balanced' compensates for near-50/50 split."),
        ("Signal capacity",
         "Strong signals degrade when capital is large enough\n"
         "   to move the market (market impact)."),
        ("Regime change",
         "Models trained on one regime (bull/bear/volatile) may\n"
         "   fail abruptly when the regime shifts."),
    ]
    for title, desc in notes:
        print(f"\n  ► {title}")
        print(f"    {desc}")

    print(f"\n{'═'*62}\n")
    return results, summary


if __name__ == "__main__":
    results, summary = main()
