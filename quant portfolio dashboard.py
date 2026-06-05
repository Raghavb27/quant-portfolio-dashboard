# ============================================================
#  Quant Portfolio Dashboard  |  app.py
#  Tools: Streamlit · yfinance · pandas · numpy · plotly
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Title ─────────────────────────────────────────────────────
st.title("📈 Quant Portfolio Dashboard")
st.markdown("Analyse stocks, visualise trends, and measure portfolio risk.")
st.divider()


# ── Helper functions ──────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices from Yahoo Finance."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns multi-level columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    return prices.pct_change().dropna()


def moving_averages(prices: pd.DataFrame, windows: list[int]) -> dict:
    """Return a dict of {window: DataFrame} moving averages."""
    return {w: prices.rolling(window=w).mean() for w in windows}


def portfolio_metrics(returns: pd.DataFrame, risk_free: float = 0.05) -> pd.DataFrame:
    """
    Compute expected return, annualised volatility, and Sharpe ratio
    for each ticker.  Assumes 252 trading days / year.
    """
    ann_return   = returns.mean() * 252
    ann_volatility = returns.std() * np.sqrt(252)
    sharpe       = (ann_return - risk_free) / ann_volatility

    metrics = pd.DataFrame({
        "Expected Return (%)": (ann_return * 100).round(2),
        "Volatility (%)":      (ann_volatility * 100).round(2),
        "Sharpe Ratio":        sharpe.round(3),
    })
    return metrics


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily returns."""
    return returns.corr().round(3)


# ── Sidebar – user inputs ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN",
        help="Use Yahoo Finance symbols, e.g. AAPL, TSLA, NVDA",
    )

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=365))
    with col_e:
        end_date = st.date_input("End",   value=date.today())

    ma_windows = st.multiselect(
        "Moving Average Windows (days)",
        options=[10, 20, 50, 100, 200],
        default=[20, 50],
    )

    risk_free_rate = st.slider(
        "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1
    ) / 100

    run = st.button("🚀 Run Analysis", use_container_width=True)

# ── Main logic ────────────────────────────────────────────────
if not run:
    st.info("👈  Enter tickers in the sidebar and click **Run Analysis** to start.")
    st.stop()

# Parse tickers
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

# Fetch data
with st.spinner("Fetching data from Yahoo Finance…"):
    try:
        prices = fetch_prices(tickers, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if prices.empty:
    st.error("No data returned. Check your tickers or date range.")
    st.stop()

# Keep only tickers that actually returned data
valid_tickers = prices.columns.tolist()
if len(valid_tickers) < len(tickers):
    missing = set(tickers) - set(valid_tickers)
    st.warning(f"No data found for: {', '.join(missing)}")

returns = daily_returns(prices)

# ── Section 1: Price Chart ────────────────────────────────────
st.subheader("1 · Historical Closing Prices")

fig_price = px.line(
    prices,
    title="Adjusted Closing Price",
    labels={"value": "Price (USD)", "variable": "Ticker", "Date": "Date"},
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig_price.update_layout(hovermode="x unified", legend_title="Ticker")
st.plotly_chart(fig_price, use_container_width=True)

# ── Section 2: Daily Returns ──────────────────────────────────
st.subheader("2 · Daily Returns")

fig_ret = px.line(
    returns,
    title="Daily Returns (%)",
    labels={"value": "Return", "variable": "Ticker"},
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig_ret.update_layout(hovermode="x unified", yaxis_tickformat=".1%")
st.plotly_chart(fig_ret, use_container_width=True)

# ── Section 3: Moving Averages ────────────────────────────────
st.subheader("3 · Moving Averages")

if ma_windows:
    selected_ticker = st.selectbox("Select ticker for MA chart", valid_tickers)
    mas = moving_averages(prices[[selected_ticker]], ma_windows)

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=prices.index, y=prices[selected_ticker],
        mode="lines", name="Price",
        line=dict(width=1.5, color="#636EFA"),
    ))
    colors_ma = ["#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
    for i, w in enumerate(sorted(ma_windows)):
        ma_series = mas[w][selected_ticker]
        fig_ma.add_trace(go.Scatter(
            x=ma_series.index, y=ma_series,
            mode="lines", name=f"MA {w}",
            line=dict(width=1.5, dash="dash", color=colors_ma[i % len(colors_ma)]),
        ))

    fig_ma.update_layout(
        title=f"{selected_ticker} – Price & Moving Averages",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", hovermode="x unified",
    )
    st.plotly_chart(fig_ma, use_container_width=True)
else:
    st.info("Select at least one MA window in the sidebar.")

# ── Section 4: Correlation Matrix ────────────────────────────
if len(valid_tickers) > 1:
    st.subheader("4 · Correlation Matrix")

    corr = correlation_matrix(returns)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation of Daily Returns",
        template="plotly_dark",
    )
    fig_corr.update_layout(width=500)
    st.plotly_chart(fig_corr, use_container_width=False)

# ── Section 5: Portfolio Metrics ──────────────────────────────
st.subheader("5 · Portfolio Metrics")

metrics = portfolio_metrics(returns, risk_free=risk_free_rate)

# Colour-code Sharpe ratio in the table
def highlight_sharpe(val):
    if val >= 1.0:
        return "background-color: #1a7a4a; color: white"
    elif val >= 0.5:
        return "background-color: #7a6a1a; color: white"
    else:
        return "background-color: #7a1a1a; color: white"

styled = metrics.style.applymap(highlight_sharpe, subset=["Sharpe Ratio"])
st.dataframe(styled, use_container_width=True)

# ── Section 6: Equal-Weight Allocation Pie ───────────────────
st.subheader("6 · Equal-Weight Portfolio Allocation")

n = len(valid_tickers)
weights = [1 / n] * n

fig_pie = px.pie(
    names=valid_tickers,
    values=weights,
    title=f"Equal-Weight Allocation ({n} assets)",
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    hole=0.35,
)
fig_pie.update_traces(textposition="inside", textinfo="label+percent")
st.plotly_chart(fig_pie, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data sourced from Yahoo Finance via yfinance · "
    "For educational purposes only · Not financial advice."
)
