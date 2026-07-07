# 📈 Quant Portfolio Dashboard

A beginner-friendly quantitative finance dashboard built with **Streamlit**, **yfinance**, and **Plotly**.
Analyse any set of stocks — visualise prices, returns, moving averages, correlations, and key portfolio metrics — all in one page.

---

## ✨ Features

| # | Feature |
|---|---------|
| 1 | Historical closing price chart |
| 2 | Daily returns chart |
| 3 | Customisable moving averages (10 / 20 / 50 / 100 / 200-day) |
| 4 | Pearson correlation heatmap |
| 5 | Portfolio metrics: Expected Return, Volatility, Sharpe Ratio |
| 6 | Equal-weight allocation pie chart |

---

## 🗂️ Project Structure

```
quant_dashboard/
├── app.py            ← main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1 · Clone / copy the files
```bash
mkdir quant_dashboard && cd quant_dashboard
# paste app.py and requirements.txt here
```

### 2 · Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3 · Install dependencies
```bash
pip install -r requirements.txt
```

### 4 · Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 🖥️ Usage

1. Enter comma-separated tickers in the sidebar (e.g. `AAPL, TSLA, NVDA`)
2. Choose a date range
3. Pick moving average windows
4. Set the risk-free rate
5. Click **Run Analysis**

---

## 🛠️ Tech Stack

- **Streamlit** — UI framework
- **yfinance** — stock data from Yahoo Finance
- **pandas / numpy** — data wrangling & maths
- **Plotly** — interactive charts

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and does not constitute financial advice.
