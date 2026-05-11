# WTI Crude Oil Valuation Model & Quantitative Dashboard

A comprehensive, Python-native quantitative portfolio application designed to evaluate the WTI crude oil market. Instead of relying purely on trend-following indicators, this model attempts to estimate a "fair value" for WTI by scoring fundamental supply/demand drivers, cross-asset correlations, and macro regime overlays. 

The output is visually mapped to a "Temperature Gauge" (ranging from "Very Cold" to "Very Hot") through an interactive, auto-refreshing dashboard.

---

## Core Philosophy & Methodology

The model calculates a base fair value by anchoring to the 50-day Simple Moving Average (SMA) and adjusting it based on a composite macro score and market volatility (OVX). 

If the calculated fair value is significantly lower than the current spot price, the model flags the market as "Hot" (overvalued). If the fair value is higher than the spot price, it flags the market as "Cold" (undervalued).

### 1. The 8-Driver Fundamental Framework
The quantitative engine scores 8 distinct parameters from -2 (bearish) to +2 (bullish). These are weighted to create a normalized composite score:

1. **US Crude Inventories (Weight: 17%):** Compares current inventories against the 5-year average. Tight inventories yield a positive score. Data is sourced from the EIA (or a simulated seasonal profile if no API key is provided).
2. **USD Strength / DXY (Weight: 12%):** A stronger dollar tightens global purchasing power, pressuring oil demand.
3. **Trend Context (Weight: 8%):** Measures extreme downside deviations from the 200-day SMA as potential fundamental cheapness.
4. **Oil Volatility / OVX (Weight: 12%):** Higher implied volatility signals stress and risk premia that can distort spot valuations.
5. **Geopolitical Risk Proxy / ITA (Weight: 13%):** Uses the iShares U.S. Aerospace & Defense ETF relative to its 200D SMA to measure supply-risk premiums.
6. **Implied Inflation Expectations / TIP vs IEF (Weight: 13%):** A rising TIP/IEF ratio points to firmer market-implied inflation expectations.
7. **Forward Growth Expectations / HYG vs LQD (Weight: 13%):** Tracks high-yield credit appetite versus investment-grade credit. High-yield outperformance implies stronger forward growth.
8. **Substitution Pressure / WTI vs UNG (Weight: 12%):** Measures when oil outruns a natural gas substitute, which can encourage fuel switching.

### 2. Macro Regime Overlays
Once the base score is calculated, the model applies critical overlays that can penalize the score and heavily discount the fair value estimate:
* **Demand Destruction Risk:** Activates when crude crosses $120/bbl. It assumes low value-added producers lose pass-through power. Above $150/bbl, the model flags "Extreme Demand Destruction," anticipating bilateral recession risks and forced shutdowns.
* **Stagflation & China Recession Monitor:** The model checks if inflation proxies are hot while growth proxies are falling. Simultaneously, it monitors the China Large-Cap ETF (FXI) against its 200D SMA. If China is weak while oil is rising, it applies a severe recession discount to the fair value.

### 3. Backtesting Engine
The model reconstructs the last 90 days of fair value estimates and tracks forward performance:
* It analyzes **5-day and 10-day forward returns** for every historical signal.
* **Win Rate Logic:** A "win" is defined as mean-reversion toward fair value within 10 days (e.g., negative returns following a "Hot" signal, or positive returns following a "Cold" signal).

### 4. Technical Overlay
While fundamental valuation drives the temperature gauge, the dashboard also integrates RSI (14), MACD (Line, Signal, Histogram), and 50/200D SMA metrics to provide immediate momentum and trend context.

---

## Architecture & Tech Stack

This project is divided into a robust REST API backend and an interactive frontend:

* **Backend Engine (`main.py`):**
  * Powered by **FastAPI** and `uvicorn[standard]`.
  * Handles asynchronous data fetching from Yahoo Finance (`yfinance`) and the EIA API (`requests`).
  * Utilizes `pandas`, `numpy`, and `scikit-learn` for rolling averages, signal generation, and historical dataframe reconstruction.
* **Frontend Dashboard (`dashboard.py`):**
  * Built completely in Python using **Dash** and **Plotly**.
  * Features an auto-refresh cycle (every 5 minutes) to update live market prices without page reloads.
  * Employs a highly customized, dark-themed UI with CSS flexbox/grid layouts handled natively in Python.

---

## Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/dozpiskin/Crude_Oil_Fairvalue_Model.git
cd Model_A
