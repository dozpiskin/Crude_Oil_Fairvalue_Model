"""
FastAPI backend for a crude-oil "Valuation Temperature Gauge".

Run locally with:
    uvicorn main:app --reload

The model is intentionally transparent for portfolio/demo purposes:
1. Fetch the latest available market data from Yahoo Finance.
2. Score five fundamental drivers on a -2 to +2 scale.
3. Convert the weighted composite into a fair-value adjustment.
4. Compare market price to fair value and map the result to a
   valuation "temperature" label for the frontend gauge.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


CRUDE_TICKER = "CL=F"
DXY_TICKER = "DX-Y.NYB"
OVX_TICKER = "^OVX"
SP500_TICKER = "^GSPC"
DEFAULT_LOOKBACK_PERIOD = "2y"

PARAMETER_WEIGHTS = {
    "inventories": 0.25,
    "usd_strength": 0.20,
    "trend_context": 0.15,
    "ovx": 0.20,
    "global_growth_proxy": 0.20,
}

SIGNAL_BACKTEST_ORDER = ["Very Cold", "Cold", "Hot", "Very Hot"]
SIGNAL_EXPECTED_DIRECTION = {
    "Very Cold": "positive",
    "Cold": "positive",
    "Hot": "negative",
    "Very Hot": "negative",
}

EIA_DEFAULT_SERIES_ID = os.getenv("EIA_CRUDE_STOCKS_SERIES_ID", "PET.WCESTUS1.W")
EIA_API_URL = "https://api.eia.gov/v2/seriesid/"


app = FastAPI(
    title="Crude Oil Valuation Temperature Gauge API",
    version="1.0.0",
    description=(
        "Academic portfolio API that scores crude-oil fundamentals, estimates "
        "fair value, and maps the valuation gap to a temperature gauge."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class MarketAsset:
    ticker: str
    name: str
    history: pd.DataFrame
    current_price: float
    last_close: float
    last_timestamp: str
    currency: str


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert arbitrary input to float while tolerating missing API values."""
    try:
        if value is None:
            return default
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return default
        return numeric_value
    except (TypeError, ValueError):
        return default


def round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    normalized_value = safe_float(value, default=None)
    if normalized_value is None:
        return None
    return round(normalized_value, digits)


def to_utc_iso(timestamp: Any) -> str:
    """Normalize timestamps to UTC ISO-8601 strings for the frontend."""
    if isinstance(timestamp, pd.Timestamp):
        ts = timestamp
    else:
        ts = pd.Timestamp(timestamp)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.isoformat()


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output and keep a clean OHLCV frame."""
    if history is None or history.empty:
        raise ValueError("Received empty history from Yahoo Finance.")

    data = history.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]) for col in data.columns]

    data.columns = [str(col).strip() for col in data.columns]
    if "Close" not in data.columns and "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]

    if "Close" not in data.columns:
        raise ValueError("Yahoo Finance history does not contain a Close column.")

    data.index = pd.to_datetime(data.index)
    data = data.dropna(subset=["Close"]).sort_index()

    if len(data) < 200:
        raise ValueError("At least 200 daily observations are required for the model.")

    return data


def download_price_history(ticker: str, period: str = DEFAULT_LOOKBACK_PERIOD) -> pd.DataFrame:
    """Download daily history for one ticker."""
    history = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_history(history)


def fetch_market_asset(ticker: str, name: str) -> MarketAsset:
    """
    Fetch historical data plus the freshest price we can get from yfinance.

    We compute indicators from the daily history, but expose `current_price`
    from `fast_info` when available to make the gauge feel more live.
    """
    history = download_price_history(ticker)
    last_close = float(history["Close"].iloc[-1])
    last_timestamp = to_utc_iso(history.index[-1])
    current_price = last_close
    currency = "USD"

    try:
        ticker_object = yf.Ticker(ticker)
        fast_info = ticker_object.fast_info
        current_price = safe_float(
            fast_info.get("lastPrice")
            or fast_info.get("regularMarketPrice")
            or fast_info.get("previousClose"),
            default=last_close,
        ) or last_close
        currency = str(fast_info.get("currency") or currency)
    except Exception as exc:  # pragma: no cover - best-effort enrichment
        logger.warning("Could not enrich %s with fast_info: %s", ticker, exc)

    return MarketAsset(
        ticker=ticker,
        name=name,
        history=history,
        current_price=float(current_price),
        last_close=last_close,
        last_timestamp=last_timestamp,
        currency=currency,
    )


def build_mock_inventory_snapshot() -> Dict[str, Any]:
    """
    Seasonality-based inventory mock used when no EIA API key is configured.

    This keeps the portfolio project demoable while making it very clear to
    the frontend whether inventories are coming from a real or simulated feed.
    """
    today = datetime.now(timezone.utc)
    day_of_year = today.timetuple().tm_yday

    # Simple sinusoidal profile to mimic the broad seasonal ebb/flow of U.S.
    # commercial stocks. Values are expressed in million barrels.
    seasonal_component = 16.0 * np.sin(2.0 * np.pi * ((day_of_year - 40) / 365.25))
    current_inventory = round(428.0 + seasonal_component, 1)
    five_year_average = 423.0

    return {
        "current_inventory_million_bbl": current_inventory,
        "five_year_average_million_bbl": five_year_average,
        "series_points": 260,
        "source": "mock",
        "series_id": None,
        "last_updated": today.isoformat(),
        "note": (
            "EIA inventory data is currently simulated because no EIA API key "
            "was found. Set EIA_API_KEY to switch to the live feed."
        ),
    }


def fetch_eia_inventory_snapshot() -> Dict[str, Any]:
    """
    Fetch weekly U.S. crude inventories from the EIA API v2 `seriesid` route.

    Default series: PET.WCESTUS1.W
    Weekly U.S. ending stocks excluding the Strategic Petroleum Reserve.

    If the call fails or no key is provided, the app falls back to a realistic
    mock so the rest of the valuation stack still works end-to-end.
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        return build_mock_inventory_snapshot()

    try:
        response = requests.get(
            f"{EIA_API_URL}{EIA_DEFAULT_SERIES_ID}",
            params={"api_key": api_key},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("response", {}).get("data", [])
        if not rows:
            raise ValueError("EIA response contained no data rows.")

        data = pd.DataFrame(rows)
        if "period" not in data.columns or "value" not in data.columns:
            raise ValueError("EIA response is missing expected fields.")

        data["period"] = pd.to_datetime(data["period"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["period", "value"]).sort_values("period")

        if data.empty:
            raise ValueError("EIA inventory series was empty after cleaning.")

        # EIA petroleum stock series are typically published in thousand barrels.
        # Convert to million barrels to make the frontend more readable.
        if data["value"].median() > 1_000:
            data["value_million_bbl"] = data["value"] / 1_000.0
        else:
            data["value_million_bbl"] = data["value"]

        trailing_window = min(len(data), 260)
        current_inventory = float(data["value_million_bbl"].iloc[-1])
        five_year_average = float(data["value_million_bbl"].tail(trailing_window).mean())

        return {
            "current_inventory_million_bbl": current_inventory,
            "five_year_average_million_bbl": five_year_average,
            "series_points": trailing_window,
            "source": "eia",
            "series_id": EIA_DEFAULT_SERIES_ID,
            "last_updated": to_utc_iso(data["period"].iloc[-1]),
            "note": "Live EIA inventory data.",
        }
    except Exception as exc:
        logger.warning("Falling back to mock inventory data because EIA failed: %s", exc)
        fallback = build_mock_inventory_snapshot()
        fallback["note"] = (
            "EIA fetch failed, so a mock inventory profile was used. "
            f"Underlying error: {exc}"
        )
        return fallback


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder-style RSI using exponentially smoothed gains/losses."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(~((average_loss == 0.0) & (average_gain > 0.0)), 100.0)
    rsi = rsi.where(~((average_gain == 0.0) & (average_loss > 0.0)), 0.0)
    rsi = rsi.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
    return rsi.fillna(50.0)


def calculate_macd(
    close: pd.Series,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=slow_window, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def describe_bias(score: int) -> str:
    if score > 0:
        return "bullish"
    if score < 0:
        return "bearish"
    return "neutral"


def score_inventory(current_inventory: float, five_year_average: float) -> Dict[str, Any]:
    deviation_pct = ((current_inventory - five_year_average) / five_year_average) * 100.0

    if deviation_pct >= 8.0:
        score = -2
    elif deviation_pct >= 3.0:
        score = -1
    elif deviation_pct >= -3.0:
        score = 0
    elif deviation_pct > -8.0:
        score = 1
    else:
        score = 2

    return {
        "key": "inventories",
        "name": "US Crude Inventories",
        "weight": PARAMETER_WEIGHTS["inventories"],
        "weight_pct": 25,
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_inventory, 2),
        "unit": "million bbl",
        "benchmark_value": round(five_year_average, 2),
        "benchmark_label": "5Y average",
        "deviation_pct": round(deviation_pct, 2),
        "rationale": (
            "High inventories imply looser physical balances and weaker spot support; "
            "tight inventories imply stronger fundamental scarcity."
        ),
    }


def score_dxy(dxy_value: float) -> Dict[str, Any]:
    if dxy_value > 105.0:
        score = -2
    elif dxy_value >= 100.0:
        score = -1
    elif dxy_value >= 95.0:
        score = 0
    elif dxy_value >= 90.0:
        score = 1
    else:
        score = 2

    return {
        "key": "usd_strength",
        "name": "USD Strength (DXY)",
        "weight": PARAMETER_WEIGHTS["usd_strength"],
        "weight_pct": 20,
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(dxy_value, 2),
        "unit": "index",
        "benchmark_label": "Rule bands",
        "benchmark_value": None,
        "deviation_pct": None,
        "rationale": (
            "Because crude is invoiced in U.S. dollars, a stronger USD usually "
            "tightens global purchasing power and pressures oil demand."
        ),
    }


def score_price_vs_200sma(current_price: float, sma_200: float) -> Dict[str, Any]:
    deviation_pct = ((current_price - sma_200) / sma_200) * 100.0

    # This factor is deliberately contrarian: price far below long-run trend is
    # treated as potentially undervalued, while price far above trend is treated
    # as overextended relative to fundamentals.
    if deviation_pct <= -10.0:
        score = 2
    elif deviation_pct <= -3.0:
        score = 1
    elif deviation_pct < 3.0:
        score = 0
    elif deviation_pct < 10.0:
        score = -1
    else:
        score = -2

    return {
        "key": "trend_context",
        "name": "Trend & Momentum Context",
        "weight": PARAMETER_WEIGHTS["trend_context"],
        "weight_pct": 15,
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(deviation_pct, 2),
        "unit": "% vs 200D SMA",
        "benchmark_value": round(sma_200, 2),
        "benchmark_label": "200D SMA",
        "deviation_pct": round(deviation_pct, 2),
        "rationale": (
            "For this valuation model, deep downside extensions below the 200-day "
            "average are interpreted as potential fundamental cheapness."
        ),
    }


def score_ovx(ovx_value: float) -> Dict[str, Any]:
    if ovx_value > 45.0:
        score = -2
    elif ovx_value >= 35.0:
        score = -1
    elif ovx_value >= 25.0:
        score = 0
    elif ovx_value >= 15.0:
        score = 1
    else:
        score = 2

    return {
        "key": "ovx",
        "name": "Oil Volatility (OVX)",
        "weight": PARAMETER_WEIGHTS["ovx"],
        "weight_pct": 20,
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(ovx_value, 2),
        "unit": "vol index",
        "benchmark_label": "Rule bands",
        "benchmark_value": None,
        "deviation_pct": None,
        "rationale": (
            "Higher implied volatility often appears alongside stress, liquidation, "
            "or geopolitical risk premia that distort spot valuation."
        ),
    }


def score_global_growth_proxy(
    spx_price: float,
    spx_sma_50: float,
    spx_sma_200: float,
) -> Dict[str, Any]:
    deviation_pct = ((spx_price - spx_sma_200) / spx_sma_200) * 100.0
    ma_spread_pct = ((spx_sma_50 - spx_sma_200) / spx_sma_200) * 100.0

    if deviation_pct <= -8.0 and ma_spread_pct < 0.0:
        score = -2
    elif deviation_pct < 0.0 or ma_spread_pct < 0.0:
        score = -1
    elif deviation_pct >= 8.0 and ma_spread_pct > 1.0:
        score = 2
    elif deviation_pct > 0.0 and ma_spread_pct >= 0.0:
        score = 1
    else:
        score = 0

    return {
        "key": "global_growth_proxy",
        "name": "Global Economic Health Proxy (S&P 500)",
        "weight": PARAMETER_WEIGHTS["global_growth_proxy"],
        "weight_pct": 20,
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(spx_price, 2),
        "unit": "index",
        "benchmark_value": round(spx_sma_200, 2),
        "benchmark_label": "200D SMA",
        "deviation_pct": round(deviation_pct, 2),
        "auxiliary_value": round(ma_spread_pct, 2),
        "auxiliary_label": "50D vs 200D SMA (%)",
        "rationale": (
            "A healthy equity trend acts as a rough demand proxy: stronger growth "
            "expectations generally support higher oil consumption."
        ),
    }


def calculate_weighted_score(parameters: List[Dict[str, Any]]) -> Dict[str, float]:
    weighted_raw = sum(parameter["score"] * parameter["weight"] for parameter in parameters)
    normalized_score = (weighted_raw / 2.0) * 100.0
    return {
        "weighted_raw": float(weighted_raw),
        "normalized_score": float(normalized_score),
    }


def derive_volatility_factor(ovx_value: float) -> float:
    """
    Convert OVX into a bounded multiplier for the fair-value adjustment.

    OVX is an annualized implied-volatility style measure. We divide by 100 to
    turn it into a dimensionless scale factor, then clip it to avoid extreme
    valuation swings in stressed markets.
    """
    return float(np.clip(ovx_value / 100.0, 0.15, 0.60))


def estimate_fair_value(
    sma_50: float,
    composite_score_normalized: float,
    volatility_factor: float,
) -> float:
    return float(sma_50 * (1.0 + (composite_score_normalized / 100.0) * volatility_factor))


def calculate_valuation_gap(current_price: float, fair_value: float) -> float:
    return float(((current_price - fair_value) / fair_value) * 100.0)


def map_temperature(gap_pct: float) -> Dict[str, str]:
    if gap_pct > 20.0:
        return {
            "label": "Very Hot",
            "emoji": "🔥",
            "description": "Severely overvalued",
            "color": "red",
        }
    if gap_pct >= 10.0:
        return {
            "label": "Hot",
            "emoji": "🌡️",
            "description": "Overvalued",
            "color": "orange",
        }
    if gap_pct >= -10.0:
        return {
            "label": "Neutral",
            "emoji": "⚖️",
            "description": "Fairly valued",
            "color": "slate",
        }
    if gap_pct >= -20.0:
        return {
            "label": "Cold",
            "emoji": "🧊",
            "description": "Undervalued",
            "color": "blue",
        }
    return {
        "label": "Very Cold",
        "emoji": "❄️❄️",
        "description": "Severely undervalued",
        "color": "cyan",
    }


def build_technical_summary(crude_asset: MarketAsset) -> Dict[str, Any]:
    close = crude_asset.history["Close"]
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    rsi_14 = calculate_rsi(close, window=14)
    macd_line, signal_line, histogram = calculate_macd(close)

    latest_rsi = float(rsi_14.iloc[-1])
    latest_macd = float(macd_line.iloc[-1])
    latest_signal = float(signal_line.iloc[-1])
    latest_histogram = float(histogram.iloc[-1])
    latest_sma_50 = float(sma_50.iloc[-1])
    latest_sma_200 = float(sma_200.iloc[-1])

    if latest_rsi >= 70.0:
        rsi_status = "overbought"
    elif latest_rsi <= 30.0:
        rsi_status = "oversold"
    else:
        rsi_status = "neutral"

    if latest_macd > latest_signal and latest_histogram > 0.0:
        macd_status = "bullish"
    elif latest_macd < latest_signal and latest_histogram < 0.0:
        macd_status = "bearish"
    else:
        macd_status = "neutral"

    recent_history = crude_asset.history.copy()
    recent_history["sma_50"] = sma_50
    recent_history["sma_200"] = sma_200
    chart_points = recent_history.tail(90)

    price_history = [
        {
            "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
            "close": round_or_none(row["Close"]),
            "sma_50": round_or_none(row["sma_50"]),
            "sma_200": round_or_none(row["sma_200"]),
        }
        for index, row in chart_points.iterrows()
    ]

    return {
        "rsi_14": {
            "value": round(latest_rsi, 2),
            "status": rsi_status,
        },
        "macd": {
            "line": round(latest_macd, 4),
            "signal": round(latest_signal, 4),
            "histogram": round(latest_histogram, 4),
            "status": macd_status,
        },
        "moving_averages": {
            "sma_50": round(latest_sma_50, 2),
            "sma_200": round(latest_sma_200, 2),
            "price_vs_sma_50_pct": round(
                ((crude_asset.current_price - latest_sma_50) / latest_sma_50) * 100.0,
                2,
            ),
            "price_vs_sma_200_pct": round(
                ((crude_asset.current_price - latest_sma_200) / latest_sma_200) * 100.0,
                2,
            ),
        },
        "price_history": price_history,
    }


def build_market_snapshot(asset: MarketAsset) -> Dict[str, Any]:
    return {
        "ticker": asset.ticker,
        "name": asset.name,
        "current_price": round(asset.current_price, 2),
        "last_close": round(asset.last_close, 2),
        "currency": asset.currency,
        "last_timestamp": asset.last_timestamp,
    }


def build_historical_model_series(
    crude_asset: MarketAsset,
    dxy_asset: MarketAsset,
    ovx_asset: MarketAsset,
    sp500_asset: MarketAsset,
    inventory_snapshot: Dict[str, Any],
    history_days: int = 90,
) -> List[Dict[str, Any]]:
    """
    Reconstruct the model's recent daily fair value series.

    Important modeling note:
    - DXY, OVX, crude price, and S&P 500 are fully historical.
    - Inventory score is held constant at the latest snapshot unless a full
      historical EIA inventory time series is added later.
    """
    crude_frame = crude_asset.history[["Close"]].copy()
    crude_frame = crude_frame.rename(columns={"Close": "actual_price"})
    crude_frame["crude_sma_50"] = crude_frame["actual_price"].rolling(50).mean()
    crude_frame["crude_sma_200"] = crude_frame["actual_price"].rolling(200).mean()

    dxy_frame = dxy_asset.history[["Close"]].rename(columns={"Close": "dxy_close"})
    ovx_frame = ovx_asset.history[["Close"]].rename(columns={"Close": "ovx_close"})

    sp500_frame = sp500_asset.history[["Close"]].copy()
    sp500_frame = sp500_frame.rename(columns={"Close": "sp500_close"})
    sp500_frame["sp500_sma_50"] = sp500_frame["sp500_close"].rolling(50).mean()
    sp500_frame["sp500_sma_200"] = sp500_frame["sp500_close"].rolling(200).mean()

    combined = crude_frame.join(dxy_frame, how="left")
    combined = combined.join(ovx_frame, how="left")
    combined = combined.join(sp500_frame, how="left")
    combined[["dxy_close", "ovx_close", "sp500_close", "sp500_sma_50", "sp500_sma_200"]] = (
        combined[["dxy_close", "ovx_close", "sp500_close", "sp500_sma_50", "sp500_sma_200"]].ffill()
    )

    inventory_score = score_inventory(
        inventory_snapshot["current_inventory_million_bbl"],
        inventory_snapshot["five_year_average_million_bbl"],
    )["score"]

    history_records: List[Dict[str, Any]] = []
    candidate_rows = combined.dropna(
        subset=[
            "actual_price",
            "crude_sma_50",
            "crude_sma_200",
            "dxy_close",
            "ovx_close",
            "sp500_close",
            "sp500_sma_50",
            "sp500_sma_200",
        ]
    ).tail(history_days).copy()

    candidate_rows["forward_return_5d_pct"] = (
        (candidate_rows["actual_price"].shift(-5) - candidate_rows["actual_price"])
        / candidate_rows["actual_price"]
    ) * 100.0
    candidate_rows["forward_return_10d_pct"] = (
        (candidate_rows["actual_price"].shift(-10) - candidate_rows["actual_price"])
        / candidate_rows["actual_price"]
    ) * 100.0

    for index, row in candidate_rows.iterrows():
        dxy_score = score_dxy(float(row["dxy_close"]))["score"]
        trend_score = score_price_vs_200sma(
            current_price=float(row["actual_price"]),
            sma_200=float(row["crude_sma_200"]),
        )["score"]
        ovx_score = score_ovx(float(row["ovx_close"]))["score"]
        growth_score = score_global_growth_proxy(
            spx_price=float(row["sp500_close"]),
            spx_sma_50=float(row["sp500_sma_50"]),
            spx_sma_200=float(row["sp500_sma_200"]),
        )["score"]

        weighted_raw = (
            inventory_score * PARAMETER_WEIGHTS["inventories"]
            + dxy_score * PARAMETER_WEIGHTS["usd_strength"]
            + trend_score * PARAMETER_WEIGHTS["trend_context"]
            + ovx_score * PARAMETER_WEIGHTS["ovx"]
            + growth_score * PARAMETER_WEIGHTS["global_growth_proxy"]
        )
        normalized_score = (weighted_raw / 2.0) * 100.0
        volatility_factor = derive_volatility_factor(float(row["ovx_close"]))
        fair_value = estimate_fair_value(
            sma_50=float(row["crude_sma_50"]),
            composite_score_normalized=normalized_score,
            volatility_factor=volatility_factor,
        )
        valuation_gap_pct = calculate_valuation_gap(float(row["actual_price"]), fair_value)
        temperature = map_temperature(valuation_gap_pct)

        history_records.append(
            {
                "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
                "actual_price": round(float(row["actual_price"]), 2),
                "fair_value": round(fair_value, 2),
                "valuation_gap_pct": round(valuation_gap_pct, 2),
                "normalized_score": round(normalized_score, 2),
                "forward_return_5d_pct": round_or_none(row["forward_return_5d_pct"]),
                "forward_return_10d_pct": round_or_none(row["forward_return_10d_pct"]),
                "temperature_label": temperature["label"],
                "temperature_color": temperature["color"],
            }
        )

    return history_records


def build_backtest_summary(history_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate forward-return performance by signal regime.

    Win rate is evaluated on the 10-day horizon:
    - Cold / Very Cold: a positive forward return is a win.
    - Hot / Very Hot: a negative forward return is a win.
    """
    history_frame = pd.DataFrame(history_records)
    signals: Dict[str, Dict[str, Any]] = {}

    for signal_label in SIGNAL_BACKTEST_ORDER:
        signal_slice = history_frame[history_frame["temperature_label"] == signal_label].copy()
        expected_direction = SIGNAL_EXPECTED_DIRECTION[signal_label]

        forward_5d = pd.to_numeric(signal_slice.get("forward_return_5d_pct"), errors="coerce").dropna()
        forward_10d = pd.to_numeric(signal_slice.get("forward_return_10d_pct"), errors="coerce").dropna()

        if forward_10d.empty:
            win_rate_pct = None
        elif expected_direction == "positive":
            win_rate_pct = float((forward_10d > 0.0).mean() * 100.0)
        else:
            win_rate_pct = float((forward_10d < 0.0).mean() * 100.0)

        signals[signal_label] = {
            "signal": signal_label,
            "expected_direction": expected_direction,
            "occurrence_count": int(len(signal_slice)),
            "valid_5d_samples": int(len(forward_5d)),
            "valid_10d_samples": int(len(forward_10d)),
            "avg_5d_return_pct": round_or_none(forward_5d.mean()) if not forward_5d.empty else None,
            "avg_10d_return_pct": (
                round_or_none(forward_10d.mean()) if not forward_10d.empty else None
            ),
            "win_rate_pct": round_or_none(win_rate_pct) if win_rate_pct is not None else None,
        }

    return {
        "signal_order": SIGNAL_BACKTEST_ORDER,
        "signals": signals,
        "win_rate_definition": (
            "Win Rate indicates the share of 10-day forward windows where price moved back "
            "toward fair value: up after Cold signals and down after Hot signals."
        ),
        "coverage": {
            "history_days": len(history_records),
            "signals_excluding_neutral": SIGNAL_BACKTEST_ORDER,
        },
    }


def build_crude_oil_valuation_response(
    crude_asset: MarketAsset,
    dxy_asset: MarketAsset,
    ovx_asset: MarketAsset,
    sp500_asset: MarketAsset,
    inventory_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    technicals = build_technical_summary(crude_asset)
    historical_model_series = build_historical_model_series(
        crude_asset=crude_asset,
        dxy_asset=dxy_asset,
        ovx_asset=ovx_asset,
        sp500_asset=sp500_asset,
        inventory_snapshot=inventory_snapshot,
        history_days=90,
    )
    backtest_summary = build_backtest_summary(historical_model_series)
    crude_sma_50 = technicals["moving_averages"]["sma_50"]
    crude_sma_200 = technicals["moving_averages"]["sma_200"]

    spx_close = sp500_asset.history["Close"]
    spx_sma_50 = float(spx_close.rolling(50).mean().iloc[-1])
    spx_sma_200 = float(spx_close.rolling(200).mean().iloc[-1])

    parameters = [
        score_inventory(
            inventory_snapshot["current_inventory_million_bbl"],
            inventory_snapshot["five_year_average_million_bbl"],
        ),
        score_dxy(dxy_asset.current_price),
        score_price_vs_200sma(crude_asset.current_price, crude_sma_200),
        score_ovx(ovx_asset.current_price),
        score_global_growth_proxy(sp500_asset.current_price, spx_sma_50, spx_sma_200),
    ]

    # Attach source metadata and weighted contributions for the frontend cards.
    for parameter in parameters:
        parameter["weighted_contribution"] = round(parameter["score"] * parameter["weight"], 4)
        if parameter["key"] == "inventories":
            parameter["source"] = inventory_snapshot["source"]
            parameter["source_note"] = inventory_snapshot["note"]
        elif parameter["key"] == "usd_strength":
            parameter["source"] = "yfinance"
        elif parameter["key"] == "trend_context":
            parameter["source"] = "yfinance"
        elif parameter["key"] == "ovx":
            parameter["source"] = "yfinance"
        elif parameter["key"] == "global_growth_proxy":
            parameter["source"] = "yfinance"

    composite = calculate_weighted_score(parameters)
    volatility_factor = derive_volatility_factor(ovx_asset.current_price)
    estimated_fair_value = estimate_fair_value(
        sma_50=crude_sma_50,
        composite_score_normalized=composite["normalized_score"],
        volatility_factor=volatility_factor,
    )
    valuation_gap_pct = calculate_valuation_gap(crude_asset.current_price, estimated_fair_value)
    temperature = map_temperature(valuation_gap_pct)

    return {
        "asset": {
            "ticker": crude_asset.ticker,
            "name": crude_asset.name,
            "display_name": "Crude Oil - WTI",
            "currency": crude_asset.currency,
            "current_price": round(crude_asset.current_price, 2),
            "last_updated": crude_asset.last_timestamp,
        },
        "market_data": {
            "crude_oil": build_market_snapshot(crude_asset),
            "dxy": build_market_snapshot(dxy_asset),
            "ovx": build_market_snapshot(ovx_asset),
            "sp500": build_market_snapshot(sp500_asset),
            "inventory_snapshot": {
                "current_inventory_million_bbl": round(
                    inventory_snapshot["current_inventory_million_bbl"], 2
                ),
                "five_year_average_million_bbl": round(
                    inventory_snapshot["five_year_average_million_bbl"], 2
                ),
                "source": inventory_snapshot["source"],
                "last_updated": inventory_snapshot["last_updated"],
                "note": inventory_snapshot["note"],
            },
        },
        "fundamentals": {
            "parameters": parameters,
            "weighted_raw_score": round(composite["weighted_raw"], 4),
            "normalized_score": round(composite["normalized_score"], 2),
            "score_scale": {
                "parameter_range": [-2, 2],
                "composite_range": [-100, 100],
            },
        },
        "technicals": technicals,
        "valuation": {
            "reference_sma_50": round(crude_sma_50, 2),
            "reference_sma_200": round(crude_sma_200, 2),
            "volatility_factor": round(volatility_factor, 4),
            "estimated_fair_value": round(estimated_fair_value, 2),
            "valuation_gap_pct": round(valuation_gap_pct, 2),
            "temperature": temperature,
            "methodology": (
                "Fair Value = 50D SMA * (1 + (Composite Score / 100) * Volatility Factor)"
            ),
        },
        "history": historical_model_series,
        "backtest_summary": backtest_summary,
        "disclaimer": (
            "This is a quantitative model for academic portfolio purposes only. "
            "Not financial advice."
        ),
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_sources": ["yfinance", inventory_snapshot["source"]],
            "notes": [
                "Yahoo Finance data may be delayed depending on the instrument and exchange.",
                "Inventory scoring uses EIA data when EIA_API_KEY is available; otherwise it falls back to a transparent mock.",
                "Historical model reconstruction uses fully historical market data, while the inventory score is held constant unless a historical EIA inventory series is added.",
                "Backtest win rate is based on whether price reverted toward fair value over the next 10 trading days.",
                "The trend-context factor is intentionally contrarian to reflect valuation stretch versus long-run trend.",
            ],
        },
    }


def fetch_all_model_inputs() -> Tuple[MarketAsset, MarketAsset, MarketAsset, MarketAsset, Dict[str, Any]]:
    """
    Fetch all required inputs sequentially.

    yfinance can behave inconsistently under concurrent access in some
    environments, so the production-safe choice for this compact model is
    to download the four symbols one after another.
    """
    crude_asset = fetch_market_asset(CRUDE_TICKER, "Crude Oil Futures (WTI)")
    dxy_asset = fetch_market_asset(DXY_TICKER, "US Dollar Index")
    ovx_asset = fetch_market_asset(OVX_TICKER, "CBOE Crude Oil Volatility Index")
    sp500_asset = fetch_market_asset(SP500_TICKER, "S&P 500 Index")
    inventory_snapshot = fetch_eia_inventory_snapshot()
    return crude_asset, dxy_asset, ovx_asset, sp500_asset, inventory_snapshot


def load_crude_oil_valuation_snapshot() -> Dict[str, Any]:
    """
    Synchronous loader shared by the API layer and Python-native dashboard UIs.
    """
    crude_asset, dxy_asset, ovx_asset, sp500_asset, inventory_snapshot = fetch_all_model_inputs()
    return build_crude_oil_valuation_response(
        crude_asset=crude_asset,
        dxy_asset=dxy_asset,
        ovx_asset=ovx_asset,
        sp500_asset=sp500_asset,
        inventory_snapshot=inventory_snapshot,
    )


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/valuation/crude-oil")
async def get_crude_oil_valuation() -> Dict[str, Any]:
    """
    Return a complete valuation payload for the crude-oil dashboard frontend.
    """
    try:
        return await asyncio.to_thread(load_crude_oil_valuation_snapshot)
    except Exception as exc:
        logger.exception("Crude oil valuation request failed.")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to build crude oil valuation snapshot: {exc}",
        ) from exc
