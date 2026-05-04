"""
FastAPI backend for a crude-oil "Valuation Temperature Gauge".

Run locally with:
    uvicorn main:app --reload

The model is intentionally transparent for portfolio/demo purposes:
1. Fetch the latest available market data from Yahoo Finance.
2. Score seven crude-oil valuation drivers on a -2 to +2 scale.
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
GEOPOLITICAL_RISK_TICKER = "ITA"
TIPS_TICKER = "TIP"
TREASURY_TICKER = "IEF"
HIGH_YIELD_TICKER = "HYG"
INVESTMENT_GRADE_TICKER = "LQD"
DEFAULT_LOOKBACK_PERIOD = "2y"

PARAMETER_WEIGHTS = {
    "inventories": 0.20,
    "usd_strength": 0.15,
    "trend_context": 0.10,
    "ovx": 0.15,
    "geopolitical_risk": 0.15,
    "implied_inflation": 0.15,
    "growth_expectations": 0.10,
}

if not np.isclose(sum(PARAMETER_WEIGHTS.values()), 1.0, atol=1e-9):
    raise ValueError("PARAMETER_WEIGHTS must sum to 1.0.")

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
    version="2.0.0",
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


@dataclass
class ModelInputs:
    crude_asset: MarketAsset
    dxy_asset: MarketAsset
    ovx_asset: MarketAsset
    geopolitical_risk_asset: MarketAsset
    tips_asset: MarketAsset
    treasury_asset: MarketAsset
    high_yield_asset: MarketAsset
    investment_grade_asset: MarketAsset
    inventory_snapshot: Dict[str, Any]


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


def weight_pct(parameter_key: str) -> float:
    return round(float(PARAMETER_WEIGHTS[parameter_key]) * 100.0, 2)


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
    elif "Adj Close" in data.columns:
        adjusted_close = pd.to_numeric(data["Adj Close"], errors="coerce")
        if adjusted_close.notna().any():
            data["Close"] = adjusted_close.combine_first(pd.to_numeric(data["Close"], errors="coerce"))

    if "Close" not in data.columns:
        raise ValueError("Yahoo Finance history does not contain a Close column.")

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data.index = pd.to_datetime(data.index)
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_convert(None)
    data = data.dropna(subset=["Close"]).sort_index()

    if len(data) < 200:
        raise ValueError("At least 200 daily observations are required for the model.")

    return data


def download_price_history(ticker: str, period: str = DEFAULT_LOOKBACK_PERIOD) -> pd.DataFrame:
    """Download daily history for one ticker."""
    try:
        history = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise ValueError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc

    try:
        return normalize_history(history)
    except Exception as exc:
        raise ValueError(f"Unable to normalize Yahoo Finance history for {ticker}: {exc}") from exc


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


def calculate_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    numerator_value = safe_float(numerator, default=None)
    denominator_value = safe_float(denominator, default=None)
    if numerator_value is None or denominator_value is None or denominator_value == 0.0:
        return None
    return float(numerator_value / denominator_value)


def calculate_ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce")
    denominator_values = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return numerator_values / denominator_values


def describe_bias(score: int) -> str:
    if score > 0:
        return "bullish"
    if score < 0:
        return "bearish"
    return "neutral"


def neutral_parameter(
    key: str,
    name: str,
    current_value: Optional[float],
    unit: str,
    benchmark_label: str,
    rationale: str,
    benchmark_value: Optional[float] = None,
    display_digits: int = 2,
) -> Dict[str, Any]:
    return {
        "key": key,
        "name": name,
        "weight": PARAMETER_WEIGHTS[key],
        "weight_pct": weight_pct(key),
        "score": 0,
        "bias": "neutral",
        "current_value": round_or_none(current_value, display_digits),
        "unit": unit,
        "benchmark_value": round_or_none(benchmark_value, display_digits),
        "benchmark_label": benchmark_label,
        "deviation_pct": None,
        "display_digits": display_digits,
        "rationale": rationale,
    }


def score_inventory(current_inventory: float, five_year_average: float) -> Dict[str, Any]:
    current_value = safe_float(current_inventory, default=None)
    average_value = safe_float(five_year_average, default=None)
    rationale = (
        "High inventories imply looser physical balances and weaker spot support; "
        "tight inventories imply stronger fundamental scarcity."
    )

    if current_value is None or average_value is None or average_value <= 0.0:
        return neutral_parameter(
            key="inventories",
            name="US Crude Inventories",
            current_value=current_value,
            unit="million bbl",
            benchmark_label="5Y average",
            benchmark_value=average_value,
            rationale=rationale,
        )

    deviation_pct = ((current_value - average_value) / average_value) * 100.0

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
        "weight_pct": weight_pct("inventories"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_value, 2),
        "unit": "million bbl",
        "benchmark_value": round(average_value, 2),
        "benchmark_label": "5Y average",
        "deviation_pct": round(deviation_pct, 2),
        "display_digits": 2,
        "rationale": rationale,
    }


def score_dxy(dxy_value: float) -> Dict[str, Any]:
    current_value = safe_float(dxy_value, default=None)
    rationale = (
        "Because crude is invoiced in U.S. dollars, a stronger USD usually "
        "tightens global purchasing power and pressures oil demand."
    )

    if current_value is None:
        return neutral_parameter(
            key="usd_strength",
            name="USD Strength (DXY)",
            current_value=current_value,
            unit="index",
            benchmark_label="Rule bands",
            rationale=rationale,
        )

    if current_value > 105.0:
        score = -2
    elif current_value >= 100.0:
        score = -1
    elif current_value >= 95.0:
        score = 0
    elif current_value >= 90.0:
        score = 1
    else:
        score = 2

    return {
        "key": "usd_strength",
        "name": "USD Strength (DXY)",
        "weight": PARAMETER_WEIGHTS["usd_strength"],
        "weight_pct": weight_pct("usd_strength"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_value, 2),
        "unit": "index",
        "benchmark_label": "Rule bands",
        "benchmark_value": None,
        "deviation_pct": None,
        "display_digits": 2,
        "rationale": rationale,
    }


def score_price_vs_200sma(current_price: float, sma_200: float) -> Dict[str, Any]:
    current_value = safe_float(current_price, default=None)
    benchmark_value = safe_float(sma_200, default=None)
    rationale = (
        "For this valuation model, deep downside extensions below the 200-day "
        "average are interpreted as potential fundamental cheapness."
    )

    if current_value is None or benchmark_value is None or benchmark_value <= 0.0:
        return neutral_parameter(
            key="trend_context",
            name="Trend & Momentum Context",
            current_value=None,
            unit="% vs 200D SMA",
            benchmark_label="200D SMA",
            benchmark_value=benchmark_value,
            rationale=rationale,
        )

    deviation_pct = ((current_value - benchmark_value) / benchmark_value) * 100.0

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
        "weight_pct": weight_pct("trend_context"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(deviation_pct, 2),
        "unit": "% vs 200D SMA",
        "benchmark_value": round(benchmark_value, 2),
        "benchmark_label": "200D SMA",
        "deviation_pct": round(deviation_pct, 2),
        "display_digits": 2,
        "rationale": rationale,
    }


def score_ovx(ovx_value: float) -> Dict[str, Any]:
    current_value = safe_float(ovx_value, default=None)
    rationale = (
        "Higher oil implied volatility often signals stress, liquidation pressure, "
        "or risk premia that can distort spot valuation."
    )

    if current_value is None:
        return neutral_parameter(
            key="ovx",
            name="Oil Volatility (OVX)",
            current_value=current_value,
            unit="vol index",
            benchmark_label="Rule bands",
            rationale=rationale,
        )

    if current_value > 45.0:
        score = -2
    elif current_value >= 35.0:
        score = -1
    elif current_value >= 25.0:
        score = 0
    elif current_value >= 15.0:
        score = 1
    else:
        score = 2

    return {
        "key": "ovx",
        "name": "Oil Volatility (OVX)",
        "weight": PARAMETER_WEIGHTS["ovx"],
        "weight_pct": weight_pct("ovx"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_value, 2),
        "unit": "vol index",
        "benchmark_label": "Rule bands",
        "benchmark_value": None,
        "deviation_pct": None,
        "display_digits": 2,
        "rationale": rationale,
    }


def score_geopolitical_risk(
    proxy_price: float,
    proxy_sma_50: float,
    proxy_sma_200: float,
) -> Dict[str, Any]:
    current_value = safe_float(proxy_price, default=None)
    sma_50_value = safe_float(proxy_sma_50, default=None)
    sma_200_value = safe_float(proxy_sma_200, default=None)
    rationale = (
        "The iShares U.S. Aerospace & Defense ETF (ITA) is used as a tradable "
        "geopolitical-risk proxy. Strength versus its own long-term trend is "
        "treated as a higher supply-risk premium for crude."
    )

    if (
        current_value is None
        or sma_50_value is None
        or sma_200_value is None
        or sma_200_value <= 0.0
    ):
        return neutral_parameter(
            key="geopolitical_risk",
            name="Geopolitical Risk Proxy (ITA)",
            current_value=current_value,
            unit="ETF price",
            benchmark_label="200D SMA",
            benchmark_value=sma_200_value,
            rationale=rationale,
        )

    deviation_pct = ((current_value - sma_200_value) / sma_200_value) * 100.0
    ma_spread_pct = ((sma_50_value - sma_200_value) / sma_200_value) * 100.0

    if deviation_pct >= 8.0 and ma_spread_pct > 1.0:
        score = 2
    elif deviation_pct > 2.0 or ma_spread_pct > 0.0:
        score = 1
    elif deviation_pct <= -8.0 and ma_spread_pct < -1.0:
        score = -2
    elif deviation_pct < -2.0 or ma_spread_pct < 0.0:
        score = -1
    else:
        score = 0

    return {
        "key": "geopolitical_risk",
        "name": "Geopolitical Risk Proxy (ITA)",
        "weight": PARAMETER_WEIGHTS["geopolitical_risk"],
        "weight_pct": weight_pct("geopolitical_risk"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_value, 2),
        "unit": "ETF price",
        "benchmark_value": round(sma_200_value, 2),
        "benchmark_label": "200D SMA",
        "deviation_pct": round(deviation_pct, 2),
        "auxiliary_value": round(ma_spread_pct, 2),
        "auxiliary_label": "50D vs 200D SMA (%)",
        "display_digits": 2,
        "rationale": rationale,
    }


def score_implied_inflation(
    tips_to_treasury_ratio: float,
    ratio_sma_50: float,
    ratio_sma_200: float,
) -> Dict[str, Any]:
    current_value = safe_float(tips_to_treasury_ratio, default=None)
    sma_50_value = safe_float(ratio_sma_50, default=None)
    sma_200_value = safe_float(ratio_sma_200, default=None)
    rationale = (
        "The TIP/IEF ratio compares Treasury Inflation-Protected Securities with "
        "intermediate nominal Treasuries. A rising ratio points to firmer "
        "market-implied inflation expectations, which usually supports nominal "
        "commodity prices."
    )

    if (
        current_value is None
        or sma_50_value is None
        or sma_200_value is None
        or sma_200_value <= 0.0
    ):
        return neutral_parameter(
            key="implied_inflation",
            name="Implied Inflation Expectations (TIP/IEF)",
            current_value=current_value,
            unit="ratio",
            benchmark_label="200D ratio",
            benchmark_value=sma_200_value,
            rationale=rationale,
            display_digits=4,
        )

    deviation_pct = ((current_value - sma_200_value) / sma_200_value) * 100.0
    ma_spread_pct = ((sma_50_value - sma_200_value) / sma_200_value) * 100.0

    if deviation_pct >= 2.5 and ma_spread_pct > 0.25:
        score = 2
    elif deviation_pct >= 0.75 or ma_spread_pct > 0.0:
        score = 1
    elif deviation_pct <= -2.5 and ma_spread_pct < -0.25:
        score = -2
    elif deviation_pct <= -0.75 or ma_spread_pct < 0.0:
        score = -1
    else:
        score = 0

    return {
        "key": "implied_inflation",
        "name": "Implied Inflation Expectations (TIP/IEF)",
        "weight": PARAMETER_WEIGHTS["implied_inflation"],
        "weight_pct": weight_pct("implied_inflation"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_value, 4),
        "unit": "ratio",
        "benchmark_value": round(sma_200_value, 4),
        "benchmark_label": "200D ratio",
        "deviation_pct": round(deviation_pct, 2),
        "auxiliary_value": round(ma_spread_pct, 2),
        "auxiliary_label": "50D vs 200D ratio (%)",
        "display_digits": 4,
        "rationale": rationale,
    }


def score_growth_expectations(
    high_yield_to_investment_grade_ratio: float,
    ratio_sma_50: float,
    ratio_sma_200: float,
) -> Dict[str, Any]:
    current_value = safe_float(high_yield_to_investment_grade_ratio, default=None)
    sma_50_value = safe_float(ratio_sma_50, default=None)
    sma_200_value = safe_float(ratio_sma_200, default=None)
    rationale = (
        "The HYG/LQD ratio tracks high-yield credit appetite versus investment-grade "
        "credit. High-yield outperformance implies tighter credit stress and "
        "stronger forward growth expectations for oil demand."
    )

    if (
        current_value is None
        or sma_50_value is None
        or sma_200_value is None
        or sma_200_value <= 0.0
    ):
        return neutral_parameter(
            key="growth_expectations",
            name="Forward Growth Expectations (HYG/LQD)",
            current_value=current_value,
            unit="ratio",
            benchmark_label="200D ratio",
            benchmark_value=sma_200_value,
            rationale=rationale,
            display_digits=4,
        )

    deviation_pct = ((current_value - sma_200_value) / sma_200_value) * 100.0
    ma_spread_pct = ((sma_50_value - sma_200_value) / sma_200_value) * 100.0

    if deviation_pct >= 3.0 and ma_spread_pct > 0.25:
        score = 2
    elif deviation_pct >= 1.0 or ma_spread_pct > 0.0:
        score = 1
    elif deviation_pct <= -3.0 and ma_spread_pct < -0.25:
        score = -2
    elif deviation_pct <= -1.0 or ma_spread_pct < 0.0:
        score = -1
    else:
        score = 0

    return {
        "key": "growth_expectations",
        "name": "Forward Growth Expectations (HYG/LQD)",
        "weight": PARAMETER_WEIGHTS["growth_expectations"],
        "weight_pct": weight_pct("growth_expectations"),
        "score": score,
        "bias": describe_bias(score),
        "current_value": round(current_value, 4),
        "unit": "ratio",
        "benchmark_value": round(sma_200_value, 4),
        "benchmark_label": "200D ratio",
        "deviation_pct": round(deviation_pct, 2),
        "auxiliary_value": round(ma_spread_pct, 2),
        "auxiliary_label": "50D vs 200D ratio (%)",
        "display_digits": 4,
        "rationale": rationale,
    }


def calculate_weighted_score(parameters: List[Dict[str, Any]]) -> Dict[str, float]:
    total_weight = sum(safe_float(parameter.get("weight"), default=0.0) or 0.0 for parameter in parameters)
    if total_weight <= 0.0:
        return {
            "weighted_raw": 0.0,
            "normalized_score": 0.0,
            "total_weight": 0.0,
        }

    weighted_raw = sum(
        (safe_float(parameter.get("score"), default=0.0) or 0.0)
        * (safe_float(parameter.get("weight"), default=0.0) or 0.0)
        for parameter in parameters
    )
    normalized_score = (weighted_raw / (2.0 * total_weight)) * 100.0
    return {
        "weighted_raw": float(weighted_raw),
        "normalized_score": float(normalized_score),
        "total_weight": float(total_weight),
    }


def derive_volatility_factor(ovx_value: float) -> float:
    """
    Convert OVX into a bounded multiplier for the fair-value adjustment.

    OVX is an annualized implied-volatility style measure. We divide by 100 to
    turn it into a dimensionless scale factor, then clip it to avoid extreme
    valuation swings in stressed markets.
    """
    normalized_ovx = safe_float(ovx_value, default=25.0) or 25.0
    return float(np.clip(normalized_ovx / 100.0, 0.15, 0.60))


def estimate_fair_value(
    sma_50: float,
    composite_score_normalized: float,
    volatility_factor: float,
) -> float:
    normalized_sma_50 = safe_float(sma_50, default=None)
    normalized_score = safe_float(composite_score_normalized, default=0.0) or 0.0
    normalized_volatility = safe_float(volatility_factor, default=0.25) or 0.25

    if normalized_sma_50 is None or normalized_sma_50 <= 0.0:
        raise ValueError("A positive 50-day SMA is required to estimate fair value.")

    return float(normalized_sma_50 * (1.0 + (normalized_score / 100.0) * normalized_volatility))


def calculate_valuation_gap(current_price: float, fair_value: float) -> float:
    normalized_price = safe_float(current_price, default=None)
    normalized_fair_value = safe_float(fair_value, default=None)

    if normalized_price is None or normalized_fair_value is None or normalized_fair_value <= 0.0:
        raise ValueError("A positive current price and fair value are required.")

    return float(((normalized_price - normalized_fair_value) / normalized_fair_value) * 100.0)


def map_temperature(gap_pct: float) -> Dict[str, str]:
    if gap_pct > 20.0:
        return {
            "label": "Very Hot",
            "emoji": "\U0001F525",
            "description": "Severely overvalued",
            "color": "red",
        }
    if gap_pct >= 10.0:
        return {
            "label": "Hot",
            "emoji": "\U0001F321\uFE0F",
            "description": "Overvalued",
            "color": "orange",
        }
    if gap_pct >= -10.0:
        return {
            "label": "Neutral",
            "emoji": "\u2696\uFE0F",
            "description": "Fairly valued",
            "color": "slate",
        }
    if gap_pct >= -20.0:
        return {
            "label": "Cold",
            "emoji": "\U0001F9CA",
            "description": "Undervalued",
            "color": "blue",
        }
    return {
        "label": "Very Cold",
        "emoji": "\u2744\uFE0F\u2744\uFE0F",
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


def add_ratio_indicators(
    frame: pd.DataFrame,
    numerator_column: str,
    denominator_column: str,
    ratio_column: str,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[ratio_column] = calculate_ratio_series(
        enriched[numerator_column],
        enriched[denominator_column],
    )
    enriched[f"{ratio_column}_sma_50"] = enriched[ratio_column].rolling(50).mean()
    enriched[f"{ratio_column}_sma_200"] = enriched[ratio_column].rolling(200).mean()
    return enriched


def build_historical_model_series(
    crude_asset: MarketAsset,
    dxy_asset: MarketAsset,
    ovx_asset: MarketAsset,
    geopolitical_risk_asset: MarketAsset,
    tips_asset: MarketAsset,
    treasury_asset: MarketAsset,
    high_yield_asset: MarketAsset,
    investment_grade_asset: MarketAsset,
    inventory_snapshot: Dict[str, Any],
    history_days: int = 90,
) -> List[Dict[str, Any]]:
    """
    Reconstruct the model's recent daily fair value series.

    Daily crude, DXY, OVX, defense ETF, TIPS, Treasury, high-yield credit, and
    investment-grade credit data are historical. Inventory score is held
    constant at the latest snapshot unless a full historical EIA inventory
    time series is added later.
    """
    crude_frame = crude_asset.history[["Close"]].copy()
    crude_frame = crude_frame.rename(columns={"Close": "actual_price"})
    crude_frame["crude_sma_50"] = crude_frame["actual_price"].rolling(50).mean()
    crude_frame["crude_sma_200"] = crude_frame["actual_price"].rolling(200).mean()

    dxy_frame = dxy_asset.history[["Close"]].rename(columns={"Close": "dxy_close"})
    ovx_frame = ovx_asset.history[["Close"]].rename(columns={"Close": "ovx_close"})

    geopolitical_frame = geopolitical_risk_asset.history[["Close"]].copy()
    geopolitical_frame = geopolitical_frame.rename(columns={"Close": "geopolitical_close"})
    geopolitical_frame["geopolitical_sma_50"] = geopolitical_frame["geopolitical_close"].rolling(50).mean()
    geopolitical_frame["geopolitical_sma_200"] = geopolitical_frame["geopolitical_close"].rolling(200).mean()

    tips_frame = tips_asset.history[["Close"]].rename(columns={"Close": "tips_close"})
    treasury_frame = treasury_asset.history[["Close"]].rename(columns={"Close": "treasury_close"})
    high_yield_frame = high_yield_asset.history[["Close"]].rename(columns={"Close": "high_yield_close"})
    investment_grade_frame = investment_grade_asset.history[["Close"]].rename(
        columns={"Close": "investment_grade_close"}
    )

    combined = crude_frame.join(dxy_frame, how="left")
    combined = combined.join(ovx_frame, how="left")
    combined = combined.join(geopolitical_frame, how="left")
    combined = combined.join(tips_frame, how="left")
    combined = combined.join(treasury_frame, how="left")
    combined = combined.join(high_yield_frame, how="left")
    combined = combined.join(investment_grade_frame, how="left")

    forward_fill_columns = [
        "dxy_close",
        "ovx_close",
        "geopolitical_close",
        "geopolitical_sma_50",
        "geopolitical_sma_200",
        "tips_close",
        "treasury_close",
        "high_yield_close",
        "investment_grade_close",
    ]
    combined[forward_fill_columns] = combined[forward_fill_columns].ffill()
    combined = add_ratio_indicators(
        combined,
        numerator_column="tips_close",
        denominator_column="treasury_close",
        ratio_column="implied_inflation_ratio",
    )
    combined = add_ratio_indicators(
        combined,
        numerator_column="high_yield_close",
        denominator_column="investment_grade_close",
        ratio_column="growth_expectations_ratio",
    )

    inventory_score = score_inventory(
        inventory_snapshot["current_inventory_million_bbl"],
        inventory_snapshot["five_year_average_million_bbl"],
    )["score"]

    required_columns = [
        "actual_price",
        "crude_sma_50",
        "crude_sma_200",
        "dxy_close",
        "ovx_close",
        "geopolitical_close",
        "geopolitical_sma_50",
        "geopolitical_sma_200",
        "implied_inflation_ratio",
        "implied_inflation_ratio_sma_50",
        "implied_inflation_ratio_sma_200",
        "growth_expectations_ratio",
        "growth_expectations_ratio_sma_50",
        "growth_expectations_ratio_sma_200",
    ]
    candidate_rows = combined.dropna(subset=required_columns).tail(history_days).copy()

    if candidate_rows.empty:
        return []

    candidate_rows["forward_return_5d_pct"] = (
        (candidate_rows["actual_price"].shift(-5) - candidate_rows["actual_price"])
        / candidate_rows["actual_price"]
    ) * 100.0
    candidate_rows["forward_return_10d_pct"] = (
        (candidate_rows["actual_price"].shift(-10) - candidate_rows["actual_price"])
        / candidate_rows["actual_price"]
    ) * 100.0

    history_records: List[Dict[str, Any]] = []
    for index, row in candidate_rows.iterrows():
        parameters = [
            {"score": inventory_score, "weight": PARAMETER_WEIGHTS["inventories"]},
            score_dxy(float(row["dxy_close"])),
            score_price_vs_200sma(
                current_price=float(row["actual_price"]),
                sma_200=float(row["crude_sma_200"]),
            ),
            score_ovx(float(row["ovx_close"])),
            score_geopolitical_risk(
                proxy_price=float(row["geopolitical_close"]),
                proxy_sma_50=float(row["geopolitical_sma_50"]),
                proxy_sma_200=float(row["geopolitical_sma_200"]),
            ),
            score_implied_inflation(
                tips_to_treasury_ratio=float(row["implied_inflation_ratio"]),
                ratio_sma_50=float(row["implied_inflation_ratio_sma_50"]),
                ratio_sma_200=float(row["implied_inflation_ratio_sma_200"]),
            ),
            score_growth_expectations(
                high_yield_to_investment_grade_ratio=float(row["growth_expectations_ratio"]),
                ratio_sma_50=float(row["growth_expectations_ratio_sma_50"]),
                ratio_sma_200=float(row["growth_expectations_ratio_sma_200"]),
            ),
        ]

        composite = calculate_weighted_score(parameters)
        normalized_score = composite["normalized_score"]
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


def empty_signal_stats(signal_label: str) -> Dict[str, Any]:
    return {
        "signal": signal_label,
        "expected_direction": SIGNAL_EXPECTED_DIRECTION[signal_label],
        "occurrence_count": 0,
        "valid_5d_samples": 0,
        "valid_10d_samples": 0,
        "avg_5d_return_pct": None,
        "avg_10d_return_pct": None,
        "win_rate_pct": None,
    }


def build_backtest_summary(history_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate forward-return performance by signal regime.

    Win rate is evaluated on the 10-day horizon:
    - Cold / Very Cold: a positive forward return is a win.
    - Hot / Very Hot: a negative forward return is a win.
    """
    history_frame = pd.DataFrame(history_records)
    signals: Dict[str, Dict[str, Any]] = {}

    if history_frame.empty or "temperature_label" not in history_frame.columns:
        signals = {signal_label: empty_signal_stats(signal_label) for signal_label in SIGNAL_BACKTEST_ORDER}
    else:
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


def current_ratio_from_assets(numerator_asset: MarketAsset, denominator_asset: MarketAsset) -> Optional[float]:
    current_ratio = calculate_ratio(numerator_asset.current_price, denominator_asset.current_price)
    if current_ratio is not None:
        return current_ratio
    return calculate_ratio(numerator_asset.last_close, denominator_asset.last_close)


def latest_rolling_value(series: pd.Series, window: int) -> Optional[float]:
    rolling_value = safe_float(series.rolling(window).mean().iloc[-1], default=None)
    return rolling_value


def attach_parameter_metadata(
    parameters: List[Dict[str, Any]],
    inventory_snapshot: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source_notes = {
        "inventories": inventory_snapshot["note"],
        "usd_strength": "Yahoo Finance DXY feed.",
        "trend_context": "Yahoo Finance WTI futures feed.",
        "ovx": "Yahoo Finance CBOE crude-oil volatility index feed.",
        "geopolitical_risk": "Yahoo Finance ITA ETF feed.",
        "implied_inflation": "Yahoo Finance TIP and IEF ETF feeds.",
        "growth_expectations": "Yahoo Finance HYG and LQD ETF feeds.",
    }

    for parameter in parameters:
        parameter["weighted_contribution"] = round_or_none(
            (safe_float(parameter.get("score"), default=0.0) or 0.0)
            * (safe_float(parameter.get("weight"), default=0.0) or 0.0),
            4,
        )
        if parameter["key"] == "inventories":
            parameter["source"] = inventory_snapshot["source"]
        else:
            parameter["source"] = "yfinance"
        parameter["source_note"] = source_notes.get(parameter["key"], "")

    return parameters


def build_crude_oil_valuation_response(model_inputs: ModelInputs) -> Dict[str, Any]:
    crude_asset = model_inputs.crude_asset
    dxy_asset = model_inputs.dxy_asset
    ovx_asset = model_inputs.ovx_asset
    geopolitical_risk_asset = model_inputs.geopolitical_risk_asset
    tips_asset = model_inputs.tips_asset
    treasury_asset = model_inputs.treasury_asset
    high_yield_asset = model_inputs.high_yield_asset
    investment_grade_asset = model_inputs.investment_grade_asset
    inventory_snapshot = model_inputs.inventory_snapshot

    technicals = build_technical_summary(crude_asset)
    crude_sma_50 = technicals["moving_averages"]["sma_50"]
    crude_sma_200 = technicals["moving_averages"]["sma_200"]

    geopolitical_close = geopolitical_risk_asset.history["Close"]
    geopolitical_sma_50 = latest_rolling_value(geopolitical_close, 50)
    geopolitical_sma_200 = latest_rolling_value(geopolitical_close, 200)

    implied_inflation_ratio_series = calculate_ratio_series(
        tips_asset.history["Close"],
        treasury_asset.history["Close"],
    ).dropna()
    implied_inflation_ratio = current_ratio_from_assets(tips_asset, treasury_asset)
    implied_inflation_sma_50 = latest_rolling_value(implied_inflation_ratio_series, 50)
    implied_inflation_sma_200 = latest_rolling_value(implied_inflation_ratio_series, 200)

    growth_expectations_ratio_series = calculate_ratio_series(
        high_yield_asset.history["Close"],
        investment_grade_asset.history["Close"],
    ).dropna()
    growth_expectations_ratio = current_ratio_from_assets(high_yield_asset, investment_grade_asset)
    growth_expectations_sma_50 = latest_rolling_value(growth_expectations_ratio_series, 50)
    growth_expectations_sma_200 = latest_rolling_value(growth_expectations_ratio_series, 200)

    parameters = [
        score_inventory(
            inventory_snapshot["current_inventory_million_bbl"],
            inventory_snapshot["five_year_average_million_bbl"],
        ),
        score_dxy(dxy_asset.current_price),
        score_price_vs_200sma(crude_asset.current_price, crude_sma_200),
        score_ovx(ovx_asset.current_price),
        score_geopolitical_risk(
            geopolitical_risk_asset.current_price,
            geopolitical_sma_50,
            geopolitical_sma_200,
        ),
        score_implied_inflation(
            implied_inflation_ratio,
            implied_inflation_sma_50,
            implied_inflation_sma_200,
        ),
        score_growth_expectations(
            growth_expectations_ratio,
            growth_expectations_sma_50,
            growth_expectations_sma_200,
        ),
    ]
    parameters = attach_parameter_metadata(parameters, inventory_snapshot)

    composite = calculate_weighted_score(parameters)
    volatility_factor = derive_volatility_factor(ovx_asset.current_price)
    estimated_fair_value = estimate_fair_value(
        sma_50=crude_sma_50,
        composite_score_normalized=composite["normalized_score"],
        volatility_factor=volatility_factor,
    )
    valuation_gap_pct = calculate_valuation_gap(crude_asset.current_price, estimated_fair_value)
    temperature = map_temperature(valuation_gap_pct)

    historical_model_series = build_historical_model_series(
        crude_asset=crude_asset,
        dxy_asset=dxy_asset,
        ovx_asset=ovx_asset,
        geopolitical_risk_asset=geopolitical_risk_asset,
        tips_asset=tips_asset,
        treasury_asset=treasury_asset,
        high_yield_asset=high_yield_asset,
        investment_grade_asset=investment_grade_asset,
        inventory_snapshot=inventory_snapshot,
        history_days=90,
    )
    backtest_summary = build_backtest_summary(historical_model_series)

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
            "geopolitical_risk": build_market_snapshot(geopolitical_risk_asset),
            "tips": build_market_snapshot(tips_asset),
            "treasury": build_market_snapshot(treasury_asset),
            "high_yield_credit": build_market_snapshot(high_yield_asset),
            "investment_grade_credit": build_market_snapshot(investment_grade_asset),
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
            "derived_proxies": {
                "implied_inflation_ratio": round_or_none(implied_inflation_ratio, 4),
                "growth_expectations_ratio": round_or_none(growth_expectations_ratio, 4),
            },
        },
        "fundamentals": {
            "parameters": parameters,
            "weighted_raw_score": round(composite["weighted_raw"], 4),
            "normalized_score": round(composite["normalized_score"], 2),
            "total_weight": round(composite["total_weight"], 4),
            "parameter_count": len(parameters),
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
                "The macro block uses ITA for geopolitical risk, TIP/IEF for implied inflation, and HYG/LQD for forward growth expectations.",
                "Inventory scoring uses EIA data when EIA_API_KEY is available; otherwise it falls back to a transparent mock.",
                "Historical model reconstruction uses fully historical market data, while the inventory score is held constant unless a historical EIA inventory series is added.",
                "Backtest win rate is based on whether price reverted toward fair value over the next 10 trading days.",
                "The trend-context factor is intentionally contrarian to reflect valuation stretch versus long-run trend.",
            ],
        },
    }


def fetch_all_model_inputs() -> ModelInputs:
    """
    Fetch all required inputs sequentially.

    yfinance can behave inconsistently under concurrent access in some
    environments, so the production-safe choice for this compact model is
    to download the symbols one after another.
    """
    crude_asset = fetch_market_asset(CRUDE_TICKER, "Crude Oil Futures (WTI)")
    dxy_asset = fetch_market_asset(DXY_TICKER, "US Dollar Index")
    ovx_asset = fetch_market_asset(OVX_TICKER, "CBOE Crude Oil Volatility Index")
    geopolitical_risk_asset = fetch_market_asset(
        GEOPOLITICAL_RISK_TICKER,
        "iShares U.S. Aerospace & Defense ETF",
    )
    tips_asset = fetch_market_asset(TIPS_TICKER, "iShares TIPS Bond ETF")
    treasury_asset = fetch_market_asset(TREASURY_TICKER, "iShares 7-10 Year Treasury Bond ETF")
    high_yield_asset = fetch_market_asset(HIGH_YIELD_TICKER, "iShares iBoxx High Yield Corporate Bond ETF")
    investment_grade_asset = fetch_market_asset(
        INVESTMENT_GRADE_TICKER,
        "iShares iBoxx Investment Grade Corporate Bond ETF",
    )
    inventory_snapshot = fetch_eia_inventory_snapshot()

    return ModelInputs(
        crude_asset=crude_asset,
        dxy_asset=dxy_asset,
        ovx_asset=ovx_asset,
        geopolitical_risk_asset=geopolitical_risk_asset,
        tips_asset=tips_asset,
        treasury_asset=treasury_asset,
        high_yield_asset=high_yield_asset,
        investment_grade_asset=investment_grade_asset,
        inventory_snapshot=inventory_snapshot,
    )


def load_crude_oil_valuation_snapshot() -> Dict[str, Any]:
    """
    Synchronous loader shared by the API layer and Python-native dashboard UIs.
    """
    model_inputs = fetch_all_model_inputs()
    return build_crude_oil_valuation_response(model_inputs)


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
