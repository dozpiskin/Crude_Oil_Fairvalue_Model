"""
FastAPI backend for a crude-oil valuation temperature gauge.

Run locally with:
    uvicorn main:app --reload
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
SUBSTITUTION_PROXY_TICKER = "UNG"
CHINA_MARKET_TICKER = "FXI"
DEFAULT_LOOKBACK_PERIOD = "2y"

PARAMETER_WEIGHTS = {
    "inventories": 0.17,
    "usd_strength": 0.12,
    "trend_context": 0.08,
    "ovx": 0.12,
    "geopolitical_risk": 0.13,
    "implied_inflation": 0.13,
    "growth_expectations": 0.13,
    "substitution_risk": 0.12,
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
    version="3.0.0",
    description=(
        "Academic portfolio API that scores crude-oil fundamentals, applies "
        "macro regime overlays, estimates fair value, and maps the valuation "
        "gap to a temperature gauge."
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
    substitution_proxy_asset: MarketAsset
    china_market_asset: MarketAsset
    inventory_snapshot: Dict[str, Any]


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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


def clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def weight_pct(parameter_key: str) -> float:
    return round(float(PARAMETER_WEIGHTS[parameter_key]) * 100.0, 2)


def to_utc_iso(timestamp: Any) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
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
    history = download_price_history(ticker)
    last_close = float(history["Close"].iloc[-1])
    last_timestamp = to_utc_iso(history.index[-1])
    current_price = last_close
    currency = "USD"

    try:
        fast_info = yf.Ticker(ticker).fast_info
        current_price = safe_float(
            fast_info.get("lastPrice")
            or fast_info.get("regularMarketPrice")
            or fast_info.get("previousClose"),
            default=last_close,
        ) or last_close
        currency = str(fast_info.get("currency") or currency)
    except Exception as exc:
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
    today = datetime.now(timezone.utc)
    day_of_year = today.timetuple().tm_yday
    seasonal_component = 16.0 * np.sin(2.0 * np.pi * ((day_of_year - 40) / 365.25))
    current_inventory = round(428.0 + seasonal_component, 1)

    return {
        "current_inventory_million_bbl": current_inventory,
        "five_year_average_million_bbl": 423.0,
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
        return {
            "current_inventory_million_bbl": float(data["value_million_bbl"].iloc[-1]),
            "five_year_average_million_bbl": float(data["value_million_bbl"].tail(trailing_window).mean()),
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
        "geopolitical-risk proxy. Strength versus its long-term trend is treated "
        "as a higher supply-risk premium for crude."
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
        "market-implied inflation expectations."
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
        "credit. High-yield outperformance implies stronger forward growth "
        "expectations for oil demand."
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


def score_substitution_risk(
    crude_to_substitute_ratio: float,
    ratio_sma_50: float,
    ratio_sma_200: float,
    crude_price: float,
) -> Dict[str, Any]:
    current_value = safe_float(crude_to_substitute_ratio, default=None)
    sma_50_value = safe_float(ratio_sma_50, default=None)
    sma_200_value = safe_float(ratio_sma_200, default=None)
    crude_price_value = safe_float(crude_price, default=None)
    rationale = (
        "The WTI/UNG ratio measures when oil outruns a gas substitute. If crude "
        "detaches upward while WTI is above the pain thresholds, multi-source "
        "facilities and fuel switching become more attractive."
    )

    if (
        current_value is None
        or sma_50_value is None
        or sma_200_value is None
        or crude_price_value is None
        or sma_200_value <= 0.0
    ):
        return neutral_parameter(
            key="substitution_risk",
            name="Substitution Pressure (WTI/UNG)",
            current_value=current_value,
            unit="ratio",
            benchmark_label="200D ratio",
            benchmark_value=sma_200_value,
            rationale=rationale,
            display_digits=4,
        )

    deviation_pct = ((current_value - sma_200_value) / sma_200_value) * 100.0
    ma_spread_pct = ((sma_50_value - sma_200_value) / sma_200_value) * 100.0

    if crude_price_value >= 150.0 and (deviation_pct >= 12.0 or ma_spread_pct >= 4.0):
        score = -2
    elif crude_price_value >= 120.0 and (deviation_pct >= 8.0 or ma_spread_pct >= 2.0):
        score = -2
    elif crude_price_value >= 120.0 and (deviation_pct >= 3.0 or ma_spread_pct > 0.0):
        score = -1
    elif crude_price_value < 95.0 and deviation_pct <= -8.0 and ma_spread_pct < 0.0:
        score = 2
    elif crude_price_value < 105.0 and deviation_pct <= -3.0:
        score = 1
    else:
        score = 0

    return {
        "key": "substitution_risk",
        "name": "Substitution Pressure (WTI/UNG)",
        "weight": PARAMETER_WEIGHTS["substitution_risk"],
        "weight_pct": weight_pct("substitution_risk"),
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
        return {"weighted_raw": 0.0, "normalized_score": 0.0, "total_weight": 0.0}

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


def build_demand_destruction_profile(current_price: float) -> Dict[str, Any]:
    price_value = safe_float(current_price, default=None)
    if price_value is None:
        return {
            "status_label": "Unavailable",
            "risk_level": "unknown",
            "threshold_1_price": 120.0,
            "threshold_2_price": 150.0,
            "current_price": None,
            "distance_to_120": None,
            "distance_to_150": None,
            "composite_score_penalty": 0.0,
            "fair_value_discount_pct": 0.0,
            "fair_value_multiplier": 1.0,
            "gauge_uplift_pct": 0.0,
            "temperature_pressure_multiplier": 1.0,
            "notes": [],
        }

    distance_to_120 = price_value - 120.0
    distance_to_150 = price_value - 150.0

    if price_value < 120.0:
        status_label = "Normal Demand"
        risk_level = "low"
        composite_score_penalty = 0.0
        fair_value_discount_pct = 0.0
        gauge_uplift_pct = 0.0
        temperature_pressure_multiplier = 1.0
        notes = [
            "Below $120, real and speculative demand can still coexist without acute demand destruction.",
        ]
    elif price_value < 150.0:
        severity = (price_value - 120.0) / 30.0
        status_label = "Demand Destruction Risk"
        risk_level = "elevated" if severity < 0.5 else "high"
        composite_score_penalty = 8.0 + (12.0 * severity)
        fair_value_discount_pct = 5.0 + (10.0 * severity)
        gauge_uplift_pct = 2.0 + (6.0 * severity)
        temperature_pressure_multiplier = 1.05 + (0.15 * severity)
        notes = [
            "Above $120, low value-added producers start losing pass-through power.",
            "Shutdown risk rises as users switch to multi-source or gas-heavy facilities.",
        ]
    else:
        severity = min((price_value - 150.0) / 30.0, 1.0)
        status_label = "Extreme Demand Destruction"
        risk_level = "extreme"
        composite_score_penalty = 30.0 + (30.0 * severity)
        fair_value_discount_pct = 20.0 + (15.0 * severity)
        gauge_uplift_pct = 10.0 + (10.0 * severity)
        temperature_pressure_multiplier = 1.25 + (0.25 * severity)
        notes = [
            "At or above $150, demand destruction becomes bilateral and recession risk dominates.",
            "Labor cost-free production, forced shutdowns, and sectoral liquidation pressure rise materially.",
        ]

    fair_value_multiplier = max(0.35, 1.0 - (fair_value_discount_pct / 100.0))

    return {
        "status_label": status_label,
        "risk_level": risk_level,
        "threshold_1_price": 120.0,
        "threshold_2_price": 150.0,
        "current_price": round(price_value, 2),
        "distance_to_120": round(distance_to_120, 2),
        "distance_to_150": round(distance_to_150, 2),
        "composite_score_penalty": round(composite_score_penalty, 2),
        "fair_value_discount_pct": round(fair_value_discount_pct, 2),
        "fair_value_multiplier": round(fair_value_multiplier, 4),
        "gauge_uplift_pct": round(gauge_uplift_pct, 2),
        "temperature_pressure_multiplier": round(temperature_pressure_multiplier, 4),
        "notes": notes,
    }


def build_stagflation_monitor(
    implied_inflation_parameter: Dict[str, Any],
    growth_expectations_parameter: Dict[str, Any],
    china_price: float,
    china_sma_200: float,
    oil_price: float,
    oil_sma_50: float,
) -> Dict[str, Any]:
    china_price_value = safe_float(china_price, default=None)
    china_sma_200_value = safe_float(china_sma_200, default=None)
    oil_price_value = safe_float(oil_price, default=None)
    oil_sma_50_value = safe_float(oil_sma_50, default=None)

    inflation_score = int(safe_float(implied_inflation_parameter.get("score"), default=0.0) or 0.0)
    growth_score = int(safe_float(growth_expectations_parameter.get("score"), default=0.0) or 0.0)
    inflation_deviation = safe_float(implied_inflation_parameter.get("deviation_pct"), default=0.0) or 0.0
    growth_deviation = safe_float(growth_expectations_parameter.get("deviation_pct"), default=0.0) or 0.0

    china_deviation_pct = None
    china_is_weak = False
    if (
        china_price_value is not None
        and china_sma_200_value is not None
        and china_sma_200_value > 0.0
    ):
        china_deviation_pct = ((china_price_value - china_sma_200_value) / china_sma_200_value) * 100.0
        china_is_weak = china_deviation_pct <= -8.0

    oil_is_rising = (
        oil_price_value is not None
        and oil_sma_50_value is not None
        and oil_sma_50_value > 0.0
        and oil_price_value > oil_sma_50_value
    )

    inflation_is_hot = inflation_score >= 1 or inflation_deviation >= 1.0
    growth_is_falling = growth_score <= -1 or growth_deviation < 0.0
    stagflation_alert = inflation_is_hot and growth_is_falling
    china_recession_risk = china_is_weak and oil_is_rising

    composite_score_penalty = 0.0
    fair_value_discount_pct = 0.0
    gauge_uplift_pct = 0.0
    notes: List[str] = []

    if stagflation_alert:
        composite_score_penalty += 8.0
        fair_value_discount_pct += 4.0
        gauge_uplift_pct += 3.0
        notes.append("Inflation expectations are elevated while forward growth is deteriorating.")

    if china_recession_risk:
        composite_score_penalty += 6.0
        fair_value_discount_pct += 4.0
        gauge_uplift_pct += 4.0
        notes.append("China is materially below its 200-day trend while oil is still rising, increasing global recession risk.")

    if stagflation_alert and china_recession_risk:
        status_label = "Stagflation + China Recession Risk"
        risk_level = "high"
    elif stagflation_alert:
        status_label = "Stagflation Alert"
        risk_level = "elevated"
    elif china_recession_risk:
        status_label = "China Demand Risk"
        risk_level = "elevated"
    else:
        status_label = "No Stagflation Alert"
        risk_level = "low"
        notes.append("Inflation and growth proxies are not yet confirming a stagflation regime.")

    fair_value_multiplier = max(0.70, 1.0 - (fair_value_discount_pct / 100.0))

    return {
        "status_label": status_label,
        "risk_level": risk_level,
        "stagflation_alert": stagflation_alert,
        "china_recession_risk": china_recession_risk,
        "china_market_ticker": CHINA_MARKET_TICKER,
        "china_market_price": round_or_none(china_price_value),
        "china_market_sma_200": round_or_none(china_sma_200_value),
        "china_market_deviation_pct": round_or_none(china_deviation_pct),
        "composite_score_penalty": round(composite_score_penalty, 2),
        "fair_value_discount_pct": round(fair_value_discount_pct, 2),
        "fair_value_multiplier": round(fair_value_multiplier, 4),
        "gauge_uplift_pct": round(gauge_uplift_pct, 2),
        "notes": notes,
    }


def apply_regime_overlays(
    current_price: float,
    sma_50: float,
    base_composite_score: float,
    volatility_factor: float,
    demand_profile: Dict[str, Any],
    stagflation_monitor: Dict[str, Any],
) -> Dict[str, Any]:
    demand_score_penalty = safe_float(demand_profile.get("composite_score_penalty"), default=0.0) or 0.0
    stagflation_score_penalty = safe_float(stagflation_monitor.get("composite_score_penalty"), default=0.0) or 0.0

    adjusted_composite_score = clamp(
        base_composite_score - demand_score_penalty - stagflation_score_penalty,
        -100.0,
        100.0,
    )
    fair_value_before_overlays = estimate_fair_value(
        sma_50=sma_50,
        composite_score_normalized=base_composite_score,
        volatility_factor=volatility_factor,
    )
    fair_value_after_score_overlay = estimate_fair_value(
        sma_50=sma_50,
        composite_score_normalized=adjusted_composite_score,
        volatility_factor=volatility_factor,
    )
    adjusted_fair_value = fair_value_after_score_overlay
    adjusted_fair_value *= safe_float(demand_profile.get("fair_value_multiplier"), default=1.0) or 1.0
    adjusted_fair_value *= safe_float(stagflation_monitor.get("fair_value_multiplier"), default=1.0) or 1.0
    adjusted_fair_value = max(adjusted_fair_value, 1.0)

    base_valuation_gap_pct = calculate_valuation_gap(current_price, fair_value_before_overlays)
    overlay_gap_pct = calculate_valuation_gap(current_price, adjusted_fair_value)
    demand_gap_push = (
        (safe_float(demand_profile.get("gauge_uplift_pct"), default=0.0) or 0.0)
        * (safe_float(demand_profile.get("temperature_pressure_multiplier"), default=1.0) or 1.0)
    )
    stagflation_gap_push = safe_float(stagflation_monitor.get("gauge_uplift_pct"), default=0.0) or 0.0
    adjusted_valuation_gap_pct = overlay_gap_pct + demand_gap_push + stagflation_gap_push

    return {
        "base_composite_score": round(base_composite_score, 2),
        "adjusted_composite_score": round(adjusted_composite_score, 2),
        "fair_value_before_overlays": round(fair_value_before_overlays, 2),
        "fair_value_after_score_overlay": round(fair_value_after_score_overlay, 2),
        "adjusted_fair_value": round(adjusted_fair_value, 2),
        "base_valuation_gap_pct": round(base_valuation_gap_pct, 2),
        "overlay_valuation_gap_pct": round(overlay_gap_pct, 2),
        "adjusted_valuation_gap_pct": round(adjusted_valuation_gap_pct, 2),
        "total_overlay_gap_push_pct": round(demand_gap_push + stagflation_gap_push, 2),
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

    price_history = [
        {
            "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
            "close": round_or_none(row["Close"]),
            "sma_50": round_or_none(row["sma_50"]),
            "sma_200": round_or_none(row["sma_200"]),
        }
        for index, row in recent_history.tail(90).iterrows()
    ]

    return {
        "rsi_14": {"value": round(latest_rsi, 2), "status": rsi_status},
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


def latest_rolling_value(series: pd.Series, window: int) -> Optional[float]:
    return safe_float(series.rolling(window).mean().iloc[-1], default=None)


def current_ratio_from_assets(numerator_asset: MarketAsset, denominator_asset: MarketAsset) -> Optional[float]:
    current_ratio = calculate_ratio(numerator_asset.current_price, denominator_asset.current_price)
    if current_ratio is not None:
        return current_ratio
    return calculate_ratio(numerator_asset.last_close, denominator_asset.last_close)


def build_parameter_map(parameters: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {parameter["key"]: parameter for parameter in parameters}


def build_historical_model_series(
    crude_asset: MarketAsset,
    dxy_asset: MarketAsset,
    ovx_asset: MarketAsset,
    geopolitical_risk_asset: MarketAsset,
    tips_asset: MarketAsset,
    treasury_asset: MarketAsset,
    high_yield_asset: MarketAsset,
    investment_grade_asset: MarketAsset,
    substitution_proxy_asset: MarketAsset,
    china_market_asset: MarketAsset,
    inventory_snapshot: Dict[str, Any],
    history_days: int = 90,
) -> List[Dict[str, Any]]:
    crude_frame = crude_asset.history[["Close"]].copy().rename(columns={"Close": "actual_price"})
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
    substitution_frame = substitution_proxy_asset.history[["Close"]].rename(
        columns={"Close": "substitution_close"}
    )
    china_frame = china_market_asset.history[["Close"]].copy().rename(columns={"Close": "china_close"})
    china_frame["china_sma_200"] = china_frame["china_close"].rolling(200).mean()

    combined = crude_frame.join(dxy_frame, how="left")
    combined = combined.join(ovx_frame, how="left")
    combined = combined.join(geopolitical_frame, how="left")
    combined = combined.join(tips_frame, how="left")
    combined = combined.join(treasury_frame, how="left")
    combined = combined.join(high_yield_frame, how="left")
    combined = combined.join(investment_grade_frame, how="left")
    combined = combined.join(substitution_frame, how="left")
    combined = combined.join(china_frame, how="left")

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
        "substitution_close",
        "china_close",
        "china_sma_200",
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
    combined = add_ratio_indicators(
        combined,
        numerator_column="actual_price",
        denominator_column="substitution_close",
        ratio_column="substitution_ratio",
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
        "substitution_ratio",
        "substitution_ratio_sma_50",
        "substitution_ratio_sma_200",
        "china_close",
        "china_sma_200",
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
            score_price_vs_200sma(float(row["actual_price"]), float(row["crude_sma_200"])),
            score_ovx(float(row["ovx_close"])),
            score_geopolitical_risk(
                float(row["geopolitical_close"]),
                float(row["geopolitical_sma_50"]),
                float(row["geopolitical_sma_200"]),
            ),
            score_implied_inflation(
                float(row["implied_inflation_ratio"]),
                float(row["implied_inflation_ratio_sma_50"]),
                float(row["implied_inflation_ratio_sma_200"]),
            ),
            score_growth_expectations(
                float(row["growth_expectations_ratio"]),
                float(row["growth_expectations_ratio_sma_50"]),
                float(row["growth_expectations_ratio_sma_200"]),
            ),
            score_substitution_risk(
                float(row["substitution_ratio"]),
                float(row["substitution_ratio_sma_50"]),
                float(row["substitution_ratio_sma_200"]),
                float(row["actual_price"]),
            ),
        ]
        parameter_map = build_parameter_map(parameters[1:])
        composite = calculate_weighted_score(parameters)
        volatility_factor = derive_volatility_factor(float(row["ovx_close"]))
        demand_profile = build_demand_destruction_profile(float(row["actual_price"]))
        stagflation_monitor = build_stagflation_monitor(
            implied_inflation_parameter=parameter_map["implied_inflation"],
            growth_expectations_parameter=parameter_map["growth_expectations"],
            china_price=float(row["china_close"]),
            china_sma_200=float(row["china_sma_200"]),
            oil_price=float(row["actual_price"]),
            oil_sma_50=float(row["crude_sma_50"]),
        )
        overlay_result = apply_regime_overlays(
            current_price=float(row["actual_price"]),
            sma_50=float(row["crude_sma_50"]),
            base_composite_score=composite["normalized_score"],
            volatility_factor=volatility_factor,
            demand_profile=demand_profile,
            stagflation_monitor=stagflation_monitor,
        )
        temperature = map_temperature(overlay_result["adjusted_valuation_gap_pct"])

        history_records.append(
            {
                "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
                "actual_price": round(float(row["actual_price"]), 2),
                "fair_value": round(overlay_result["adjusted_fair_value"], 2),
                "base_fair_value": round(overlay_result["fair_value_before_overlays"], 2),
                "valuation_gap_pct": round(overlay_result["adjusted_valuation_gap_pct"], 2),
                "base_valuation_gap_pct": round(overlay_result["base_valuation_gap_pct"], 2),
                "normalized_score": round(overlay_result["adjusted_composite_score"], 2),
                "base_normalized_score": round(overlay_result["base_composite_score"], 2),
                "forward_return_5d_pct": round_or_none(row["forward_return_5d_pct"]),
                "forward_return_10d_pct": round_or_none(row["forward_return_10d_pct"]),
                "temperature_label": temperature["label"],
                "temperature_color": temperature["color"],
                "demand_destruction_status": demand_profile["status_label"],
                "stagflation_status": stagflation_monitor["status_label"],
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
                "avg_10d_return_pct": round_or_none(forward_10d.mean()) if not forward_10d.empty else None,
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
        "substitution_risk": "Yahoo Finance WTI and UNG feeds.",
    }

    for parameter in parameters:
        parameter["weighted_contribution"] = round_or_none(
            (safe_float(parameter.get("score"), default=0.0) or 0.0)
            * (safe_float(parameter.get("weight"), default=0.0) or 0.0),
            4,
        )
        parameter["source"] = inventory_snapshot["source"] if parameter["key"] == "inventories" else "yfinance"
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
    substitution_proxy_asset = model_inputs.substitution_proxy_asset
    china_market_asset = model_inputs.china_market_asset
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

    substitution_ratio_series = calculate_ratio_series(
        crude_asset.history["Close"],
        substitution_proxy_asset.history["Close"],
    ).dropna()
    substitution_ratio = current_ratio_from_assets(crude_asset, substitution_proxy_asset)
    substitution_sma_50 = latest_rolling_value(substitution_ratio_series, 50)
    substitution_sma_200 = latest_rolling_value(substitution_ratio_series, 200)

    china_close = china_market_asset.history["Close"]
    china_sma_200 = latest_rolling_value(china_close, 200)

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
        score_substitution_risk(
            substitution_ratio,
            substitution_sma_50,
            substitution_sma_200,
            crude_asset.current_price,
        ),
    ]
    parameters = attach_parameter_metadata(parameters, inventory_snapshot)
    parameter_map = build_parameter_map(parameters)

    composite = calculate_weighted_score(parameters)
    volatility_factor = derive_volatility_factor(ovx_asset.current_price)
    demand_profile = build_demand_destruction_profile(crude_asset.current_price)
    stagflation_monitor = build_stagflation_monitor(
        implied_inflation_parameter=parameter_map["implied_inflation"],
        growth_expectations_parameter=parameter_map["growth_expectations"],
        china_price=china_market_asset.current_price,
        china_sma_200=china_sma_200,
        oil_price=crude_asset.current_price,
        oil_sma_50=crude_sma_50,
    )
    overlay_result = apply_regime_overlays(
        current_price=crude_asset.current_price,
        sma_50=crude_sma_50,
        base_composite_score=composite["normalized_score"],
        volatility_factor=volatility_factor,
        demand_profile=demand_profile,
        stagflation_monitor=stagflation_monitor,
    )
    temperature = map_temperature(overlay_result["adjusted_valuation_gap_pct"])

    historical_model_series = build_historical_model_series(
        crude_asset=crude_asset,
        dxy_asset=dxy_asset,
        ovx_asset=ovx_asset,
        geopolitical_risk_asset=geopolitical_risk_asset,
        tips_asset=tips_asset,
        treasury_asset=treasury_asset,
        high_yield_asset=high_yield_asset,
        investment_grade_asset=investment_grade_asset,
        substitution_proxy_asset=substitution_proxy_asset,
        china_market_asset=china_market_asset,
        inventory_snapshot=inventory_snapshot,
        history_days=90,
    )
    backtest_summary = build_backtest_summary(historical_model_series)

    regime_notes = []
    regime_notes.extend(demand_profile["notes"])
    regime_notes.extend(stagflation_monitor["notes"])

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
            "substitution_proxy": build_market_snapshot(substitution_proxy_asset),
            "china_market": build_market_snapshot(china_market_asset),
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
                "substitution_ratio": round_or_none(substitution_ratio, 4),
                "china_market_deviation_pct": round_or_none(stagflation_monitor["china_market_deviation_pct"]),
            },
        },
        "fundamentals": {
            "parameters": parameters,
            "weighted_raw_score": round(composite["weighted_raw"], 4),
            "normalized_score": round(overlay_result["adjusted_composite_score"], 2),
            "base_normalized_score": round(overlay_result["base_composite_score"], 2),
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
            "base_estimated_fair_value": round(overlay_result["fair_value_before_overlays"], 2),
            "estimated_fair_value": round(overlay_result["adjusted_fair_value"], 2),
            "fair_value_after_score_overlay": round(overlay_result["fair_value_after_score_overlay"], 2),
            "valuation_gap_pct": round(overlay_result["adjusted_valuation_gap_pct"], 2),
            "base_valuation_gap_pct": round(overlay_result["base_valuation_gap_pct"], 2),
            "temperature": temperature,
            "demand_destruction": demand_profile,
            "stagflation_monitor": stagflation_monitor,
            "regime_notes": regime_notes,
            "methodology": (
                "Base Fair Value = 50D SMA * (1 + (Composite Score / 100) * Volatility Factor). "
                "Demand-destruction and stagflation overlays then penalize the score, apply "
                "fair-value discounts, and add bearish pressure to the valuation gap."
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
                "The macro block uses ITA for geopolitical risk, TIP/IEF for implied inflation, HYG/LQD for growth, UNG for substitution pressure, and FXI as a China recession monitor.",
                "Demand-destruction overlays activate above $120 and intensify materially at $150.",
                "Inventory scoring uses EIA data when EIA_API_KEY is available; otherwise it falls back to a transparent mock.",
                "Historical model reconstruction uses fully historical market data, while the inventory score is held constant unless a historical EIA inventory series is added.",
                "Backtest win rate is based on whether price reverted toward fair value over the next 10 trading days.",
            ],
        },
    }


def fetch_all_model_inputs() -> ModelInputs:
    crude_asset = fetch_market_asset(CRUDE_TICKER, "Crude Oil Futures (WTI)")
    dxy_asset = fetch_market_asset(DXY_TICKER, "US Dollar Index")
    ovx_asset = fetch_market_asset(OVX_TICKER, "CBOE Crude Oil Volatility Index")
    geopolitical_risk_asset = fetch_market_asset(
        GEOPOLITICAL_RISK_TICKER,
        "iShares U.S. Aerospace & Defense ETF",
    )
    tips_asset = fetch_market_asset(TIPS_TICKER, "iShares TIPS Bond ETF")
    treasury_asset = fetch_market_asset(TREASURY_TICKER, "iShares 7-10 Year Treasury Bond ETF")
    high_yield_asset = fetch_market_asset(
        HIGH_YIELD_TICKER,
        "iShares iBoxx High Yield Corporate Bond ETF",
    )
    investment_grade_asset = fetch_market_asset(
        INVESTMENT_GRADE_TICKER,
        "iShares iBoxx Investment Grade Corporate Bond ETF",
    )
    substitution_proxy_asset = fetch_market_asset(
        SUBSTITUTION_PROXY_TICKER,
        "United States Natural Gas Fund",
    )
    china_market_asset = fetch_market_asset(
        CHINA_MARKET_TICKER,
        "iShares China Large-Cap ETF",
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
        substitution_proxy_asset=substitution_proxy_asset,
        china_market_asset=china_market_asset,
        inventory_snapshot=inventory_snapshot,
    )


def load_crude_oil_valuation_snapshot() -> Dict[str, Any]:
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
    try:
        return await asyncio.to_thread(load_crude_oil_valuation_snapshot)
    except Exception as exc:
        logger.exception("Crude oil valuation request failed.")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to build crude oil valuation snapshot: {exc}",
        ) from exc
