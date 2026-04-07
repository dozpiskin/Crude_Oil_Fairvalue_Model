"""
Python-native dashboard for the WTI Crude Oil Valuation Model.

Run locally with:
    python dashboard.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update

from main import load_crude_oil_valuation_snapshot


logger = logging.getLogger(__name__)

REFRESH_INTERVAL_MS = 5 * 60 * 1000

COLOR_BG = "#050816"
COLOR_PANEL = "rgba(8, 15, 31, 0.82)"
COLOR_PANEL_SUBTLE = "rgba(10, 18, 32, 0.62)"
COLOR_BORDER = "rgba(255, 255, 255, 0.10)"
COLOR_TEXT = "#d9e3f0"
COLOR_MUTED = "#94a3b8"
COLOR_WHITE = "#ffffff"
COLOR_CYAN = "#67e8f9"
COLOR_CYAN_SOFT = "rgba(103, 232, 249, 0.12)"
COLOR_ROSE = "#fb7185"
COLOR_ROSE_SOFT = "rgba(251, 113, 133, 0.12)"
COLOR_EMERALD = "#6ee7b7"
COLOR_ORANGE = "#fb923c"
COLOR_SLATE = "#cbd5e1"

TEMPERATURE_THEMES = {
    "Very Hot": {
        "accent": COLOR_ROSE,
        "accent_soft": "rgba(251, 113, 133, 0.18)",
        "badge_bg": "rgba(251, 113, 133, 0.15)",
        "border": "rgba(251, 113, 133, 0.32)",
        "emoji": "\U0001F525",
    },
    "Hot": {
        "accent": COLOR_ORANGE,
        "accent_soft": "rgba(251, 146, 60, 0.18)",
        "badge_bg": "rgba(251, 146, 60, 0.15)",
        "border": "rgba(251, 146, 60, 0.30)",
        "emoji": "\U0001F321\uFE0F",
    },
    "Neutral": {
        "accent": COLOR_SLATE,
        "accent_soft": "rgba(203, 213, 225, 0.12)",
        "badge_bg": "rgba(203, 213, 225, 0.10)",
        "border": "rgba(203, 213, 225, 0.24)",
        "emoji": "\u2696\uFE0F",
    },
    "Cold": {
        "accent": COLOR_CYAN,
        "accent_soft": "rgba(103, 232, 249, 0.16)",
        "badge_bg": "rgba(103, 232, 249, 0.12)",
        "border": "rgba(103, 232, 249, 0.28)",
        "emoji": "\U0001F9CA",
    },
    "Very Cold": {
        "accent": "#a5f3fc",
        "accent_soft": "rgba(165, 243, 252, 0.18)",
        "badge_bg": "rgba(165, 243, 252, 0.14)",
        "border": "rgba(165, 243, 252, 0.30)",
        "emoji": "\u2744\uFE0F\u2744\uFE0F",
    },
}

BIAS_THEMES = {
    "bullish": {"bg": "rgba(103, 232, 249, 0.12)", "text": "#cffafe"},
    "bearish": {"bg": "rgba(251, 113, 133, 0.12)", "text": "#ffe4e6"},
    "neutral": {"bg": "rgba(203, 213, 225, 0.10)", "text": "#e2e8f0"},
}

STATUS_THEMES = {
    "bullish": {"bg": "rgba(103, 232, 249, 0.12)", "text": "#cffafe"},
    "bearish": {"bg": "rgba(251, 113, 133, 0.12)", "text": "#ffe4e6"},
    "neutral": {"bg": "rgba(203, 213, 225, 0.10)", "text": "#e2e8f0"},
    "oversold": {"bg": "rgba(103, 232, 249, 0.12)", "text": "#cffafe"},
    "overbought": {"bg": "rgba(251, 113, 133, 0.12)", "text": "#ffe4e6"},
}

PAGE_STYLE = {
    "minHeight": "100vh",
    "padding": "24px",
    "background": (
        "radial-gradient(circle at top left, rgba(8, 145, 178, 0.14), transparent 28%),"
        "radial-gradient(circle at top right, rgba(244, 63, 94, 0.12), transparent 26%),"
        "linear-gradient(180deg, #07111f 0%, #040814 45%, #050816 100%)"
    ),
    "color": COLOR_TEXT,
    "fontFamily": "'Segoe UI', 'Trebuchet MS', sans-serif",
}

WRAPPER_STYLE = {
    "maxWidth": "1440px",
    "margin": "0 auto",
    "display": "flex",
    "flexDirection": "column",
    "gap": "24px",
}

TOOLBAR_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "justifyContent": "space-between",
    "alignItems": "center",
    "gap": "16px",
    "padding": "18px 22px",
    "border": f"1px solid {COLOR_BORDER}",
    "background": "rgba(3, 7, 18, 0.55)",
    "backdropFilter": "blur(18px)",
    "borderRadius": "24px",
    "boxShadow": "0 24px 80px rgba(2, 6, 23, 0.38)",
}

PANEL_STYLE = {
    "border": f"1px solid {COLOR_BORDER}",
    "background": COLOR_PANEL,
    "borderRadius": "28px",
    "padding": "22px",
    "boxShadow": "0 24px 80px rgba(2, 6, 23, 0.38)",
    "backdropFilter": "blur(18px)",
}

SUB_PANEL_STYLE = {
    "border": f"1px solid {COLOR_BORDER}",
    "background": COLOR_PANEL_SUBTLE,
    "borderRadius": "22px",
    "padding": "16px",
}


def app_panel(children: Any, style: Optional[Dict[str, Any]] = None) -> html.Div:
    combined_style = dict(PANEL_STYLE)
    if style:
        combined_style.update(style)
    return html.Div(children, style=combined_style)


def sub_panel(children: Any, style: Optional[Dict[str, Any]] = None) -> html.Div:
    combined_style = dict(SUB_PANEL_STYLE)
    if style:
        combined_style.update(style)
    return html.Div(children, style=combined_style)


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def format_percent(value: Optional[float], digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def format_score(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    if float(value).is_integer():
        return f"{sign}{int(value)}"
    return f"{sign}{value:.1f}"


def format_timestamp(value: Optional[str]) -> str:
    if not value:
        return "N/A"
    try:
        timestamp = pd.Timestamp(value).tz_convert("UTC")
        return timestamp.strftime("%b %d, %Y %H:%M UTC")
    except Exception:
        return str(value)


def get_temperature_theme(label: str) -> Dict[str, str]:
    return TEMPERATURE_THEMES.get(label, TEMPERATURE_THEMES["Neutral"])


def get_bias_theme(bias: str) -> Dict[str, str]:
    return BIAS_THEMES.get(bias, BIAS_THEMES["neutral"])


def get_status_theme(status: str) -> Dict[str, str]:
    return STATUS_THEMES.get(status, STATUS_THEMES["neutral"])


def return_color(value: Optional[float]) -> str:
    if value is None:
        return COLOR_MUTED
    if value > 0:
        return COLOR_EMERALD
    if value < 0:
        return COLOR_ROSE
    return COLOR_WHITE


def win_rate_color(value: Optional[float]) -> str:
    if value is None:
        return COLOR_MUTED
    if value > 50:
        return COLOR_EMERALD
    if value < 50:
        return COLOR_ROSE
    return COLOR_WHITE


def build_temperature_gauge_figure(valuation: Dict[str, Any]) -> go.Figure:
    clipped_gap = max(-20.0, min(20.0, valuation["valuation_gap_pct"]))
    theme = get_temperature_theme(valuation["temperature"]["label"])

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clipped_gap,
            number={"suffix": "%", "font": {"size": 42, "color": COLOR_WHITE}},
            title={"text": "Valuation Gap", "font": {"size": 18, "color": COLOR_MUTED}},
            gauge={
                "axis": {
                    "range": [-20, 20],
                    "tickvals": [-20, -10, 0, 10, 20],
                    "ticktext": ["Very Cold", "Cold", "Neutral", "Hot", "Very Hot"],
                    "tickcolor": COLOR_MUTED,
                    "tickfont": {"color": COLOR_MUTED, "size": 11},
                },
                "bar": {"color": theme["accent"], "thickness": 0.36},
                "borderwidth": 0,
                "bgcolor": "rgba(15, 23, 42, 0.82)",
                "steps": [
                    {"range": [-20, -10], "color": "rgba(34, 211, 238, 0.18)"},
                    {"range": [-10, 10], "color": "rgba(203, 213, 225, 0.12)"},
                    {"range": [10, 20], "color": "rgba(251, 113, 133, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": COLOR_WHITE, "width": 4},
                    "thickness": 0.85,
                    "value": clipped_gap,
                },
            },
        )
    )
    figure.update_layout(
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    return figure


def build_price_vs_fair_value_figure(history: List[Dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(history)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["actual_price"],
            mode="lines",
            name="Actual Price",
            line={"color": COLOR_CYAN, "width": 3},
            hovertemplate="Date: %{x}<br>Actual: $%{y:.2f}<br>Gap: %{customdata:.2f}%<extra></extra>",
            customdata=frame["valuation_gap_pct"],
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["fair_value"],
            mode="lines",
            name="Model Fair Value",
            line={"color": "#fda4af", "width": 2.5, "dash": "dash"},
            hovertemplate="Date: %{x}<br>Fair Value: $%{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_MUTED},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={"showgrid": False, "tickfont": {"color": COLOR_MUTED}, "title": ""},
        yaxis={
            "gridcolor": "rgba(148, 163, 184, 0.14)",
            "zeroline": False,
            "tickprefix": "$",
            "tickfont": {"color": COLOR_MUTED},
            "title": "",
        },
        height=360,
    )
    return figure


def build_gap_history_figure(history: List[Dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(history)
    frame["hot_gap"] = frame["valuation_gap_pct"].where(frame["valuation_gap_pct"] > 10)
    frame["neutral_gap"] = frame["valuation_gap_pct"].where(
        frame["valuation_gap_pct"].between(-10, 10)
    )
    frame["cold_gap"] = frame["valuation_gap_pct"].where(frame["valuation_gap_pct"] < -10)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["hot_gap"],
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(251, 113, 133, 0.30)",
            name="Hot Zone",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["neutral_gap"],
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.18)",
            name="Neutral Zone",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["cold_gap"],
            mode="lines",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(34, 211, 238, 0.28)",
            name="Cold Zone",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["valuation_gap_pct"],
            mode="lines",
            name="Valuation Gap",
            line={"color": COLOR_WHITE, "width": 2.2},
            hovertemplate=(
                "Date: %{x}<br>Gap: %{y:.2f}%<br>Temperature: %{customdata}<extra></extra>"
            ),
            customdata=frame["temperature_label"],
        )
    )
    figure.add_hline(y=10, line={"color": COLOR_ROSE, "dash": "dash", "width": 1.4})
    figure.add_hline(y=0, line={"color": "rgba(148, 163, 184, 0.35)", "dash": "dot", "width": 1.2})
    figure.add_hline(y=-10, line={"color": COLOR_CYAN, "dash": "dash", "width": 1.4})
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_MUTED},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={"showgrid": False, "title": ""},
        yaxis={
            "gridcolor": "rgba(148, 163, 184, 0.14)",
            "ticksuffix": "%",
            "zeroline": False,
            "title": "",
        },
        height=360,
    )
    return figure


def status_badge(text: str, status: str) -> html.Span:
    theme = get_status_theme(status)
    return html.Span(
        text.upper(),
        style={
            "display": "inline-block",
            "padding": "6px 10px",
            "borderRadius": "999px",
            "background": theme["bg"],
            "color": theme["text"],
            "fontSize": "11px",
            "fontWeight": 700,
            "letterSpacing": "0.08em",
        },
    )


def bias_badge(text: str) -> html.Span:
    theme = get_bias_theme(text)
    return html.Span(
        text.upper(),
        style={
            "display": "inline-block",
            "padding": "6px 10px",
            "borderRadius": "999px",
            "background": theme["bg"],
            "color": theme["text"],
            "fontSize": "11px",
            "fontWeight": 700,
            "letterSpacing": "0.08em",
        },
    )


def metric_block(
    label: str,
    value: str,
    subtitle: Optional[str] = None,
    value_color: str = COLOR_WHITE,
    value_font_size: str = "30px",
) -> html.Div:
    children = [
        html.Div(label.upper(), style={"fontSize": "11px", "letterSpacing": "0.18em", "color": COLOR_MUTED}),
        html.Div(value, style={"fontSize": value_font_size, "fontWeight": 700, "color": value_color, "marginTop": "8px"}),
    ]
    if subtitle:
        children.append(html.Div(subtitle, style={"fontSize": "13px", "color": COLOR_MUTED, "marginTop": "4px"}))
    return sub_panel(children)


def build_header_panel(snapshot: Dict[str, Any]) -> html.Div:
    asset = snapshot["asset"]
    valuation = snapshot["valuation"]
    temperature = valuation["temperature"]
    theme = get_temperature_theme(temperature["label"])

    return app_panel(
        [
            html.Div(
                [
                    html.Div("Quantitative Dashboard", style={"fontSize": "12px", "letterSpacing": "0.26em", "color": "#bae6fd", "textTransform": "uppercase", "fontWeight": 700}),
                    html.H1(
                        "WTI Crude Oil Valuation Model",
                        style={"margin": "10px 0 0", "fontSize": "42px", "lineHeight": 1.05, "color": COLOR_WHITE},
                    ),
                    html.P(
                        "A Python-native valuation dashboard blending macro proxies, inventories, volatility, and trend context into a transparent fair-value estimate.",
                        style={"maxWidth": "980px", "fontSize": "16px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "14px"},
                    ),
                ]
            ),
            html.Div(
                [
                    sub_panel(
                        [
                            html.Div("Live Price", style={"fontSize": "11px", "letterSpacing": "0.18em", "color": COLOR_MUTED, "textTransform": "uppercase"}),
                            html.Div(
                                [
                                    html.Span(format_currency(asset["current_price"]), style={"fontSize": "42px", "fontWeight": 800, "color": COLOR_WHITE}),
                                    html.Span(
                                        f'{temperature["label"]} {theme["emoji"]}',
                                        style={
                                            "padding": "8px 14px",
                                            "borderRadius": "999px",
                                            "background": theme["badge_bg"],
                                            "color": COLOR_WHITE,
                                            "fontWeight": 700,
                                            "marginLeft": "12px",
                                            "display": "inline-block",
                                        },
                                    ),
                                ],
                                style={"marginTop": "10px", "display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "12px"},
                            ),
                            html.Div(
                                f'Valuation gap {format_percent(valuation["valuation_gap_pct"])}',
                                style={"marginTop": "10px", "fontSize": "14px", "color": COLOR_MUTED},
                            ),
                        ]
                    ),
                    sub_panel(
                        [
                            html.Div(
                                f'Last updated {format_timestamp(asset["last_updated"])}',
                                style={"fontSize": "14px", "color": "#cbd5e1"},
                            ),
                            html.Div(
                                temperature["description"],
                                style={"fontSize": "14px", "color": "#cbd5e1", "marginTop": "10px"},
                            ),
                            html.Div(
                                f'Dashboard snapshot generated {format_timestamp(snapshot["meta"]["generated_at"])}',
                                style={"fontSize": "13px", "color": COLOR_MUTED, "marginTop": "12px"},
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                    "gap": "14px",
                    "marginTop": "22px",
                },
            ),
        ]
    )


def build_temperature_section(snapshot: Dict[str, Any]) -> html.Div:
    asset = snapshot["asset"]
    valuation = snapshot["valuation"]
    fundamentals = snapshot["fundamentals"]
    theme = get_temperature_theme(valuation["temperature"]["label"])

    left_column = app_panel(
        [
            html.Div("Temperature Gauge", style={"fontSize": "12px", "letterSpacing": "0.26em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
            html.H2(
                f'{valuation["temperature"]["label"]} {theme["emoji"]}',
                style={"margin": "14px 0 0", "fontSize": "40px", "lineHeight": 1.08, "color": COLOR_WHITE},
            ),
            html.P(
                "The model compares WTI spot price with an estimated fair value derived from the 50-day average, the weighted macro score, and an OVX-based volatility factor.",
                style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "12px"},
            ),
            dcc.Graph(
                figure=build_temperature_gauge_figure(valuation),
                config={"displayModeBar": False, "responsive": True},
                style={"height": "320px", "marginTop": "10px"},
            ),
            html.Div(
                [
                    metric_block("Current Price", format_currency(asset["current_price"])),
                    metric_block("Estimated Fair Value", format_currency(valuation["estimated_fair_value"])),
                    metric_block("Valuation Gap", format_percent(valuation["valuation_gap_pct"])),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "14px",
                },
            ),
        ],
        style={"border": f'1px solid {theme["border"]}'},
    )

    right_column = app_panel(
        [
            html.Div("Model Lens", style={"fontSize": "11px", "letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLOR_MUTED}),
            html.P(
                "This gauge is deliberately valuation-oriented rather than purely trend-following. Positive gaps imply the market is trading above the model's fair value estimate, while negative gaps imply potential undervaluation.",
                style={"fontSize": "15px", "lineHeight": 1.75, "color": "#cbd5e1", "marginTop": "12px"},
            ),
            html.Div(
                [
                    metric_block("Composite Score", format_score(fundamentals["normalized_score"])),
                    metric_block("Volatility Factor", format_number(valuation["volatility_factor"], 2)),
                    metric_block("Premium vs Fair Value", format_currency(asset["current_price"] - valuation["estimated_fair_value"])),
                    metric_block("50D Reference", format_currency(valuation["reference_sma_50"])),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "14px",
                    "marginTop": "18px",
                },
            ),
        ]
    )

    return html.Div(
        [left_column, right_column],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
            "gap": "24px",
        },
    )


def build_technical_panel(technicals: Dict[str, Any]) -> html.Div:
    return app_panel(
        [
            html.Div("Technical Overlay", style={"fontSize": "12px", "letterSpacing": "0.26em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
            html.H3("RSI, MACD, and Trend Levels", style={"margin": "12px 0 0", "fontSize": "30px", "color": COLOR_WHITE}),
            html.P(
                "These signals do not drive the full model on their own, but they help frame whether valuation extremes are occurring alongside momentum strength or stress.",
                style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "10px"},
            ),
            html.Div(
                [
                    sub_panel(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("RSI (14)", style={"fontSize": "11px", "letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLOR_MUTED}),
                                            html.Div(format_number(technicals["rsi_14"]["value"]), style={"fontSize": "38px", "fontWeight": 800, "color": COLOR_WHITE, "marginTop": "8px"}),
                                        ]
                                    ),
                                    status_badge(technicals["rsi_14"]["status"], technicals["rsi_14"]["status"]),
                                ],
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "12px"},
                            ),
                            html.Div(
                                style={
                                    "height": "8px",
                                    "borderRadius": "999px",
                                    "background": "linear-gradient(90deg, #67e8f9 0%, #fcd34d 50%, #fb7185 100%)",
                                    "marginTop": "16px",
                                    "width": f'{max(0, min(100, technicals["rsi_14"]["value"]))}%',
                                }
                            ),
                        ]
                    ),
                    sub_panel(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("MACD", style={"fontSize": "11px", "letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLOR_MUTED}),
                                            html.Div(format_number(technicals["macd"]["line"], 3), style={"fontSize": "28px", "fontWeight": 700, "color": COLOR_WHITE, "marginTop": "8px"}),
                                        ]
                                    ),
                                    status_badge(technicals["macd"]["status"], technicals["macd"]["status"]),
                                ],
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "12px"},
                            ),
                            html.Div(
                                [
                                    metric_block("Signal Line", format_number(technicals["macd"]["signal"], 3)),
                                    metric_block("Histogram", format_number(technicals["macd"]["histogram"], 3)),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px", "marginTop": "16px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            metric_block("50D SMA", format_currency(technicals["moving_averages"]["sma_50"])),
                            metric_block("200D SMA", format_currency(technicals["moving_averages"]["sma_200"])),
                            metric_block("vs 50D SMA", format_percent(technicals["moving_averages"]["price_vs_sma_50_pct"])),
                            metric_block("vs 200D SMA", format_percent(technicals["moving_averages"]["price_vs_sma_200_pct"])),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))", "gap": "12px"},
                    ),
                ],
                style={"display": "grid", "gap": "14px", "marginTop": "18px"},
            ),
        ]
    )


def build_charts_section(snapshot: Dict[str, Any]) -> html.Div:
    history = snapshot["history"]
    latest_gap = history[-1]["valuation_gap_pct"] if history else None
    max_divergence = None
    if history:
        max_divergence = max(history, key=lambda row: abs(row["valuation_gap_pct"]))["valuation_gap_pct"]

    price_chart = app_panel(
        [
            html.Div("Historical Performance", style={"fontSize": "12px", "letterSpacing": "0.26em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
            html.H3("Price vs. Model Fair Value", style={"margin": "12px 0 0", "fontSize": "30px", "color": COLOR_WHITE}),
            html.P(
                "This chart reconstructs the model's fair value over the last 90 days, making it easy to inspect where price diverged from the valuation estimate and whether it later converged.",
                style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "10px"},
            ),
            html.Div(
                [
                    metric_block("Latest Gap", format_percent(latest_gap)),
                    metric_block("Largest 90D Divergence", format_percent(max_divergence)),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "12px", "marginTop": "18px"},
            ),
            dcc.Graph(
                figure=build_price_vs_fair_value_figure(history),
                config={"displayModeBar": False, "responsive": True},
                style={"height": "380px", "marginTop": "12px"},
            ),
        ]
    )

    gap_chart = app_panel(
        [
            html.Div("Historical Performance", style={"fontSize": "12px", "letterSpacing": "0.26em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
            html.H3("Temperature and Gap History", style={"margin": "12px 0 0", "fontSize": "30px", "color": COLOR_WHITE}),
            html.P(
                "Hot zones mark periods when the market ran far above model fair value, while Cold zones highlight pricing that fell materially below the model estimate.",
                style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "10px"},
            ),
            dcc.Graph(
                figure=build_gap_history_figure(history),
                config={"displayModeBar": False, "responsive": True},
                style={"height": "432px", "marginTop": "12px"},
            ),
        ]
    )

    return html.Div(
        [price_chart, gap_chart],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
            "gap": "24px",
        },
    )


def build_parameter_card(parameter: Dict[str, Any]) -> html.Div:
    deviation_display = (
        format_percent(parameter["deviation_pct"])
        if parameter["deviation_pct"] is not None
        else format_score(parameter["weighted_contribution"])
    )
    benchmark_display = (
        "Rule-based"
        if parameter["benchmark_value"] is None
        else format_number(parameter["benchmark_value"])
    )

    extras = []
    if parameter.get("auxiliary_label"):
        extras.append(
            sub_panel(
                [
                    html.Div(parameter["auxiliary_label"].upper(), style={"fontSize": "11px", "letterSpacing": "0.14em", "color": COLOR_MUTED}),
                    html.Div(format_percent(parameter["auxiliary_value"]), style={"fontSize": "20px", "fontWeight": 700, "color": COLOR_WHITE, "marginTop": "8px"}),
                ],
                style={"marginTop": "12px"},
            )
        )

    return app_panel(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(parameter["name"], style={"fontSize": "16px", "fontWeight": 700, "color": COLOR_WHITE}),
                            html.Div(
                                f'Weight {parameter["weight_pct"]}%',
                                style={"fontSize": "11px", "letterSpacing": "0.18em", "color": COLOR_MUTED, "textTransform": "uppercase", "marginTop": "4px"},
                            ),
                        ]
                    ),
                    bias_badge(parameter["bias"]),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "12px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Current Value", style={"fontSize": "11px", "letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLOR_MUTED}),
                            html.Div(format_number(parameter["current_value"]), style={"fontSize": "34px", "fontWeight": 800, "color": COLOR_WHITE, "marginTop": "10px"}),
                            html.Div(parameter["unit"], style={"fontSize": "13px", "color": COLOR_MUTED, "marginTop": "4px"}),
                        ]
                    ),
                    sub_panel(
                        [
                            html.Div("Score", style={"fontSize": "11px", "letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLOR_MUTED}),
                            html.Div(format_score(parameter["score"]), style={"fontSize": "34px", "fontWeight": 800, "color": COLOR_WHITE, "marginTop": "8px"}),
                        ],
                        style={
                            "minWidth": "110px",
                            "textAlign": "center",
                            "background": get_bias_theme(parameter["bias"])["bg"],
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end", "gap": "12px", "marginTop": "18px"},
            ),
            html.Div(
                [
                    metric_block(parameter["benchmark_label"], benchmark_display),
                    metric_block("Deviation / Contribution", deviation_display),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px", "marginTop": "16px"},
            ),
            *extras,
            html.P(
                parameter["rationale"],
                style={"fontSize": "14px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "16px"},
            ),
        ]
    )


def build_fundamentals_section(snapshot: Dict[str, Any]) -> html.Div:
    parameters = snapshot["fundamentals"]["parameters"]
    inventory_snapshot = snapshot["market_data"]["inventory_snapshot"]

    return html.Div(
        [
            app_panel(
                [
                    html.Div("Fundamental Dashboard", style={"fontSize": "12px", "letterSpacing": "0.26em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
                    html.H3("Five-Driver Scoring Framework", style={"margin": "12px 0 0", "fontSize": "30px", "color": COLOR_WHITE}),
                    html.P(
                        "Each parameter is scored from -2 to +2, weighted into a normalized macro score, and then fed into the fair-value equation to estimate how stretched current WTI pricing may be.",
                        style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "10px"},
                    ),
                    sub_panel(
                        [
                            html.Div(f'Inventory source: {inventory_snapshot["source"]}', style={"fontSize": "14px", "fontWeight": 700, "color": COLOR_WHITE}),
                            html.Div(inventory_snapshot["note"], style={"fontSize": "13px", "lineHeight": 1.6, "color": COLOR_MUTED, "marginTop": "8px"}),
                        ],
                        style={"marginTop": "18px"},
                    ),
                ]
            ),
            html.Div(
                [build_parameter_card(parameter) for parameter in parameters],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "24px",
                },
            ),
        ],
        style={"display": "grid", "gap": "24px"},
    )


def build_backtest_card(signal: str, stats: Dict[str, Any]) -> html.Div:
    expects_positive = stats["expected_direction"] == "positive"
    direction_text = "Expect upside reversion" if expects_positive else "Expect downside reversion"
    direction_color = COLOR_CYAN if expects_positive else COLOR_ROSE

    avg_5d = format_percent(stats["avg_5d_return_pct"]) if stats["avg_5d_return_pct"] is not None else "Insufficient data"
    avg_10d = format_percent(stats["avg_10d_return_pct"]) if stats["avg_10d_return_pct"] is not None else "Insufficient data"
    win_rate = f'{stats["win_rate_pct"]:.1f}%' if stats["win_rate_pct"] is not None else "Insufficient data"

    return app_panel(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(signal, style={"fontSize": "28px", "fontWeight": 800, "color": COLOR_WHITE}),
                            html.Div(
                                direction_text,
                                style={
                                    "marginTop": "10px",
                                    "display": "inline-block",
                                    "padding": "7px 12px",
                                    "borderRadius": "999px",
                                    "background": f"{direction_color}22",
                                    "color": COLOR_WHITE,
                                    "fontSize": "11px",
                                    "fontWeight": 700,
                                    "letterSpacing": "0.12em",
                                    "textTransform": "uppercase",
                                },
                            ),
                        ]
                    ),
                    sub_panel(
                        [
                            html.Div("Count", style={"fontSize": "11px", "letterSpacing": "0.18em", "textTransform": "uppercase", "color": COLOR_MUTED}),
                            html.Div(str(stats["occurrence_count"]), style={"fontSize": "32px", "fontWeight": 800, "color": COLOR_WHITE, "marginTop": "8px"}),
                        ],
                        style={"textAlign": "center", "minWidth": "100px"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "12px"},
            ),
            html.Div(
                [
                    metric_block(
                        "Avg 5D Return",
                        avg_5d,
                        f'{stats["valid_5d_samples"]} valid windows',
                        value_color=return_color(stats["avg_5d_return_pct"]),
                        value_font_size="26px",
                    ),
                    metric_block(
                        "Avg 10D Return",
                        avg_10d,
                        f'{stats["valid_10d_samples"]} valid windows',
                        value_color=return_color(stats["avg_10d_return_pct"]),
                        value_font_size="26px",
                    ),
                    metric_block(
                        "Win Rate",
                        win_rate,
                        "10D mean-reversion test",
                        value_color=win_rate_color(stats["win_rate_pct"]),
                        value_font_size="26px",
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px", "marginTop": "18px"},
            ),
            html.Div(
                (
                    f"{signal} is treated as undervaluation, so positive forward returns count as successful reversion toward fair value."
                    if expects_positive
                    else f"{signal} is treated as overvaluation, so negative forward returns count as successful downside mean reversion."
                ),
                style={
                    "marginTop": "16px",
                    "padding": "14px 16px",
                    "borderRadius": "18px",
                    "background": "rgba(2, 6, 23, 0.55)",
                    "border": f"1px solid {COLOR_BORDER}",
                    "fontSize": "14px",
                    "lineHeight": 1.7,
                    "color": "#cbd5e1",
                },
            ),
        ]
    )


def build_backtest_section(snapshot: Dict[str, Any]) -> html.Div:
    backtest_summary = snapshot["backtest_summary"]

    return html.Div(
        [
            app_panel(
                [
                    html.Div("Backtest Summary", style={"fontSize": "12px", "letterSpacing": "0.26em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
                    html.H3("Forward Returns by Temperature Signal", style={"margin": "12px 0 0", "fontSize": "30px", "color": COLOR_WHITE}),
                    html.P(
                        "This panel measures what happened after the model flashed a non-neutral valuation signal. Cold signals seek positive forward returns, while Hot signals seek negative returns as price mean-reverts toward fair value.",
                        style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "10px"},
                    ),
                    html.Div(
                        [
                            sub_panel(
                                [
                                    html.Div("Win Rate Logic", style={"fontSize": "14px", "fontWeight": 700, "color": COLOR_WHITE}),
                                    html.Div(
                                        "Win Rate indicates % of time price reverted towards fair value within 10 days.",
                                        style={"fontSize": "13px", "lineHeight": 1.6, "color": "#cbd5e1", "marginTop": "8px"},
                                    ),
                                ],
                                style={"background": COLOR_CYAN_SOFT},
                            ),
                            sub_panel(
                                [
                                    html.Div("Interpretation", style={"fontSize": "14px", "fontWeight": 700, "color": COLOR_WHITE}),
                                    html.Div(
                                        "Hot signals should ideally be followed by negative returns, while Cold signals should ideally be followed by positive returns.",
                                        style={"fontSize": "13px", "lineHeight": 1.6, "color": "#cbd5e1", "marginTop": "8px"},
                                    ),
                                ]
                            ),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "14px", "marginTop": "18px"},
                    ),
                    sub_panel(
                        html.Div(backtest_summary["win_rate_definition"], style={"fontSize": "14px", "lineHeight": 1.7, "color": "#cbd5e1"}),
                        style={"marginTop": "14px"},
                    ),
                ]
            ),
            html.Div(
                [
                    build_backtest_card(signal, backtest_summary["signals"][signal])
                    for signal in backtest_summary["signal_order"]
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "24px",
                },
            ),
        ],
        style={"display": "grid", "gap": "24px"},
    )


def build_error_banner(message: Optional[str]) -> Optional[html.Div]:
    if not message:
        return None

    return app_panel(
        [
            html.Div("Last refresh failed", style={"fontSize": "16px", "fontWeight": 700, "color": "#fde68a"}),
            html.Div(message, style={"fontSize": "14px", "lineHeight": 1.7, "color": "#fef3c7", "marginTop": "8px"}),
        ],
        style={
            "border": "1px solid rgba(251, 191, 36, 0.30)",
            "background": "rgba(251, 191, 36, 0.10)",
        },
    )


def build_loading_state() -> html.Div:
    return app_panel(
        [
            html.H2("Loading valuation model", style={"margin": 0, "fontSize": "34px", "color": COLOR_WHITE}),
            html.P(
                "Pulling live WTI, DXY, OVX, and S&P 500 data, then rebuilding the fair-value and backtest history from Python.",
                style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "12px"},
            ),
        ],
        style={"textAlign": "center", "padding": "48px 22px"},
    )


def build_error_state(message: str) -> html.Div:
    return app_panel(
        [
            html.H2("Unable to load data", style={"margin": 0, "fontSize": "34px", "color": COLOR_WHITE}),
            html.P(message, style={"fontSize": "15px", "lineHeight": 1.7, "color": "#cbd5e1", "marginTop": "12px"}),
        ],
        style={"textAlign": "center", "padding": "48px 22px"},
    )


def build_footer(snapshot: Dict[str, Any]) -> html.Div:
    return app_panel(
        html.Div(
            snapshot["disclaimer"],
            style={"fontSize": "14px", "lineHeight": 1.7, "color": "#cbd5e1"},
        ),
        style={"borderLeft": f"4px solid {COLOR_CYAN}"},
    )


def build_dashboard(snapshot: Dict[str, Any], error_message: Optional[str] = None) -> List[html.Div]:
    content: List[html.Div] = []
    error_banner = build_error_banner(error_message)
    if error_banner is not None:
        content.append(error_banner)

    content.extend(
        [
            build_header_panel(snapshot),
            html.Div(
                [
                    build_temperature_section(snapshot),
                    build_technical_panel(snapshot["technicals"]),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
                    "gap": "24px",
                },
            ),
            build_charts_section(snapshot),
            build_backtest_section(snapshot),
            build_fundamentals_section(snapshot),
            build_footer(snapshot),
        ]
    )
    return content


app = Dash(__name__, title="WTI Crude Oil Valuation Model")
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body, #react-entry-point, #_dash-app-content { margin: 0; min-height: 100%; background: #050816; }
            * { box-sizing: border-box; }
            button { font: inherit; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    [
        dcc.Interval(id="auto-refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dcc.Store(id="valuation-store"),
        dcc.Store(id="error-store"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("100% Python Runtime", style={"fontSize": "12px", "letterSpacing": "0.22em", "textTransform": "uppercase", "color": "#bae6fd", "fontWeight": 700}),
                        html.Div("Dash + Plotly + FastAPI Quant Engine", style={"fontSize": "14px", "color": COLOR_MUTED, "marginTop": "6px"}),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Auto refresh every 5 minutes", style={"fontSize": "14px", "color": COLOR_MUTED}),
                        html.Button(
                            "Refresh Data",
                            id="refresh-button",
                            n_clicks=0,
                            style={
                                "marginLeft": "12px",
                                "border": f"1px solid {COLOR_BORDER}",
                                "background": "rgba(255, 255, 255, 0.05)",
                                "color": COLOR_WHITE,
                                "padding": "10px 16px",
                                "borderRadius": "16px",
                                "cursor": "pointer",
                                "fontWeight": 700,
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "12px"},
                ),
            ],
            style=TOOLBAR_STYLE,
        ),
        dcc.Loading(
            html.Div(id="dashboard-content"),
            color=COLOR_CYAN,
            type="circle",
        ),
    ],
    style=PAGE_STYLE,
)


@app.callback(
    Output("valuation-store", "data"),
    Output("error-store", "data"),
    Input("auto-refresh", "n_intervals"),
    Input("refresh-button", "n_clicks"),
)
def refresh_snapshot(_: int, __: int) -> tuple[Any, Any]:
    try:
        snapshot = load_crude_oil_valuation_snapshot()
        return snapshot, None
    except Exception as exc:  # pragma: no cover - interactive runtime path
        logger.exception("Dashboard refresh failed.")
        return no_update, f"Unable to refresh dashboard snapshot: {exc}"


@app.callback(
    Output("dashboard-content", "children"),
    Input("valuation-store", "data"),
    Input("error-store", "data"),
)
def render_dashboard_content(snapshot: Optional[Dict[str, Any]], error_message: Optional[str]) -> Any:
    if snapshot is None and error_message:
        return build_error_state(error_message)
    if snapshot is None:
        return build_loading_state()
    return html.Div(build_dashboard(snapshot, error_message), style=WRAPPER_STYLE)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
