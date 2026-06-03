import json
import html
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(APP_DIR, "watchlist.json")
CHART_PERIODS = {
    "1W": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "1Y": "1y",
}


def load_watchlist() -> list[str]:
    if not os.path.exists(WATCHLIST_PATH):
        return []
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_watchlist(tickers: list[str]) -> None:
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(tickers, f, indent=2)


def remove_from_watchlist(ticker: str) -> None:
    watchlist = load_watchlist()
    if ticker in watchlist:
        watchlist.remove(ticker)
        save_watchlist(watchlist)
    if st.session_state.get("selected_ticker") == ticker:
        st.session_state.selected_ticker = None


def clear_data_caches() -> None:
    get_stock_data.clear()
    get_chart_data.clear()
    get_stock_news.clear()


def render_watchlist_summary(
    summary_slot,
    total_stocks: int,
    signal_counts: dict[str, int],
) -> None:
    with summary_slot.container():
        st.markdown("**Watchlist Summary**")
        st.caption(f"Total stocks tracked: {total_stocks}")
        st.caption(
            " | ".join(
                [
                    f"BUY: {signal_counts['BUY']}",
                    f"SELL: {signal_counts['SELL']}",
                    f"HOLD: {signal_counts['HOLD']}",
                ]
            )
        )


@st.cache_data(ttl=60)
def get_stock_data(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist is None or hist.empty or len(hist) < 2:
            return None

        hist_1y = stock.history(period="1y")
        if hist_1y is None or hist_1y.empty:
            return None

        closes = hist["Close"]
        price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        change_pct = ((price - prev_close) / prev_close) * 100

        ma20 = float(closes.rolling(20).mean().iloc[-1])
        ma50 = float(closes.rolling(50).mean().iloc[-1])
        if pd.isna(ma20) or pd.isna(ma50):
            return None

        week_52_high = float(hist_1y["High"].max())
        week_52_low = float(hist_1y["Low"].min())
        volume = int(hist["Volume"].iloc[-1])

        info = stock.info
        name = info.get("longName") or info.get("shortName") or ticker

        return {
            "name": name,
            "price": price,
            "change_pct": change_pct,
            "volume": volume,
            "ma50": ma50,
            "ma20": ma20,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_chart_data(ticker: str, chart_range: str) -> pd.DataFrame | None:
    period = CHART_PERIODS.get(chart_range, CHART_PERIODS["1M"])

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            return None

        required_columns = ["Open", "High", "Low", "Close"]
        hist = hist[required_columns].dropna()
        if hist.empty:
            return None

        hist["MA20"] = hist["Close"].rolling(20, min_periods=1).mean()
        hist["MA50"] = hist["Close"].rolling(50, min_periods=1).mean()
        return hist
    except Exception:
        return None


def format_news_date(timestamp: int | float | None) -> str:
    if not timestamp:
        return ""

    date = datetime.fromtimestamp(timestamp)
    return f"{date:%b} {date.day}, {date:%Y}"


@st.cache_data(ttl=300)
def get_stock_news(ticker: str) -> list[dict[str, str]]:
    try:
        articles = yf.Ticker(ticker).news
        news_items = []

        for article in articles[:8]:
            content = article.get("content", article)
            title = content.get("title", "")
            publisher = content.get("publisher", "")
            link = content.get("link") or content.get("canonicalUrl", {}).get("url", "")
            published_at = (
                content.get("providerPublishTime")
                or content.get("pubDate")
                or content.get("displayTime")
            )

            if isinstance(published_at, str):
                try:
                    published_at = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    published_at = None

            if title and link:
                news_items.append(
                    {
                        "title": title,
                        "publisher": publisher,
                        "link": link,
                        "providerPublishTime": format_news_date(published_at),
                    }
                )

        return news_items
    except Exception:
        return []


def get_signal(
    price: float,
    ma20: float,
    ma50: float,
    week_52_high: float,
    week_52_low: float,
) -> dict[str, str]:
    near_52w_high = price >= week_52_high * 0.97

    if price > ma50 and ma20 > ma50 and not near_52w_high:
        return {
            "signal": "BUY",
            "reason": (
                "Price is above the 50-day average with the 20-day average above the 50-day "
                "(uptrend), and the stock is not close to its 52-week high."
            ),
        }

    if price < ma50 and ma20 < ma50:
        return {
            "signal": "SELL",
            "reason": (
                "Price is below the 50-day average with the 20-day average below the 50-day "
                "(downtrend)."
            ),
        }

    if near_52w_high and price > ma50 and ma20 > ma50:
        reason = (
            "The uptrend looks healthy, but price is within 3% of the 52-week high, "
            "so momentum may be stretched."
        )
    elif price > ma50 and ma20 <= ma50:
        reason = (
            "Price is above the 50-day average, but the 20-day average has not crossed "
            "above the 50-day yet."
        )
    elif price <= ma50 and ma20 >= ma50:
        reason = (
            "The 20-day average is still above the 50-day, but price has slipped below "
            "the 50-day average."
        )
    else:
        reason = (
            "Moving averages do not show a clear uptrend or downtrend right now."
        )

    return {"signal": "HOLD", "reason": reason}


def render_stock_card(ticker: str) -> str | None:
    with st.spinner(f"Loading {ticker}..."):
        data = get_stock_data(ticker)

    if data is None:
        st.error(f"Could not load data for **{ticker}**. Check the symbol and try Refresh.")
        if st.button("Remove", key=f"remove_{ticker}"):
            remove_from_watchlist(ticker)
            st.rerun()
        return None

    if st.button(
        data["name"],
        key=f"select_{ticker}",
        use_container_width=True,
        type="tertiary",
    ):
        st.session_state.selected_ticker = ticker
    st.caption(ticker)

    st.metric(
        label="Price",
        value=f"${data['price']:.2f}",
        delta=f"{data['change_pct']:.2f}%",
    )

    signal_info = get_signal(
        data["price"],
        data["ma20"],
        data["ma50"],
        data["week_52_high"],
        data["week_52_low"],
    )
    signal = signal_info["signal"]

    badge = {"BUY": "🟢 **BUY**", "SELL": "🔴 **SELL**", "HOLD": "🟡 **HOLD**"}[signal]
    st.markdown(f"### {badge}")
    st.caption(signal_info["reason"])

    if st.button("Remove", key=f"remove_{ticker}"):
        remove_from_watchlist(ticker)
        st.rerun()

    return signal


def render_chart_section() -> None:
    selected_ticker = st.session_state.get("selected_ticker")
    if selected_ticker is None:
        return

    if st.session_state.chart_range not in CHART_PERIODS:
        st.session_state.chart_range = "1M"

    st.divider()
    st.header(f"\U0001F4CA {selected_ticker} \u2014 Price Chart")

    range_cols = st.columns(4)
    for range_label, col in zip(CHART_PERIODS, range_cols):
        with col:
            button_type = (
                "primary"
                if st.session_state.chart_range == range_label
                else "secondary"
            )
            if st.button(
                range_label,
                key=f"chart_range_{range_label}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state.chart_range = range_label
                st.rerun()

    hist = get_chart_data(selected_ticker, st.session_state.chart_range)
    if hist is None:
        st.error(f"Could not load chart data for **{selected_ticker}**. Try Refresh.")
    else:
        current_price = float(hist["Close"].iloc[-1])

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="OHLC",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["MA20"],
                mode="lines",
                name="20-day MA",
                line={"color": "orange", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["MA50"],
                mode="lines",
                name="50-day MA",
                line={"color": "blue", "width": 2},
            )
        )
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Current ${current_price:.2f}",
            annotation_position="top left",
        )
        fig.update_layout(
            height=520,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    if st.button("Close chart", key="close_chart"):
        st.session_state.selected_ticker = None
        st.rerun()


def render_news_section() -> None:
    selected_ticker = st.session_state.get("selected_ticker")
    if selected_ticker is None:
        return

    st.divider()
    st.header(f"\U0001F5DE\ufe0f Latest News \u2014 {selected_ticker}")

    articles = get_stock_news(selected_ticker)
    if not articles:
        st.info(f"No recent news found for {selected_ticker}")
        return

    for article in articles:
        title = article["title"].replace("[", "\\[").replace("]", "\\]")
        publisher = html.escape(article.get("publisher", ""))
        date = html.escape(article.get("providerPublishTime", ""))
        details = " - ".join(item for item in [publisher, date] if item)

        with st.container(border=True):
            st.markdown(f"**[{title}]({article['link']})**")
            if details:
                st.markdown(
                    f'<p style="color: gray; font-size: 0.85em;">{details}</p>',
                    unsafe_allow_html=True,
                )


def run_auto_refresh(interval_seconds: int) -> None:
    @st.fragment(run_every=interval_seconds)
    def auto_refresh_fragment() -> None:
        now = time.time()
        last_refresh = st.session_state.setdefault("last_auto_refresh", now)

        if now - last_refresh >= interval_seconds:
            st.session_state.last_auto_refresh = now
            clear_data_caches()
            st.rerun()

    auto_refresh_fragment()


st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.session_state.setdefault("selected_ticker", None)
st.session_state.setdefault("chart_range", "1M")
st.session_state.setdefault("last_auto_refresh", time.time())

st.markdown(
    """
    <style>
    div[class*="st-key-select_"] button {
        background: transparent;
        border: 0;
        box-shadow: none;
        color: inherit;
        font-size: 1.15rem;
        font-weight: 700;
        justify-content: flex-start;
        line-height: 1.25;
        min-height: auto;
        padding: 0;
        text-align: left;
        white-space: normal;
        word-break: break-word;
    }

    div[class*="st-key-select_"] button:hover {
        background: transparent;
        color: #2563eb;
        text-decoration: underline;
    }

    div[class*="st-key-select_"] button:focus {
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("📈 My Stock Dashboard")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("Refresh"):
        st.session_state.last_auto_refresh = time.time()
        clear_data_caches()
        st.rerun()

with st.sidebar:
    st.subheader("Watchlist")
    ticker_input = st.text_input("Stock ticker", placeholder="e.g. AAPL or TSLA")
    if st.button("Add to Watchlist"):
        ticker = ticker_input.strip().upper()
        if not ticker:
            st.warning("Enter a ticker symbol first.")
        else:
            watchlist = load_watchlist()
            if ticker in watchlist:
                st.info(f"{ticker} is already on your watchlist.")
            else:
                watchlist.append(ticker)
                save_watchlist(watchlist)
                st.success(f"Added {ticker} to your watchlist.")
                st.rerun()
    summary_slot = st.empty()
    st.divider()
    auto_refresh_seconds = int(
        st.number_input(
            "Auto-refresh every (seconds)",
            min_value=30,
            value=60,
            step=15,
        )
    )

if st.session_state.get("auto_refresh_seconds") != auto_refresh_seconds:
    st.session_state.auto_refresh_seconds = auto_refresh_seconds
    st.session_state.last_auto_refresh = time.time()

run_auto_refresh(auto_refresh_seconds)

watchlist = load_watchlist()

st.subheader("Your Watchlist")

signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

if not watchlist:
    st.info("Add a stock in the sidebar to get started.")
else:
    for row_start in range(0, len(watchlist), 3):
        row_tickers = watchlist[row_start : row_start + 3]
        cols = st.columns(3)
        for col, ticker in zip(cols, row_tickers):
            with col:
                with st.container(border=True):
                    signal = render_stock_card(ticker)
                    if signal in signal_counts:
                        signal_counts[signal] += 1

render_watchlist_summary(summary_slot, len(watchlist), signal_counts)

render_chart_section()
render_news_section()

st.markdown(
    '<p style="color: gray; font-size: 0.85em;">'
    "Signals are based on simple moving average logic and are for educational purposes only. "
    "Not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
