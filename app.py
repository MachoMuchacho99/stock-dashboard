import json
import html
import os
import sys
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_PATH = os.path.join(APP_DIR, "watchlist.json")
PORTFOLIO_PATH = os.path.join(APP_DIR, "portfolio.json")
ALERTS_PATH = os.path.join(APP_DIR, "alerts.json")
NOTES_PATH = os.path.join(APP_DIR, "notes.json")
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


def load_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_PATH):
        return {}
    try:
        with open(PORTFOLIO_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_portfolio(portfolio: dict) -> None:
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2)


def load_alerts() -> list[dict]:
    if not os.path.exists(ALERTS_PATH):
        return []
    try:
        with open(ALERTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_alerts(alerts: list[dict]) -> None:
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)


def load_notes() -> dict[str, str]:
    if not os.path.exists(NOTES_PATH):
        return {}
    try:
        with open(NOTES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_notes(notes: dict[str, str]) -> None:
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def clear_data_caches() -> None:
    get_stock_data.clear()
    get_chart_data.clear()
    get_intraday_data.clear()
    get_stock_news.clear()
    get_hot_stocks.clear()
    get_market_news.clear()
    get_earnings_data.clear()
    get_analyst_data.clear()
    get_correlation_data.clear()


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

        # Bollinger Bands (20-day, ±2σ)
        bb_std = hist["Close"].rolling(20, min_periods=1).std().fillna(0)
        hist["BB_upper"] = hist["MA20"] + 2 * bb_std
        hist["BB_lower"] = hist["MA20"] - 2 * bb_std

        # RSI (14-period)
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        hist["RSI"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        hist["MACD"] = ema12 - ema26
        hist["MACD_signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
        hist["MACD_hist"] = hist["MACD"] - hist["MACD_signal"]

        return hist
    except Exception:
        return None


WATCHLIST_CHART_RANGES = {
    "1m": {"period": "1d", "interval": "1m"},
    "1D": {"period": "1d", "interval": "5m"},
    "5D": {"period": "5d", "interval": "30m"},
    "1M": {"period": "1mo", "interval": "1d"},
    "6M": {"period": "6mo", "interval": "1d"},
    "1Y": {"period": "1y", "interval": "1d"},
    "5Y": {"period": "5y", "interval": "1wk"},
}

ROBINHOOD_LINE_RANGES = {
    "1D": {"period": "1d", "interval": "5m"},
    "1W": {"period": "5d", "interval": "30m"},
    "1M": {"period": "1mo", "interval": "1d"},
    "3M": {"period": "3mo", "interval": "1d"},
    "1Y": {"period": "1y", "interval": "1d"},
}


@st.cache_data(ttl=60)
def get_intraday_data(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist is None or hist.empty:
            return None

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        hist = hist[required_columns].dropna()
        if hist.empty:
            return None

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
                        "_ts": float(published_at) if isinstance(published_at, (int, float)) else 0.0,
                    }
                )

        return news_items
    except Exception:
        return []


HOT_STOCKS_FALLBACK = ["AAPL", "TSLA", "NVDA", "AMD", "SPY", "AMZN", "META", "MSFT"]


@st.cache_data(ttl=300)
def get_hot_stocks() -> list[dict]:
    symbols: list[str] = []
    try:
        result = yf.screen("most_actives")
        quotes = result.get("quotes", []) if isinstance(result, dict) else []
        for quote in quotes:
            sym = quote.get("symbol")
            if sym:
                symbols.append(sym)
    except Exception:
        symbols = []

    if not symbols:
        symbols = list(HOT_STOCKS_FALLBACK)

    rows = []
    for sym in symbols[:20]:
        data = get_stock_data(sym)
        if data is None:
            continue
        rows.append(
            {
                "Ticker": sym,
                "Price": round(data["price"], 2),
                "Change %": round(data["change_pct"], 2),
                "Volume": data["volume"],
            }
        )

    rows.sort(key=lambda r: r["Volume"], reverse=True)
    return rows


def render_hot_stocks() -> None:
    st.subheader("🔥 Most Active Stocks")
    st.caption("Sorted by trading volume (highest first).")

    with st.spinner("Loading hot stocks..."):
        rows = get_hot_stocks()

    if not rows:
        st.info("Could not load hot stocks right now. Try Refresh.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        },
    )


@st.cache_data(ttl=300)
def get_market_news(extra_tickers: tuple[str, ...]) -> list[dict]:
    source_tickers = ["SPY", "QQQ"] + list(extra_tickers)
    seen_titles: set[str] = set()
    combined: list[dict] = []

    for sym in source_tickers:
        for article in get_stock_news(sym):
            key = article["title"].strip().lower()
            if not key or key in seen_titles:
                continue
            seen_titles.add(key)
            combined.append(article)

    combined.sort(key=lambda a: a.get("_ts", 0.0), reverse=True)
    return combined[:15]


def render_market_news(watchlist: list[str]) -> None:
    st.subheader("📰 Market News")

    extra = tuple(watchlist[:3])
    with st.spinner("Loading market news..."):
        articles = get_market_news(extra)

    if not articles:
        st.info("No market news available right now. Try Refresh.")
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


@st.cache_data(ttl=3600)
def get_sector_info(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        return {
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_earnings_data(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if ts and isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts)
            return {"ticker": ticker, "date": dt, "date_str": dt.strftime("%b %d, %Y")}
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_analyst_data(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        recommendation = info.get("recommendationKey", "")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        num_analysts = info.get("numberOfAnalystOpinions")
        if not recommendation:
            return None
        label_map = {
            "strong_buy": "Strong Buy",
            "buy": "Buy",
            "hold": "Hold",
            "underperform": "Underperform",
            "sell": "Sell",
        }
        return {
            "recommendation": label_map.get(recommendation, recommendation.title()),
            "target_mean": target_mean,
            "target_high": target_high,
            "target_low": target_low,
            "num_analysts": num_analysts,
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_correlation_data(tickers: tuple[str, ...]) -> pd.DataFrame | None:
    try:
        if len(tickers) < 2:
            return None
        frames = {}
        for ticker in tickers:
            hist = yf.Ticker(ticker).history(period="3mo")
            if hist is not None and not hist.empty:
                frames[ticker] = hist["Close"]
        if len(frames) < 2:
            return None
        df = pd.DataFrame(frames).dropna()
        return df.pct_change().dropna().corr()
    except Exception:
        return None


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


BACKTEST_PERIODS = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}


@st.cache_data(ttl=300)
def run_backtest(ticker: str, period: str, starting_cash: float) -> dict | None:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist is None or hist.empty or len(hist) < 60:
        return None

    closes = hist["Close"]
    ma20 = closes.rolling(20).mean()
    ma50 = closes.rolling(50).mean()
    high_52 = closes.rolling(252, min_periods=1).max()
    low_52 = closes.rolling(252, min_periods=1).min()

    cash = float(starting_cash)
    shares = 0
    trades = 0
    dates: list[str] = []
    values: list[float] = []

    for i in range(len(closes)):
        price = float(closes.iloc[i])
        m20 = ma20.iloc[i]
        m50 = ma50.iloc[i]

        if not (pd.isna(m20) or pd.isna(m50)):
            signal = get_signal(
                price,
                float(m20),
                float(m50),
                float(high_52.iloc[i]),
                float(low_52.iloc[i]),
            )["signal"]

            if signal == "BUY" and shares == 0:
                qty = int(cash // price)
                if qty > 0:
                    shares += qty
                    cash -= qty * price
                    trades += 1
            elif signal == "SELL" and shares > 0:
                cash += shares * price
                shares = 0
                trades += 1

        dates.append(closes.index[i].strftime("%Y-%m-%d"))
        values.append(cash + shares * price)

    final_price = float(closes.iloc[-1])
    final_value = cash + shares * final_price
    total_return = (final_value - starting_cash) / starting_cash * 100

    first_price = float(closes.iloc[0])
    bh_shares = starting_cash / first_price
    bh_values = [bh_shares * float(closes.iloc[i]) for i in range(len(closes))]
    bh_return = (bh_values[-1] - starting_cash) / starting_cash * 100

    return {
        "final_value": final_value,
        "total_return": total_return,
        "trades": trades,
        "dates": dates,
        "values": values,
        "bh_values": bh_values,
        "bh_return": bh_return,
    }


def render_backtester(watchlist: list[str]) -> None:
    st.divider()
    st.subheader("⏮️ Strategy Backtester")

    if not watchlist:
        st.info("Add a stock to the watchlist to run a backtest.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.selectbox("Ticker", watchlist, key="bt_ticker")
    with col2:
        period_label = st.selectbox(
            "Period", list(BACKTEST_PERIODS.keys()), index=1, key="bt_period"
        )
    with col3:
        starting_cash = st.number_input(
            "Starting cash ($)",
            min_value=1.0,
            value=10000.0,
            step=100.0,
            key="bt_cash",
        )

    if st.button("Run Backtest", key="bt_run"):
        with st.spinner(f"Backtesting {ticker}..."):
            result = run_backtest(
                ticker, BACKTEST_PERIODS[period_label], float(starting_cash)
            )

        if result is None:
            st.error("Not enough historical data to run a backtest for this ticker.")
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Final Value", f"${result['final_value']:,.2f}")
        m2.metric("Total Return %", f"{result['total_return']:+.2f}%")
        m3.metric("Number of Trades", result["trades"])

        st.metric("Buy & Hold Return %", f"{result['bh_return']:+.2f}%")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=result["dates"],
                y=result["values"],
                name="Strategy",
                line={"color": "#16a34a"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=result["dates"],
                y=result["bh_values"],
                name="Buy & Hold",
                line={"color": "#3b82f6"},
            )
        )
        fig.update_layout(
            height=360,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            yaxis_title="Portfolio Value ($)",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Backtests use historical data and do not guarantee future results."
        )


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
        width="stretch",
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

    notes = load_notes()
    if notes.get(ticker, "").strip():
        st.caption(f"📝 {notes[ticker][:80]}{'…' if len(notes[ticker]) > 80 else ''}")

    if st.button("Remove", key=f"remove_{ticker}"):
        remove_from_watchlist(ticker)
        st.rerun()

    return signal


def render_portfolio_section(watchlist: list[str]) -> None:
    portfolio = load_portfolio()
    if not portfolio:
        return

    st.subheader("📁 My Portfolio")

    rows = []
    total_value = 0.0
    for ticker, holding in portfolio.items():
        shares = holding.get("shares", 0)
        avg_buy = holding.get("avg_buy_price", 0)
        data = get_stock_data(ticker)
        current_price = data["price"] if data else None

        if current_price is None:
            continue

        total_val = shares * current_price
        gain_loss = (current_price - avg_buy) * shares
        gain_loss_pct = ((current_price - avg_buy) / avg_buy) * 100 if avg_buy else 0
        total_value += total_val

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Avg Buy Price": avg_buy,
            "Current Price": current_price,
            "Total Value": total_val,
            "Gain/Loss ($)": gain_loss,
            "Gain/Loss (%)": gain_loss_pct,
        })

    if not rows:
        st.info("No portfolio data available. Add holdings in the sidebar.")
        return

    header_cols = st.columns([1, 1, 1.5, 1.5, 1.5, 1.5, 1.5])
    headers = ["Ticker", "Shares", "Avg Buy Price", "Current Price", "Total Value", "Gain/Loss ($)", "Gain/Loss (%)"]
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    for row in rows:
        gl = row["Gain/Loss ($)"]
        gl_pct = row["Gain/Loss (%)"]
        color = "#16a34a" if gl >= 0 else "#dc2626"
        sign = "+" if gl >= 0 else ""

        row_cols = st.columns([1, 1, 1.5, 1.5, 1.5, 1.5, 1.5])
        row_cols[0].markdown(f"**{row['Ticker']}**")
        row_cols[1].write(f"{row['Shares']:.4g}")
        row_cols[2].write(f"${row['Avg Buy Price']:.2f}")
        row_cols[3].write(f"${row['Current Price']:.2f}")
        row_cols[4].write(f"${row['Total Value']:,.2f}")
        row_cols[5].markdown(
            f'<span style="color:{color}; font-weight:600;">{sign}${gl:,.2f}</span>',
            unsafe_allow_html=True,
        )
        row_cols[6].markdown(
            f'<span style="color:{color}; font-weight:600;">{sign}{gl_pct:.2f}%</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    st.divider()


def render_alerts_section(watchlist: list[str]) -> None:
    alerts = load_alerts()
    if not alerts:
        return

    st.subheader("🔔 Active Alerts")

    alerts_changed = False
    for i, alert in enumerate(alerts):
        ticker = alert["ticker"]
        target = alert["target_price"]
        condition = alert["condition"]

        data = get_stock_data(ticker)
        current_price = data["price"] if data else None

        triggered = False
        if current_price is not None:
            if condition == "above" and current_price > target:
                triggered = True
            elif condition == "below" and current_price < target:
                triggered = True

        cond_label = "risen above" if condition == "above" else "fallen below"

        col_msg, col_btn = st.columns([5, 1])
        with col_msg:
            if triggered:
                st.success(f"**{ticker}** has {cond_label} ${target:.2f} ✅")
            else:
                with st.container(border=True):
                    st.markdown(
                        f"**{ticker}** — {cond_label.capitalize()} **${target:.2f}**"
                        + (f"  _(current: ${current_price:.2f})_" if current_price else "")
                    )
        with col_btn:
            if st.button("Remove", key=f"alert_remove_{i}"):
                alerts.pop(i)
                save_alerts(alerts)
                alerts_changed = True
                break

    if alerts_changed:
        st.rerun()

    st.divider()


def render_chart_section() -> None:
    selected_ticker = st.session_state.get("selected_ticker")
    if selected_ticker is None:
        return

    if st.session_state.chart_range not in CHART_PERIODS:
        st.session_state.chart_range = "1M"

    st.divider()
    st.header(f"\U0001F4CA {selected_ticker} — Price Chart")

    # Time range buttons
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
                width="stretch",
                type=button_type,
            ):
                st.session_state.chart_range = range_label
                st.rerun()

    # Indicator toggle buttons
    st.caption("Technical Indicators")
    ind_cols = st.columns(3)
    with ind_cols[0]:
        bb_type = "primary" if st.session_state.show_bollinger else "secondary"
        if st.button("Bollinger Bands", key="toggle_bb", width="stretch", type=bb_type):
            st.session_state.show_bollinger = not st.session_state.show_bollinger
            st.rerun()
    with ind_cols[1]:
        rsi_type = "primary" if st.session_state.show_rsi else "secondary"
        if st.button("RSI", key="toggle_rsi", width="stretch", type=rsi_type):
            st.session_state.show_rsi = not st.session_state.show_rsi
            st.rerun()
    with ind_cols[2]:
        macd_type = "primary" if st.session_state.show_macd else "secondary"
        if st.button("MACD", key="toggle_macd", width="stretch", type=macd_type):
            st.session_state.show_macd = not st.session_state.show_macd
            st.rerun()

    hist = get_chart_data(selected_ticker, st.session_state.chart_range)
    if hist is None:
        st.error(f"Could not load chart data for **{selected_ticker}**. Try Refresh.")
    else:
        show_bb = st.session_state.show_bollinger
        show_rsi = st.session_state.show_rsi
        show_macd = st.session_state.show_macd

        current_price = float(hist["Close"].iloc[-1])

        # Build subplot grid
        subplot_rows = 1 + int(show_rsi) + int(show_macd)
        if subplot_rows > 1:
            extra = subplot_rows - 1
            main_h = 0.55
            sub_h = (1.0 - main_h) / extra
            row_heights = [main_h] + [sub_h] * extra
            subplot_titles = [""] + (["RSI (14)"] if show_rsi else []) + (["MACD (12,26,9)"] if show_macd else [])
            fig = make_subplots(
                rows=subplot_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=row_heights,
                subplot_titles=subplot_titles,
            )
        else:
            fig = make_subplots(rows=1, cols=1)

        # --- Main chart (row 1) ---
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="OHLC",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist["MA20"],
                mode="lines", name="MA20",
                line={"color": "orange", "width": 1.5},
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist["MA50"],
                mode="lines", name="MA50",
                line={"color": "#3b82f6", "width": 1.5},
            ),
            row=1, col=1,
        )

        # Bollinger Bands overlay
        if show_bb:
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist["BB_upper"],
                    mode="lines", name="BB Upper",
                    line={"color": "rgba(168,85,247,0.6)", "width": 1, "dash": "dot"},
                    showlegend=True,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist["BB_lower"],
                    mode="lines", name="BB Lower",
                    line={"color": "rgba(168,85,247,0.6)", "width": 1, "dash": "dot"},
                    fill="tonexty",
                    fillcolor="rgba(168,85,247,0.07)",
                    showlegend=True,
                ),
                row=1, col=1,
            )

        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"${current_price:.2f}",
            annotation_position="top left",
            row=1, col=1,
        )

        # --- RSI panel ---
        current_row = 2
        if show_rsi:
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist["RSI"],
                    mode="lines", name="RSI",
                    line={"color": "#f59e0b", "width": 1.5},
                ),
                row=current_row, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(220,38,38,0.5)",
                          row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(22,163,74,0.5)",
                          row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)
            current_row += 1

        # --- MACD panel ---
        if show_macd:
            macd_colors = [
                "#16a34a" if v >= 0 else "#dc2626"
                for v in hist["MACD_hist"]
            ]
            fig.add_trace(
                go.Bar(
                    x=hist.index, y=hist["MACD_hist"],
                    name="MACD Hist",
                    marker_color=macd_colors,
                    opacity=0.6,
                ),
                row=current_row, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist["MACD"],
                    mode="lines", name="MACD",
                    line={"color": "#3b82f6", "width": 1.5},
                ),
                row=current_row, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hist.index, y=hist["MACD_signal"],
                    mode="lines", name="Signal",
                    line={"color": "#f97316", "width": 1.5},
                ),
                row=current_row, col=1,
            )

        total_height = 520 + 180 * (int(show_rsi) + int(show_macd))
        fig.update_layout(
            height=total_height,
            margin={"l": 20, "r": 20, "t": 30, "b": 20},
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)

        st.plotly_chart(fig, width="stretch")

    if st.button("Close chart", key="close_chart"):
        st.session_state.selected_ticker = None
        st.rerun()


def render_news_section() -> None:
    selected_ticker = st.session_state.get("selected_ticker")
    if selected_ticker is None:
        return

    st.divider()
    st.header(f"\U0001F5DE️ Latest News — {selected_ticker}")

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


def render_analyst_ratings(watchlist: list[str]) -> None:
    if not watchlist:
        return

    st.divider()
    st.subheader("🔬 Analyst Ratings")

    rows = []
    for ticker in watchlist:
        data = get_stock_data(ticker)
        analyst = get_analyst_data(ticker)
        if not data or not analyst:
            continue
        price = data["price"]
        target = analyst["target_mean"]
        upside = ((target - price) / price * 100) if target else None
        rows.append({
            "ticker": ticker,
            "recommendation": analyst["recommendation"],
            "target_mean": target,
            "target_high": analyst["target_high"],
            "target_low": analyst["target_low"],
            "num_analysts": analyst["num_analysts"],
            "price": price,
            "upside": upside,
        })

    if not rows:
        st.info("No analyst data available.")
        return

    rec_color = {
        "Strong Buy": "#16a34a",
        "Buy": "#4ade80",
        "Hold": "#f59e0b",
        "Underperform": "#f97316",
        "Sell": "#dc2626",
    }

    header_cols = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1.5])
    for col, h in zip(header_cols, ["Ticker", "Consensus", "Price Target", "Range", "Upside", "# Analysts"]):
        col.markdown(f"**{h}**")

    for row in rows:
        color = rec_color.get(row["recommendation"], "inherit")
        cols = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1.5])
        cols[0].markdown(f"**{row['ticker']}**")
        cols[1].markdown(f'<span style="color:{color};font-weight:600;">{row["recommendation"]}</span>', unsafe_allow_html=True)
        cols[2].write(f"${row['target_mean']:.2f}" if row["target_mean"] else "—")
        lo = f"${row['target_low']:.0f}" if row["target_low"] else "—"
        hi = f"${row['target_high']:.0f}" if row["target_high"] else "—"
        cols[3].write(f"{lo} – {hi}")
        if row["upside"] is not None:
            up_color = "#16a34a" if row["upside"] >= 0 else "#dc2626"
            sign = "+" if row["upside"] >= 0 else ""
            cols[4].markdown(f'<span style="color:{up_color};font-weight:600;">{sign}{row["upside"]:.1f}%</span>', unsafe_allow_html=True)
        else:
            cols[4].write("—")
        cols[5].write(str(row["num_analysts"]) if row["num_analysts"] else "—")


def render_correlation_matrix(watchlist: list[str]) -> None:
    if len(watchlist) < 2:
        return

    st.divider()
    st.subheader("🔗 Correlation Matrix")
    st.caption("3-month daily return correlations. Values near 1.0 = move together, near -1.0 = move opposite.")

    corr = get_correlation_data(tuple(watchlist))
    if corr is None or corr.empty:
        st.info("Not enough data to build correlation matrix.")
        return

    tickers = list(corr.columns)
    z = corr.values.tolist()
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            colorbar={"title": "r"},
        )
    )
    fig.update_layout(
        height=max(300, 80 * len(tickers)),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")


def render_notes_section(watchlist: list[str]) -> None:
    if not watchlist:
        return

    notes = load_notes()
    tickers_with_notes = [t for t in watchlist if notes.get(t, "").strip()]
    if not tickers_with_notes:
        return

    st.divider()
    st.subheader("📝 My Notes")
    for ticker in tickers_with_notes:
        with st.container(border=True):
            st.markdown(f"**{ticker}**")
            st.write(notes[ticker])


def render_live_chart(ticker: str) -> None:
    st.session_state.setdefault("watchlist_line_ranges", {})
    st.session_state.watchlist_line_ranges.setdefault(ticker, "1D")

    selected_refresh_range = st.session_state.watchlist_line_ranges.get(ticker, "1D")
    refresh_seconds = 10 if selected_refresh_range == "1D" else 30

    @st.fragment(run_every=refresh_seconds)
    def _live_chart() -> None:
        ranges = ROBINHOOD_LINE_RANGES
        st.session_state.watchlist_line_ranges.setdefault(ticker, "1D")
        if st.session_state.watchlist_line_ranges[ticker] not in ranges:
            st.session_state.watchlist_line_ranges[ticker] = "1D"

        active_range = st.session_state.watchlist_line_ranges[ticker]
        active_color = st.session_state.get(f"chart_accent_{ticker}", "#00C805")
        range_cols = st.columns(len(ranges))
        for range_label, col in zip(ranges, range_cols):
            with col:
                is_active = active_range == range_label
                if is_active:
                    st.markdown(
                        f"""
                        <style>
                        div[class*="st-key-wlchart_{ticker}_{range_label}"] button {{
                            border-color: {active_color};
                            color: {active_color};
                            font-weight: 800;
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                if st.button(
                    range_label,
                    key=f"wlchart_{ticker}_{range_label}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.watchlist_line_ranges[ticker] = range_label
                    st.rerun()

        selected_range = st.session_state.watchlist_line_ranges[ticker]
        range_cfg = ranges[selected_range]

        st.session_state.setdefault("stop_loss_prices", {})
        stop_loss = st.number_input(
            "Stop-loss price ($)",
            min_value=0.0,
            value=float(st.session_state.stop_loss_prices.get(ticker, 0.0)),
            step=0.01,
            key=f"stop_loss_{ticker}",
        )
        st.session_state.stop_loss_prices[ticker] = stop_loss

        get_intraday_data.clear()
        hist = get_intraday_data(ticker, range_cfg["period"], range_cfg["interval"])

        if hist is None or hist.empty:
            st.info("No data available for this range right now.")
            return

        current_price = float(hist["Close"].iloc[-1])
        baseline_price = float(hist["Open"].iloc[0])
        change_amount = current_price - baseline_price
        change_pct = (change_amount / baseline_price * 100) if baseline_price else 0.0
        change_color = "#00C805" if change_amount >= 0 else "#FF5000"
        st.session_state[f"chart_accent_{ticker}"] = change_color
        change_sign = "+" if change_amount >= 0 else ""
        direction_icon = "▲" if change_amount >= 0 else "▼"
        range_label = "Today" if selected_range == "1D" else selected_range
        updated_at = datetime.now().strftime("%I:%M:%S %p")

        st.markdown(
            f"""
            <style>
            @keyframes pulse-live {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.45; }}
                100% {{ opacity: 1; }}
            }}
            .live-row {{
                align-items: baseline;
                display: flex;
                gap: 1rem;
                margin: 0.25rem 0 0;
            }}
            .live-price {{
                font-size: 2.6rem;
                font-weight: 800;
                letter-spacing: -0.04em;
            }}
            .live-change {{
                color: {change_color};
                font-size: 1.1rem;
                font-weight: 700;
            }}
            .live-badge {{
                animation: pulse-live 1.3s ease-in-out infinite;
                background: rgba(220, 38, 38, 0.12);
                border: 1px solid rgba(220, 38, 38, 0.35);
                border-radius: 999px;
                color: #dc2626;
                font-size: 0.8rem;
                font-weight: 800;
                padding: 0.15rem 0.55rem;
            }}
            .live-updated {{
                color: gray;
                font-size: 0.85rem;
                margin: -0.25rem 0 0.5rem;
            }}
            </style>
            <div class="live-row">
                <span class="live-price">${current_price:,.2f}</span>
                <span class="live-badge">🔴 LIVE</span>
            </div>
            <div class="live-change">
                {direction_icon} {change_sign}${abs(change_amount):,.2f}
                ({change_sign}{change_pct:.2f}%) {range_label}
            </div>
            <div class="live-updated">Updated {updated_at}</div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        close_min = float(hist["Close"].min())
        close_max = float(hist["Close"].max())
        plot_min = min(close_min, stop_loss) if stop_loss > 0 else close_min
        plot_max = max(close_max, stop_loss) if stop_loss > 0 else close_max
        padding = max((plot_max - plot_min) * 0.12, plot_max * 0.002)
        y_floor = plot_min - padding
        gradient_color = (
            "rgba(0, 200, 5, 0.22)"
            if change_amount >= 0
            else "rgba(255, 80, 0, 0.22)"
        )
        transparent_color = (
            "rgba(0, 200, 5, 0)"
            if change_amount >= 0
            else "rgba(255, 80, 0, 0)"
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=[y_floor] * len(hist),
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                line={"color": change_color, "width": 3, "shape": "spline"},
                fill="tonexty",
                fillgradient={
                    "type": "vertical",
                    "colorscale": [
                        [0, transparent_color],
                        [1, gradient_color],
                    ],
                },
                hovertemplate="$%{y:,.2f}<extra></extra>",
                name=ticker,
            )
        )

        visible = hist.copy()
        visible["MA20"] = visible["Close"].rolling(20).mean()
        visible["MA50"] = visible["Close"].rolling(50).mean()
        ma_diff = visible["MA20"] - visible["MA50"]
        prev_diff = ma_diff.shift(1)
        buy_mask = (prev_diff <= 0) & (ma_diff > 0)
        sell_mask = (prev_diff >= 0) & (ma_diff < 0)
        buys = visible[buy_mask.fillna(False)]
        sells = visible[sell_mask.fillna(False)]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["Close"],
                    mode="markers+text",
                    marker={"symbol": "triangle-up", "size": 14, "color": "#00C805"},
                    text=["BUY"] * len(buys),
                    textposition="top center",
                    textfont={"color": "#00C805", "size": 11},
                    hovertemplate="BUY<br>$%{y:,.2f}<extra></extra>",
                    name="BUY",
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["Close"],
                    mode="markers+text",
                    marker={"symbol": "triangle-down", "size": 14, "color": "#FF5000"},
                    text=["SELL"] * len(sells),
                    textposition="bottom center",
                    textfont={"color": "#FF5000", "size": 11},
                    hovertemplate="SELL<br>$%{y:,.2f}<extra></extra>",
                    name="SELL",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[hist.index[-1]],
                y=[current_price],
                mode="markers",
                marker={
                    "color": change_color,
                    "size": 9,
                    "line": {"color": "white", "width": 2},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if stop_loss > 0:
            fig.add_hline(
                y=stop_loss,
                line_dash="dash",
                line_color="#FF5000",
                line_width=1,
            )

        fig.update_layout(
            height=520,
            hovermode="x",
            margin={"l": 8, "r": 8, "t": 10, "b": 8},
            xaxis_rangeslider_visible=False,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont={"color": "rgba(148, 163, 184, 0.72)", "size": 11},
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(148, 163, 184, 0.55)",
            spikethickness=1,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            tickfont={"color": "rgba(148, 163, 184, 0.72)", "size": 11},
            range=[y_floor, plot_max + padding],
        )

        st.plotly_chart(fig, width="stretch")

        if stop_loss > 0:
            if current_price <= stop_loss:
                st.warning("⚠️ Price has hit your stop-loss level.")
            else:
                distance_pct = ((current_price - stop_loss) / stop_loss) * 100
                st.caption(
                    f"Current price is {distance_pct:.1f}% above your stop-loss."
                )

    _live_chart()


def render_market_heatmap(watchlist: list[str]) -> None:
    st.divider()
    st.subheader("📊 Watchlist Charts")

    if len(watchlist) < 1:
        st.info("Add a stock to see charts.")
        return

    st.session_state.setdefault("watchlist_chart_ranges", {})

    # Default the first stock open; keep open_chart valid against the watchlist.
    if st.session_state.get("open_chart") not in watchlist:
        st.session_state.open_chart = watchlist[0]

    for ticker in watchlist:
        st.session_state.watchlist_chart_ranges.setdefault(ticker, "1M")

        is_open = st.session_state.open_chart == ticker
        label = ticker
        if is_open:
            data = get_stock_data(ticker)
            name = data["name"] if data else ticker
            label = f"{ticker} — {name}"

        if st.button(
            f"{'▼' if is_open else '▶'} {label}",
            key=f"accordion_{ticker}",
            width="stretch",
            type="primary" if is_open else "secondary",
        ):
            if is_open:
                st.session_state.open_chart = None
            else:
                st.session_state.open_chart = ticker
                # Fetch fresh data the moment a stock is unfolded.
                get_intraday_data.clear()
            st.rerun()

        # Only the open stock fetches/renders a chart and auto-refreshes.
        if is_open:
            render_live_chart(ticker)


def build_export_csv(watchlist: list[str]) -> bytes:
    portfolio = load_portfolio()
    rows = []
    for ticker in watchlist:
        data = get_stock_data(ticker)
        if data is None:
            continue
        signal_info = get_signal(
            data["price"], data["ma20"], data["ma50"],
            data["week_52_high"], data["week_52_low"],
        )
        holding = portfolio.get(ticker, {})
        shares = holding.get("shares", 0)
        avg_buy = holding.get("avg_buy_price", 0)
        gain_loss = (data["price"] - avg_buy) * shares if shares else None
        gain_loss_pct = ((data["price"] - avg_buy) / avg_buy * 100) if (shares and avg_buy) else None

        rows.append({
            "Ticker": ticker,
            "Name": data["name"],
            "Price": round(data["price"], 2),
            "Change (%)": round(data["change_pct"], 2),
            "MA20": round(data["ma20"], 2),
            "MA50": round(data["ma50"], 2),
            "52W High": round(data["week_52_high"], 2),
            "52W Low": round(data["week_52_low"], 2),
            "Signal": signal_info["signal"],
            "Shares": shares or "",
            "Avg Buy Price": avg_buy or "",
            "Gain/Loss ($)": round(gain_loss, 2) if gain_loss is not None else "",
            "Gain/Loss (%)": round(gain_loss_pct, 2) if gain_loss_pct is not None else "",
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def render_export_section(watchlist: list[str]) -> None:
    if not watchlist:
        return

    st.divider()
    csv_bytes = build_export_csv(watchlist)
    filename = f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    st.download_button(
        label="⬇️ Export Watchlist to CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def render_earnings_calendar(watchlist: list[str]) -> None:
    if not watchlist:
        return

    st.divider()
    st.subheader("📅 Earnings Calendar")

    entries = []
    for ticker in watchlist:
        result = get_earnings_data(ticker)
        if result:
            entries.append(result)

    if not entries:
        st.info("No upcoming earnings dates found.")
        return

    now = datetime.now()
    entries.sort(key=lambda x: x["date"])

    header_cols = st.columns([1, 2, 2])
    header_cols[0].markdown("**Ticker**")
    header_cols[1].markdown("**Earnings Date**")
    header_cols[2].markdown("**Days Away**")

    for entry in entries:
        days_away = (entry["date"] - now).days
        if days_away < 0:
            days_label = f"{abs(days_away)}d ago"
            color = "gray"
        elif days_away == 0:
            days_label = "Today"
            color = "#f59e0b"
        elif days_away <= 7:
            days_label = f"{days_away}d"
            color = "#f59e0b"
        else:
            days_label = f"{days_away}d"
            color = "inherit"

        row_cols = st.columns([1, 2, 2])
        row_cols[0].markdown(f"**{entry['ticker']}**")
        row_cols[1].write(entry["date_str"])
        row_cols[2].markdown(
            f'<span style="color:{color};">{days_label}</span>',
            unsafe_allow_html=True,
        )


# --- Kronos Forecast integration (shared engine in Projects\Kronos-Forecast) ---
KRONOS_FORECAST_SRC = r"C:\Users\river\Projects\Kronos-Forecast\src"


def _import_kronos_forecast():
    """Lazily import the shared Kronos engine; heavy deps load on demand only."""
    if KRONOS_FORECAST_SRC not in sys.path:
        sys.path.insert(0, KRONOS_FORECAST_SRC)
    import kronos_forecast
    from kronos_forecast import charts as kf_charts

    return kronos_forecast, kf_charts


@st.cache_data(ttl=3600, show_spinner=False)
def get_kronos_forecast(ticker: str, timeframe: str, pred_len: int, n_paths: int):
    kf, _ = _import_kronos_forecast()
    df = kf.fetch_ohlcv(ticker, timeframe)
    return kf.forecast(df, pred_len=pred_len, n_paths=n_paths)


def render_forecast_tab(watchlist: list[str]) -> None:
    st.subheader("🔮 Kronos Forecast")
    st.caption(
        "Probabilistic candle forecasts from the Kronos foundation model — "
        "N sampled futures aggregated into a median path, uncertainty cone and "
        "direction signal. Research model, not investment advice."
    )

    try:
        _, kf_charts = _import_kronos_forecast()
    except Exception as exc:
        st.warning(
            "Kronos engine not available. Set it up once, then reload:\n\n"
            "1. `C:\\Users\\river\\Projects\\Kronos-Forecast` must exist "
            "(see its README.md)\n"
            "2. Launch the dashboard via `run.bat` so it uses the Kronos venv\n\n"
            f"Import error: `{exc}`"
        )
        return

    options = watchlist if watchlist else ["AAPL"]
    sel_cols = st.columns([2, 1.2, 1.2, 1.2, 1])
    with sel_cols[0]:
        base = st.selectbox("Ticker", options + ["Custom…"], key="kf_ticker_select")
        if base == "Custom…":
            ticker = st.text_input(
                "Custom symbol", value="", key="kf_ticker_custom",
                placeholder="e.g. NVDA or EURUSD=X",
            ).strip().upper()
        else:
            ticker = base
    with sel_cols[1]:
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"], key="kf_tf")
    with sel_cols[2]:
        pred_len = st.select_slider(
            "Horizon (candles)", options=[6, 12, 24, 36, 48], value=24, key="kf_horizon"
        )
    with sel_cols[3]:
        n_paths = st.select_slider(
            "Sample paths", options=[10, 20, 30, 40], value=30, key="kf_paths"
        )
    with sel_cols[4]:
        st.write("")
        generate = st.button("Generate", type="primary", width="stretch", key="kf_go")

    if generate:
        if not ticker:
            st.warning("Pick or type a symbol first.")
            return
        try:
            with st.spinner(f"Sampling {n_paths} futures for {ticker}… "
                            "(first run downloads the model)"):
                result = get_kronos_forecast(ticker, timeframe, int(pred_len), int(n_paths))
            st.session_state["kf_last"] = (ticker, timeframe, result)
        except Exception as exc:
            st.error(f"Forecast failed for {ticker}: {exc}")
            return

    cached = st.session_state.get("kf_last")
    if not cached:
        st.info("Pick a ticker and hit **Generate** — forecasts are cached for an hour.")
        return

    r_ticker, r_tf, result = cached

    arrow = kf_charts.DIRECTION_ARROWS[result.direction]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signal", f"{arrow} {result.direction.upper()}")
    m2.metric("P(up)", f"{result.p_up:.0%}")
    m3.metric("Confidence", f"{result.confidence:.0%}")
    m4.metric(
        "Median move",
        f"{result.expected_move:+.2%}",
        help=f"Median path return over {result.meta['pred_len']} candles",
    )

    st.plotly_chart(
        kf_charts.build_forecast_figure(
            result, title=f"{r_ticker} · {r_tf} · {result.meta['n_paths']} sampled futures"
        ),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": True},
    )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.plotly_chart(
            kf_charts.build_prob_gauge(result), width="stretch",
            config={"displayModeBar": False},
        )
    with detail_cols[1]:
        st.plotly_chart(
            kf_charts.build_return_histogram(result), width="stretch",
            config={"displayModeBar": False},
        )

    for note in result.meta.get("notes", []):
        st.caption(f"⚠ {note}")
    st.caption(
        f"Model {result.meta['model_id']} on {result.meta['device']} · "
        f"context {result.meta['context_len']} candles · "
        f"inference {result.meta['elapsed_s']}s · cached ≤ 1h"
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
st.session_state.setdefault("show_bollinger", False)
st.session_state.setdefault("show_rsi", False)
st.session_state.setdefault("show_macd", False)

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

watchlist = load_watchlist()

with st.sidebar:
    st.subheader("📈 Stock Dashboard")
    if st.button("🔄 Refresh", width="stretch"):
        st.session_state.last_auto_refresh = time.time()
        clear_data_caches()
        st.rerun()

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

tab_watchlist, tab_charts, tab_portfolio, tab_hot, tab_news, tab_forecast = st.tabs(
    ["📋 Watchlist", "📊 Charts", "📁 Portfolio", "🔥 Hot Stocks", "📰 News", "🔮 Forecast"]
)

with tab_watchlist:
    st.subheader("Add to Watchlist")
    add_cols = st.columns([3, 1])
    with add_cols[0]:
        ticker_input = st.text_input(
            "Stock ticker",
            placeholder="e.g. AAPL or TSLA",
            label_visibility="collapsed",
        )
    with add_cols[1]:
        add_clicked = st.button("Add", width="stretch")
    if add_clicked:
        ticker = ticker_input.strip().upper()
        if not ticker:
            st.warning("Enter a ticker symbol first.")
        else:
            wl = load_watchlist()
            if ticker in wl:
                st.info(f"{ticker} is already on your watchlist.")
            else:
                wl.append(ticker)
                save_watchlist(wl)
                st.success(f"Added {ticker} to your watchlist.")
                st.rerun()

    summary_slot = st.empty()
    st.divider()

    st.subheader("Your Watchlist")

    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    if not watchlist:
        st.info("Add a stock above to get started.")
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

    # Price Alerts
    st.divider()
    st.subheader("🔔 Price Alerts")
    if watchlist:
        alert_cols = st.columns([2, 2, 2, 1])
        with alert_cols[0]:
            alert_ticker = st.selectbox(
                "Ticker", watchlist, key="alert_ticker_select"
            )
        with alert_cols[1]:
            alert_target = st.number_input(
                "Target price ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                key="alert_target_input",
            )
        with alert_cols[2]:
            alert_condition_label = st.selectbox(
                "Condition",
                ["Rises above", "Falls below"],
                key="alert_condition_select",
            )
        with alert_cols[3]:
            st.write("")
            add_alert_clicked = st.button("Add Alert", width="stretch")
        alert_condition = "above" if alert_condition_label == "Rises above" else "below"
        if add_alert_clicked:
            alerts = load_alerts()
            alerts.append({
                "ticker": alert_ticker,
                "target_price": alert_target,
                "condition": alert_condition,
            })
            save_alerts(alerts)
            st.success(f"Alert added for {alert_ticker}.")
            st.rerun()
    else:
        st.caption("Add stocks to your watchlist first.")

    render_alerts_section(watchlist)

    # Notes
    st.divider()
    st.subheader("📝 Notes")
    if watchlist:
        note_ticker = st.selectbox("Ticker", watchlist, key="note_ticker_select")
        existing_notes = load_notes()
        note_text = st.text_area(
            "Note",
            value=existing_notes.get(note_ticker, ""),
            height=100,
            key="note_text_input",
            placeholder="Entry thesis, reminders, price targets...",
        )
        note_btn_cols = st.columns([1, 1, 4])
        with note_btn_cols[0]:
            if st.button("Save Note", width="stretch"):
                existing_notes[note_ticker] = note_text
                save_notes(existing_notes)
                st.success(f"Note saved for {note_ticker}.")
                st.rerun()
        with note_btn_cols[1]:
            if note_ticker in existing_notes and st.button("Delete Note", width="stretch"):
                del existing_notes[note_ticker]
                save_notes(existing_notes)
                st.rerun()
    else:
        st.caption("Add stocks to your watchlist first.")

    render_notes_section(watchlist)

    render_analyst_ratings(watchlist)
    render_earnings_calendar(watchlist)

with tab_charts:
    render_market_heatmap(watchlist)
    render_chart_section()
    render_news_section()
    render_correlation_matrix(watchlist)
    render_backtester(watchlist)

with tab_portfolio:
    st.subheader("📁 Add to Portfolio")
    if watchlist:
        pf_cols = st.columns([2, 2, 2, 1])
        with pf_cols[0]:
            portfolio_ticker = st.selectbox(
                "Ticker", watchlist, key="portfolio_ticker_select"
            )
        with pf_cols[1]:
            portfolio_shares = st.number_input(
                "Shares owned",
                min_value=0.01,
                value=1.0,
                step=0.01,
                key="portfolio_shares_input",
            )
        with pf_cols[2]:
            portfolio_avg_price = st.number_input(
                "Avg buy price ($)",
                min_value=0.01,
                value=1.0,
                step=0.01,
                key="portfolio_avg_price_input",
            )
        with pf_cols[3]:
            st.write("")
            save_pf_clicked = st.button("Save", width="stretch")
        if save_pf_clicked:
            portfolio = load_portfolio()
            portfolio[portfolio_ticker] = {
                "shares": portfolio_shares,
                "avg_buy_price": portfolio_avg_price,
            }
            save_portfolio(portfolio)
            st.success(f"Saved {portfolio_ticker} to portfolio.")
            st.rerun()
    else:
        st.caption("Add stocks to your watchlist first.")

    st.divider()
    render_portfolio_section(watchlist)
    render_export_section(watchlist)

with tab_hot:
    render_hot_stocks()

with tab_news:
    render_market_news(watchlist)

with tab_forecast:
    render_forecast_tab(watchlist)

st.markdown(
    '<p style="color: gray; font-size: 0.85em;">'
    "Signals are based on simple moving average logic and are for educational purposes only. "
    "Not financial advice."
    "</p>",
    unsafe_allow_html=True,
)
