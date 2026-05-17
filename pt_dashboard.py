#!/usr/bin/env python3
"""
PowerTrader AI — Streamlit Analytics Dashboard

Reads hub_data/ IPC files and backtest_results/ in real time.
Run alongside pt_hub.py:

    streamlit run pt_dashboard.py

Optional env var to point at a non-default hub_data directory:
    POWERTRADER_HUB_DIR=/path/to/hub_data streamlit run pt_dashboard.py
"""

import os
import json
import time
import glob
from datetime import datetime
from typing import Optional

import streamlit as st

# ── page config (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="PowerTrader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── dark-theme CSS override ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background-color: #070B10; color: #C7D1DB; }
    [data-testid="stSidebar"]  { background-color: #0B1220; }
    [data-testid="stMetricValue"] { color: #00FF66; font-size: 1.6rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
    .signal-bar { border-radius: 4px; display: inline-block; height: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUB_DIR = os.environ.get("POWERTRADER_HUB_DIR", os.path.join(BASE_DIR, "hub_data"))

TRADER_STATUS_PATH = os.path.join(HUB_DIR, "trader_status.json")
TRADE_HISTORY_PATH = os.path.join(HUB_DIR, "trade_history.jsonl")
PNL_LEDGER_PATH    = os.path.join(HUB_DIR, "pnl_ledger.json")
PAPER_ACCOUNT_PATH = os.path.join(HUB_DIR, "paper_account.json")
ACCT_HISTORY_PATH  = os.path.join(HUB_DIR, "account_value_history.jsonl")
SIGNAL_LONG_PATH   = os.path.join(BASE_DIR, "long_dca_signal.txt")
SIGNAL_SHORT_PATH  = os.path.join(BASE_DIR, "short_dca_signal.txt")
BACKTEST_RESULTS_DIR = os.path.join(BASE_DIR, "backtest_results")
GUI_SETTINGS_PATH  = os.path.join(BASE_DIR, "gui_settings.json")

# ── helpers ──────────────────────────────────────────────────────────────

def _read_json(path: str) -> Optional[dict]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _read_int_file(path: str) -> int:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
    except Exception:
        pass
    return 0

def _read_jsonl_tail(path: str, n: int = 200) -> list:
    """Read last n lines of a JSONL file efficiently."""
    lines = []
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            lines = lines[-n:]
    except Exception:
        pass
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def _read_all_jsonl(path: str) -> list:
    return _read_jsonl_tail(path, n=10_000)

def _coins_from_settings() -> list:
    s = _read_json(GUI_SETTINGS_PATH) or {}
    coins = s.get("coins", ["BTC", "ETH", "XRP", "BNB", "DOGE"])
    return [str(c).strip().upper() for c in coins if str(c).strip()]

def _fmt_price(price) -> str:
    try:
        p = float(price)
    except Exception:
        return "N/A"
    if p == 0:
        return "$0"
    if abs(p) >= 1.0:
        return f"${p:,.2f}"
    return f"${p:.6f}"

def _signal_bar_html(level: int, color: str) -> str:
    """Render a 7-segment bar showing DCA signal intensity."""
    segments = []
    for i in range(1, 8):
        bg = color if i <= level else "#243044"
        segments.append(
            f'<span class="signal-bar" style="width:14px;background:{bg};margin:1px;opacity:0.9;"></span>'
        )
    return "".join(segments)

# ── sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png",
        width=48,
    )
    st.title("PowerTrader AI")
    st.caption(f"Hub dir: `{HUB_DIR}`")
    st.divider()

    refresh_sec = st.slider("Auto-refresh (seconds)", 1, 30, 3)
    show_paper = st.checkbox("Show Paper Account", value=os.path.isfile(PAPER_ACCOUNT_PATH))
    st.divider()

    # Backtest report picker
    st.subheader("Backtest Reports")
    report_dirs = sorted(glob.glob(os.path.join(BACKTEST_RESULTS_DIR, "*")), reverse=True)
    report_labels = [os.path.basename(d) for d in report_dirs]
    selected_report = st.selectbox(
        "Load report",
        ["— none —"] + report_labels,
        label_visibility="collapsed",
    ) if report_labels else "— none —"

    st.divider()
    st.caption(f"⏱ Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ── TABS ─────────────────────────────────────────────────────────────────
tab_live, tab_trades, tab_backtest = st.tabs(
    ["📊 Live Dashboard", "📋 Trade Log", "🔬 Backtest Report"]
)

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD
# ════════════════════════════════════════════════════════════════════════
with tab_live:
    status  = _read_json(TRADER_STATUS_PATH)
    pnl     = _read_json(PNL_LEDGER_PATH)
    paper   = _read_json(PAPER_ACCOUNT_PATH) if show_paper else None

    account = (status or {}).get("account", {})
    total_value    = account.get("total_account_value") or (paper and (paper["cash"] + sum(
        v["qty"] * 0 for v in paper.get("holdings", {}).values()  # placeholder; no live price here
    )))
    buying_power   = account.get("buying_power", paper["cash"] if paper else None)
    holdings_value = account.get("holdings_sell_value", 0.0)
    pct_in_trade   = account.get("percent_in_trade", 0.0)
    realized_pnl   = (pnl or {}).get("total_realized_profit_usd", 0.0)

    is_paper = os.path.isfile(PAPER_ACCOUNT_PATH) and show_paper

    # ── header badge ──
    mode_badge = "🟡 PAPER MODE" if is_paper else "🟢 LIVE"
    st.markdown(f"### {mode_badge} &nbsp; Account Summary")

    # ── KPI row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        val = f"${float(total_value):,.2f}" if total_value is not None else "N/A"
        st.metric("Total Account Value", val)
    with col2:
        bp = f"${float(buying_power):,.2f}" if buying_power is not None else "N/A"
        st.metric("Buying Power", bp)
    with col3:
        hv = f"${float(holdings_value):,.2f}" if holdings_value else "$0.00"
        st.metric("Holdings Value", hv)
    with col4:
        pnl_color = "normal" if realized_pnl >= 0 else "inverse"
        st.metric("Realized PnL", f"${realized_pnl:+,.2f}")

    st.progress(min(float(pct_in_trade) / 100.0, 1.0), text=f"Portfolio deployment: {pct_in_trade:.1f}%")

    st.divider()

    # ── DCA Signal Gauges ──
    st.subheader("DCA Signal Intensity")
    coins = _coins_from_settings()
    long_sig  = _read_int_file(SIGNAL_LONG_PATH)
    short_sig = _read_int_file(SIGNAL_SHORT_PATH)

    sig_cols = st.columns(min(len(coins), 5))
    for i, coin in enumerate(coins[:5]):
        with sig_cols[i]:
            # For multi-coin the signal files are per-coin; BTC uses root dir
            coin_long  = _read_int_file(
                os.path.join(BASE_DIR if coin == "BTC" else os.path.join(BASE_DIR, coin), "long_dca_signal.txt")
            ) if coin != "BTC" else long_sig
            coin_short = _read_int_file(
                os.path.join(BASE_DIR if coin == "BTC" else os.path.join(BASE_DIR, coin), "short_dca_signal.txt")
            ) if coin != "BTC" else short_sig

            st.markdown(f"**{coin}**")
            st.markdown(
                f"Long &nbsp; {_signal_bar_html(coin_long, '#3399FF')} &nbsp; **{coin_long}**",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"Short {_signal_bar_html(coin_short, '#FF9900')} &nbsp; **{coin_short}**",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Equity Curve ──
    st.subheader("Equity Curve")
    history = _read_all_jsonl(ACCT_HISTORY_PATH)
    if history:
        import pandas as pd
        df_eq = pd.DataFrame(history)
        df_eq["time"] = pd.to_datetime(df_eq["ts"], unit="s")
        df_eq = df_eq.set_index("time")[["total_account_value"]].rename(
            columns={"total_account_value": "Account Value ($)"}
        )
        st.line_chart(df_eq, height=280)
    else:
        st.info("No equity history yet — start trading or run a backtest.")

    st.divider()

    # ── Open Positions ──
    positions = (status or {}).get("positions", {})
    if positions:
        st.subheader("Open Positions")
        import pandas as pd
        rows = []
        for sym, pos in positions.items():
            rows.append({
                "Coin": sym,
                "Quantity": pos.get("quantity", 0),
                "Avg Cost": _fmt_price(pos.get("avg_cost_basis", 0)),
                "Current Price": _fmt_price(pos.get("current_price", 0)),
                "P/L %": f"{pos.get('gain_loss_pct_sell', 0):+.2f}%",
                "Value": _fmt_price(pos.get("value", 0)),
                "DCA Levels": pos.get("dca_levels_triggered", 0),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif status:
        st.info("No open positions.")

    # ── Paper Account Detail ──
    if paper and show_paper:
        st.divider()
        st.subheader("🟡 Paper Account")
        p_cash = paper.get("cash", 0.0)
        p_holdings = paper.get("holdings", {})
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Paper Cash", f"${p_cash:,.2f}")
        with c2:
            st.metric("Open Positions", len(p_holdings))
        if p_holdings:
            import pandas as pd
            rows = [
                {"Coin": coin, "Qty": d["qty"], "Avg Cost Basis": _fmt_price(d["cost_basis"])}
                for coin, d in p_holdings.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADE LOG
# ════════════════════════════════════════════════════════════════════════
with tab_trades:
    trades = _read_jsonl_tail(TRADE_HISTORY_PATH, n=500)
    if trades:
        import pandas as pd
        df = pd.DataFrame(trades)
        if "ts" in df.columns:
            df["time"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        keep = [c for c in ["time", "side", "symbol", "qty", "price", "avg_cost_basis", "pnl_pct", "realized_profit_usd", "tag", "order_id"] if c in df.columns]
        df = df[keep].sort_values("time", ascending=False) if "time" in df.columns else df[keep]

        # Color buy/sell
        def _color_side(val):
            if val == "buy":
                return "color: #3399FF"
            if val == "sell":
                return "color: #FF6666"
            return ""

        st.subheader(f"Trade History ({len(df)} trades)")
        st.dataframe(
            df.style.applymap(_color_side, subset=["side"]) if "side" in df.columns else df,
            use_container_width=True,
            hide_index=True,
        )

        # PnL summary
        if "realized_profit_usd" in df.columns:
            total = df["realized_profit_usd"].dropna().sum()
            wins  = (df["realized_profit_usd"].dropna() > 0).sum()
            total_closed = df["realized_profit_usd"].notna().sum()
            win_rate = (wins / total_closed * 100) if total_closed else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Realized PnL", f"${total:+,.2f}")
            m2.metric("Closed Trades", str(total_closed))
            m3.metric("Win Rate", f"{win_rate:.1f}%")
    else:
        st.info("No trades recorded yet.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST REPORT
# ════════════════════════════════════════════════════════════════════════
with tab_backtest:
    if selected_report and selected_report != "— none —":
        report_dir = os.path.join(BACKTEST_RESULTS_DIR, selected_report)
        report_html = os.path.join(report_dir, "analytics_report.html")
        events_jsonl = os.path.join(report_dir, "backtest_events.jsonl")

        if os.path.isfile(report_html):
            with open(report_html, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.subheader(f"Analytics Report — {selected_report}")
            st.components.v1.html(html_content, height=900, scrolling=True)
        else:
            st.warning(f"No analytics_report.html found in `{selected_report}`.")
            st.info("Run `python pt_analyze.py backtest_results/" + selected_report + "` to generate it.")

        # Event log summary
        if os.path.isfile(events_jsonl):
            with st.expander("Backtest Event Log (last 50 events)"):
                events = _read_jsonl_tail(events_jsonl, n=50)
                import pandas as pd
                if events:
                    df_ev = pd.DataFrame(events)
                    if "timestamp" in df_ev.columns:
                        df_ev["time"] = pd.to_datetime(df_ev["timestamp"], unit="s").dt.strftime("%H:%M:%S")
                    st.dataframe(df_ev, use_container_width=True, hide_index=True)
    else:
        st.info("Select a backtest run from the sidebar to view its analytics report.")
        st.markdown(
            "**Generate a report first:**\n"
            "```bash\n"
            "python pt_replay.py --warm-cache --start-date 2024-01-01 --end-date 2024-02-01 --coins BTC\n"
            "python pt_replay.py --backtest  --start-date 2024-01-01 --end-date 2024-02-01 --coins BTC --speed 10.0\n"
            "python pt_analyze.py backtest_results/<timestamped-dir>\n"
            "```"
        )

# ── auto-refresh ──────────────────────────────────────────────────────
time.sleep(refresh_sec)
st.rerun()
