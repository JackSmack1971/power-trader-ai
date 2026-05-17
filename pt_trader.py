import base64
import datetime
import json
import uuid
import time
import math
import sys
from typing import Any, Dict, Optional
import requests
from nacl.signing import SigningKey
import os
import colorama
from colorama import Fore, Style
import traceback
import argparse
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from kucoin.client import Market as _KucoinMarket

# KuCoin market client used in paper mode for live price data (no auth needed)
_kucoin_market = _KucoinMarket(url='https://api.kucoin.com')

# Optional alert dispatching — silently disabled if pt_alerts not present or
# env vars TELEGRAM_TOKEN / DISCORD_WEBHOOK_URL are unset.
try:
    import pt_alerts as _alerts
    _ALERTS_ENABLED = True
except ImportError:
    _ALERTS_ENABLED = False


def _fire_alert(alert_type: str, message: str, coin: str = "", force: bool = False) -> None:
    if _ALERTS_ENABLED:
        try:
            _alerts.send(alert_type, message, coin=coin, force=force)
        except Exception:
            pass

# -----------------------------
# REPLAY MODE GLOBALS (set by main block)
# -----------------------------
REPLAY_MODE = False
REPLAY_OUTPUT_DIR = None

# -----------------------------
# PAPER TRADING GLOBALS (set by main block)
# -----------------------------
PAPER_MODE = False
PAPER_BALANCE = 10_000.0   # default starting cash; overridden by --paper-balance

# Detect --paper early so the credential check below can be skipped at import time
_EARLY_PAPER_MODE = '--paper' in sys.argv

# -----------------------------
# GUI HUB OUTPUTS
# -----------------------------
HUB_DATA_DIR = os.environ.get("POWERTRADER_HUB_DIR", os.path.join(os.path.dirname(__file__), "hub_data"))
os.makedirs(HUB_DATA_DIR, exist_ok=True)

TRADER_STATUS_PATH = os.path.join(HUB_DATA_DIR, "trader_status.json")
TRADE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "trade_history.jsonl")
PNL_LEDGER_PATH = os.path.join(HUB_DATA_DIR, "pnl_ledger.json")
ACCOUNT_VALUE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "account_value_history.jsonl")
OPTIMIZER_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimizer_config.json")


# Initialize colorama
colorama.init(autoreset=True)

# -----------------------------
# GUI SETTINGS (coins list + main_neural_dir)
# -----------------------------
_GUI_SETTINGS_PATH = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"gui_settings.json"
)

_gui_settings_cache = {
	"mtime": None,
	"coins": ['BTC', 'ETH', 'XRP', 'BNB', 'DOGE'],  # fallback defaults
	"main_neural_dir": None,
}

def _load_gui_settings() -> dict:
	"""
	Returns a dict with coins and main_neural_dir.  Priority order:
	1. POWERTRADER_COINS env var (comma-separated, Docker-friendly)
	2. gui_settings.json (hot-reloaded by mtime)
	3. Hardcoded fallback defaults
	"""
	env_coins_raw = os.environ.get("POWERTRADER_COINS", "").strip()
	if env_coins_raw:
		env_coins = [c.strip().upper() for c in env_coins_raw.split(",") if c.strip()]
		if env_coins:
			result = dict(_gui_settings_cache)
			result["coins"] = env_coins
			return result

	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			return dict(_gui_settings_cache)

		mtime = os.path.getmtime(_GUI_SETTINGS_PATH)
		if _gui_settings_cache["mtime"] == mtime:
			return dict(_gui_settings_cache)

		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}

		coins = data.get("coins", None)
		if not isinstance(coins, list) or not coins:
			coins = list(_gui_settings_cache["coins"])
		coins = [str(c).strip().upper() for c in coins if str(c).strip()]
		if not coins:
			coins = list(_gui_settings_cache["coins"])

		main_neural_dir = data.get("main_neural_dir", None)
		if isinstance(main_neural_dir, str):
			main_neural_dir = main_neural_dir.strip() or None
		else:
			main_neural_dir = None

		_gui_settings_cache["mtime"] = mtime
		_gui_settings_cache["coins"] = coins
		_gui_settings_cache["main_neural_dir"] = main_neural_dir

		return {
			"mtime": mtime,
			"coins": list(coins),
			"main_neural_dir": main_neural_dir,
		}
	except Exception:
		return dict(_gui_settings_cache)

def _build_base_paths(main_dir_in: str, coins_in: list) -> dict:
	"""
	Safety rule:
	- BTC uses main_dir directly
	- other coins use <main_dir>/<SYM> ONLY if that folder exists
	  (no fallback to BTC folder — avoids corrupting BTC data)
	"""
	out = {"BTC": main_dir_in}
	try:
		for sym in coins_in:
			sym = str(sym).strip().upper()
			if not sym:
				continue
			if sym == "BTC":
				out["BTC"] = main_dir_in
				continue
			sub = os.path.join(main_dir_in, sym)
			if os.path.isdir(sub):
				out[sym] = sub
	except Exception:
		pass
	return out


# Live globals (will be refreshed inside manage_trades())
crypto_symbols = ['BTC', 'ETH', 'XRP', 'BNB', 'DOGE']

# Default main_dir behavior if settings are missing
main_dir = os.getcwd()
base_paths = {"BTC": main_dir}

_last_settings_mtime = None

def _refresh_paths_and_symbols():
	"""
	Hot-reload coins + main_neural_dir while trader is running.
	Updates globals: crypto_symbols, main_dir, base_paths
	"""
	global crypto_symbols, main_dir, base_paths, _last_settings_mtime

	s = _load_gui_settings()
	mtime = s.get("mtime", None)

	# If settings file doesn't exist, keep current defaults
	if mtime is None:
		return

	if _last_settings_mtime == mtime:
		return

	_last_settings_mtime = mtime

	coins = s.get("coins") or list(crypto_symbols)
	mndir = s.get("main_neural_dir") or main_dir

	# Keep it safe if folder isn't real on this machine
	if not os.path.isdir(mndir):
		mndir = os.getcwd()

	crypto_symbols = list(coins)
	main_dir = mndir
	base_paths = _build_base_paths(main_dir, crypto_symbols)


#API STUFF
API_KEY = ""
BASE64_PRIVATE_KEY = ""

try:
    with open('r_key.txt', 'r', encoding='utf-8') as f:
        API_KEY = (f.read() or "").strip()
    with open('r_secret.txt', 'r', encoding='utf-8') as f:
        BASE64_PRIVATE_KEY = (f.read() or "").strip()
except Exception:
    API_KEY = ""
    BASE64_PRIVATE_KEY = ""

if not API_KEY or not BASE64_PRIVATE_KEY:
    if not _EARLY_PAPER_MODE:
        print(
            "\n[PowerTrader] Robinhood API credentials not found.\n"
            "Open the GUI and go to Settings → Robinhood API → Setup / Update.\n"
            "That wizard will generate your keypair, tell you where to paste the public key on Robinhood,\n"
            "and will save r_key.txt + r_secret.txt so this trader can authenticate.\n"
            "To run without credentials, use: python pt_trader.py --paper\n"
        )
        raise SystemExit(1)

class CryptoAPITrading:
    def __init__(self):
        # keep a copy of the folder map (same idea as trader.py)
        self.path_map = dict(base_paths)

        if not PAPER_MODE:
            self.api_key = API_KEY
            private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
            self.private_key = SigningKey(private_key_seed)
            self.base_url = "https://trading.robinhood.com"
        else:
            self.api_key = ""
            self.private_key = None
            self.base_url = ""

        self.dca_levels_triggered = {}  # Track DCA levels for each crypto
        self.dca_levels = [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]  # Moved to instance variable

        # --- Trailing profit margin (per-coin state) ---
        # Each coin keeps its own trailing PM line, peak, and "was above line" flag.
        self.trailing_pm = {}  # { "BTC": {"active": bool, "line": float, "peak": float, "was_above": bool}, ... }
        self.trailing_gap_pct = 0.5  # 0.5% trail gap behind peak
        self.pm_start_pct_no_dca = 5.0
        self.pm_start_pct_with_dca = 2.5

        if not PAPER_MODE:
            self.cost_basis = self.calculate_cost_basis()
            self.initialize_dca_levels()
        else:
            self._paper: Dict[str, Any] = {}
            self._init_paper_account()
            self.cost_basis = {
                coin: data["cost_basis"]
                for coin, data in self._paper["holdings"].items()
            }

        # GUI hub persistence
        self._pnl_ledger = self._load_pnl_ledger()

        # Cache last known bid/ask per symbol so transient API misses don't zero out account value
        self._last_good_bid_ask = {}

        # Cache last *complete* account snapshot so transient holdings/price misses can't write a bogus low value
        self._last_good_account_snapshot = {
            "total_account_value": None,
            "buying_power": None,
            "holdings_sell_value": None,
            "holdings_buy_value": None,
            "percent_in_trade": None,
        }

        # --- DCA rate-limit (per trade, per coin, rolling 24h window) ---
        self.max_dca_buys_per_24h = 2
        self.dca_window_seconds = 24 * 60 * 60
        self._dca_buy_ts = {}         # { "BTC": [ts, ts, ...] } (DCA buys only)
        self._dca_last_sell_ts = {}   # { "BTC": ts_of_last_sell }
        self._seed_dca_window_from_history()

        # ── risk overlays (defaults; overridden by optimizer_config.json) ──
        self.atr_sizing_enabled    = False
        self.atr_sizing_min_signal = 6      # only scale size when signal >= this
        self.atr_period            = 14
        self.atr_target_vol_pct    = 2.0    # target 2 % daily vol per position
        self.kelly_sizing_enabled  = False
        self.kelly_min_trades      = 20     # need at least N closed trades
        self.stop_loss_enabled     = False
        self.stop_loss_atr_mult    = 2.5    # stop at entry − mult×ATR
        self.stop_loss_min_signal  = 6      # only attach stop on high-intensity entries
        self._stop_levels: Dict[str, float] = {}  # symbol → stop price

        # ── correlation-aware position limits ──
        self.correlation_limit_enabled   = False
        self.max_correlated_positions    = 3     # max simultaneous holdings with high pairwise correlation
        self.correlation_threshold       = 0.85  # Pearson r above which coins are "correlated"
        self.correlation_window          = 30    # candles of 1-hour data used for correlation calc

        self._apply_optimizer_config()






    def _atomic_write_json(self, path: str, data: dict) -> None:
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optimizer config & risk overlays
    # ------------------------------------------------------------------

    def _apply_optimizer_config(self) -> None:
        """Override instance trading params from optimizer_config.json if present."""
        try:
            if not os.path.isfile(OPTIMIZER_CONFIG_PATH):
                return
            with open(OPTIMIZER_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            scalar_overrides = {
                "buy_threshold":           int,
                "pm_start_pct_no_dca":     float,
                "pm_start_pct_with_dca":   float,
                "trailing_gap_pct":        float,
                "max_dca_buys_per_24h":    int,
                "atr_sizing_enabled":      bool,
                "atr_sizing_min_signal":   int,
                "atr_period":              int,
                "atr_target_vol_pct":      float,
                "kelly_sizing_enabled":    bool,
                "kelly_min_trades":        int,
                "stop_loss_enabled":           bool,
                "stop_loss_atr_mult":          float,
                "stop_loss_min_signal":        int,
                "correlation_limit_enabled":   bool,
                "max_correlated_positions":    int,
                "correlation_threshold":       float,
                "correlation_window":          int,
            }
            for key, cast in scalar_overrides.items():
                if key in cfg:
                    setattr(self, key, cast(cfg[key]))
            if "buy_threshold" in cfg:
                self._buy_threshold = int(cfg["buy_threshold"])
        except Exception:
            pass

    def _calc_atr(self, symbol: str) -> float:
        """Return ATR(atr_period) in USD using recent 1-hour KuCoin candles."""
        try:
            candles = _kucoin_market.get_kline(f"{symbol}-USDT", "1hour", limit=self.atr_period + 2)
            if not candles or len(candles) < self.atr_period + 1:
                return 0.0
            candles = list(reversed(candles))
            trs = []
            for i in range(1, len(candles)):
                high       = float(candles[i][3])
                low        = float(candles[i][4])
                prev_close = float(candles[i - 1][2])
                trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
            return sum(trs[-self.atr_period :]) / self.atr_period if trs else 0.0
        except Exception:
            return 0.0

    def _calc_kelly_fraction(self) -> float:
        """Compute half-Kelly multiplier from closed-trade PnL history.
        Returns 1.0 when there are insufficient trades."""
        try:
            pnl_list: list = []
            if os.path.isfile(TRADE_HISTORY_PATH):
                with open(TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            t = json.loads(line)
                            if t.get("side") == "sell" and t.get("pnl_pct") is not None:
                                pnl_list.append(float(t["pnl_pct"]))
                        except Exception:
                            pass
            if len(pnl_list) < self.kelly_min_trades:
                return 1.0
            wins   = [p for p in pnl_list if p > 0]
            losses = [abs(p) for p in pnl_list if p < 0]
            if not wins or not losses:
                return 1.0
            w_rate   = len(wins)   / len(pnl_list)
            l_rate   = 1.0 - w_rate
            avg_win  = sum(wins)   / len(wins)
            avg_loss = sum(losses) / len(losses)
            kelly    = (w_rate / avg_loss) - (l_rate / avg_win)
            return max(0.1, min(kelly / 2.0, 2.0))
        except Exception:
            return 1.0

    def _apply_risk_sizing(self, symbol: str, base_usd: float, signal_level: int) -> float:
        """Scale a base USD allocation by ATR volatility and/or Kelly fraction.
        Only applies overlays when signal_level >= the configured minimum."""
        result = base_usd
        try:
            if self.atr_sizing_enabled and signal_level >= self.atr_sizing_min_signal:
                atr = self._calc_atr(symbol)
                if atr > 0:
                    price = float((_kucoin_market.get_ticker(f"{symbol}-USDT") or {}).get("price", 0) or 0)
                    if price > 0:
                        current_vol_pct = (atr / price) * 100.0
                        scale = self.atr_target_vol_pct / max(current_vol_pct, 0.01)
                        scale = max(0.25, min(scale, 2.0))
                        result *= scale

            if self.kelly_sizing_enabled:
                result *= self._calc_kelly_fraction()
        except Exception:
            pass
        return max(result, 1.0)

    def _calc_correlation_matrix(self, symbols: list) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Pearson correlation of recent hourly close returns.
        Returns nested dict: matrix[coinA][coinB] = r.  Missing pairs default to 0.0."""
        matrix: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
        if len(symbols) < 2:
            return matrix
        closes: Dict[str, list] = {}
        for sym in symbols:
            try:
                candles = _kucoin_market.get_kline(f"{sym}-USDT", "1hour", limit=self.correlation_window + 2)
                if candles and len(candles) >= 4:
                    prices = [float(c[2]) for c in reversed(candles)]  # index 2 = close
                    closes[sym] = prices
            except Exception:
                pass
        for a in symbols:
            for b in symbols:
                if a == b:
                    matrix[a][b] = 1.0
                    continue
                if b in matrix[a]:
                    continue
                if a not in closes or b not in closes:
                    matrix[a][b] = matrix[b][a] = 0.0
                    continue
                pa, pb = closes[a], closes[b]
                n = min(len(pa), len(pb)) - 1
                if n < 3:
                    matrix[a][b] = matrix[b][a] = 0.0
                    continue
                ra = [pa[i+1]/pa[i] - 1 for i in range(n)]
                rb = [pb[i+1]/pb[i] - 1 for i in range(n)]
                mean_a = sum(ra) / n
                mean_b = sum(rb) / n
                cov = sum((ra[i]-mean_a)*(rb[i]-mean_b) for i in range(n))
                var_a = sum((x-mean_a)**2 for x in ra)
                var_b = sum((x-mean_b)**2 for x in rb)
                denom = (var_a * var_b) ** 0.5
                r = cov / denom if denom > 1e-12 else 0.0
                matrix[a][b] = matrix[b][a] = max(-1.0, min(1.0, r))
        return matrix

    def _is_correlated_entry_blocked(self, candidate: str, held_symbols: list) -> bool:
        """Return True when adding *candidate* would push the number of highly-correlated
        positions above max_correlated_positions.  Always returns False when the feature
        is disabled or when there are fewer current holdings than the limit."""
        if not self.correlation_limit_enabled:
            return False
        if len(held_symbols) < self.max_correlated_positions:
            return False
        try:
            all_syms = list({candidate} | set(held_symbols))
            matrix = self._calc_correlation_matrix(all_syms)
            correlated_count = 1  # candidate itself
            for sym in held_symbols:
                r = matrix.get(candidate, {}).get(sym, 0.0)
                if r >= self.correlation_threshold:
                    correlated_count += 1
            return correlated_count > self.max_correlated_positions
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Paper account state helpers
    # ------------------------------------------------------------------

    def _paper_account_path(self) -> str:
        return os.path.join(HUB_DATA_DIR, "paper_account.json")

    def _init_paper_account(self) -> None:
        """Load persisted paper account or create a fresh one."""
        path = self._paper_account_path()
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._paper = json.load(f)
                print(f"[PAPER] Loaded existing paper account: ${self._paper.get('cash', 0):.2f} cash")
                return
        except Exception:
            pass
        self._paper = {
            "cash": PAPER_BALANCE,
            "holdings": {},   # {"BTC": {"qty": float, "cost_basis": float}}
            "created_ts": time.time(),
        }
        self._save_paper_account()
        print(f"[PAPER] New paper account created: ${PAPER_BALANCE:.2f} starting balance")

    def _save_paper_account(self) -> None:
        self._atomic_write_json(self._paper_account_path(), self._paper)

    def _paper_price(self, symbol: str) -> float:
        """Fetch live KuCoin spot price for a symbol like 'BTC-USD'."""
        coin = symbol.replace("-USD", "")
        try:
            ticker = _kucoin_market.get_ticker(f"{coin}-USDT")
            return float(ticker["price"])
        except Exception:
            return 0.0

    def _append_jsonl(self, path: str, obj: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")
        except Exception:
            pass

    def _load_pnl_ledger(self) -> dict:
        try:
            if os.path.isfile(PNL_LEDGER_PATH):
                with open(PNL_LEDGER_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"total_realized_profit_usd": 0.0, "last_updated_ts": time.time()}

    def _save_pnl_ledger(self) -> None:
        try:
            self._pnl_ledger["last_updated_ts"] = time.time()
            self._atomic_write_json(PNL_LEDGER_PATH, self._pnl_ledger)
        except Exception:
            pass

    def _record_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        """
        Minimal local ledger for GUI:
        - append trade_history.jsonl
        - update pnl_ledger.json on sells (using estimated price * qty)
        - store the exact PnL% at the moment for DCA buys / sells (for GUI trade history)
        """
        ts = time.time()
        realized = None
        if side.lower() == "sell" and price is not None and avg_cost_basis is not None:
            try:
                realized = (float(price) - float(avg_cost_basis)) * float(qty)
                self._pnl_ledger["total_realized_profit_usd"] = float(self._pnl_ledger.get("total_realized_profit_usd", 0.0)) + float(realized)
            except Exception:
                realized = None

        entry = {
            "ts": ts,
            "side": side,
            "tag": tag,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "avg_cost_basis": avg_cost_basis,
            "pnl_pct": pnl_pct,
            "realized_profit_usd": realized,
            "order_id": order_id,
        }
        self._append_jsonl(TRADE_HISTORY_PATH, entry)
        if realized is not None:
            self._save_pnl_ledger()


    def _write_trader_status(self, status: dict) -> None:
        self._atomic_write_json(TRADER_STATUS_PATH, status)

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def _fmt_price(price: float) -> str:
        """
        Dynamic decimal formatting by magnitude:
        - >= 1.0   -> 2 decimals (BTC/ETH/etc won't show 8 decimals)
        - <  1.0   -> enough decimals to show meaningful digits (based on first non-zero),
                     then trim trailing zeros.
        """
        try:
            p = float(price)
        except Exception:
            return "N/A"

        if p == 0:
            return "0"

        ap = abs(p)

        if ap >= 1.0:
            decimals = 2
        else:
            # Example:
            # 0.5      -> decimals ~ 4 (prints "0.5" after trimming zeros)
            # 0.05     -> 5
            # 0.005    -> 6
            # 0.000012 -> 8
            decimals = int(-math.floor(math.log10(ap))) + 3
            decimals = max(2, min(12, decimals))

        s = f"{p:.{decimals}f}"

        # Trim useless trailing zeros for cleaner output (0.5000 -> 0.5)
        if "." in s:
            s = s.rstrip("0").rstrip(".")

        return s


    @staticmethod
    def _read_long_dca_signal(symbol: str) -> int:
        """Read long DCA signal via IPC bridge (Redis or file fallback).

        Used for:
        - Start gate: start trades at level 3+
        - DCA assist: levels 4-7 map to trader DCA stages 0-3
        """
        from pt_ipc import ipc  # late import — bridge initialises on first use
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, main_dir if sym == "BTC" else os.path.join(main_dir, sym))
        return ipc.read_signal(sym, "long", base_dir=folder)

    @staticmethod
    def _read_short_dca_signal(symbol: str) -> int:
        """Read short DCA signal via IPC bridge (Redis or file fallback).

        Used for:
        - Start gate: start trades at level 3+
        - DCA assist: levels 4-7 map to trader DCA stages 0-3
        """
        from pt_ipc import ipc
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, main_dir if sym == "BTC" else os.path.join(main_dir, sym))
        return ipc.read_signal(sym, "short", base_dir=folder)





    def initialize_dca_levels(self):

        """
        Initializes the DCA levels_triggered dictionary based on the number of buy orders
        that have occurred after the first buy order following the most recent sell order
        for each cryptocurrency.
        """
        holdings = self.get_holdings()
        if not holdings or "results" not in holdings:
            print("No holdings found. Skipping DCA levels initialization.")
            return

        for holding in holdings.get("results", []):
            symbol = holding["asset_code"]

            full_symbol = f"{symbol}-USD"
            orders = self.get_orders(full_symbol)
            
            if not orders or "results" not in orders:
                print(f"No orders found for {full_symbol}. Skipping.")
                continue

            # Filter for filled buy and sell orders
            filled_orders = [
                order for order in orders["results"]
                if order["state"] == "filled" and order["side"] in ["buy", "sell"]
            ]
            
            if not filled_orders:
                print(f"No filled buy or sell orders for {full_symbol}. Skipping.")
                continue

            # Sort orders by creation time in ascending order (oldest first)
            filled_orders.sort(key=lambda x: x["created_at"])

            # Find the timestamp of the most recent sell order
            most_recent_sell_time = None
            for order in reversed(filled_orders):
                if order["side"] == "sell":
                    most_recent_sell_time = order["created_at"]
                    break

            # Determine the cutoff time for buy orders
            if most_recent_sell_time:
                # Find all buy orders after the most recent sell
                relevant_buy_orders = [
                    order for order in filled_orders
                    if order["side"] == "buy" and order["created_at"] > most_recent_sell_time
                ]
                if not relevant_buy_orders:
                    print(f"No buy orders after the most recent sell for {full_symbol}.")
                    self.dca_levels_triggered[symbol] = []
                    continue
                print(f"Most recent sell for {full_symbol} at {most_recent_sell_time}.")
            else:
                # If no sell orders, consider all buy orders
                relevant_buy_orders = [
                    order for order in filled_orders
                    if order["side"] == "buy"
                ]
                if not relevant_buy_orders:
                    print(f"No buy orders for {full_symbol}. Skipping.")
                    self.dca_levels_triggered[symbol] = []
                    continue
                print(f"No sell orders found for {full_symbol}. Considering all buy orders.")

            # Ensure buy orders are sorted by creation time ascending
            relevant_buy_orders.sort(key=lambda x: x["created_at"])

            # Identify the first buy order in the relevant list
            first_buy_order = relevant_buy_orders[0]
            first_buy_time = first_buy_order["created_at"]

            # Count the number of buy orders after the first buy
            buy_orders_after_first = [
                order for order in relevant_buy_orders
                if order["created_at"] > first_buy_time
            ]

            triggered_levels_count = len(buy_orders_after_first)

            # Track DCA by stage index (0, 1, 2, ...) rather than % values.
            # This makes neural-vs-hardcoded clean, and allows repeating the -50% stage indefinitely.
            self.dca_levels_triggered[symbol] = list(range(triggered_levels_count))
            print(f"Initialized DCA stages for {symbol}: {triggered_levels_count}")


    def _seed_dca_window_from_history(self) -> None:
        """
        Seeds in-memory DCA buy timestamps from TRADE_HISTORY_PATH so the 24h limit
        works across restarts.

        Uses the local GUI trade history (tag == "DCA") and resets per trade at the most recent sell.
        """
        now_ts = time.time()
        cutoff = now_ts - float(getattr(self, "dca_window_seconds", 86400))

        self._dca_buy_ts = {}
        self._dca_last_sell_ts = {}

        if not os.path.isfile(TRADE_HISTORY_PATH):
            return

        try:
            with open(TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    ts = obj.get("ts", None)
                    side = str(obj.get("side", "")).lower()
                    tag = obj.get("tag", None)
                    sym_full = str(obj.get("symbol", "")).upper().strip()
                    base = sym_full.split("-")[0].strip() if sym_full else ""
                    if not base:
                        continue

                    try:
                        ts_f = float(ts)
                    except Exception:
                        continue

                    if side == "sell":
                        prev = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
                        if ts_f > prev:
                            self._dca_last_sell_ts[base] = ts_f

                    elif side == "buy" and tag == "DCA":
                        self._dca_buy_ts.setdefault(base, []).append(ts_f)

        except Exception:
            return

        # Keep only DCA buys after the last sell (current trade) and within rolling 24h
        for base, ts_list in list(self._dca_buy_ts.items()):
            last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
            kept = [t for t in ts_list if (t > last_sell) and (t >= cutoff)]
            kept.sort()
            self._dca_buy_ts[base] = kept


    def _dca_window_count(self, base_symbol: str, now_ts: Optional[float] = None) -> int:
        """
        Count of DCA buys for this coin within rolling 24h in the *current trade*.
        Current trade boundary = most recent sell we observed for this coin.
        """
        base = str(base_symbol).upper().strip()
        if not base:
            return 0

        now = float(now_ts if now_ts is not None else time.time())
        cutoff = now - float(getattr(self, "dca_window_seconds", 86400))
        last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)

        ts_list = list(self._dca_buy_ts.get(base, []) or [])
        ts_list = [t for t in ts_list if (t > last_sell) and (t >= cutoff)]
        self._dca_buy_ts[base] = ts_list
        return len(ts_list)


    def _note_dca_buy(self, base_symbol: str, ts: Optional[float] = None) -> None:
        base = str(base_symbol).upper().strip()
        if not base:
            return
        t = float(ts if ts is not None else time.time())
        self._dca_buy_ts.setdefault(base, []).append(t)
        self._dca_window_count(base, now_ts=t)  # prune in-place


    def _reset_dca_window_for_trade(self, base_symbol: str, sold: bool = False, ts: Optional[float] = None) -> None:
        base = str(base_symbol).upper().strip()
        if not base:
            return
        if sold:
            self._dca_last_sell_ts[base] = float(ts if ts is not None else time.time())
        self._dca_buy_ts[base] = []


    def make_api_request(self, method: str, path: str, body: Optional[str] = "") -> Any:

        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)

            response.raise_for_status()
            return response.json()
        except requests.HTTPError as http_err:
            try:
                # Parse and return the JSON error response
                error_response = response.json()
                return error_response  # Return the JSON error for further handling
            except Exception:
                return None
        except Exception:
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def get_account(self) -> Any:
        if PAPER_MODE:
            return {"buying_power": str(self._paper["cash"])}
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    def get_holdings(self) -> Any:
        if PAPER_MODE:
            results = []
            for coin, data in self._paper["holdings"].items():
                if float(data.get("qty", 0)) > 0:
                    results.append({
                        "asset_code": coin,
                        "total_quantity": str(data["qty"]),
                    })
            return {"results": results}
        path = "/api/v1/crypto/trading/holdings/"
        return self.make_api_request("GET", path)

    def get_trading_pairs(self) -> Any:
        if PAPER_MODE:
            return [
                {"symbol": f"{coin}-USD", "asset_code": coin, "quote_currency": "USD"}
                for coin in crypto_symbols
            ]
        path = "/api/v1/crypto/trading/trading_pairs/"
        response = self.make_api_request("GET", path)

        if not response or "results" not in response:
            return []

        trading_pairs = response.get("results", [])
        if not trading_pairs:
            return []

        return trading_pairs

    def get_orders(self, symbol: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/?symbol={symbol}"
        return self.make_api_request("GET", path)

    def calculate_cost_basis(self):
        holdings = self.get_holdings()
        if not holdings or "results" not in holdings:
            return {}

        active_assets = {holding["asset_code"] for holding in holdings.get("results", [])}
        current_quantities = {
            holding["asset_code"]: float(holding["total_quantity"])
            for holding in holdings.get("results", [])
        }

        cost_basis = {}

        for asset_code in active_assets:
            orders = self.get_orders(f"{asset_code}-USD")
            if not orders or "results" not in orders:
                continue

            # Get all filled buy orders, sorted from most recent to oldest
            buy_orders = [
                order for order in orders["results"]
                if order["side"] == "buy" and order["state"] == "filled"
            ]
            buy_orders.sort(key=lambda x: x["created_at"], reverse=True)

            remaining_quantity = current_quantities[asset_code]
            total_cost = 0.0

            for order in buy_orders:
                for execution in order.get("executions", []):
                    quantity = float(execution["quantity"])
                    price = float(execution["effective_price"])

                    if remaining_quantity <= 0:
                        break

                    # Use only the portion of the quantity needed to match the current holdings
                    if quantity > remaining_quantity:
                        total_cost += remaining_quantity * price
                        remaining_quantity = 0
                    else:
                        total_cost += quantity * price
                        remaining_quantity -= quantity

                if remaining_quantity <= 0:
                    break

            if current_quantities[asset_code] > 0:
                cost_basis[asset_code] = total_cost / current_quantities[asset_code]
            else:
                cost_basis[asset_code] = 0.0

        return cost_basis

    def get_price(self, symbols: list) -> Dict[str, float]:
        buy_prices = {}
        sell_prices = {}
        valid_symbols = []

        # Paper mode: fetch live prices from KuCoin (no Robinhood auth needed)
        if PAPER_MODE:
            for symbol in symbols:
                if symbol == "USDC-USD":
                    continue
                price = self._paper_price(symbol)
                if price > 0:
                    buy_prices[symbol] = price
                    sell_prices[symbol] = price
                    valid_symbols.append(symbol)
                    try:
                        self._last_good_bid_ask[symbol] = {"ask": price, "bid": price, "ts": time.time()}
                    except Exception:
                        pass
            return buy_prices, sell_prices, valid_symbols

        # In replay mode, read prices from backtest_state.json
        if REPLAY_MODE:
            state_file = "replay_data/backtest_state.json"
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        state = json.load(f)
                    prices = state.get("prices", {})

                    for symbol in symbols:
                        if symbol == "USDC-USD":
                            continue

                        # Extract coin from symbol (e.g., "BTC-USD" -> "BTC")
                        coin = symbol.replace("-USD", "")

                        if coin in prices:
                            price_data = prices[coin]
                            price = float(price_data.get("close", 0.0))

                            if price > 0.0:
                                # Use close price for both buy and sell (simplified)
                                buy_prices[symbol] = price
                                sell_prices[symbol] = price
                                valid_symbols.append(symbol)
                except Exception as e:
                    print(f"ERROR: Failed to read replay state: {e}")

            return buy_prices, sell_prices, valid_symbols

        # Live mode: use Robinhood API
        for symbol in symbols:
            if symbol == "USDC-USD":
                continue

            path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
            response = self.make_api_request("GET", path)

            if response and "results" in response:
                result = response["results"][0]
                ask = float(result["ask_inclusive_of_buy_spread"])
                bid = float(result["bid_inclusive_of_sell_spread"])

                buy_prices[symbol] = ask
                sell_prices[symbol] = bid
                valid_symbols.append(symbol)

                # Update cache for transient failures later
                try:
                    self._last_good_bid_ask[symbol] = {"ask": ask, "bid": bid, "ts": time.time()}
                except Exception:
                    pass
            else:
                # Fallback to cached bid/ask so account value never drops due to a transient miss
                cached = None
                try:
                    cached = self._last_good_bid_ask.get(symbol)
                except Exception:
                    cached = None

                if cached:
                    ask = float(cached.get("ask", 0.0) or 0.0)
                    bid = float(cached.get("bid", 0.0) or 0.0)
                    if ask > 0.0 and bid > 0.0:
                        buy_prices[symbol] = ask
                        sell_prices[symbol] = bid
                        valid_symbols.append(symbol)

        return buy_prices, sell_prices, valid_symbols


    def place_buy_order(
        self,
        client_order_id: str,
        side: str,
        order_type: str,
        symbol: str,
        amount_in_usd: float,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> Any:
        # Fetch the current price of the asset
        current_buy_prices, current_sell_prices, valid_symbols = self.get_price([symbol])
        current_price = current_buy_prices[symbol]
        asset_quantity = amount_in_usd / current_price

        # Paper mode: simulate fill against live KuCoin price, update paper account
        if PAPER_MODE:
            rounded_quantity = round(asset_quantity, 8)
            coin = symbol.replace("-USD", "")

            if self._paper["cash"] < amount_in_usd:
                print(f"[PAPER] Insufficient cash (${self._paper['cash']:.2f}) for ${amount_in_usd:.2f} buy")
                return None

            # Weighted-average cost basis
            if coin in self._paper["holdings"] and self._paper["holdings"][coin]["qty"] > 0:
                old_qty = self._paper["holdings"][coin]["qty"]
                old_cb = self._paper["holdings"][coin]["cost_basis"]
                new_cb = (old_qty * old_cb + rounded_quantity * current_price) / (old_qty + rounded_quantity)
                self._paper["holdings"][coin]["cost_basis"] = new_cb
                self._paper["holdings"][coin]["qty"] = round(old_qty + rounded_quantity, 8)
            else:
                self._paper["holdings"][coin] = {"qty": rounded_quantity, "cost_basis": current_price}

            self._paper["cash"] = round(self._paper["cash"] - amount_in_usd, 8)
            self._save_paper_account()
            self.cost_basis[coin] = self._paper["holdings"][coin]["cost_basis"]

            self._record_trade(
                side="buy", symbol=symbol, qty=float(rounded_quantity),
                price=float(current_price),
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=tag, order_id=client_order_id,
            )
            print(f"[PAPER] BUY  {rounded_quantity:.8f} {coin} @ ${current_price:.2f}  cash left: ${self._paper['cash']:.2f}")
            _fire_alert("TRADE", f"[PAPER] BUY {rounded_quantity:.6f} {coin} @ ${current_price:,.4f}  tag={tag or 'ENTRY'}", coin=coin, force=True)
            return {
                "id": client_order_id, "state": "filled", "side": side,
                "symbol": symbol, "executed_notional": amount_in_usd,
                "cumulative_quantity": rounded_quantity,
            }

        # In replay mode, write order to sim_orders.jsonl and return mock fill
        if REPLAY_MODE:
            rounded_quantity = round(asset_quantity, 8)

            order = {
                "ts": time.time(),
                "client_order_id": client_order_id,
                "side": side,
                "symbol": symbol,
                "qty": rounded_quantity,
                "order_type": order_type
            }

            # Write order to sim_orders.jsonl
            os.makedirs("replay_data", exist_ok=True)
            with open("replay_data/sim_orders.jsonl", "a") as f:
                f.write(json.dumps(order) + "\n")

            # Record trade for GUI history
            self._record_trade(
                side="buy",
                symbol=symbol,
                qty=float(rounded_quantity),
                price=float(current_price),
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=tag,
                order_id=client_order_id,
            )

            # Return instant mock fill
            return {
                "id": client_order_id,
                "state": "filled",
                "side": side,
                "symbol": symbol,
                "executed_notional": amount_in_usd,
                "cumulative_quantity": rounded_quantity
            }

        # Live mode: use Robinhood API
        max_retries = 5
        retries = 0

        while retries < max_retries:
            retries += 1
            try:
                # Default precision to 8 decimals initially
                rounded_quantity = round(asset_quantity, 8)

                body = {
                    "client_order_id": client_order_id,
                    "side": side,
                    "type": order_type,
                    "symbol": symbol,
                    "market_order_config": {
                        "asset_quantity": f"{rounded_quantity:.8f}"  # Start with 8 decimal places
                    }
                }

                path = "/api/v1/crypto/trading/orders/"
                response = self.make_api_request("POST", path, json.dumps(body))
                if response and "errors" not in response:
                    # Record for GUI history (estimated fill at current_price)
                    try:
                        order_id = response.get("id", None) if isinstance(response, dict) else None
                    except Exception:
                        order_id = None
                    self._record_trade(
                        side="buy",
                        symbol=symbol,
                        qty=float(rounded_quantity),
                        price=float(current_price),
                        avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                        pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                        tag=tag,
                        order_id=order_id,
                    )
                    coin = symbol.replace("-USD", "")
                    _fire_alert("TRADE", f"BUY {rounded_quantity:.6f} {coin} @ ${current_price:,.4f}  tag={tag or 'ENTRY'}", coin=coin, force=True)
                    return response  # Successfully placed order

            except Exception as e:
                pass #print(traceback.format_exc())
                

            # Check for precision errors
            if response and "errors" in response:
                for error in response["errors"]:
                    if "has too much precision" in error.get("detail", ""):
                        # Extract required precision directly from the error message
                        detail = error["detail"]
                        nearest_value = detail.split("nearest ")[1].split(" ")[0]

                        decimal_places = len(nearest_value.split(".")[1].rstrip("0"))
                        asset_quantity = round(asset_quantity, decimal_places)
                        break
                    elif "must be greater than or equal to" in error.get("detail", ""):
                        return None

        return None


    def place_sell_order(
        self,
        client_order_id: str,
        side: str,
        order_type: str,
        symbol: str,
        asset_quantity: float,
        expected_price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> Any:
        # Paper mode: simulate fill against live KuCoin price, update paper account
        if PAPER_MODE:
            rounded_quantity = round(asset_quantity, 8)
            coin = symbol.replace("-USD", "")
            current_price = self._paper_price(symbol)
            if current_price <= 0:
                print(f"[PAPER] Could not get price for {symbol}, skipping sell")
                return None

            proceeds = rounded_quantity * current_price
            self._paper["cash"] = round(self._paper["cash"] + proceeds, 8)

            if coin in self._paper["holdings"]:
                new_qty = round(self._paper["holdings"][coin]["qty"] - rounded_quantity, 8)
                if new_qty <= 0:
                    del self._paper["holdings"][coin]
                    self.cost_basis.pop(coin, None)
                else:
                    self._paper["holdings"][coin]["qty"] = new_qty

            self._save_paper_account()
            self._record_trade(
                side="sell", symbol=symbol, qty=float(rounded_quantity),
                price=float(current_price),
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=tag, order_id=client_order_id,
            )
            print(f"[PAPER] SELL {rounded_quantity:.8f} {coin} @ ${current_price:.2f}  cash: ${self._paper['cash']:.2f}")
            pnl_str = f"  pnl={pnl_pct:+.2f}%" if pnl_pct is not None else ""
            _fire_alert("TRADE", f"[PAPER] SELL {rounded_quantity:.6f} {coin} @ ${current_price:,.4f}{pnl_str}  tag={tag or 'SELL'}", coin=coin, force=True)
            return {
                "id": client_order_id, "state": "filled", "side": side,
                "symbol": symbol, "cumulative_quantity": rounded_quantity,
            }

        # In replay mode, write order to sim_orders.jsonl and return mock fill
        if REPLAY_MODE:
            rounded_quantity = round(asset_quantity, 8)

            order = {
                "ts": time.time(),
                "client_order_id": client_order_id,
                "side": side,
                "symbol": symbol,
                "qty": rounded_quantity,
                "order_type": order_type
            }

            # Write order to sim_orders.jsonl
            os.makedirs("replay_data", exist_ok=True)
            with open("replay_data/sim_orders.jsonl", "a") as f:
                f.write(json.dumps(order) + "\n")

            # Record trade for GUI history
            self._record_trade(
                side="sell",
                symbol=symbol,
                qty=float(rounded_quantity),
                price=float(expected_price) if expected_price is not None else None,
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=tag,
                order_id=client_order_id,
            )

            # Return instant mock fill
            return {
                "id": client_order_id,
                "state": "filled",
                "side": side,
                "symbol": symbol,
                "cumulative_quantity": rounded_quantity
            }

        # Live mode: use Robinhood API
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            "market_order_config": {
                "asset_quantity": f"{asset_quantity:.8f}"
            }
        }

        path = "/api/v1/crypto/trading/orders/"

        response = self.make_api_request("POST", path, json.dumps(body))

        if response and isinstance(response, dict) and "errors" not in response:
            order_id = response.get("id", None)
            self._record_trade(
                side="sell",
                symbol=symbol,
                qty=float(asset_quantity),
                price=float(expected_price) if expected_price is not None else None,
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=tag,
                order_id=order_id,
            )
            coin = symbol.replace("-USD", "")
            pnl_str = f"  pnl={pnl_pct:+.2f}%" if pnl_pct is not None else ""
            price_str = f"${float(expected_price):,.4f}" if expected_price is not None else "market"
            _fire_alert("TRADE", f"SELL {asset_quantity:.6f} {coin} @ {price_str}{pnl_str}  tag={tag or 'SELL'}", coin=coin, force=True)

        return response



    def manage_trades(self):
        trades_made = False  # Flag to track if any trade was made in this iteration

        # Hot-reload coins list + paths from GUI settings while running
        try:
            _refresh_paths_and_symbols()
            self.path_map = dict(base_paths)
        except Exception:
            pass

        # Fetch account details
        account = self.get_account()
        # Fetch holdings
        holdings = self.get_holdings()
        # Fetch trading pairs
        trading_pairs = self.get_trading_pairs()

        # Use the stored cost_basis instead of recalculating
        cost_basis = self.cost_basis
        # Fetch current prices
        symbols = [holding["asset_code"] + "-USD" for holding in holdings.get("results", [])]

        # ALSO fetch prices for tracked coins even if not currently held (so GUI can show bid/ask lines)
        for s in crypto_symbols:
            full = f"{s}-USD"
            if full not in symbols:
                symbols.append(full)

        current_buy_prices, current_sell_prices, valid_symbols = self.get_price(symbols)

        # Calculate total account value (robust: never drop a held coin to $0 on transient API misses)
        snapshot_ok = True

        # buying power
        try:
            buying_power = float(account.get("buying_power", 0))
        except Exception:
            buying_power = 0.0
            snapshot_ok = False

        # holdings list (treat missing/invalid holdings payload as transient error)
        try:
            holdings_list = holdings.get("results", None) if isinstance(holdings, dict) else None
            if not isinstance(holdings_list, list):
                holdings_list = []
                snapshot_ok = False
        except Exception:
            holdings_list = []
            snapshot_ok = False

        holdings_buy_value = 0.0
        holdings_sell_value = 0.0

        for holding in holdings_list:
            try:
                asset = holding.get("asset_code")
                if asset == "USDC":
                    continue

                qty = float(holding.get("total_quantity", 0.0))
                if qty <= 0.0:
                    continue

                sym = f"{asset}-USD"
                bp = float(current_buy_prices.get(sym, 0.0) or 0.0)
                sp = float(current_sell_prices.get(sym, 0.0) or 0.0)

                # If any held asset is missing a usable price this tick, do NOT allow a new "low" snapshot
                if bp <= 0.0 or sp <= 0.0:
                    snapshot_ok = False
                    continue

                holdings_buy_value += qty * bp
                holdings_sell_value += qty * sp
            except Exception:
                snapshot_ok = False
                continue

        total_account_value = buying_power + holdings_sell_value
        in_use = (holdings_sell_value / total_account_value) * 100 if total_account_value > 0 else 0.0

        # If this tick is incomplete, fall back to last known-good snapshot so the GUI chart never gets a bogus dip.
        if (not snapshot_ok) or (total_account_value <= 0.0):
            last = getattr(self, "_last_good_account_snapshot", None) or {}
            if last.get("total_account_value") is not None:
                total_account_value = float(last["total_account_value"])
                buying_power = float(last.get("buying_power", buying_power or 0.0))
                holdings_sell_value = float(last.get("holdings_sell_value", holdings_sell_value or 0.0))
                holdings_buy_value = float(last.get("holdings_buy_value", holdings_buy_value or 0.0))
                in_use = float(last.get("percent_in_trade", in_use or 0.0))
        else:
            # Save last complete snapshot
            self._last_good_account_snapshot = {
                "total_account_value": float(total_account_value),
                "buying_power": float(buying_power),
                "holdings_sell_value": float(holdings_sell_value),
                "holdings_buy_value": float(holdings_buy_value),
                "percent_in_trade": float(in_use),
            }

        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n--- Account Summary ---")
        print(f"Total Account Value: ${total_account_value:.2f}")
        print(f"Holdings Value: ${holdings_sell_value:.2f}")
        print(f"Percent In Trade: {in_use:.2f}%")
        print(
            f"Trailing PM: start +{self.pm_start_pct_no_dca:.2f}% (no DCA) / +{self.pm_start_pct_with_dca:.2f}% (with DCA) "
            f"| gap {self.trailing_gap_pct:.2f}%"
        )
        print("\n--- Current Trades ---")

        positions = {}
        for holding in holdings.get("results", []):
            symbol = holding["asset_code"]
            full_symbol = f"{symbol}-USD"

            if full_symbol not in valid_symbols or symbol == "USDC":
                continue

            quantity = float(holding["total_quantity"])
            current_buy_price = current_buy_prices.get(full_symbol, 0)
            current_sell_price = current_sell_prices.get(full_symbol, 0)
            avg_cost_basis = cost_basis.get(symbol, 0)

            if avg_cost_basis > 0:
                gain_loss_percentage_buy = ((current_buy_price - avg_cost_basis) / avg_cost_basis) * 100
                gain_loss_percentage_sell = ((current_sell_price - avg_cost_basis) / avg_cost_basis) * 100
            else:
                gain_loss_percentage_buy = 0
                gain_loss_percentage_sell = 0
                print(f"  Warning: Average Cost Basis is 0 for {symbol}, Gain/Loss calculation skipped.")

            value = quantity * current_sell_price
            triggered_levels_count = len(self.dca_levels_triggered.get(symbol, []))
            triggered_levels = triggered_levels_count  # Number of DCA levels triggered

            # Determine the next DCA trigger for this coin (hardcoded % and optional neural level)
            next_stage = triggered_levels_count  # stage 0 == first DCA after entry (trade starts at neural level 3)

            # Hardcoded % for this stage (repeat -50% after we reach it)
            hard_next = self.dca_levels[next_stage] if next_stage < len(self.dca_levels) else self.dca_levels[-1]

            # Neural DCA only applies to first 4 DCA stages:
            # stage 0-> neural 4, stage 1->5, stage 2->6, stage 3->7
            if next_stage < 4:
                neural_next = next_stage + 4
                next_dca_display = f"{hard_next:.2f}% / N{neural_next}"
            else:
                next_dca_display = f"{hard_next:.2f}%"

            # --- DCA DISPLAY LINE (pick whichever trigger line is higher: NEURAL vs HARD) ---
            # Hardcoded gives an actual price line: cost_basis * (1 + hard_next%).
            # Neural is level-based; for display we treat it as "higher" only once its condition is already met.
            dca_line_source = "HARD"
            dca_line_price = 0.0
            dca_line_pct = 0.0

            if avg_cost_basis > 0:
                # Hardcoded trigger line price
                hard_line_price = avg_cost_basis * (1.0 + (hard_next / 100.0))
                dca_line_price = hard_line_price

                # If neural is already satisfied for this stage, then neural is effectively the "higher/earlier" trigger.
                # For display purposes, treat that as an immediate line at current price (i.e., DCA is ready NOW).
                if next_stage < 4:
                    neural_level_needed_disp = next_stage + 4
                    neural_level_now_disp = self._read_long_dca_signal(symbol)

                    neural_ready_now = (gain_loss_percentage_buy < 0) and (neural_level_now_disp >= neural_level_needed_disp)
                    if neural_ready_now:
                        neural_line_price = current_buy_price
                        if neural_line_price > dca_line_price:
                            dca_line_price = neural_line_price
                            dca_line_source = f"NEURAL N{neural_level_needed_disp}"

                # PnL% shown alongside DCA is the normal buy-side PnL%
                # (same calculation as GUI "Buy Price PnL": current buy/ask vs avg cost basis)
                dca_line_pct = gain_loss_percentage_buy



            dca_line_price_disp = self._fmt_price(dca_line_price) if avg_cost_basis > 0 else "N/A"

            # Set color code:
            # - DCA is green if we're above the chosen DCA line, red if we're below it
            # - SELL stays based on profit vs cost basis (your original behavior)
            if dca_line_pct >= 0:
                color = Fore.GREEN
            else:
                color = Fore.RED

            if gain_loss_percentage_sell >= 0:
                color2 = Fore.GREEN
            else:
                color2 = Fore.RED

            # --- Trailing PM display (per-coin, isolated) ---
            # Display uses current state if present; otherwise shows the base PM start line.
            trail_status = "N/A"
            pm_start_pct_disp = 0.0
            base_pm_line_disp = 0.0
            trail_line_disp = 0.0
            trail_peak_disp = 0.0
            above_disp = False
            dist_to_trail_pct = 0.0

            if avg_cost_basis > 0:
                pm_start_pct_disp = self.pm_start_pct_no_dca if int(triggered_levels) == 0 else self.pm_start_pct_with_dca
                base_pm_line_disp = avg_cost_basis * (1.0 + (pm_start_pct_disp / 100.0))

                state = self.trailing_pm.get(symbol)
                if state is None:
                    trail_line_disp = base_pm_line_disp
                    trail_peak_disp = 0.0
                    active_disp = False
                else:
                    trail_line_disp = float(state.get("line", base_pm_line_disp))
                    trail_peak_disp = float(state.get("peak", 0.0))
                    active_disp = bool(state.get("active", False))

                above_disp = current_sell_price >= trail_line_disp
                # If we're already above the line, trailing is effectively "on/armed" (even if active flips this tick)
                trail_status = "ON" if (active_disp or above_disp) else "OFF"

                if trail_line_disp > 0:
                    dist_to_trail_pct = ((current_sell_price - trail_line_disp) / trail_line_disp) * 100.0
            file = open(symbol+'_current_price.txt', 'w+')
            file.write(str(current_buy_price))
            file.close()
            positions[symbol] = {
                "quantity": quantity,
                "avg_cost_basis": avg_cost_basis,
                "current_buy_price": current_buy_price,
                "current_sell_price": current_sell_price,
                "gain_loss_pct_buy": gain_loss_percentage_buy,
                "gain_loss_pct_sell": gain_loss_percentage_sell,
                "value_usd": value,
                "dca_triggered_stages": int(triggered_levels_count),
                "next_dca_display": next_dca_display,
                "dca_line_price": float(dca_line_price) if dca_line_price else 0.0,
                "dca_line_source": dca_line_source,
                "dca_line_pct": float(dca_line_pct) if dca_line_pct else 0.0,
                "trail_active": True if (trail_status == "ON") else False,
                "trail_line": float(trail_line_disp) if trail_line_disp else 0.0,
                "trail_peak": float(trail_peak_disp) if trail_peak_disp else 0.0,
                "dist_to_trail_pct": float(dist_to_trail_pct) if dist_to_trail_pct else 0.0,
            }


            print(
                f"\nSymbol: {symbol}"
                f"  |  DCA: {color}{dca_line_pct:+.2f}%{Style.RESET_ALL} @ {self._fmt_price(current_buy_price)} (Line: {dca_line_price_disp} {dca_line_source} | Next: {next_dca_display})"
                f"  |  Gain/Loss SELL: {color2}{gain_loss_percentage_sell:.2f}%{Style.RESET_ALL} @ {self._fmt_price(current_sell_price)}"
                f"  |  DCA Levels Triggered: {triggered_levels}"
                f"  |  Trade Value: ${value:.2f}"
            )




            if avg_cost_basis > 0:
                print(
                    f"  Trailing Profit Margin"
                    f"  |  Line: {self._fmt_price(trail_line_disp)}"
                    f"  |  Above: {above_disp}"
                )
            else:
                print("  PM/Trail: N/A (avg_cost_basis is 0)")



            # --- ATR stop-loss (optional, high-intensity entries only) ---
            if self.stop_loss_enabled and symbol in self._stop_levels:
                stop_price = self._stop_levels[symbol]
                if current_sell_price <= stop_price:
                    print(
                        f"  STOP-LOSS triggered for {symbol}: sell price {current_sell_price:.8f} "
                        f"<= stop {stop_price:.8f}."
                    )
                    response = self.place_sell_order(
                        str(uuid.uuid4()),
                        "sell",
                        "market",
                        full_symbol,
                        quantity,
                        expected_price=current_sell_price,
                        avg_cost_basis=avg_cost_basis,
                        pnl_pct=gain_loss_percentage_sell,
                        tag="STOP_LOSS",
                    )
                    if response and "errors" not in response:
                        trades_made = True
                        self._stop_levels.pop(symbol, None)
                        self.trailing_pm.pop(symbol, None)
                        self._reset_dca_window_for_trade(symbol, sold=True)
                        print(f"  Stop-loss sell executed for {symbol}.")
                        time.sleep(5)
                        holdings = self.get_holdings()
                        continue

            # --- Trailing profit margin (0.5% trail gap) ---
            # PM "start line" is the normal 5% / 2.5% line (depending on DCA levels hit).
            # Trailing activates once price is ABOVE the PM start line, then line follows peaks up
            # by 0.5%. Forced sell happens ONLY when price goes from ABOVE the trailing line to BELOW it.
            if avg_cost_basis > 0:
                pm_start_pct = self.pm_start_pct_no_dca if int(triggered_levels) == 0 else self.pm_start_pct_with_dca
                base_pm_line = avg_cost_basis * (1.0 + (pm_start_pct / 100.0))
                trail_gap = self.trailing_gap_pct / 100.0  # 0.5% => 0.005

                state = self.trailing_pm.get(symbol)
                if state is None:
                    state = {"active": False, "line": base_pm_line, "peak": 0.0, "was_above": False}
                    self.trailing_pm[symbol] = state
                else:
                    # Never let the line be below the (possibly updated) base PM start line
                    if state.get("line", 0.0) < base_pm_line:
                        state["line"] = base_pm_line

                # Use SELL price because that's what you actually get when you market sell
                above_now = current_sell_price >= state["line"]

                # Activate trailing once we first get above the base PM line
                if (not state["active"]) and above_now:
                    state["active"] = True
                    state["peak"] = current_sell_price

                # If active, update peak and move trailing line up behind it
                if state["active"]:
                    if current_sell_price > state["peak"]:
                        state["peak"] = current_sell_price

                    new_line = state["peak"] * (1.0 - trail_gap)
                    if new_line < base_pm_line:
                        new_line = base_pm_line
                    if new_line > state["line"]:
                        state["line"] = new_line

                    # Forced sell on cross from ABOVE -> BELOW trailing line
                    if state["was_above"] and (current_sell_price < state["line"]):
                        print(
                            f"  Trailing PM hit for {symbol}. "
                            f"Sell price {current_sell_price:.8f} fell below trailing line {state['line']:.8f}."
                        )
                        response = self.place_sell_order(
                            str(uuid.uuid4()),
                            "sell",
                            "market",
                            full_symbol,
                            quantity,
                            expected_price=current_sell_price,
                            avg_cost_basis=avg_cost_basis,
                            pnl_pct=gain_loss_percentage_sell,
                            tag="TRAIL_SELL",
                        )


                        trades_made = True
                        self.trailing_pm.pop(symbol, None)  # clear per-coin trailing state on exit
                        self._stop_levels.pop(symbol, None)  # clear stop level on exit

                        # Trade ended -> reset rolling 24h DCA window for this coin
                        self._reset_dca_window_for_trade(symbol, sold=True)

                        print(f"  Successfully sold {quantity} {symbol}.")
                        time.sleep(5)
                        holdings = self.get_holdings()
                        continue

                # Save this tick’s position relative to the line (needed for “above -> below” detection)
                state["was_above"] = above_now

            # DCA (NEURAL or hardcoded %, whichever hits first for the current stage)
            # Trade starts at neural level 3 => trader is at stage 0.
            # Neural-driven DCA stages (max 4):
            #   stage 0 => neural 4 OR -2.5%
            #   stage 1 => neural 5 OR -5.0%
            #   stage 2 => neural 6 OR -10.0%
            #   stage 3 => neural 7 OR -20.0%
            # After that: hardcoded only (-30, -40, -50, then repeat -50 forever).
            current_stage = len(self.dca_levels_triggered.get(symbol, []))

            # Hardcoded loss % for this stage (repeat last level after list ends)
            hard_level = self.dca_levels[current_stage] if current_stage < len(self.dca_levels) else self.dca_levels[-1]
            hard_hit = gain_loss_percentage_buy <= hard_level

            # Neural trigger only for first 4 DCA stages
            neural_level_needed = None
            neural_level_now = None
            neural_hit = False
            if current_stage < 4:
                neural_level_needed = current_stage + 4
                neural_level_now = self._read_long_dca_signal(symbol)

                # Keep it sane: don't DCA from neural if we're not even below cost basis.
                neural_hit = (gain_loss_percentage_buy < 0) and (neural_level_now >= neural_level_needed)

            if hard_hit or neural_hit:
                if neural_hit and hard_hit:
                    reason = f"NEURAL L{neural_level_now}>=L{neural_level_needed} OR HARD {hard_level:.2f}%"
                elif neural_hit:
                    reason = f"NEURAL L{neural_level_now}>=L{neural_level_needed}"
                else:
                    reason = f"HARD {hard_level:.2f}%"

                print(f"  DCAing {symbol} (stage {current_stage + 1}) via {reason}.")

                print(f"  Current Value: ${value:.2f}")
                dca_amount = value * 2
                print(f"  DCA Amount: ${dca_amount:.2f}")
                print(f"  Buying Power: ${buying_power:.2f}")

                recent_dca = self._dca_window_count(symbol)
                if recent_dca >= int(getattr(self, "max_dca_buys_per_24h", 2)):
                    print(
                        f"  Skipping DCA for {symbol}. "
                        f"Already placed {recent_dca} DCA buys in the last 24h (max {self.max_dca_buys_per_24h})."
                    )

                elif dca_amount <= buying_power:
                    response = self.place_buy_order(
                        str(uuid.uuid4()),
                        "buy",
                        "market",
                        full_symbol,
                        dca_amount,
                        avg_cost_basis=avg_cost_basis,
                        pnl_pct=gain_loss_percentage_buy,
                        tag="DCA",
                    )

                    print(f"  Buy Response: {response}")
                    if response and "errors" not in response:
                        # record that we completed THIS stage (no matter what triggered it)
                        self.dca_levels_triggered.setdefault(symbol, []).append(current_stage)

                        # Only record a DCA buy timestamp on success (so skips never advance anything)
                        self._note_dca_buy(symbol)

                        trades_made = True
                        print(f"  Successfully placed DCA buy order for {symbol}.")
                    else:
                        print(f"  Failed to place DCA buy order for {symbol}.")
                else:
                    print(f"  Skipping DCA for {symbol}. Not enough funds.")

            else:
                pass


        # --- ensure GUI gets bid/ask lines even for coins not currently held ---
        try:
            for sym in crypto_symbols:
                if sym in positions:
                    continue

                full_symbol = f"{sym}-USD"
                if full_symbol not in valid_symbols or sym == "USDC":
                    continue

                current_buy_price = current_buy_prices.get(full_symbol, 0.0)
                current_sell_price = current_sell_prices.get(full_symbol, 0.0)

                # keep the per-coin current price file behavior for consistency
                try:
                    file = open(sym + '_current_price.txt', 'w+')
                    file.write(str(current_buy_price))
                    file.close()
                except Exception:
                    pass

                positions[sym] = {
                    "quantity": 0.0,
                    "avg_cost_basis": 0.0,
                    "current_buy_price": current_buy_price,
                    "current_sell_price": current_sell_price,
                    "gain_loss_pct_buy": 0.0,
                    "gain_loss_pct_sell": 0.0,
                    "value_usd": 0.0,
                    "dca_triggered_stages": int(len(self.dca_levels_triggered.get(sym, []))),
                    "next_dca_display": "",
                    "dca_line_price": 0.0,
                    "dca_line_source": "N/A",
                    "dca_line_pct": 0.0,
                    "trail_active": False,
                    "trail_line": 0.0,
                    "trail_peak": 0.0,
                    "dist_to_trail_pct": 0.0,
                }
        except Exception:
            pass

        if not trading_pairs:
            return



        allocation_in_usd = total_account_value * (0.00005/len(crypto_symbols))
        if allocation_in_usd < 0.5:
            allocation_in_usd = 0.5

        holding_full_symbols = [f"{h['asset_code']}-USD" for h in holdings.get("results", [])]

        start_index = 0
        while start_index < len(crypto_symbols):
            base_symbol = crypto_symbols[start_index].upper().strip()
            full_symbol = f"{base_symbol}-USD"

            # Skip if already held
            if full_symbol in holding_full_symbols:
                start_index += 1
                continue

            # Neural signals are used as a "permission to start" gate.
            buy_count = self._read_long_dca_signal(base_symbol)
            sell_count = self._read_short_dca_signal(base_symbol)

            # Entry gate: long must be >= buy_threshold and short must be 0.
            # buy_threshold defaults to 3 and can be overridden by optimizer_config.json.
            entry_threshold = int(getattr(self, "_buy_threshold", 3))
            if not (buy_count >= entry_threshold and sell_count == 0):
                start_index += 1
                continue

            # Correlation-aware portfolio limit: skip entry when adding this coin
            # would exceed the configured threshold of highly-correlated positions.
            held_coins = [h["asset_code"] for h in (holdings.get("results", []) if isinstance(holdings, dict) else [])]
            if self._is_correlated_entry_blocked(base_symbol, held_coins):
                print(f"[correlation] skipping {base_symbol}: correlated positions at limit")
                start_index += 1
                continue

            _fire_alert("SIGNAL", f"Entry signal: {base_symbol}  long={buy_count}/7  short={sell_count}/7  threshold={entry_threshold}", coin=base_symbol)

            # Apply risk-overlay sizing before placing the entry order.
            sized_allocation = self._apply_risk_sizing(base_symbol, allocation_in_usd, buy_count)

            response = self.place_buy_order(
                str(uuid.uuid4()),
                "buy",
                "market",
                full_symbol,
                sized_allocation,
            )

            if response and "errors" not in response:
                trades_made = True
                # Do NOT pre-trigger any DCA levels. Hardcoded DCA will mark levels only when it hits your loss thresholds.
                self.dca_levels_triggered[base_symbol] = []

                # Fresh trade -> clear any rolling 24h DCA window for this coin
                self._reset_dca_window_for_trade(base_symbol, sold=False)

                # Reset trailing PM state for this coin (fresh trade, fresh trailing logic)
                self.trailing_pm.pop(base_symbol, None)

                # Set ATR stop-loss level if enabled and signal meets threshold.
                if self.stop_loss_enabled and buy_count >= self.stop_loss_min_signal:
                    try:
                        entry_atr = self._calc_atr(base_symbol)
                        buy_prices, _, _ = self.get_price([full_symbol])
                        entry_price = float((buy_prices or {}).get(full_symbol, 0) or 0)
                        if entry_price > 0 and entry_atr > 0:
                            self._stop_levels[base_symbol] = entry_price - self.stop_loss_atr_mult * entry_atr
                    except Exception:
                        pass

                print(
                    f"Starting new trade for {full_symbol} (AI start signal long={buy_count}, short={sell_count}). "
                    f"Allocating ${sized_allocation:.2f}."
                )
                time.sleep(5)
                holdings = self.get_holdings()
                holding_full_symbols = [f"{h['asset_code']}-USD" for h in holdings.get("results", [])]


            start_index += 1

        # If any trades were made, recalculate the cost basis
        if trades_made:
            time.sleep(5)
            print("Trades were made in this iteration. Recalculating cost basis...")
            new_cost_basis = self.calculate_cost_basis()
            if new_cost_basis:
                self.cost_basis = new_cost_basis
                print("Cost basis recalculated successfully.")
            else:
                print("Failed to recalculcate cost basis.")
            self.initialize_dca_levels()

        # --- GUI HUB STATUS WRITE ---
        try:
            status = {
                "timestamp": time.time(),
                "account": {
                    "total_account_value": total_account_value,
                    "buying_power": buying_power,
                    "holdings_sell_value": holdings_sell_value,
                    "holdings_buy_value": holdings_buy_value,
                    "percent_in_trade": in_use,
                    # trailing PM config (matches what's printed above current trades)
                    "pm_start_pct_no_dca": float(getattr(self, "pm_start_pct_no_dca", 0.0)),
                    "pm_start_pct_with_dca": float(getattr(self, "pm_start_pct_with_dca", 0.0)),
                    "trailing_gap_pct": float(getattr(self, "trailing_gap_pct", 0.0)),
                },
                "positions": positions,
            }
            self._append_jsonl(
                ACCOUNT_VALUE_HISTORY_PATH,
                {"ts": status["timestamp"], "total_account_value": total_account_value},
            )
            self._write_trader_status(status)
        except Exception:
            pass




    def run(self):
        while True:
            try:
                self.manage_trades()
                time.sleep(0.5)
            except Exception as e:
                print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerTrader AI - Execution Engine")
    parser.add_argument('--replay', action='store_true', help='Run in backtesting replay mode')
    parser.add_argument('--replay-output-dir', default='backtest_results/latest',
                        help='Output directory for replay mode')
    parser.add_argument('--paper', action='store_true',
                        help='Paper trading mode: live KuCoin prices, simulated fills, no Robinhood keys needed')
    parser.add_argument('--paper-balance', type=float, default=10_000.0,
                        help='Starting cash balance for paper trading (default: $10,000)')
    args = parser.parse_args()

    REPLAY_MODE = args.replay
    REPLAY_OUTPUT_DIR = args.replay_output_dir
    PAPER_MODE = args.paper
    PAPER_BALANCE = args.paper_balance

    if REPLAY_MODE:
        HUB_DATA_DIR = os.path.join(REPLAY_OUTPUT_DIR, "hub_data")
        os.makedirs(HUB_DATA_DIR, exist_ok=True)

        current_module = sys.modules[__name__]
        current_module.HUB_DATA_DIR = HUB_DATA_DIR
        current_module.TRADER_STATUS_PATH = os.path.join(HUB_DATA_DIR, "trader_status.json")
        current_module.TRADE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "trade_history.jsonl")
        current_module.PNL_LEDGER_PATH = os.path.join(HUB_DATA_DIR, "pnl_ledger.json")
        current_module.ACCOUNT_VALUE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "account_value_history.jsonl")

        print("=" * 60)
        print(f"REPLAY MODE ACTIVE - Output to {HUB_DATA_DIR}")
        print("=" * 60)

    if PAPER_MODE:
        print("=" * 60)
        print("PAPER TRADING MODE ACTIVE")
        print(f"  Starting balance : ${PAPER_BALANCE:,.2f}")
        print(f"  Price source     : KuCoin (live)")
        print(f"  Order execution  : Simulated (no Robinhood API calls)")
        print(f"  Account state    : {os.path.join(HUB_DATA_DIR, 'paper_account.json')}")
        print("=" * 60)

    trading_bot = CryptoAPITrading()
    trading_bot.run()
