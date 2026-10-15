"""yfinance fetching: quote/forward fields daily, statements weekly.

Every function returns plain dicts of numbers so the cache is a flat table.
Failures never raise; they leave fields empty and record an error string.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _retry(fn, tries: int = 3, base_wait: float = 2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # yfinance raises many types, incl. rate limits
            last = e
            time.sleep(base_wait * (2 ** i))
    raise last


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _row(df: pd.DataFrame | None, *names) -> pd.Series | None:
    """First matching line item as a numeric Series, newest period first."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index)
                return s.sort_index(ascending=False)
    return None


def _first(s: pd.Series | None, i: int = 0):
    if s is None or len(s) <= i:
        return np.nan
    return float(s.iloc[i])


# --------------------------------------------------------------------------- info

INFO_FIELDS = {
    "price": ("currentPrice", "regularMarketPrice", "previousClose"),
    "shares": ("sharesOutstanding", "impliedSharesOutstanding"),
    "mcap_info": ("marketCap",),
    "ev_info": ("enterpriseValue",),
    "fwd_eps": ("forwardEps",),
    "ttm_eps": ("trailingEps",),
    "target_mean": ("targetMeanPrice",),
    "n_analysts": ("numberOfAnalystOpinions",),
    "rec_mean": ("recommendationMean",),
    "pb_info": ("priceToBook",),
    "revenue_info": ("totalRevenue",),
}


def fetch_info(symbol: str) -> dict:
    out = {"symbol": symbol, "info_fetched_at": _now(), "info_error": ""}
    t = yf.Ticker(f"{symbol}.NS")
    try:
        info = _retry(lambda: t.info) or {}
    except Exception as e:
        out["info_error"] = str(e)[:200]
        return out
    for key, names in INFO_FIELDS.items():
        out[key] = np.nan
        for n in names:
            v = _num(info.get(n))
            if not np.isnan(v):
                out[key] = v
                break
    # .info sometimes omits shares/market cap for Indian stocks; fast_info usually has them.
    if np.isnan(out["shares"]) or np.isnan(out["mcap_info"]):
        try:
            fi = t.fast_info
            if np.isnan(out["shares"]):
                out["shares"] = _num(fi.get("shares"))
            if np.isnan(out["mcap_info"]):
                out["mcap_info"] = _num(fi.get("market_cap"))
            if np.isnan(out["price"]):
                out["price"] = _num(fi.get("last_price"))
        except Exception:
            pass
    out["yf_industry"] = info.get("industry") or ""
    out["yf_sector"] = info.get("sector") or ""
    out["currency"] = info.get("financialCurrency") or info.get("currency") or ""
    if not out.get("price") or np.isnan(out.get("price", np.nan)):
        out["info_error"] = "no price"
    return out


# ------------------------------------------------------------------- statements

def _ttm(q: pd.Series | None, max_age_days: int = 200) -> float:
    """Sum of the last 4 quarters, only if they are recent and consecutive."""
    if q is None or len(q) < 4:
        return np.nan
    q4 = q.iloc[:4]
    newest, oldest = q4.index[0], q4.index[3]
    age = (pd.Timestamp.now() - newest).days
    span = (newest - oldest).days
    if age > max_age_days or not (250 <= span <= 300):
        return np.nan
    return float(q4.sum())


def fetch_fundamentals(symbol: str) -> dict:
    out = {"symbol": symbol, "fund_fetched_at": _now(), "fund_error": ""}
    t = yf.Ticker(f"{symbol}.NS")
    frames = {}
    for name, getter in {
        "is_a": lambda: t.income_stmt,
        "is_q": lambda: t.quarterly_income_stmt,
        "bs": lambda: t.balance_sheet,
        "cf": lambda: t.cashflow,
    }.items():
        try:
            frames[name] = _retry(getter)
        except Exception as e:
            frames[name] = None
            out["fund_error"] += f"{name}: {str(e)[:80]}; "

    a, q, b, c = frames["is_a"], frames["is_q"], frames["bs"], frames["cf"]
    # Yahoo sometimes returns empty frames instead of an error when it throttles.
    if all(f is None or getattr(f, "empty", True) for f in (a, b)):
        out["fund_error"] = (out["fund_error"] + "empty statements").strip()
        return out

    ebit = _row(a, "EBIT", "Operating Income")
    ebit_q = _row(q, "EBIT", "Operating Income")
    rev = _row(a, "Total Revenue", "Operating Revenue")
    rev_q = _row(q, "Total Revenue", "Operating Revenue")
    ni = _row(a, "Net Income Common Stockholders", "Net Income")
    interest = _row(a, "Interest Expense", "Interest Expense Non Operating")
    debt = _row(b, "Total Debt")
    cash = _row(b, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    equity = _row(b, "Stockholders Equity", "Common Stock Equity")
    assets = _row(b, "Total Assets")
    cur_liab = _row(b, "Current Liabilities")
    cfo = _row(c, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")

    ebit_ttm = _ttm(ebit_q)
    out["ebit"] = ebit_ttm if not np.isnan(ebit_ttm) else _first(ebit)
    out["ebit_basis"] = "TTM" if not np.isnan(ebit_ttm) else ("FY" if ebit is not None else "")
    rev_ttm = _ttm(rev_q)
    out["revenue"] = rev_ttm if not np.isnan(rev_ttm) else _first(rev)
    out["fy_end"] = ebit.index[0].strftime("%Y-%m-%d") if ebit is not None else (
        ni.index[0].strftime("%Y-%m-%d") if ni is not None else "")

    out["net_income"] = _first(ni)
    out["net_income_prev"] = _first(ni, 1)
    out["interest"] = abs(_first(interest)) if interest is not None else np.nan
    out["debt"] = _first(debt)
    out["cash"] = _first(cash)
    out["equity"] = _first(equity)
    out["equity_prev"] = _first(equity, 1)
    out["assets"] = _first(assets)
    out["cur_liab"] = _first(cur_liab)

    # 3-year sums for earnings quality (only if 3 years of both exist)
    if cfo is not None and ni is not None and len(cfo) >= 3 and len(ni) >= 3:
        out["cfo_3y"] = float(cfo.iloc[:3].sum())
        out["ni_3y"] = float(ni.iloc[:3].sum())
    else:
        out["cfo_3y"] = out["ni_3y"] = np.nan

    # Book value growth: oldest available up to 3 years back
    if equity is not None and len(equity) >= 2:
        k = min(3, len(equity) - 1)
        out["equity_old"] = float(equity.iloc[k])
        out["equity_years"] = k
    else:
        out["equity_old"], out["equity_years"] = np.nan, 0

    # Own history: annual EBIT/EV and P/B at each fiscal year end.
    out["hist_ebit_yield"] = "[]"
    out["hist_pb"] = "[]"
    try:
        info_shares = _num((t.fast_info or {}).get("shares"))
    except Exception:
        info_shares = np.nan
    try:
        px = _retry(lambda: t.history(period="6y", interval="1mo", auto_adjust=False))
        px = px["Close"].dropna() if px is not None and not px.empty else None
    except Exception:
        px = None
    if px is not None and not np.isnan(info_shares):
        px.index = pd.to_datetime(px.index).tz_localize(None)
        hy, hp = [], []
        for d in (equity.index if equity is not None else []):
            near = px[px.index <= d + pd.Timedelta(days=5)]
            if near.empty:
                continue
            mcap = float(near.iloc[-1]) * info_shares   # split-adjusted price x current shares
            eq = equity.get(d, np.nan)
            if eq and eq > 0:
                hp.append(round(mcap / eq, 4))
            e = ebit.get(d, np.nan) if ebit is not None else np.nan
            dbt = debt.get(d, 0.0) if debt is not None else 0.0
            csh = cash.get(d, 0.0) if cash is not None else 0.0
            ev = mcap + (0 if np.isnan(dbt) else dbt) - (0 if np.isnan(csh) else csh)
            if not np.isnan(e) and ev > 0:
                hy.append(round(100 * e / ev, 4))
        out["hist_ebit_yield"] = json.dumps(hy)
        out["hist_pb"] = json.dumps(hp)
    return out


# ------------------------------------------------------------------------ batch

def fetch_many(symbols, fn, workers: int = 4, label: str = "", progress_every: int = 50) -> pd.DataFrame:
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, s): s for s in symbols}
        for f in as_completed(futs):
            s = futs[f]
            try:
                rows.append(f.result())
            except Exception as e:  # should not happen; fn swallows errors
                rows.append({"symbol": s, f"{label}_error": str(e)[:200]})
            done += 1
            if done % progress_every == 0 or done == len(symbols):
                log.info("%s: %d/%d", label, done, len(symbols))
    return pd.DataFrame(rows)
