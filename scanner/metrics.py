"""Turn raw fetched fields into the per-stock metrics the scoring step uses.

Output is threshold-free: the app re-scores it instantly when sidebar values change.
Percent-type metrics are stored in percent units (18.5 means 18.5%).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import METRIC_BOUNDS


def classify(nse_industry: str, yf_industry: str) -> tuple[str, str, bool]:
    """Return (bucket, primary metric, is_financial)."""
    if (nse_industry or "").strip() == "Financial Services":
        y = (yf_industry or "").lower()
        if "bank" in y:
            return "Banks", "pb", True
        if "insurance" in y:
            return "Insurers", "pb", True
        if any(k in y for k in ("asset management", "capital markets", "financial data", "exchange")):
            return "AMCs, brokers & exchanges", "pe", True
        return "NBFCs & HFCs", "pb", True
    return (nse_industry or "Unclassified").strip(), "ebit_yield", False


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return np.where(np.isfinite(r), r, np.nan)


def _hist_pct(hist_json: str, current: float, higher_is_cheaper: bool) -> tuple[float, int]:
    """% of past fiscal-year readings that were more expensive than today."""
    try:
        h = [float(x) for x in json.loads(hist_json or "[]")]
    except (TypeError, ValueError):
        h = []
    if len(h) < 4 or current is None or np.isnan(current):
        return np.nan, len(h)
    h = np.array(h)
    pricier = (h < current) if higher_is_cheaper else (h > current)
    return float(100 * pricier.mean()), len(h)


def build_metrics(raw: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.merge(raw, on="symbol", how="left")
    for col in ("yf_industry", "info_error", "fund_error", "ebit_basis", "fy_end",
                "hist_ebit_yield", "hist_pb", "info_fetched_at", "fund_fetched_at"):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("")
    num_cols = ["price", "shares", "mcap_info", "ev_info", "fwd_eps", "ttm_eps", "target_mean",
                "n_analysts", "rec_mean", "pb_info", "revenue_info", "ebit", "revenue", "net_income",
                "net_income_prev", "interest", "debt", "cash", "equity", "equity_prev", "assets",
                "cur_liab", "cfo_3y", "ni_3y", "equity_old", "equity_years"]
    for c in num_cols:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    cls = [classify(a, b) for a, b in zip(df["nse_industry"], df["yf_industry"])]
    df["bucket"] = [c[0] for c in cls]
    df["metric"] = [c[1] for c in cls]
    df["is_financial"] = [c[2] for c in cls]

    # Market cap and EV
    mcap = df["price"] * df["shares"]
    df["mcap"] = np.where(mcap.notna(), mcap, df["mcap_info"])
    df["mcap_cr"] = df["mcap"] / 1e7
    have_bs = df["debt"].notna() | df["cash"].notna()
    ev_bs = df["mcap"] + df["debt"].fillna(0) - df["cash"].fillna(0)
    df["ev"] = np.where(have_bs & df["mcap"].notna(), ev_bs, df["ev_info"])
    df["ev_source"] = np.where(have_bs & df["mcap"].notna(), "mcap + debt - cash", "yfinance EV")

    # Valuation candidates
    df["ebit_yield"] = 100 * _div(df["ebit"], df["ev"])
    df.loc[df["ev"] <= 0, "ebit_yield"] = np.nan
    pb_calc = _div(df["mcap"], df["equity"])
    df["pb"] = np.where(df["equity"] > 0, pb_calc, np.nan)
    df["pb"] = np.where(pd.isna(df["pb"]), df["pb_info"], df["pb"])
    df["pe"] = np.where(df["ttm_eps"] > 0, _div(df["price"], df["ttm_eps"]), np.nan)
    rev = df["revenue"].where(df["revenue"].notna(), df["revenue_info"])
    df["ev_sales"] = np.where((rev > 0) & (df["ev"] > 0), _div(df["ev"], rev), np.nan)

    # Loss-makers among non-financials switch to EV/Sales and are marked speculative
    loss = (~df["is_financial"]) & (df["ebit"] <= 0)
    df.loc[loss, "metric"] = "ev_sales"
    df["speculative"] = loss

    # Primary metric value with sanity bounds
    vals, flags = [], []
    for m, row in zip(df["metric"], df.itertuples()):
        v = getattr(row, m)
        lo, hi = METRIC_BOUNDS[m]
        if v is None or pd.isna(v):
            vals.append(np.nan); flags.append("missing")
        elif not (lo < v < hi) and not (m == "ebit_yield" and lo <= v <= hi):
            vals.append(np.nan); flags.append(f"rejected ({v:.2f} out of bounds)")
        else:
            vals.append(float(v)); flags.append("")
    df["metric_value"] = vals
    df["metric_flag"] = flags

    # Own-history percentile (information only)
    pcts, ns = [], []
    for m, v, hy, hp in zip(df["metric"], df["metric_value"], df["hist_ebit_yield"], df["hist_pb"]):
        if m == "ebit_yield":
            p, n = _hist_pct(hy, v, True)
        elif m == "pb":
            p, n = _hist_pct(hp, v, False)
        else:
            p, n = np.nan, 0
        pcts.append(p); ns.append(n)
    df["own_hist_cheaper_than_pct"] = pcts
    df["own_hist_years"] = ns

    # Quality inputs
    cap_emp = df["assets"] - df["cur_liab"]
    df["roce"] = np.where(cap_emp > 0, 100 * _div(df["ebit"], cap_emp), np.nan)
    avg_eq = df[["equity", "equity_prev"]].mean(axis=1)
    df["roe"] = np.where(avg_eq > 0, 100 * _div(df["net_income"], avg_eq), np.nan)
    df["roa"] = np.where(df["assets"] > 0, 100 * _div(df["net_income"], df["assets"]), np.nan)
    df["de"] = np.where(df["equity"] > 0, _div(df["debt"].fillna(0), df["equity"]), np.nan)
    df.loc[df["debt"].isna() & df["equity"].notna(), "de"] = np.nan
    df["icr"] = np.where(df["interest"] > 0, _div(df["ebit"], df["interest"]), np.nan)
    df["no_interest"] = (df["interest"].isna() | (df["interest"] == 0)) & (df["de"] < 0.1)
    df["cfo_pat"] = np.where(df["ni_3y"] > 0, _div(df["cfo_3y"], df["ni_3y"]), np.nan)
    yrs = df["equity_years"].replace(0, np.nan)
    ratio = _div(df["equity"], df["equity_old"])
    df["bv_cagr"] = np.where((ratio > 0) & yrs.notna(), 100 * (np.power(ratio, 1 / yrs) - 1), np.nan)

    # Forward inputs
    df["fwd_eps_g"] = np.where(df["ttm_eps"] > 0, 100 * (_div(df["fwd_eps"], df["ttm_eps"]) - 1), np.nan)
    df["upside"] = np.where(df["price"] > 0, 100 * (_div(df["target_mean"], df["price"]) - 1), np.nan)
    fwd_pe = np.where(df["fwd_eps"] > 0, _div(df["price"], df["fwd_eps"]), np.nan)
    df["fwd_pe"] = fwd_pe
    df["fwd_peg"] = np.where(df["fwd_eps_g"] > 0, _div(fwd_pe, df["fwd_eps_g"]), np.nan)

    # Market-cap band by rank inside the universe (SEBI-style: top 100 / next 150 / rest)
    rank = df["mcap"].rank(ascending=False, method="first")
    df["cap_band"] = np.select([rank <= 100, rank <= 250], ["Large", "Mid"], "Small")
    df.loc[df["mcap"].isna(), "cap_band"] = "Unknown"
    return df
