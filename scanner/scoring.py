"""Gates, Z-scores, forward score and verdicts. Pure pandas, no network.

score(metrics, thresholds, trendlyne) runs in well under a second for 500 stocks,
so the app calls it on every sidebar change.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import METRICS, Thresholds

PASS, FAIL, NODATA = "pass", "fail", "no data"


# ------------------------------------------------------------------ valuation

def robust_z(x: pd.Series, winsor_pct: float = 2.0) -> pd.Series:
    """(x - median) / (1.4826 * MAD) after winsorising; NaN where undefined."""
    v = x.astype(float)
    ok = v.dropna()
    if len(ok) < 3:
        return pd.Series(np.nan, index=x.index)
    lo, hi = np.percentile(ok, [winsor_pct, 100 - winsor_pct])
    w = v.clip(lo, hi)
    med = w.median()
    mad = (w - med).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(np.nan, index=x.index)
    return (w - med) / (1.4826 * mad)


def add_valuation(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    df = df.copy()
    df["z_group"] = ""
    df["peers"] = 0
    df["z"] = np.nan
    has = df["metric_value"].notna()

    # Group by (bucket, metric); small groups pool with every stock on that metric.
    counts = df[has].groupby(["bucket", "metric"]).size()
    key = []
    for b, m, h in zip(df["bucket"], df["metric"], has):
        if h and counts.get((b, m), 0) >= th.min_peers:
            key.append(f"{b} · {METRICS[m][0]}")
        else:
            key.append(f"All stocks · {METRICS[m][0]}")
    df["z_group"] = key

    for g, idx in df[has].groupby("z_group").groups.items():
        z = robust_z(df.loc[idx, "metric_value"], th.winsor_pct)
        m = df.loc[idx[0], "metric"]
        if METRICS[m][1]:          # higher is cheaper -> flip so negative = cheap
            z = -z
        df.loc[idx, "z"] = z
        df.loc[idx, "peers"] = len(idx)
        df.loc[idx, "group_median"] = df.loc[idx, "metric_value"].median()

    df["valuation"] = np.select(
        [df["z"].isna(), df["z"] <= th.cheap_z, df["z"] > th.expensive_z],
        ["n/a", "Cheap", "Expensive"], "Fair")
    return df


# -------------------------------------------------------------------- quality

def _check(v, ok) -> str:
    if v is None or pd.isna(v):
        return NODATA
    return PASS if ok else FAIL


def add_quality(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    df = df.copy()
    roce_med = df[~df["is_financial"]].groupby("bucket")["roce"].transform("median")
    df["roce_bucket_median"] = roce_med

    core, s1, s2, s3 = [], [], [], []
    for r, med in zip(df.itertuples(), roce_med.reindex(df.index)):
        if r.is_financial:
            core.append(_check(r.roe, (r.roe or 0) >= th.roe_min))
            s1.append(_check(r.roa, (r.roa or 0) >= th.roa_min))
            s2.append(_check(r.bv_cagr, (r.bv_cagr or 0) >= th.bv_cagr_min))
            s3.append("")
        else:
            roce_ok = (r.roce >= th.roce_min) or (
                th.roce_or_sector_median and pd.notna(med) and r.roce >= med)
            core.append(_check(r.roce, roce_ok))
            s1.append(_check(r.de, (r.de if pd.notna(r.de) else 99) <= th.de_max))
            if r.no_interest:
                s2.append(PASS)
            else:
                s2.append(_check(r.icr, (r.icr if pd.notna(r.icr) else -1) >= th.icr_min))
            s3.append(_check(r.cfo_pat, (r.cfo_pat if pd.notna(r.cfo_pat) else -1) >= th.cfo_pat_min))
    df["q_core"], df["q_s1"], df["q_s2"], df["q_s3"] = core, s1, s2, s3

    status, nodata_n = [], []
    for r in df.itertuples():
        sup = [x for x in (r.q_s1, r.q_s2, r.q_s3) if x]
        need = 2
        passes = sum(x == PASS for x in sup)
        nd = sum(x == NODATA for x in sup) + (r.q_core == NODATA)
        nodata_n.append(nd)
        if r.q_core == NODATA:
            status.append(NODATA)
        elif r.q_core == FAIL:
            status.append(FAIL)
        elif passes >= need:
            status.append(PASS)
        elif passes + sum(x == NODATA for x in sup) >= need:
            status.append(NODATA)
        else:
            status.append(FAIL)
    df["quality"] = status
    df["quality_nodata"] = nodata_n
    df["thin_flag"] = df["quality_nodata"] >= 2
    return df


# -------------------------------------------------------------------- forward

def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


TL_SYMBOL_COLS = ["nsecode", "nsesymbol", "symbol", "ticker", "nseticker", "stockcode"]
TL_FIELDS = {
    "tl_eps_g": "fwdepsgrowthttm",
    "tl_upside": "fwdupside12m",
    "tl_upgrade": "fwdanalystupgrade3m",
    "tl_peg": "fwdpegttm",
    "tl_grid": "fwdgridscore",
    "tl_grid_v2": "fwdgridscorev2",
}


def parse_trendlyne(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Map a Trendlyne export to symbol + tl_* columns. Returns (frame, note)."""
    cols = {_norm(c): c for c in raw.columns}
    sym = next((cols[c] for c in TL_SYMBOL_COLS if c in cols), None)
    if sym is None:
        return pd.DataFrame(columns=["symbol"]), (
            "No symbol column found (expected one of: NSE Code, Symbol, Ticker).")
    out = pd.DataFrame({"symbol": raw[sym].astype(str).str.strip().str.upper()
                        .str.replace(r"^NSE:", "", regex=True)})
    found = []
    for new, want in TL_FIELDS.items():
        if want in cols:
            out[new] = pd.to_numeric(
                raw[cols[want]].astype(str).str.replace("%", "").str.replace(",", ""),
                errors="coerce")
            found.append(cols[want])
    out = out.drop_duplicates("symbol")
    return out, f"{len(out)} rows; fields used: {', '.join(found) or 'none'}"


def _sig(v, pos_ok, neg_ok) -> int:
    if v is None or pd.isna(v):
        return 0
    if pos_ok:
        return 1
    if neg_ok:
        return -1
    return 0


def add_forward(df: pd.DataFrame, th: Thresholds, tl: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    tl_cols = list(TL_FIELDS)
    if tl is not None and not tl.empty:
        df = df.drop(columns=[c for c in tl_cols if c in df], errors="ignore").merge(tl, on="symbol", how="left")
    for c in tl_cols:
        if c not in df:
            df[c] = np.nan

    covered = df["n_analysts"].fillna(0) >= th.min_analysts
    rows = {k: [] for k in ("f_eps", "f_up", "f_rec", "f_peg", "f_src")}
    for r, cov in zip(df.itertuples(), covered):
        used_tl = False

        def pick(yf_val, tl_val):
            nonlocal used_tl
            if pd.notna(tl_val):
                used_tl = True
                return tl_val, True
            return (yf_val if cov else np.nan), False

        v, _ = pick(r.fwd_eps_g, r.tl_eps_g)
        rows["f_eps"].append(_sig(v, pd.notna(v) and v >= th.eps_g_pos, pd.notna(v) and v < th.eps_g_neg))
        v, _ = pick(r.upside, r.tl_upside)
        rows["f_up"].append(_sig(v, pd.notna(v) and v >= th.upside_pos, pd.notna(v) and v < th.upside_neg))
        if pd.notna(r.tl_upgrade):
            used_tl = True
            rows["f_rec"].append(int(np.sign(r.tl_upgrade)))
        else:
            v = r.rec_mean if cov else np.nan
            rows["f_rec"].append(_sig(v, pd.notna(v) and v <= th.rec_pos, pd.notna(v) and v >= th.rec_neg))
        v, _ = pick(r.fwd_peg, r.tl_peg)
        rows["f_peg"].append(_sig(v, pd.notna(v) and 0 < v <= th.peg_pos, pd.notna(v) and v > th.peg_neg))
        rows["f_src"].append("Trendlyne" if used_tl else ("yfinance" if cov else "No coverage"))
    for k, v in rows.items():
        df[k] = v
    df["fwd_score"] = df[["f_eps", "f_up", "f_rec", "f_peg"]].sum(axis=1).astype(int)
    df["fwd_coverage"] = df["f_src"]
    return df


# -------------------------------------------------------------------- verdict

def add_verdict(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    df = df.copy()
    verdict, why = [], []
    for r in df.itertuples():
        if pd.isna(r.z):
            name = METRICS[r.metric][0]
            if pd.notna(r.metric_value):
                msg = f"only {r.peers} stocks on {name}; too few to compare"
            else:
                msg = f"no usable {name} ({r.metric_flag or 'missing'})"
            verdict.append("Data: Thin"); why.append(msg)
        elif r.z > th.cheap_z:
            verdict.append("Not cheap"); why.append(f"{r.valuation} vs peers (Z {r.z:+.2f})")
        elif r.quality == FAIL:
            verdict.append("Value trap risk"); why.append("cheap, but quality gate failed")
        elif r.fwd_score < 0:
            verdict.append("Value trap risk"); why.append(f"cheap, but forward score {r.fwd_score}")
        elif (r.fwd_score >= th.prime_min_fwd and r.fwd_coverage != "No coverage"
              and r.quality == PASS and not r.thin_flag and not r.speculative):
            verdict.append("Prime"); why.append("cheap, quality passed, forward support")
        else:
            reasons = []
            if r.quality == NODATA or r.thin_flag:
                reasons.append("quality data thin")
            if r.speculative:
                reasons.append("loss-maker on EV/Sales")
            if r.fwd_coverage == "No coverage":
                reasons.append("no analyst coverage")
            elif r.fwd_score < th.prime_min_fwd:
                reasons.append(f"forward score {r.fwd_score}")
            verdict.append("Value-Watch"); why.append("cheap; " + ", ".join(reasons or ["awaiting forward lift"]))
    df["verdict"] = verdict
    df["verdict_reason"] = why

    order = {"Prime": 0, "Value-Watch": 1, "Value trap risk": 2, "Not cheap": 3, "Data: Thin": 4}
    df["_o"] = df["verdict"].map(order)
    df = df.sort_values(["_o", "z", "fwd_score"], ascending=[True, True, False]).drop(columns="_o")
    df["rank"] = df.groupby("verdict").cumcount() + 1
    df.loc[~df["verdict"].isin(["Prime", "Value-Watch"]), "rank"] = np.nan
    return df.reset_index(drop=True)


def score(metrics: pd.DataFrame, th: Thresholds | None = None, tl: pd.DataFrame | None = None) -> pd.DataFrame:
    th = th or Thresholds()
    df = add_valuation(metrics, th)
    df = add_quality(df, th)
    df = add_forward(df, th, tl)
    return add_verdict(df, th)
