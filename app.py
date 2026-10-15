"""Nifty 500 Value + Forward Scanner — Streamlit viewer.

Reads the file written by the nightly job (data/metrics_latest.parquet) and
re-scores it instantly with the thresholds set in the sidebar.
© Dr Shantanu Samanta
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scanner.config import METRICS, METRICS_FILE, RUN_META, VERDICTS, Thresholds
from scanner.scoring import parse_trendlyne, score

st.set_page_config(page_title="Nifty 500 Value + Forward Scanner", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
@media (max-width: 640px) {
  .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
  div[data-testid="stMetricValue"] {font-size: 1.2rem;}
}
.small-note {font-size: 0.85rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

VERDICT_COLORS = {
    "Prime": "#1a9e5c", "Value-Watch": "#3b7dd8", "Value trap risk": "#d64545",
    "Not cheap": "#9aa0a6", "Data: Thin": "#c9a227",
}
QUALITY_LABELS = {
    False: [("q_core", "ROCE"), ("q_s1", "Debt/Equity"), ("q_s2", "Interest cover"), ("q_s3", "CFO/PAT (3y)")],
    True: [("q_core", "ROE"), ("q_s1", "ROA"), ("q_s2", "Book value CAGR")],
}


# ----------------------------------------------------------------------- data

@st.cache_data(show_spinner=False)
def load_metrics(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(METRICS_FILE)


def load_meta() -> dict:
    try:
        with open(RUN_META) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt_pct(v, d=1):
    return "—" if v is None or pd.isna(v) else f"{v:.{d}f}%"


def fmt_num(v, d=2):
    return "—" if v is None or pd.isna(v) else f"{v:,.{d}f}"


# -------------------------------------------------------------------- sidebar

def sidebar() -> tuple[Thresholds, pd.DataFrame | None]:
    d = Thresholds()
    st.sidebar.header("Thresholds")
    st.sidebar.caption("Changes re-score instantly. Defaults match the decision matrix.")
    if st.sidebar.button("Reset to defaults", width="stretch"):
        for k in list(st.session_state.keys()):
            if k.startswith("th_"):
                del st.session_state[k]
        st.rerun()

    with st.sidebar.expander("Valuation", expanded=False):
        cheap_z = st.slider("Cheap if Z ≤", -2.0, 0.0, d.cheap_z, 0.1, key="th_cheap_z")
        exp_z = st.slider("Expensive if Z >", 0.0, 2.0, d.expensive_z, 0.1, key="th_exp_z")
        min_peers = st.number_input("Min peers per sector bucket", 3, 30, d.min_peers, key="th_min_peers")
    with st.sidebar.expander("Quality — non-financials", expanded=False):
        roce_min = st.number_input("ROCE ≥ (%)", 0.0, 50.0, d.roce_min, 1.0, key="th_roce")
        roce_med = st.checkbox("…or ROCE ≥ sector median", d.roce_or_sector_median, key="th_roce_med")
        de_max = st.number_input("Debt/Equity ≤", 0.0, 5.0, d.de_max, 0.1, key="th_de")
        icr_min = st.number_input("Interest cover ≥ (×)", 0.0, 20.0, d.icr_min, 0.5, key="th_icr")
        cfo_min = st.number_input("CFO ÷ PAT, 3-year ≥", 0.0, 2.0, d.cfo_pat_min, 0.1, key="th_cfo")
    with st.sidebar.expander("Quality — financials", expanded=False):
        roe_min = st.number_input("ROE ≥ (%)", 0.0, 40.0, d.roe_min, 1.0, key="th_roe")
        roa_min = st.number_input("ROA ≥ (%)", 0.0, 10.0, d.roa_min, 0.1, key="th_roa")
        bv_min = st.number_input("Book value CAGR ≥ (%)", 0.0, 40.0, d.bv_cagr_min, 1.0, key="th_bv")
    with st.sidebar.expander("Forward bias", expanded=False):
        min_an = st.number_input("Min analysts for coverage", 1, 20, d.min_analysts, key="th_an")
        eps_pos = st.number_input("Fwd EPS growth +1 if ≥ (%)", 0.0, 50.0, d.eps_g_pos, 1.0, key="th_epsp")
        eps_neg = st.number_input("Fwd EPS growth −1 if < (%)", -50.0, 20.0, d.eps_g_neg, 1.0, key="th_epsn")
        up_pos = st.number_input("Target upside +1 if ≥ (%)", 0.0, 60.0, d.upside_pos, 1.0, key="th_upp")
        up_neg = st.number_input("Target upside −1 if < (%)", -50.0, 20.0, d.upside_neg, 1.0, key="th_upn")
        rec_pos = st.number_input("Rating +1 if ≤ (1 = strong buy)", 1.0, 3.0, d.rec_pos, 0.1, key="th_recp")
        rec_neg = st.number_input("Rating −1 if ≥", 2.5, 5.0, d.rec_neg, 0.1, key="th_recn")
        peg_pos = st.number_input("Fwd PEG +1 if ≤", 0.1, 3.0, d.peg_pos, 0.1, key="th_pegp")
        peg_neg = st.number_input("Fwd PEG −1 if >", 0.5, 5.0, d.peg_neg, 0.1, key="th_pegn")
        prime_fwd = st.number_input("Forward score needed for Prime", 0, 4, d.prime_min_fwd, key="th_prime")

    th = Thresholds(
        cheap_z=cheap_z, expensive_z=exp_z, min_peers=int(min_peers), winsor_pct=d.winsor_pct,
        roce_min=roce_min, roce_or_sector_median=roce_med, de_max=de_max, icr_min=icr_min,
        cfo_pat_min=cfo_min, roe_min=roe_min, roa_min=roa_min, bv_cagr_min=bv_min,
        min_analysts=int(min_an), eps_g_pos=eps_pos, eps_g_neg=eps_neg, upside_pos=up_pos,
        upside_neg=up_neg, rec_pos=rec_pos, rec_neg=rec_neg, peg_pos=peg_pos, peg_neg=peg_neg,
        prime_min_fwd=int(prime_fwd),
    )

    st.sidebar.header("Trendlyne CSV (optional)")
    up = st.sidebar.file_uploader("Forward-screener export", type=["csv"],
                                  help="Its forward fields replace yfinance values for the stocks it covers.")
    tl = None
    if up is not None:
        try:
            tl, note = parse_trendlyne(pd.read_csv(up))
            (st.sidebar.success if len(tl) else st.sidebar.error)(note)
        except Exception as e:
            st.sidebar.error(f"Could not read the CSV: {e}")
            tl = None
    return th, tl


# ----------------------------------------------------------------------- tabs

def tab_shortlist(s: pd.DataFrame):
    counts = s["verdict"].value_counts()
    cols = st.columns(5)
    for c, v in zip(cols, VERDICTS):
        c.metric(v, int(counts.get(v, 0)))

    f1, f2, f3 = st.columns([2, 2, 1.4])
    verdicts = f1.multiselect("Verdict", VERDICTS, default=["Prime", "Value-Watch"])
    buckets = f2.multiselect("Sector bucket", sorted(s["bucket"].unique()))
    bands = f3.multiselect("Market cap", ["Large", "Mid", "Small"])
    v = s[s["verdict"].isin(verdicts)] if verdicts else s
    if buckets:
        v = v[v["bucket"].isin(buckets)]
    if bands:
        v = v[v["cap_band"].isin(bands)]

    view = pd.DataFrame({
        "Rank": v["rank"],
        "Symbol": v["symbol"],
        "Company": v["company"],
        "Verdict": v["verdict"],
        "Bucket": v["bucket"],
        "Cap": v["cap_band"],
        "Metric": v["metric"].map(lambda m: METRICS[m][0]),
        "Value": v["metric_value"],
        "Z vs peers": v["z"],
        "Own-history cheaper than %": v["own_hist_cheaper_than_pct"],
        "Quality": v["quality"],
        "Fwd score": v["fwd_score"],
        "Fwd source": v["fwd_coverage"],
        "Price ₹": v["price"],
        "Mcap ₹ cr": v["mcap_cr"],
        "Why": v["verdict_reason"],
    })
    st.dataframe(
        view, hide_index=True, width="stretch", height=min(640, 38 + 35 * max(len(view), 1)),
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d", width="small"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Z vs peers": st.column_config.NumberColumn(format="%+.2f", help="Negative = cheaper than sector peers"),
            "Own-history cheaper than %": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f%%",
                help="Share of past fiscal years when this stock was more expensive than today (needs 4+ years)"),
            "Fwd score": st.column_config.NumberColumn(format="%+d", help="−4 to +4"),
            "Price ₹": st.column_config.NumberColumn(format="%.2f"),
            "Mcap ₹ cr": st.column_config.NumberColumn(format="%,.0f"),
        },
    )
    st.caption(f"{len(view)} stocks shown. Prime and Value-Watch are ranked cheapest Z first, "
               "forward score breaks ties.")

    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", view.to_csv(index=False).encode(), "nifty500_scan.csv",
                       "text/csv", width="stretch")
    tv = ",".join(f"NSE:{x}" for x in v["symbol"])
    c2.download_button("TradingView watchlist (.txt)", tv.encode(), "tradingview_watchlist.txt",
                       "text/plain", width="stretch",
                       help="In TradingView: Watchlist menu → Import list → choose this file")


def tab_sectors(s: pd.DataFrame, th: Thresholds):
    st.markdown("Cheap is relative to sector peers. If most of a bucket looks cheap, the whole sector "
                "may be de-rating rather than any one stock being undervalued.")
    has = s[s["z"].notna()]
    g = has.groupby("bucket").agg(
        Metric=("metric", lambda x: ", ".join(sorted({METRICS[m][0] for m in x}))),
        Stocks=("symbol", "count"),
        Median_value=("metric_value", "median"),
        Cheap=("z", lambda x: int((x <= th.cheap_z).sum())),
        Prime=("verdict", lambda x: int((x == "Prime").sum())),
        Value_Watch=("verdict", lambda x: int((x == "Value-Watch").sum())),
        Trap=("verdict", lambda x: int((x == "Value trap risk").sum())),
    ).reset_index()
    g["Cheap share %"] = 100 * g["Cheap"] / g["Stocks"]
    g["Flag"] = np.where(g["Cheap share %"] >= 40, "Possible sector-wide de-rating", "")
    g = g.sort_values("Cheap share %", ascending=False).rename(columns={
        "bucket": "Bucket", "Median_value": "Median value", "Value_Watch": "Value-Watch",
        "Trap": "Value trap"})
    st.dataframe(g, hide_index=True, width="stretch", column_config={
        "Median value": st.column_config.NumberColumn(format="%.2f"),
        "Cheap share %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
    })

    order = g["Bucket"].tolist()
    fig = px.strip(has, x="z", y="bucket", color="verdict", color_discrete_map=VERDICT_COLORS,
                   category_orders={"bucket": order, "verdict": VERDICTS},
                   hover_data={"symbol": True, "metric_value": ":.2f", "z": ":+.2f", "bucket": False})
    fig.add_vline(x=th.cheap_z, line_dash="dash", line_color="#888",
                  annotation_text=f"cheap ≤ {th.cheap_z}", annotation_position="top")
    fig.update_layout(height=max(420, 26 * len(order)), margin=dict(l=8, r=8, t=30, b=8),
                      xaxis_title="Z vs sector peers (negative = cheaper)", yaxis_title=None,
                      legend=dict(orientation="h", y=-0.12), xaxis=dict(range=[-4, 4]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Z is clipped to ±4 on this chart. Buckets with fewer than "
               f"{th.min_peers} stocks on their metric are compared against all stocks on that metric.")


def tab_audit(s: pd.DataFrame, th: Thresholds):
    opts = s.sort_values("symbol")
    label = opts["symbol"] + " — " + opts["company"]
    default = 0
    if "audit_symbol" in st.session_state and st.session_state["audit_symbol"] in label.values:
        default = int(np.where(label.values == st.session_state["audit_symbol"])[0][0])
    pick = st.selectbox("Stock", label.tolist(), index=default, key="audit_symbol")
    r = opts.iloc[label.tolist().index(pick)]

    color = VERDICT_COLORS.get(r["verdict"], "#888")
    st.markdown(f"### {r['symbol']} · <span style='color:{color}'>{r['verdict']}</span>", unsafe_allow_html=True)
    st.markdown(f"{r['verdict_reason']}  \n<span class='small-note'>{r['company']} · {r['bucket']} · "
                f"{r['cap_band']} cap · price ₹{fmt_num(r['price'])} · mcap ₹{fmt_num(r['mcap_cr'], 0)} cr"
                f"</span>", unsafe_allow_html=True)

    st.markdown("#### 1. Valuation")
    mname = METRICS[r["metric"]][0]
    val = pd.DataFrame([
        ["Metric used", mname + (" (loss-maker, speculative)" if r["speculative"] else "")],
        ["Value", fmt_num(r["metric_value"])],
        ["Compared within", f"{r['z_group']} ({int(r['peers'])} stocks)"],
        ["Peer median", fmt_num(r.get("group_median"))],
        ["Z vs peers", fmt_num(r["z"]) + f"  → {r['valuation']} (cheap if ≤ {th.cheap_z})"],
        ["Own history", (f"cheaper than {r['own_hist_cheaper_than_pct']:.0f}% of the last "
                         f"{int(r['own_hist_years'])} fiscal year-ends") if pd.notna(r['own_hist_cheaper_than_pct'])
         else "fewer than 4 years of data"],
        ["EV built from", r["ev_source"]],
        ["EBIT basis", r["ebit_basis"] or "—"],
        ["Data flag", r["metric_flag"] or "none"],
    ], columns=["Item", "Value"])
    st.dataframe(val, hide_index=True, width="stretch")

    st.markdown(f"#### 2. Quality — {r['quality']}")
    fin = bool(r["is_financial"])
    if fin:
        rows = [
            ["ROE (core)", fmt_pct(r["roe"]), f"≥ {th.roe_min}%", r["q_core"]],
            ["ROA", fmt_pct(r["roa"], 2), f"≥ {th.roa_min}%", r["q_s1"]],
            ["Book value CAGR", fmt_pct(r["bv_cagr"]), f"≥ {th.bv_cagr_min}%", r["q_s2"]],
        ]
        rule = "ROE must pass, plus both supporting checks."
    else:
        med = r.get("roce_bucket_median")
        roce_rule = f"≥ {th.roce_min}%" + (f" or ≥ sector median {fmt_pct(med)}" if th.roce_or_sector_median else "")
        rows = [
            ["ROCE (core)", fmt_pct(r["roce"]), roce_rule, r["q_core"]],
            ["Debt/Equity", fmt_num(r["de"]), f"≤ {th.de_max}", r["q_s1"]],
            ["Interest cover", "no debt" if r["no_interest"] else fmt_num(r["icr"], 1) + "×",
             f"≥ {th.icr_min}×", r["q_s2"]],
            ["CFO ÷ PAT, 3-year", fmt_num(r["cfo_pat"]), f"≥ {th.cfo_pat_min}", r["q_s3"]],
        ]
        rule = "ROCE must pass, plus two of the three supporting checks."
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Value", "Threshold", "Result"]),
                 hide_index=True, width="stretch")
    st.caption(rule + " 'no data' is not a fail; two or more 'no data' checks block Prime.")

    st.markdown(f"#### 3. Forward bias — score {int(r['fwd_score']):+d} ({r['fwd_coverage']})")
    tl = lambda c: pd.notna(r.get(c))
    frows = [
        ["Forward EPS growth", fmt_pct(r["tl_eps_g"]) + " (Trendlyne)" if tl("tl_eps_g") else fmt_pct(r["fwd_eps_g"]),
         f"+1 ≥ {th.eps_g_pos}%, −1 < {th.eps_g_neg}%", int(r["f_eps"])],
        ["Target upside", fmt_pct(r["tl_upside"]) + " (Trendlyne)" if tl("tl_upside") else fmt_pct(r["upside"]),
         f"+1 ≥ {th.upside_pos}%, −1 < {th.upside_neg}%", int(r["f_up"])],
        ["Analyst rating", (f"net upgrades {r['tl_upgrade']:+.0f} (Trendlyne)" if tl("tl_upgrade")
                            else fmt_num(r["rec_mean"])), f"+1 ≤ {th.rec_pos}, −1 ≥ {th.rec_neg}", int(r["f_rec"])],
        ["Forward PEG", fmt_num(r["tl_peg"]) + " (Trendlyne)" if tl("tl_peg") else fmt_num(r["fwd_peg"]),
         f"+1 ≤ {th.peg_pos}, −1 > {th.peg_neg}", int(r["f_peg"])],
    ]
    st.dataframe(pd.DataFrame(frows, columns=["Signal", "Value", "Rule", "Score"]),
                 hide_index=True, width="stretch")
    n = r["n_analysts"]
    st.caption(f"Analysts covering: {int(n) if pd.notna(n) else 0}. Below {th.min_analysts}, yfinance "
               "signals score 0 (neutral, not negative).")
    if tl("tl_grid") or tl("tl_grid_v2"):
        st.caption(f"Trendlyne FwdGridScore {fmt_num(r['tl_grid'])} · V2 {fmt_num(r['tl_grid_v2'])} "
                   "(information only, not in the score).")

    with st.expander("Raw inputs and data dates"):
        raw_cols = ["ebit", "revenue", "net_income", "debt", "cash", "equity", "assets", "cur_liab",
                    "interest", "cfo_3y", "ni_3y", "fwd_eps", "ttm_eps", "target_mean", "fy_end",
                    "info_fetched_at", "fund_fetched_at", "yf_industry"]
        rr = pd.DataFrame({"Field": raw_cols,
                           "Value": [r.get(c) if not isinstance(r.get(c), float)
                                     else (f"{r.get(c):,.0f}" if abs(r.get(c)) > 1000 else fmt_num(r.get(c)))
                                     for c in raw_cols]})
        rr["Value"] = rr["Value"].astype(str)
        st.dataframe(rr, hide_index=True, width="stretch")
        st.caption("Amounts in ₹ as reported by yfinance. Source: Yahoo Finance via yfinance.")


def tab_health(m: pd.DataFrame, meta: dict):
    if meta:
        c = st.columns(4)
        c[0].metric("Last run", meta.get("run_at_ist", "—"))
        c[1].metric("Stocks fetched", f"{meta.get('info_ok', 0)}/{meta.get('symbols', 0)}")
        c[2].metric("Statements refreshed", meta.get("fundamentals_refreshed", 0))
        c[3].metric("Run time", f"{meta.get('seconds', 0)//60} min")
        st.caption(f"Universe: {meta.get('universe', '')}, {meta.get('universe_source', '')}")
        if meta.get("info_failed"):
            st.warning("No quote data for: " + ", ".join(meta["info_failed"]))
        if meta.get("info_stale"):
            st.info(f"{len(meta['info_stale'])} stocks use the previous run's price and analyst data "
                    "(Yahoo throttled them tonight): " + ", ".join(meta["info_stale"][:40])
                    + (" …" if len(meta["info_stale"]) > 40 else ""))

    fields = {
        "Price": "price", "Market cap": "mcap", "EBIT": "ebit", "Total debt": "debt", "Cash": "cash",
        "Equity": "equity", "ROCE": "roce", "ROE": "roe", "CFO/PAT (3y)": "cfo_pat",
        "Forward EPS": "fwd_eps", "Target price": "target_mean", "Analyst count": "n_analysts",
        "Valuation metric": "metric_value",
    }
    rows = []
    for name, col in fields.items():
        for band in ["Large", "Mid", "Small"]:
            sub = m[m["cap_band"] == band]
            if col in ("ebit", "roce", "cfo_pat"):
                sub = sub[~sub["is_financial"]]
            rows.append([name, band, 100 * sub[col].notna().mean() if len(sub) else np.nan])
    cov = pd.DataFrame(rows, columns=["Field", "Band", "Coverage"]).pivot(index="Field", columns="Band", values="Coverage")
    cov = cov.reindex(list(fields))[["Large", "Mid", "Small"]].reset_index()
    st.markdown("**Coverage by market-cap band** (EBIT, ROCE and CFO/PAT counted for non-financials only)")
    st.dataframe(cov, hide_index=True, width="stretch", column_config={
        b: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%") for b in ["Large", "Mid", "Small"]
    })
    cov_an = (m["n_analysts"].fillna(0) >= 3).groupby(m["cap_band"]).mean() * 100
    st.caption("Stocks with 3+ analysts: " + " · ".join(f"{b} {cov_an.get(b, 0):.0f}%" for b in ["Large", "Mid", "Small"]))

    errs = m[(m["fund_error"] != "") | (m["metric_flag"].str.startswith("rejected"))]
    if len(errs):
        with st.expander(f"{len(errs)} stocks with fetch errors or rejected values"):
            st.dataframe(errs[["symbol", "fund_error", "metric", "metric_flag"]], hide_index=True,
                         width="stretch")


# ----------------------------------------------------------------------- main

def main():
    st.title("Nifty 500 Value + Forward Scanner")
    st.markdown("<span class='small-note'>Cheap vs sector peers → quality gate → forward analyst bias. "
                "Entry timing stays with VMS·ValQ on TradingView.</span>", unsafe_allow_html=True)

    if not os.path.exists(METRICS_FILE):
        st.error("No scan file yet. Run the GitHub Actions workflow 'Nightly scan' once "
                 "(Actions tab → Nightly scan → Run workflow), or locally: `python run_scan.py`.")
        st.stop()

    th, tl = sidebar()
    m = load_metrics(os.path.getmtime(METRICS_FILE))
    meta = load_meta()
    with st.spinner("Scoring…"):
        s = score(m, th, tl)

    st.caption(f"Data as of {meta.get('run_at_ist', 'unknown')} · {len(s)} stocks"
               + (" · Trendlyne CSV applied" if tl is not None and len(tl) else ""))

    t1, t2, t3, t4 = st.tabs(["Shortlist", "Sector map", "Stock audit", "Data health"])
    with t1:
        tab_shortlist(s)
    with t2:
        tab_sectors(s, th)
    with t3:
        tab_audit(s, th)
    with t4:
        tab_health(m, meta)

    st.divider()
    st.markdown("<span class='small-note'>© Dr Shantanu Samanta · Data: Yahoo Finance via yfinance, "
                "NSE index constituents, optional Trendlyne export. A screening tool for research, "
                "not investment advice.</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
