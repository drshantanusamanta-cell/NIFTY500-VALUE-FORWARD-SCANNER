"""Nightly scan: refresh universe, fetch yfinance data, compute metrics, archive a scored copy.

Usage:
    python run_scan.py                 # full Nifty 500 run
    python run_scan.py --limit 40      # quick test on the first 40 symbols
    python run_scan.py --force-fundamentals
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from scanner.config import (ARCHIVE_DIR, FUNDAMENTALS_MAX_AGE_DAYS, METRICS_FILE, RAW_CACHE,
                            RUN_META, Thresholds)
from scanner.fetch import fetch_fundamentals, fetch_info, fetch_many
from scanner.metrics import build_metrics
from scanner.scoring import score
from scanner.universe import refresh_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_scan")

IST = timezone(timedelta(hours=5, minutes=30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N symbols (testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force-fundamentals", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.dirname(RAW_CACHE), exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    universe, uni_note = refresh_universe()
    if args.limit:
        universe = universe.head(args.limit)
    symbols = universe["symbol"].tolist()
    log.info("Universe: %d symbols (%s)", len(symbols), uni_note)

    cache = pd.read_parquet(RAW_CACHE) if os.path.exists(RAW_CACHE) else pd.DataFrame(columns=["symbol"])

    # 1) Quote + forward fields: every run, with slow retries for throttled symbols
    info = fetch_many(symbols, fetch_info, workers=args.workers, label="info")
    for attempt in (1, 2):
        bad = info.loc[info["info_error"].fillna("") != "", "symbol"].tolist()
        if not bad:
            break
        log.info("Quote retry pass %d for %d symbols after a pause", attempt, len(bad))
        time.sleep(60 * attempt)
        retry = fetch_many(bad, fetch_info, workers=1, label=f"info-retry{attempt}")
        info = pd.concat([info[~info["symbol"].isin(bad)], retry], ignore_index=True)

    # Still failing: reuse the last good quote from the cache and mark it stale
    info["info_stale"] = False
    if len(cache) and "price" in cache:
        good_old = cache[cache["price"].notna()].set_index("symbol")
        info_cols = [c for c in info.columns if c not in ("symbol", "info_stale")]
        fixed = []
        for i, r in info.iterrows():
            if r.get("info_error") and r["symbol"] in good_old.index:
                old = good_old.loc[r["symbol"]]
                for c in info_cols:
                    if c in old.index:
                        info.at[i, c] = old[c]
                info.at[i, "info_stale"] = True
                fixed.append(r["symbol"])
        if fixed:
            log.info("Using previous quotes for %d symbols: %s", len(fixed), ", ".join(fixed[:20]))

    # 2) Statements: only when missing or older than the max age
    fund_cols = [c for c in cache.columns if c != "symbol" and not c.startswith("info_")
                 and c not in info.columns]
    old_fund = cache[["symbol"] + fund_cols] if fund_cols else pd.DataFrame(columns=["symbol"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=FUNDAMENTALS_MAX_AGE_DAYS)
    if args.force_fundamentals or "fund_fetched_at" not in old_fund:
        stale = symbols
    else:
        fetched = pd.to_datetime(old_fund.set_index("symbol")["fund_fetched_at"], errors="coerce", utc=True)
        errored = old_fund.set_index("symbol").get("fund_error", pd.Series(dtype=str)).fillna("")
        stale = [s for s in symbols
                 if s not in fetched.index or pd.isna(fetched.get(s)) or fetched[s] < cutoff
                 or errored.get(s, "") != ""]
    log.info("Fundamentals to refresh: %d of %d", len(stale), len(symbols))
    new_fund = fetch_many(stale, fetch_fundamentals, workers=args.workers, label="fundamentals") \
        if stale else pd.DataFrame(columns=["symbol"])

    # Slow retry pass for anything Yahoo throttled (empty or failed statements)
    for attempt in (1, 2):
        if new_fund.empty or "fund_error" not in new_fund:
            break
        bad = new_fund.loc[new_fund["fund_error"].fillna("") != "", "symbol"].tolist()
        if not bad:
            break
        log.info("Retry pass %d for %d symbols after a pause", attempt, len(bad))
        time.sleep(60 * attempt)
        retry = fetch_many(bad, fetch_fundamentals, workers=1, label=f"retry{attempt}")
        new_fund = pd.concat([new_fund[~new_fund["symbol"].isin(bad)], retry], ignore_index=True)
    keep_old = old_fund[~old_fund["symbol"].isin(new_fund["symbol"])] if len(old_fund) else old_fund
    fund = pd.concat([keep_old, new_fund], ignore_index=True)

    raw = info.merge(fund, on="symbol", how="left")
    # Keep cached rows for symbols outside this run (e.g. --limit) so the cache never shrinks
    if len(cache):
        others = cache[~cache["symbol"].isin(raw["symbol"])]
        raw_all = pd.concat([others, raw], ignore_index=True)
    else:
        raw_all = raw
    raw_all.to_parquet(RAW_CACHE, index=False)

    metrics = build_metrics(raw, universe)
    metrics.to_parquet(METRICS_FILE, index=False)

    scored = score(metrics, Thresholds())
    today = datetime.now(IST).strftime("%Y-%m-%d")
    keep = ["symbol", "company", "bucket", "metric", "metric_value", "z", "valuation", "quality",
            "fwd_score", "fwd_coverage", "verdict", "rank", "price", "mcap_cr"]
    scored[keep].to_parquet(f"{ARCHIVE_DIR}/scan_{today}.parquet", index=False)

    info_fail = info.loc[info["info_error"].fillna("") != "", "symbol"].tolist()
    meta = {
        "run_at_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        "universe": "Nifty 500",
        "universe_source": uni_note,
        "symbols": len(symbols),
        "info_ok": len(symbols) - len(info_fail),
        "info_failed": info_fail,
        "info_stale": info.loc[info["info_stale"] == True, "symbol"].tolist(),  # noqa: E712
        "fundamentals_refreshed": len(stale),
        "verdict_counts": scored["verdict"].value_counts().to_dict(),
        "seconds": round(time.time() - t0),
    }
    with open(RUN_META, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Done in %ss: %s", meta["seconds"], meta["verdict_counts"])


if __name__ == "__main__":
    main()
