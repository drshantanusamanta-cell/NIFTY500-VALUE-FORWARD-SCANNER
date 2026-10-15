"""Nifty 500 universe: NSE constituent file, with the committed copy as fallback."""
import io
import logging

import pandas as pd
import requests

from .config import NIFTY500_URLS, UNIVERSE_CSV

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/octet-stream,*/*",
}


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={
        "Company Name": "company",
        "Industry": "nse_industry",
        "Symbol": "symbol",
        "Series": "series",
        "ISIN Code": "isin",
    })
    need = {"company", "nse_industry", "symbol"}
    if not need.issubset(df.columns):
        raise ValueError(f"Unexpected Nifty 500 columns: {list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df = df[df["symbol"] != ""].drop_duplicates("symbol")
    return df[["symbol", "company", "nse_industry"] + [c for c in ("isin",) if c in df.columns]]


def refresh_universe(path: str = UNIVERSE_CSV, timeout: int = 30) -> tuple[pd.DataFrame, str]:
    """Download the latest list; on any failure use the committed copy.

    Returns the universe and a short note on where it came from.
    """
    for url in NIFTY500_URLS:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            df = _tidy(pd.read_csv(io.StringIO(r.text)))
            if len(df) < 450:
                raise ValueError(f"only {len(df)} rows")
            raw = pd.read_csv(io.StringIO(r.text))
            raw.to_csv(path, index=False)
            return df, f"downloaded from {url}"
        except Exception as e:  # network blocks, format changes
            log.warning("Universe download failed from %s: %s", url, e)
    return load_universe(path), f"used saved copy ({path})"


def load_universe(path: str = UNIVERSE_CSV) -> pd.DataFrame:
    return _tidy(pd.read_csv(path))
