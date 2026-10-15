"""Default thresholds and fixed settings for the scanner.

Every number in Thresholds can be changed from the app's sidebar. The nightly
job scores its archive file with these defaults.
"""
from dataclasses import dataclass, asdict, fields

DATA_DIR = "data"
UNIVERSE_CSV = f"{DATA_DIR}/nifty500.csv"
RAW_CACHE = f"{DATA_DIR}/cache/raw.parquet"
METRICS_FILE = f"{DATA_DIR}/metrics_latest.parquet"
RUN_META = f"{DATA_DIR}/run_meta.json"
ARCHIVE_DIR = f"{DATA_DIR}/archive"

NIFTY500_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]

# Statements change quarterly, so they are re-fetched only when older than this.
FUNDAMENTALS_MAX_AGE_DAYS = 7

# Metric keys -> display names and direction (True = higher value is cheaper).
METRICS = {
    "ebit_yield": ("EBIT/EV %", True),
    "pb": ("P/B", False),
    "pe": ("P/E", False),
    "ev_sales": ("EV/Sales", False),
}

# Sanity bounds: values outside are treated as bad data, not as signals.
METRIC_BOUNDS = {
    "ebit_yield": (-50.0, 50.0),
    "pb": (0.0, 100.0),
    "pe": (0.0, 500.0),
    "ev_sales": (0.0, 200.0),
}

VERDICTS = ["Prime", "Value-Watch", "Value trap risk", "Not cheap", "Data: Thin"]


@dataclass
class Thresholds:
    # Valuation
    cheap_z: float = -0.5          # Z at or below this = cheap
    expensive_z: float = 0.5       # Z above this = expensive (between = fair)
    min_peers: int = 8             # smaller groups pool into all stocks on the same metric
    winsor_pct: float = 2.0        # winsorise at this and (100 - this) percentile

    # Quality, non-financials (percent units where relevant)
    roce_min: float = 12.0
    roce_or_sector_median: bool = True   # also pass if ROCE >= bucket median
    de_max: float = 1.0
    icr_min: float = 3.0
    cfo_pat_min: float = 0.7

    # Quality, financials
    roe_min: float = 12.0
    roa_min: float = 1.0
    bv_cagr_min: float = 10.0

    # Forward bias
    min_analysts: int = 3
    eps_g_pos: float = 12.0        # forward EPS growth % for +1
    eps_g_neg: float = 0.0         # below this = -1
    upside_pos: float = 15.0       # target upside % for +1
    upside_neg: float = 0.0
    rec_pos: float = 2.2           # recommendationMean at or below = +1 (1 strong buy .. 5 sell)
    rec_neg: float = 3.3           # at or above = -1
    peg_pos: float = 1.0
    peg_neg: float = 2.0
    prime_min_fwd: int = 1         # forward score needed for Prime

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})
