import numpy as np
import pandas as pd
import pytest

from scanner.config import Thresholds
from scanner.metrics import build_metrics, classify
from scanner.scoring import parse_trendlyne, robust_z, score


def test_classify():
    assert classify("Financial Services", "Banks - Regional") == ("Banks", "pb", True)
    assert classify("Financial Services", "Insurance - Life") == ("Insurers", "pb", True)
    assert classify("Financial Services", "Asset Management") == ("AMCs, brokers & exchanges", "pe", True)
    assert classify("Financial Services", "Credit Services") == ("NBFCs & HFCs", "pb", True)
    assert classify("Capital Goods", "Specialty Industrial Machinery") == ("Capital Goods", "ebit_yield", False)


def test_robust_z_median_zero_and_scale():
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    z = robust_z(x, winsor_pct=0)
    assert z.iloc[4] == pytest.approx(0.0)
    # MAD of 1..9 around 5 is 2 -> scale 2.9652
    assert z.iloc[8] == pytest.approx(4 / (1.4826 * 2))


def test_robust_z_resists_outlier():
    x = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 1000], dtype=float)
    z = robust_z(x, winsor_pct=2)
    assert abs(z.iloc[4]) < 0.01
    assert np.isfinite(z).all()


def _universe(n=10):
    rows = [{"symbol": f"NF{i}", "company": f"Nonfin {i}", "nse_industry": "Capital Goods"} for i in range(n)]
    rows += [{"symbol": f"BK{i}", "company": f"Bank {i}", "nse_industry": "Financial Services"} for i in range(n)]
    return pd.DataFrame(rows)


def _raw(n=10):
    rows = []
    for i in range(n):
        # NF0 has the highest EBIT/EV (cheapest); good quality; strong forward data
        rows.append(dict(symbol=f"NF{i}", price=100.0, shares=1e7, ebit=(20 - i) * 1e7, debt=1e8, cash=0.0,
                         equity=5e8, equity_prev=4.5e8, assets=1e9, cur_liab=2e8, interest=1e7,
                         net_income=(15 - i) * 1e7, cfo_3y=4e8, ni_3y=3.5e8, yf_industry="Machinery",
                         fwd_eps=12.0, ttm_eps=10.0, target_mean=125.0, n_analysts=10, rec_mean=1.8,
                         equity_old=3e8, equity_years=3))
        rows.append(dict(symbol=f"BK{i}", price=100.0, shares=1e7, equity=(4 + i) * 1e8, equity_prev=(3.6 + i) * 1e8,
                         assets=1e10, net_income=2e8, debt=5e9, cash=1e9, yf_industry="Banks - Regional",
                         fwd_eps=11.0, ttm_eps=10.0, target_mean=105.0, n_analysts=1, rec_mean=2.0,
                         equity_old=2e8 + i * 1e8, equity_years=3))
    return pd.DataFrame(rows)


def test_end_to_end_verdicts():
    m = build_metrics(_raw(), _universe())
    s = score(m, Thresholds())
    nf0 = s.set_index("symbol").loc["NF0"]
    assert nf0["metric"] == "ebit_yield"
    assert nf0["z"] < 0                      # highest EBIT/EV -> cheapest -> negative Z
    assert nf0["verdict"] == "Prime"
    nf9 = s.set_index("symbol").loc["NF9"]
    assert nf9["verdict"] == "Not cheap"
    # Banks: 1 analyst -> no coverage -> never Prime
    banks = s[s["bucket"] == "Banks"]
    assert (banks["fwd_coverage"] == "No coverage").all()
    assert "Prime" not in set(banks["verdict"])
    cheap_banks = banks[banks["z"] <= -0.5]
    assert len(cheap_banks) > 0


def test_quality_fail_makes_value_trap():
    raw = _raw()
    raw.loc[raw["symbol"] == "NF0", ["ebit", "net_income"]] = [20e7, 1e6]
    raw.loc[raw["symbol"] == "NF0", ["assets", "cur_liab"]] = [9e9, 1e8]   # ROCE ~2%
    m = build_metrics(raw, _universe())
    s = score(m, Thresholds(roce_or_sector_median=False)).set_index("symbol")
    assert s.loc["NF0", "q_core"] == "fail"
    assert s.loc["NF0", "verdict"] == "Value trap risk"


def test_trendlyne_override():
    m = build_metrics(_raw(), _universe())
    tl_raw = pd.DataFrame({"NSE Code": ["BK0"], "FwdEPSGrowthTTM": ["25%"], "FwdUpside12M": [30],
                           "FwdAnalystUpgrade3M": [2], "FwdPEGTTM": [0.6]})
    tl, note = parse_trendlyne(tl_raw)
    assert "FwdEPSGrowthTTM" in note
    s = score(m, Thresholds(), tl).set_index("symbol")
    assert s.loc["BK0", "fwd_coverage"] == "Trendlyne"
    assert s.loc["BK0", "fwd_score"] == 4


def test_loss_maker_never_prime():
    raw = _raw()
    raw.loc[raw["symbol"] == "NF0", "ebit"] = -5e7
    raw.loc[raw["symbol"] == "NF0", "revenue"] = 1e9
    m = build_metrics(raw, _universe())
    s = score(m, Thresholds(min_peers=1)).set_index("symbol")
    assert s.loc["NF0", "metric"] == "ev_sales"
    assert s.loc["NF0", "verdict"] != "Prime"
