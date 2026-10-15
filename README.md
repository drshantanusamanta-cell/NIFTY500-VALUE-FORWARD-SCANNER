# Nifty 500 Value + Forward Scanner

Finds Nifty 500 stocks that are cheap against their own sector, pass a quality check, and have non-negative forward analyst bias. It produces a ranked shortlist you can export straight to a TradingView watchlist, where VMS·ValQ times the entry.

© Dr Shantanu Samanta · A screening tool for research, not investment advice.

## How it works

1. **Valuation.** Each stock is valued on one metric chosen by its sector:
   - Non-financials: EBIT/EV
   - Banks, NBFCs and insurers: P/B
   - AMCs, brokers and exchanges: P/E
   - Loss-makers: EV/Sales, marked speculative

   The metric is turned into a robust Z-score (median and MAD) within the stock's NSE sector bucket. Z ≤ −0.5 counts as cheap.
2. **Quality gate.**
   - Non-financials: ROCE must pass, plus two of D/E, interest cover and 3-year CFO/PAT.
   - Financials: ROE must pass, plus both ROA and book-value CAGR.
3. **Forward bias.** Four signals each score −1, 0 or +1, giving a total from −4 to +4: forward EPS growth, target upside, analyst rating and forward PEG. With fewer than 3 analysts, a stock scores 0 (neutral). An uploaded Trendlyne CSV overrides yfinance for the stocks it covers.
4. **Verdict.** Each stock gets one:
   - **Prime**: cheap, quality passed, forward score +1 or more
   - **Value-Watch**: cheap and sound, but no forward lift yet
   - **Value trap risk**: cheap, but quality fails or estimates are falling
   - **Not cheap**
   - **Data: Thin**

   Prime and Value-Watch are ranked cheapest Z first.

A GitHub Actions job runs the scan at **7:15 pm IST, Monday to Friday**. It commits `data/metrics_latest.parquet`, and the Streamlit app only reads that file, so it opens fast and never hits Yahoo rate limits. Financial statements are re-fetched once a week; prices and analyst data every run.

## What's in the repo

| File | What it does |
| --- | --- |
| `app.py` | Streamlit app: Shortlist, Sector map, Stock audit, Data health tabs; thresholds in the sidebar |
| `run_scan.py` | The nightly job: universe, fetch, metrics, archive |
| `scanner/config.py` | Default thresholds (all editable in the sidebar) |
| `scanner/universe.py` | Downloads the Nifty 500 list from NSE; falls back to `data/nifty500.csv` |
| `scanner/fetch.py` | yfinance fetching with retries and caching |
| `scanner/metrics.py` | EBIT/EV, P/B, ROCE, ROE, D/E, forward fields, sector buckets |
| `scanner/scoring.py` | Z-scores, gates, forward score, verdicts, Trendlyne merge |
| `.github/workflows/nightly.yml` | The schedule |
| `tests/` | Unit tests for the maths (`pytest`) |
| `data/` | Universe, latest scan, cache, and a dated archive of every scan |

## What you need

- A GitHub account (free)
- A Streamlit Community Cloud account (free, sign in with GitHub at share.streamlit.io)
- No API keys. yfinance and the NSE file are free and need no login.

## Deploy (about 10 minutes)

### Step 1: Put the code on GitHub
1. On github.com, click **New** to create a repository. Name it `nifty500-value-forward-scanner`. It can be private or public, but **do not** tick "Add a README".
2. On the empty repo page, click **uploading an existing file**.
3. Unzip the download on your computer. Drag **everything inside** the `nifty500-value-forward-scanner` folder into the browser, including the `.github`, `.streamlit` and `data` folders, then click **Commit changes**.
   - If your file browser hides folders starting with a dot, upload `.github/workflows/nightly.yml` and `.streamlit/config.toml` separately with **Add file → Create new file**, typing the full path as the file name and pasting in the contents.

### Step 2: Let the nightly job save results
1. In the repo, open **Settings → Actions → General**.
2. Under **Workflow permissions**, choose **Read and write permissions** and click **Save**.
3. Open the **Actions** tab. If asked, click **I understand my workflows, go ahead and enable them**.
4. Click **Nightly scan → Run workflow** once to check it works. A full run takes roughly 10–25 minutes.

### Step 3: Deploy the app
1. Go to share.streamlit.io and click **Create app → Deploy a public app from GitHub**.
2. Pick your repo, branch `main`, main file `app.py`, and click **Deploy**.
3. Each nightly run commits new data, and the app picks it up automatically.

## Run it on your own computer (optional)

```bash
pip install -r requirements.txt
python run_scan.py --limit 40   # quick test on 40 stocks; omit --limit for all 500
streamlit run app.py
python -m pytest -q             # unit tests
```

## Using the Trendlyne CSV

In the sidebar, upload your forward-screener export. The app reads `NSE Code` (or `Symbol`/`Ticker`) plus any of these columns: `FwdEPSGrowthTTM`, `FwdUpside12M`, `FwdAnalystUpgrade3M`, `FwdPEGTTM`, `FwdGridScore`, `FwdGridScore_V2`. Its values replace yfinance values for the signals it covers. The grid scores are shown in the Stock audit tab but do not enter the score, so the same analyst data isn't counted twice.

## Known limits

- yfinance statements for some smaller companies are patchy. The Data health tab shows coverage, and the Stock audit tab shows every raw input.
- P/B for insurers stands in for P/EV, which isn't in free data.
- Cheapness is measured against today's sector peers. The stock's own history (4–5 fiscal year-ends from yfinance) is shown for information only.
- There is no backtest yet. Every scan is archived in `data/archive/`, so a forward test builds up from day one.
- Yahoo occasionally rate-limits GitHub's servers. If a run fails, the previous scan stays live; re-run it from the Actions tab.
