# SEC EDGAR point-in-time research data

The EDGAR package stores issuer filing history in `datasets/quantlab/edgar/edgar.sqlite3` under the project root.
The research join uses a read-only SQLite connection. It performs no HTTP requests and has no broker or trading
dependencies.

## Download and update

Set the process environment variable `LAT_SEC_USER_AGENT` to `"Name email"`, replacing both parts with your own
contact details as required by SEC fair-access policy. The fetch exits with an error if the variable is unset.
It does not read contact information from application settings, the Keychain, or a config file.

```bash
uv run --frozen python -m services.market_data.edgar fetch AAPL 0000789019 --start 2014-01-01 --end 2025-11-20
```

Use CIKs for former or ambiguous tickers. Select the issuer and filing-date interval explicitly. Use several years
of prior filings to warm up seasonal EPS and year-over-year accounting features. The downloader uses the same
eight-request-per-second budget across both SEC origins and fetches sharing that project's cache. It retries
transient failures with exponential backoff and honors `Retry-After`. The local cache reuses current submissions
and companyfacts for one hour, the ticker directory for one day, and archived submissions and filing documents
indefinitely. Repeat the same command to resume. A filing enters the store once its required payload has parsed.

## Data model and timing

- **Identity.** An accession belongs to one issuer CIK. Historical symbol evidence comes from visible inline XBRL
  `dei:TradingSymbol` values on 10-K/10-Q covers and issuer symbols on Form 4. Those observations apply forward from
  their filing's available session. The current `company_tickers.json` directory applies forward from its cached
  retrieval time. It does not assign current tickers to historical filings. One CIK can retain multiple share
  classes from cover evidence. A ticker claimed concurrently by multiple CIKs is left unmapped. A name or ticker
  change between observed disclosures remains unknown until the next observation, and older issuers without
  dated symbol evidence may have no historical coverage.
- **Availability.** Filings retain acceptance time, filed date, form, accession, and reporting period. XBRL facts
  carry their own period start and end, and Form 4 rows carry transaction dates. Fetch uses the Eastern
  acceptance timestamp on the accession's archived filing detail page. The first XNYS session opening at or
  after acceptance is `available_at` and is stored in UTC. Pre-open acceptance can join that session.
  Acceptance after its open waits until the next session. The New York calendar handles weekends, holidays,
  and daylight saving. A missing archive acceptance timestamp stops that issuer's fetch with the accession
  in the error. A date-only filing label never sets availability.
- **Versions.** XBRL concepts are filtered by periodic accession, unit, and period. If a 10-K/A, 10-Q/A, or later
  comparative statement revises a period, its own accession and availability control when the new value is used.
  An exact duplicate ingest is idempotent. For Form 4/A, the original submission date and reporting owner set
  must identify one original filing. A linked amendment replaces the original's rows from its own availability.
  An ambiguous amendment is retained in the store and excluded from feature aggregation.
- **Earnings markers.** The issuer submissions index supplies 8-K item 2.02 and periodic report filing dates.
  Both count as public events, so an announcement and a later periodic filing can each reset the event clock.
  Amendments do not reset it. EPS becomes available only from the periodic accession that reports it. Item 2.02
  recognition does not infer an EPS value from the 8-K exhibit.
- **Form 4.** Non-derivative transaction codes P and S capture disclosed purchases and sales. Grants, exercises,
  and derivative rows are excluded. Shares, reported price, owner CIK, and relationship roles are retained. A
  jointly filed row contributes its shares once and each distinct reporting owner can contribute to cluster count.
  The codes can also include private trades, which the structured code alone cannot distinguish.

## Daily features

`services.quantlab.events.attach_events` joins by dated ticker evidence and session date. Unknown values remain
`NaN`. The factor DSL exposes the following zero-argument functions.

| Function | Definition |
| --- | --- |
| `sue()` | Latest quarterly EPS minus the same fiscal quarter's EPS one year earlier, divided by the sample standard deviation of the previous four to eight seasonal EPS changes. Annual EPS minus the first three direct quarterly EPS values estimates Q4 when direct Q4 EPS is absent. |
| `earnings_announcement()` | One on the first available session of an earnings event and zero on subsequent covered sessions. |
| `earnings_window()` | One during the announcement session and the following four sessions. |
| `earnings_drift_age()` | Number of trading sessions since the latest available event. |
| `accruals()` | Trailing four-quarter net income minus operating cash flow, scaled by average current and year-ago assets. |
| `asset_growth()` | Fiscal-period assets divided by year-ago assets minus one. |
| `gross_profitability()` | Trailing four-quarter gross profit, or revenue minus cost of revenue, divided by period-end assets. |
| `net_share_issuance()` | Fiscal-period outstanding shares divided by year-ago shares minus one. Split-driven changes in reported shares can affect this feature. |
| `insider_net_buy_ratio()` | Net P/S shares divided by total P/S shares over the trailing 30 calendar days by transaction date, using only filings available at the session open. |
| `insider_cluster_buy()` | One when at least three distinct insiders disclosed qualifying purchases in that same window. |

Fiscal quarters are matched by end date with a tolerance for 52/53-week differences. Cumulative accounting flows
can be differenced from a matching fiscal-year start to reconstruct quarters. XBRL coverage depends on available
standard concepts and accepted units. Issuer-level share and accounting values can be common to more than one
listed class. A covered insider window with zero qualifying observed transactions is zero, which can reflect
sparse local filing history. Current consensus, guidance text, broker and alternative data are outside this dataset.

## Controlled comparison

`L_events` uses family K's purge through the label's final execution open, expanding annual folds, out-of-sample
edge calibration, five-session horizon, cost-aware allocation, simulator, and development gates. The four fixed
ridge arms are price-only, price plus accounting fundamentals, price plus insider activity, and all features
including earnings. EDGAR fields are cross-sectionally ranked. Missing fields map to neutral scores in the
comparison so each arm starts with the same price-defined universe and training eligibility.

The family records all four variants in the normal append-only trial ledger and reports matched out-of-sample
net CAGR, Sharpe, cost drag, and their differences from price-only in
`experiments/quantlab/runs/<run_id>/events_comparison.json`. The input panel is sliced before any EDGAR feature
join. The evidence fingerprint is based on the development prices and aligned development features. Existing
promotion gates and operator requirements apply to this family.
