# forecast-service

Nightly job that fills in `sales_forecast` (and `sales_forecast_exclusions`), read by
`react-finance-dashboard`'s Sales Forecast tab (`GET /api/sales-forecast`). Not a
long-running worker - it's a small FastAPI app with one real endpoint, `POST /run`, that
does one full catalog pass and returns. Something else (n8n, cron, curl) has to actually
call it on a schedule; this service has no scheduler of its own.

## What it does

Per SKU, over roughly the last 2 years of `net revenue` (order revenue net of discounts
**and** refunds - the same figure the dashboard calls "net revenue" everywhere else):

1. **Classify growth stage** - `new` / `growth` / `mature` / `plateau` / `declining`,
   unless the SKU has a manual `stage_override` set on the Sales Forecast tab, which
   always wins.
2. **Strip outliers** - a local-median/MAD check flags genuine one-off spikes (Prime Day,
   a bulk order) and excludes them from the fit, recorded in `sales_forecast_exclusions`
   so the tab can show why.
3. **Fit a model for that stage** and project 180 days (~6 months) forward:
   - `new` → logistic growth curve (S-shaped ramp toward a ceiling, not a straight line)
   - `growth` / `declining` → damped-trend ETS
   - `mature` / `plateau` → damped-trend ETS **with weekly seasonality**
   - **end-of-life** (checkbox on the tab) → no fitted curve at all: sell at the current
     run rate until FBA sellable stock runs out (same velocity math `/api/inventory` uses
     for its "days of inventory left"), then zero - no restock assumed.
4. **Blend with prior-year seasonality**, for any SKU with 380+ days of history (not
   end-of-life): a damped trend flattens out by design over a 6-month horizon and never
   reproduces a real yearly cycle (a Christmas bump, a summer dip) on its own, however
   much history it's fitted on. This recenters the point estimate onto PY's same-date
   revenue (7-day-smoothed, scaled by this year's trailing-56d vs PY's growth factor),
   ramping from "trust the fitted model" (day 1) to "trust the PY shape" (day 28+). The
   fitted model's own band width is kept, just recentered - `model_used` gets a
   `+py_blend` suffix when this applied. Below 380 days of history, forecast stays purely
   the stage-based fit from step 3.
5. Writes `forecast_revenue` + an uncertainty band (`low_revenue`/`high_revenue`) per day,
   replacing that SKU's previous forecast rows. The band is deliberately tight near-term -
   `0.75 × recent residual std`, widening by `sqrt(1 + days_out/45)` - so "tomorrow" reads
   as mostly determined by actual recent volatility rather than a wide hedge, and only
   opens up gradually further into the horizon. Every fit type (flat fallback, logistic,
   ETS) uses this same formula; ETS's own prediction interval is intentionally not used
   here, since it factors in parameter-estimation uncertainty on top of noise and ran
   noticeably wider, especially for SKUs without much history.

A SKU with no sale in the last 180 days is skipped (dormant/delisted), unless the user has
explicitly configured it (an override or the end-of-life flag).

Revenue-only - no unit/ASP split, no price elasticity. PVM already covers price-vs-volume
historically; this only projects the top-line number forward.

## Environment variables

- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` - same database
  `react-finance-dashboard` uses, and the same variable names its Node service's `Pool`
  config reads (`server/index.js`) - copy the exact same values over from that service.
  This service reads `v_sku_revenue`, `v_refunds_by_date`, `vat_divisor()`,
  `amazon_order_lines`/`amazon_orders`, `amazon_inventory_snapshots`, `sku_forecast_config`
  and writes `sales_forecast` / `sales_forecast_exclusions`. (A single `DATABASE_URL` is
  also accepted if set, as an alternative - but this account's Railway services use the
  five separate vars, not that.)
- `API_KEY` - optional. If set, `POST /run` requires header `x-api-key: <API_KEY>`. If
  unset, the endpoint is open to anyone who has the URL (same opt-in convention as this
  repo's `Postgres-Access` proxy).
- `PORT` - set by Railway automatically.

## Deploy (Railway)

Standalone repo - deploys as its own Railway service, separate from
`react-finance-dashboard`'s Node app (and the two live in different repos, so there's no
Root Directory setting to worry about):

1. In Railway, add a new service from this GitHub repo (`sales-forecast-service`).
2. Set `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` (copy the exact values from
   the dashboard's Node service) and, optionally, `API_KEY`.
3. Deploy - Nixpacks auto-detects Python from `requirements.txt` and installs it
   correctly on its own; there's no custom `nixpacks.toml` here (an earlier one that
   pinned `nixPkgs = ["python311"]` without `pip` broke the build - Nixpacks' built-in
   Python provider gets this right without help, so it was just removed rather than
   fixed forward).
4. Confirm `GET https://<this-service>.up.railway.app/health` returns `{"ok": true}`.

## Triggering a run

Nothing runs automatically - call it:

```
curl -X POST https://<this-service>.up.railway.app/run \
  -H "x-api-key: <API_KEY>"   # omit if API_KEY isn't set
```

A full-catalog run is expected to take low tens of seconds, not minutes; the request
blocks until it's done and returns a summary (`skus_processed`, `skus_skipped_inactive`,
per-stage counts, rows written).

### Scheduling it nightly (n8n)

This repo already uses n8n for scheduled jobs. Suggested setup:

1. New n8n workflow → **Schedule Trigger** node, e.g. daily at 02:00.
2. **HTTP Request** node: `POST` to `https://<this-service>.up.railway.app/run`, header
   `x-api-key` if set.
3. Activate the workflow.

Nothing in the dashboard depends on the exact schedule - the Sales Forecast tab just
reads whatever `sales_forecast` currently holds, and shows `has_forecast: false` /
"not generated yet" until the first run completes.

## Local development

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
DB_HOST=... DB_PORT=5432 DB_NAME=... DB_USER=... DB_PASSWORD=... uvicorn main:app --reload
```

`forecast.py` has no database dependency and can be exercised directly with a synthetic
pandas DataFrame - see the docstrings on `run_for_sku`, `classify_stage`, `strip_outliers`,
`fit_new`, `fit_ets`, and `eol_forecast` for the shape each expects.
