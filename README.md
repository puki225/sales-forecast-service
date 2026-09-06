# forecast-service

Nightly job that fills in `sales_forecast` (and `sales_forecast_exclusions`), read by
`react-finance-dashboard`'s Sales Forecast tab (`GET /api/sales-forecast`). Not a
long-running worker - it's a small FastAPI app with one real endpoint, `POST /run`, that
does one full catalog pass and returns. Something else (n8n, cron, curl) has to actually
call it on a schedule; this service has no scheduler of its own.

## What it does

Per SKU, over roughly the last 2 years of `net revenue` (order revenue net of discounts
**and** refunds - the same figure the dashboard calls "net revenue" everywhere else):

1. **Fill supply-shortage gaps** - a run of 14+ consecutive £0 days (a stockout, not just
   quiet demand) gets replaced, for fitting purposes only, with a straight-line estimate
   between the trailing and leading baseline either side of it, so a 2-week-to-several-
   month shortage doesn't get learned as "this SKU's demand fell to zero" (it would
   otherwise drag down both the stage classification below and the fitted trend/level for
   however long the gap lasted). Logged in `sales_forecast_exclusions` as one row per gap.
   Only a gap fully enclosed by real data on both sides is filled - one still running
   through the most recent day is left as real zeros, since there's no way to know from
   history alone whether/when it restocks (that's what `stage_override`/end-of-life are
   for).
2. **Classify growth stage** - `new` / `growth` / `mature` / `plateau` / `declining`,
   unless the SKU has a manual `stage_override` set on the Sales Forecast tab, which
   always wins.
3. **Strip outliers** - a local-median/MAD check flags genuine one-off spikes (Prime Day,
   Black Friday, a bulk order) and excludes them from the fit, recorded in
   `sales_forecast_exclusions` so the tab can show why. A spike that lands inside a known
   recurring Amazon sales-event window (Black Friday/Cyber Monday - calendar-fixed around
   the 4th Thursday of November; Prime Day/Prime Big Deal Days - no fixed date, so a
   broad July/October window stands in) is tagged as such and its real value is kept
   around separately for step 5 below, rather than being discarded outright like a random
   one-off spike (a bulk order, a data glitch) is.
4. **Fit a model for that stage** and project 180 days (~6 months) forward:
   - `new` → logistic growth curve (S-shaped ramp toward a ceiling, not a straight line)
   - `growth` / `declining` → damped-trend ETS
   - `mature` / `plateau` → damped-trend ETS **with weekly seasonality**
   - **end-of-life** (checkbox on the tab) gets the exact same fit as everything else here
     - a SKU on its last units still sees a real Black Friday, so there's no reason to
     forecast it flat. The only difference end-of-life makes is in step 7 below: once
     projected inventory runs out, sales stop abruptly rather than fading out or
     continuing past what's actually sellable.
5. **Blend with prior-year seasonality**, for any SKU with 380+ days of history: a damped
   trend flattens out by design over a 6-month horizon and never reproduces a real yearly
   cycle (a Christmas bump, a summer dip) on its own, however much history it's fitted on.
   This recenters the point estimate onto PY's same-date
   revenue, scaled by this year's trailing-56d vs PY's growth factor, ramping from "trust
   the fitted model" (day 1) to "trust the PY shape" (day 28+). `model_used` gets a
   `+py_blend` suffix when this applied. Below 380 days of history, forecast stays purely
   the stage-based fit from step 4.
   - For most days, the PY value is a ±3-day centered average around the matching PY date,
     same as before - smooths single-day noise out of the seasonal shape.
   - For a target date whose matching PY window contains an *actual* detected recurring-
     event spike (step 3's Black Friday/Cyber Monday/Prime Day tagging - not just any day
     that happens to fall in the same calendar month), that day's real peak value is used
     directly instead of being averaged away, so a SKU that sold well last Black Friday is
     forecast to sell well on the equivalent date this year too, rather than that day
     quietly reverting to baseline the way a straight damped-trend fit would.
   - Below 380 days, there's no PY comparison of this SKU's own to blend with - but a
     **catalog-wide seasonal index**, built once per run from every SKU that *does* clear
     380 days (`pipeline.build_catalog_seasonal_index` - excluding end-of-life SKUs, whose
     current trajectory is an intentional wind-down, not representative demand), applies
     instead when one is available: each contributing SKU's own PY-implied point is
     measured as a multiple of its own recent 28-day baseline, and those multiples are
     averaged **equal-weighted** across contributors - deliberately not revenue-weighted,
     since revenue-weighting is exactly what made the catalog look falsely flat before this
     existed (a few large SKUs' totals swamping everyone else's in any pooled view). Same
     day-1-to-day-28 trust ramp as the PY blend; `model_used` gets a `+catalog_seasonal`
     suffix when this applied.
6. **Texture the point forecast with daily noise**, then **compute the band from that
   noisy point** - in that order. `_volatility()` measures the trailing actual day-to-day
   volatility from differenced daily values (so a steady trend doesn't get mistaken for
   noise); `add_daily_noise()` adds i.i.d. noise scaled to 0.7x that figure, so the line
   reads as a plausible day-by-day sales path instead of a suspiciously smooth curve; not
   seeded, a fresh pattern every run rather than an identical one every night.
   `band_from_point()` then builds `low_revenue`/`high_revenue` as that *same* noisy point
   ± `0.5 × volatility`, widening by `sqrt(1 + days_out/45)` - which makes `low <= point
   <= high` a hard guarantee by construction (not just "usually true"), and gives the
   band's own edges the same jagged, real-looking texture as the line, rather than a
   smooth curve sitting under a jagged one. An earlier version computed the band from the
   pre-noise point using each model's own residual std, which decoupled the two badly
   enough that the noisy line routinely poked outside its own band.
7. **End-of-life depletion cutoff** - the one and only place the end-of-life checkbox
   changes anything: once projected inventory (`sellable / daily_velocity_units`, the same
   velocity math `/api/inventory` uses for its "days of inventory left") would run out,
   `forecast_revenue`/`low_revenue`/`high_revenue` are hard-clipped to 0 from that day
   forward - no restock assumed, and no noise/band wobble around 0 either. Everything
   before this point (stage fit, PY/catalog seasonality, noise, band) is untouched -
   `model_used` just gets a `+eol_cutoff` suffix appended. A SKU flagged end-of-life with
   no usable sellable/velocity figures forecasts exactly like any other SKU, uncapped -
   there's nothing to cut off against.
8. Writes `forecast_revenue` + that band per day, replacing that SKU's previous forecast
   rows.

A SKU with no sale in the last 180 days is skipped (dormant/delisted), unless the user has
explicitly configured it (an override or the end-of-life flag). Note this is a *trailing*
window check against the raw data, independent of step 1's gap-fill (which only touches
enclosed historical gaps) - a SKU still mid-shortage today with no resolution yet in the
data stays correctly excluded here rather than silently forecast as if it were selling.

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
`fit_new`, and `fit_ets` for the shape each expects.
