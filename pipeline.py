"""
Orchestrates one end-to-end forecasting run: pull history + config from Postgres, fit a
90-day forecast per SKU, write results back. This is the only module that combines db.py
(I/O) with forecast.py (pure modeling) - keeps the modeling code testable without a
database and the DB access free of modeling assumptions.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import db
import forecast as fc

logger = logging.getLogger("forecast_pipeline")

HISTORY_LOOKBACK_DAYS = 730  # 2 years - enough for a yearly-seasonal SKU, bounded so the
# pull doesn't keep growing forever as more history accumulates.
HORIZON_DAYS = 180  # ~6 months - a forecast is only actually useful at this range, per
# the tab's own design discussion; the UI lets you view it at daily/weekly/monthly
# granularity and doesn't force looking at all 180 raw daily points at once.

CATALOG_INDEX_BASELINE_DAYS = 28  # each contributing SKU's own recent-run-rate baseline,
# the denominator its PY-implied point gets measured against before averaging into the
# catalog-wide index - short enough to reflect "current" run rate, long enough not to be
# thrown by a single noisy day.
CATALOG_INDEX_CLIP = (0.1, 10.0)  # bounds each SKU's own contributed ratio before
# averaging - guards the whole-catalog index against one thin/noisy contributor implying
# an absurd multiplier for every young SKU that borrows it.


def build_catalog_seasonal_index(daily, config, today, horizon):
    """A length-`horizon` array: at each day out, how many times a typical SKU's own
    recent baseline the catalog as a whole tends to sell on that calendar date - built
    from every SKU with enough history for its own PY comparison (same PY_MIN_HISTORY_DAYS
    gate run_for_sku already applies), equal-weighted across contributors rather than
    revenue-weighted, so the SKU that happens to be the single biggest earner doesn't
    define "typical" seasonality for everyone else - it's exactly a few large SKUs
    dominating a plain revenue-summed view that made the whole catalog look artificially
    flat before this existed. End-of-life SKUs are excluded from contributing: their
    current trajectory is an intentional wind-down, not representative demand. Returns
    None if no SKU qualifies (e.g. an entirely new catalog) - callers should treat that as
    "no catalog signal available" and fall back to whatever they'd otherwise do."""
    ratios = []
    for sku, df_sku in daily.groupby("sku"):
        df_sku = df_sku[df_sku["date"] < today]
        if df_sku.empty:
            continue
        cfg = config.loc[sku] if sku in config.index else None
        if cfg is not None and bool(cfg["is_end_of_life"]):
            continue
        series = fc.reindex_daily(df_sku[["date", "revenue"]], today)
        if len(series) < fc.PY_MIN_HISTORY_DAYS:
            continue
        gap_filled, gap_dates, _ = fc.detect_and_fill_gaps(series)
        cleaned, event_retained, event_dates, _ = fc.strip_outliers(gap_filled, exclude_dates=gap_dates)
        py_point, has_py = fc.py_naive_forecast(cleaned, event_retained, event_dates, horizon)
        if not has_py:
            continue
        baseline = float(cleaned.iloc[-CATALOG_INDEX_BASELINE_DAYS:].mean())
        if baseline <= 0:
            continue
        ratios.append(np.clip(py_point / baseline, *CATALOG_INDEX_CLIP))

    if not ratios:
        return None
    return np.mean(ratios, axis=0)


def _eol_velocity(row):
    """Same PY-seasonal-adjusted velocity formula as /api/inventory's "days of inventory
    left" figure (server/index.js, GET /api/inventory) - ported here rather than shared,
    so a SKU's end-of-life depletion date agrees with what the Inventory tab already says
    about it instead of computing a second, disagreeing answer from different logic."""
    cy, py_trailing, py_forward = row["cy_trailing_units"], row["py_trailing_units"], row["py_forward_units"]
    if py_forward > 0:
        growth = (cy - py_trailing) / py_trailing if py_trailing > 0 else 0.0
        return max(0.0, (py_forward / 90) * (1 + growth))
    if cy > 0:
        return cy / 90
    return 0.0


def run(conn):
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    min_date = (today - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)).date()

    daily = db.fetch_daily_revenue(conn, min_date)
    daily["date"] = pd.to_datetime(daily["date"])
    config = db.fetch_sku_config(conn).set_index("sku")
    eol_raw = db.fetch_eol_inputs(conn).set_index("sku")
    eol_velocity = {sku: {"sellable": float(row["sellable"] or 0), "daily_velocity_units": _eol_velocity(row)}
                     for sku, row in eol_raw.iterrows()}
    catalog_index = build_catalog_seasonal_index(daily, config, today, HORIZON_DAYS)

    all_forecast_rows = []
    all_exclusion_rows = []
    summary = {"skus_processed": 0, "skus_skipped_inactive": 0, "stages": {}, "end_of_life": 0}

    for sku, df_sku in daily.groupby("sku"):
        df_sku = df_sku[df_sku["date"] < today]
        if df_sku.empty:
            continue

        cfg = config.loc[sku] if sku in config.index else None
        stage_override = cfg["stage_override"] if cfg is not None else None
        is_eol = bool(cfg["is_end_of_life"]) if cfg is not None else False

        last_sale_age_days = (today - df_sku["date"].max()).days
        user_configured = bool(stage_override) or is_eol
        if last_sale_age_days > fc.ACTIVITY_WINDOW_DAYS and not user_configured:
            summary["skus_skipped_inactive"] += 1
            continue

        eol_inputs = None
        if is_eol and sku in eol_velocity:
            trailing_7d = df_sku[df_sku["date"] >= today - pd.Timedelta(days=7)]["revenue"].mean()
            eol_inputs = dict(eol_velocity[sku])
            eol_inputs["daily_run_rate"] = float(trailing_7d) if pd.notna(trailing_7d) else 0.0

        try:
            rows, exclusions, stage_used = fc.run_for_sku(
                sku, df_sku[["date", "revenue"]], today,
                stage_override, is_eol, eol_inputs, horizon=HORIZON_DAYS,
                catalog_index=catalog_index,
            )
        except Exception:
            logger.exception("Forecast failed for SKU %s - skipping it this run", sku)
            continue

        all_forecast_rows.extend(rows)
        all_exclusion_rows.extend(exclusions)
        summary["skus_processed"] += 1
        summary["stages"][stage_used] = summary["stages"].get(stage_used, 0) + 1
        if is_eol:
            summary["end_of_life"] += 1

    db.write_forecast(conn, all_forecast_rows)
    db.write_exclusions(conn, all_exclusion_rows)
    summary["forecast_rows_written"] = len(all_forecast_rows)
    summary["exclusion_rows_written"] = len(all_exclusion_rows)
    return summary
