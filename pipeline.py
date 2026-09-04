"""
Orchestrates one end-to-end forecasting run: pull history + config from Postgres, fit a
90-day forecast per SKU, write results back. This is the only module that combines db.py
(I/O) with forecast.py (pure modeling) - keeps the modeling code testable without a
database and the DB access free of modeling assumptions.
"""
import logging
from datetime import datetime, timezone

import pandas as pd

import db
import forecast as fc

logger = logging.getLogger("forecast_pipeline")

HISTORY_LOOKBACK_DAYS = 730  # 2 years - enough for a yearly-seasonal SKU, bounded so the
# pull doesn't keep growing forever as more history accumulates.
HORIZON_DAYS = 90  # covers the 30/60/90-day rollups the Sales Forecast tab shows.


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
