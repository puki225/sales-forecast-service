"""
Core forecasting pipeline: per-SKU stage classification, outlier stripping, model fit,
and end-of-life depletion. Pure functions operating on pandas/numpy - no DB calls in this
file, so it can be unit-tested and reasoned about without a live Postgres connection.

Revenue-only forecast (no unit/price split) - PVM already covers price-vs-volume
historically; this only projects the top-line number forward. Every SKU gets a `horizon`
day-by-day forecast; the API layer sums it into 30/60/90-day rollups.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

VALID_STAGES = ["new", "growth", "mature", "plateau", "declining"]

# A SKU with no sales in this many trailing days is treated as dormant/delisted and
# skipped, unless the user has explicitly configured it (override or end-of-life) - no
# point spending a forecast row on a product nobody's decided is still active.
ACTIVITY_WINDOW_DAYS = 180

# Outlier stripping: local-median + MAD (median absolute deviation), robust to the very
# spikes it's trying to catch (unlike a mean/stdev z-score, which the spike itself
# distorts). Threshold is deliberately conservative (7 MADs) - this should only catch
# genuine one-offs like a Prime Day spike, not ordinary week-to-week variance the model
# should be allowed to learn from.
OUTLIER_MAD_THRESHOLD = 7.0
OUTLIER_WINDOW_DAYS = 28


def reindex_daily(df_sku, today):
    """df_sku: columns [date, revenue, units] for one SKU, sparse (only days with orders).
    Returns a continuous daily series from first sale through yesterday, 0-filled on
    no-order days - required for the seasonal/trend models below, which assume an evenly
    spaced series."""
    first_date = df_sku["date"].min()
    last_date = today - timedelta(days=1)
    idx = pd.date_range(first_date, last_date, freq="D")
    s = df_sku.set_index("date")["revenue"].reindex(idx, fill_value=0.0)
    s.index.name = "date"
    return s


def classify_stage(series):
    """Auto-classification, used only when the user hasn't set stage_override. Simple by
    design - the Sales Forecast tab lets a user correct a misclassification directly
    rather than this trying to be clever."""
    age_days = len(series)
    if age_days < 90:
        return "new"
    window = min(56, age_days // 2)
    last = series.iloc[-window:].sum()
    prior = series.iloc[-2 * window:-window].sum() if age_days >= 2 * window else None
    growth_rate = (last - prior) / prior if prior and prior > 0 else None
    if growth_rate is None:
        return "mature"
    if growth_rate > 0.10:
        return "growth"
    if growth_rate < -0.10:
        return "declining"
    if age_days > 270:
        return "plateau"
    return "mature"


def strip_outliers(series):
    """Returns (cleaned_series, excluded_dates_with_reason) - cleaned has each flagged
    spike replaced by its local rolling median so the fit below never sees it, while the
    original dates/reasons are kept separately for the sales_forecast_exclusions table
    (and the "⚑ N periods excluded" note on the Sales Forecast tab)."""
    rolling_median = series.rolling(OUTLIER_WINDOW_DAYS, min_periods=7, center=True).median()
    resid = series - rolling_median
    mad = resid.abs().rolling(OUTLIER_WINDOW_DAYS, min_periods=7, center=True).median()
    mad_safe = mad.replace(0, np.nan)
    z = (resid / (1.4826 * mad_safe)).abs()
    is_outlier = (z > OUTLIER_MAD_THRESHOLD) & rolling_median.notna()

    cleaned = series.copy()
    cleaned[is_outlier] = rolling_median[is_outlier]

    exclusions = []
    for d in series.index[is_outlier]:
        multiple = series[d] / rolling_median[d] if rolling_median[d] else float("inf")
        exclusions.append({"date": d, "reason": f"Revenue {multiple:.1f}x local trend"})
    return cleaned, exclusions


def _logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))


def _flat_fallback(series, horizon, band_pct=0.25):
    tail = series.iloc[-14:] if len(series) >= 1 else series
    base = float(tail.mean()) if len(tail) else 0.0
    point = np.full(horizon, base)
    band = base * band_pct
    widen = np.sqrt(1 + np.arange(horizon) / 30)
    low = np.clip(point - band * widen, 0, None)
    high = point + band * widen
    return point, low, high, "flat_fallback"


def fit_new(series, horizon):
    """New SKUs rarely have enough history for a seasonal/trend time-series model, and
    growth is naturally S-shaped (ramping toward a ceiling, not linear) - fit a logistic
    growth curve on a 7-day-smoothed series instead."""
    if len(series) < 10:
        return _flat_fallback(series, horizon, band_pct=0.4)
    t = np.arange(len(series))
    y_smooth = series.rolling(7, min_periods=1, center=True).mean().values
    y_max = max(float(y_smooth.max()), 1.0)
    try:
        popt, _ = curve_fit(
            _logistic, t, y_smooth,
            p0=[y_max * 3, 0.05, len(t) / 2],
            bounds=([y_max, 0.001, -len(t)], [y_max * 10 + 1, 2, len(t) * 3]),
            maxfev=5000,
        )
        t_future = np.arange(len(t), len(t) + horizon)
        point = _logistic(t_future, *popt)
        resid_std = float(np.std(series.values - _logistic(t, *popt)))
        widen = np.sqrt(1 + np.arange(horizon) / 30)
        low = np.clip(point - 1.28 * resid_std * widen, 0, None)
        high = point + 1.28 * resid_std * widen
        return point, low, high, "logistic_growth"
    except Exception:
        return _flat_fallback(series, horizon, band_pct=0.4)


def fit_ets(series, horizon, seasonal):
    """Damped-trend ETS for growth/mature/plateau/declining. Seasonal=True adds a
    7-day (weekly) seasonal component - needs at least 3 full cycles of history to be
    worth trusting, otherwise falls back to the non-seasonal damped-trend fit.
    ETSModel must be given the pandas Series (not .values) - get_prediction()'s
    summary_frame() reaches for the input's index internally and raises a bare
    AttributeError against a plain ndarray."""
    n = len(series)
    use_seasonal = seasonal and n >= 21
    try:
        model = ETSModel(
            series, error="add", trend="add", damped_trend=True,
            seasonal="add" if use_seasonal else None,
            seasonal_periods=7 if use_seasonal else None,
        )
        fit = model.fit(disp=False)
        pred = fit.get_prediction(start=n, end=n + horizon - 1)
        summary = pred.summary_frame(alpha=0.2)  # 80% interval
        point = np.clip(summary["mean"].values, 0, None)
        low = np.clip(summary["pi_lower"].values, 0, None)
        high = np.clip(summary["pi_upper"].values, 0, None)
        model_used = "ets_seasonal" if use_seasonal else "ets_damped_trend"
        return point, low, high, model_used
    except Exception:
        return _flat_fallback(series, horizon, band_pct=0.3)


def eol_forecast(daily_run_rate, depletion_days, horizon):
    """Sell at the current run rate until inventory (from the same velocity/sellable
    figures the Inventory tab uses) runs out, then stop - no restock assumed. Reflects the
    end-of-life checkbox on the Sales Forecast tab exactly, not a fitted curve."""
    days = np.arange(horizon)
    point = np.where(days < depletion_days, daily_run_rate, 0.0)
    band = daily_run_rate * 0.15
    low = np.where(days < depletion_days, np.clip(point - band, 0, None), 0.0)
    high = np.where(days < depletion_days, point + band, 0.0)
    return point, low, high, "eol_depletion"


def run_for_sku(sku, hist_df, today, stage_override, is_end_of_life, eol_inputs, horizon=90):
    """Returns (forecast_rows, exclusion_rows, stage_used). eol_inputs is a dict with
    sellable/daily_velocity_units/daily_run_rate, or None if unavailable (falls back to a
    non-EOL fit even if the checkbox is set, rather than fail outright)."""
    series = reindex_daily(hist_df, today)
    stage_used = stage_override if stage_override in VALID_STAGES else classify_stage(series)

    if is_end_of_life and eol_inputs and eol_inputs.get("daily_velocity_units", 0) > 0:
        depletion_days = eol_inputs["sellable"] / eol_inputs["daily_velocity_units"]
        point, low, high, model_used = eol_forecast(eol_inputs["daily_run_rate"], depletion_days, horizon)
        exclusions = []
    else:
        cleaned, exclusions = strip_outliers(series)
        if stage_used == "new":
            point, low, high, model_used = fit_new(cleaned, horizon)
        elif stage_used in ("mature", "plateau"):
            point, low, high, model_used = fit_ets(cleaned, horizon, seasonal=True)
        else:  # growth, declining
            point, low, high, model_used = fit_ets(cleaned, horizon, seasonal=False)

    generated_at = pd.Timestamp.utcnow()
    forecast_rows = [
        {
            "sku": sku,
            "forecast_date": (today + timedelta(days=i)).date(),
            "forecast_revenue": round(float(point[i]), 2),
            "low_revenue": round(float(low[i]), 2),
            "high_revenue": round(float(high[i]), 2),
            "stage_used": stage_used,
            "model_used": model_used,
            "generated_at": generated_at,
        }
        for i in range(horizon)
    ]
    exclusion_rows = [
        {"sku": sku, "excluded_date": e["date"].date(), "reason": e["reason"]}
        for e in exclusions
    ]
    return forecast_rows, exclusion_rows, stage_used
