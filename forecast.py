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


def _flat_fallback(series, horizon):
    tail = series.iloc[-14:] if len(series) >= 1 else series
    base = float(tail.mean()) if len(tail) else 0.0
    return np.full(horizon, base), "flat_fallback"


def fit_new(series, horizon):
    """New SKUs rarely have enough history for a seasonal/trend time-series model, and
    growth is naturally S-shaped (ramping toward a ceiling, not linear) - fit a logistic
    growth curve on a 7-day-smoothed series instead."""
    if len(series) < 10:
        return _flat_fallback(series, horizon)
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
        return point, "logistic_growth"
    except Exception:
        return _flat_fallback(series, horizon)


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
        point = np.clip(fit.forecast(horizon).values, 0, None)
        model_used = "ets_seasonal" if use_seasonal else "ets_damped_trend"
        return point, model_used
    except Exception:
        return _flat_fallback(series, horizon)


# Minimum history for a PY (prior-year) blend to be trustworthy - one full year plus
# enough slack either side to compute a trailing-56d growth factor and a centered PY
# window near the very first comparison date.
PY_MIN_HISTORY_DAYS = 380
PY_TRAILING_DAYS = 56
PY_SHIFT_DAYS = 364  # 52 whole weeks, not 365 - keeps weekday alignment intact (same
# reasoning PVM's period presets already use for their YoY comparisons).
PY_WINDOW_DAYS = 3  # +/- days averaged around the matching PY date, to smooth single-day
# noise out of the seasonal shape rather than reading one PY Tuesday too literally.
PY_GROWTH_FACTOR_BOUNDS = (0.3, 3.0)  # guards against a near-zero PY trailing window
# (or a genuine outlier week) implying an absurd multiplier.


def py_naive_forecast(series, horizon):
    """Prior-year seasonal shape (same calendar dates 364 days back, centered 7-day
    average to smooth noise) scaled by this year's trailing-56d vs PY's same-56d growth
    factor. Returns (point array, True) or (None, False) if there isn't a full PY of
    history yet. This is what actually carries seasonality forward past the point a
    damped-trend ETS fit has flattened out - ETS's trend decays toward zero by design, so
    on its own it never reproduces a real yearly cycle (a Christmas bump, a summer dip),
    no matter how much history it's given."""
    if len(series) < PY_MIN_HISTORY_DAYS:
        return None, False
    last_date = series.index[-1]
    py_trailing_end = last_date - pd.Timedelta(days=PY_SHIFT_DAYS)
    py_trailing_start = py_trailing_end - pd.Timedelta(days=PY_TRAILING_DAYS - 1)
    py_trailing = series.loc[py_trailing_start:py_trailing_end].sum()
    if py_trailing <= 0:
        return None, False
    this_trailing = series.iloc[-PY_TRAILING_DAYS:].sum()
    growth_factor = np.clip(this_trailing / py_trailing, *PY_GROWTH_FACTOR_BOUNDS)

    point = np.zeros(horizon)
    for i in range(horizon):
        target_date = last_date + timedelta(days=1 + i)
        py_date = target_date - pd.Timedelta(days=PY_SHIFT_DAYS)
        window = series.loc[py_date - pd.Timedelta(days=PY_WINDOW_DAYS): py_date + pd.Timedelta(days=PY_WINDOW_DAYS)]
        py_val = float(window.mean()) if len(window) else 0.0
        point[i] = max(0.0, py_val * growth_factor)
    return point, True


def blend_with_py(series, point, horizon):
    """Recenters a stage-based fit's point estimate onto the PY-naive seasonal estimate
    where PY history exists, ramping from "trust the fitted model" (day 1, real current
    momentum) to "trust the PY shape" (day 28+, where a damped trend has already gone
    flat and PY is the only remaining source of real signal) over the first 4 weeks."""
    py_point, has_py = py_naive_forecast(series, horizon)
    if not has_py:
        return point, False
    ramp = np.clip(np.arange(horizon) / 28, 0, 1)
    blended = ramp * py_point + (1 - ramp) * point
    return blended, True


# One volatility measure drives both the day-to-day noise texture AND the band width -
# using two different figures (as an earlier version did: each model's own residual std
# for the band, raw actual std for the noise) meant the band's width had no reliable
# relationship to how much noise actually got added to the point, so the noisy line
# routinely poked outside its own "uncertainty" band - backwards for something meant to
# bound the forecast. Deriving both from the SAME number, and computing the band from the
# POST-noise point (not the smooth pre-noise one), makes containment automatic instead of
# probabilistic, and gives the band's own edges the same jagged, real-looking texture as
# the line instead of a smooth curve sitting under a jagged one.
NOISE_FRACTION = 0.7  # "a bit of" the measured volatility, not full-strength
NOISE_WINDOW_DAYS = 28
# band_from_point() below builds the band AS point +/- this half-width, so low <= point
# <= high holds for ANY positive BAND_Z - containment comes from the construction, not
# from BAND_Z being large enough to "cover" the noise. An earlier version raised this to
# 1.4 specifically to make the band comfortably wider than the noise before an (at the
# time, still separate) containment clip kicked in - that reasoning stopped applying the
# moment the band started being computed from the noisy point directly, and 1.4 just made
# the shaded area unnecessarily wide with nothing to show for it. Tightened back down.
BAND_Z = 0.5  # multiple of the measured volatility - purely a "how much wiggle room to
# display" dial now, not a safety margin.
BAND_RAMP_DAYS = 45  # width at day t scales by sqrt(1 + t/45): ~1.15x by day 7,
# ~1.4x by day 45, ~2.2x by day 180 (end of a 6-month horizon).


def _volatility(series):
    """Trailing actual day-to-day volatility, measured from DIFFERENCED daily values
    (day[i] - day[i-1]), not the raw level std of the window - a fast-ramping short series
    (e.g. a brand-new SKU trending hard upward over its first few weeks) has a raw level
    std dominated by the trend itself, not by actual noise. Differencing cancels a steady
    trend out and isolates the noise; dividing by sqrt(2) undoes the variance-doubling
    that differencing two independent noise terms introduces, recovering the per-day
    noise sigma. Returns 0.0 (no noise, no band) when there's too little data to measure
    it from - never a guessed percentage."""
    tail = series.iloc[-NOISE_WINDOW_DAYS:]
    if len(tail) < 3:
        return 0.0
    diffs = tail.diff().dropna()
    if len(diffs) < 2:
        return 0.0
    std = float(diffs.std()) / np.sqrt(2)
    return std if np.isfinite(std) and std > 0 else 0.0


def add_daily_noise(point, std, horizon):
    """Textures a smooth point forecast with i.i.d. noise scaled to `std` (see
    _volatility), so the forecast reads as a plausible day-by-day sales path instead of a
    suspiciously tidy trend line. Not seeded - a fresh, independent pattern each run is
    more honest than an identical one appearing every night."""
    if horizon <= 0 or std <= 0:
        return point
    noise = np.random.default_rng().normal(0, std * NOISE_FRACTION, horizon)
    return np.clip(point + noise, 0, None)


def band_from_point(point, std, horizon):
    """The uncertainty band, computed FROM the (already noisy) point - not the other way
    around - so low <= point <= high holds by construction, always, not just on average.
    Widens gradually with horizon (BAND_RAMP_DAYS); the day-to-day texture that gives both
    edges their jagged look comes along for free, since they're just `point` offset by a
    slowly-changing half-width."""
    widen = np.sqrt(1 + np.arange(horizon) / BAND_RAMP_DAYS)
    half = max(BAND_Z * std, 1e-9) * widen  # half > 0 always, so low <= point <= high
    # holds by construction below - no separate clamp needed to enforce it.
    low = np.clip(point - half, 0, None)
    high = point + half
    return low, high


def eol_forecast(daily_run_rate, depletion_days, horizon):
    """Sell at the current run rate until inventory (from the same velocity/sellable
    figures the Inventory tab uses) runs out, then stop - no restock assumed. Reflects the
    end-of-life checkbox on the Sales Forecast tab exactly, not a fitted curve."""
    days = np.arange(horizon)
    point = np.where(days < depletion_days, daily_run_rate, 0.0)
    return point, "eol_depletion"


def run_for_sku(sku, hist_df, today, stage_override, is_end_of_life, eol_inputs, horizon=90):
    """Returns (forecast_rows, exclusion_rows, stage_used). eol_inputs is a dict with
    sellable/daily_velocity_units/daily_run_rate, or None if unavailable (falls back to a
    non-EOL fit even if the checkbox is set, rather than fail outright)."""
    series = reindex_daily(hist_df, today)
    stage_used = stage_override if stage_override in VALID_STAGES else classify_stage(series)

    if is_end_of_life and eol_inputs and eol_inputs.get("daily_velocity_units", 0) > 0:
        depletion_days = eol_inputs["sellable"] / eol_inputs["daily_velocity_units"]
        point, model_used = eol_forecast(eol_inputs["daily_run_rate"], depletion_days, horizon)
        std = _volatility(series)
        point = add_daily_noise(point, std, horizon)
        low, high = band_from_point(point, std, horizon)
        # Noise (and its band) must not leak past the hard sell-out cutoff - once
        # inventory is gone, revenue is exactly 0, not a small random wobble around 0.
        past_cutoff = np.arange(horizon) >= depletion_days
        point = np.where(past_cutoff, 0.0, point)
        low = np.where(past_cutoff, 0.0, low)
        high = np.where(past_cutoff, 0.0, high)
        exclusions = []
    else:
        cleaned, exclusions = strip_outliers(series)
        if stage_used == "new":
            point, model_used = fit_new(cleaned, horizon)
        elif stage_used in ("mature", "plateau"):
            point, model_used = fit_ets(cleaned, horizon, seasonal=True)
        else:  # growth, declining
            point, model_used = fit_ets(cleaned, horizon, seasonal=False)
        point, blended = blend_with_py(cleaned, point, horizon)
        if blended:
            model_used += "+py_blend"
        std = _volatility(cleaned)
        point = add_daily_noise(point, std, horizon)
        low, high = band_from_point(point, std, horizon)

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
