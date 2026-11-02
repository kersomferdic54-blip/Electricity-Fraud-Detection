"""Data loading, cleaning and feature engineering for electricity fraud detection.

Every customer is described by a daily consumption series (kWh per day).
Features are designed to work for any history length >= MIN_DAYS, so the
same code serves both model training (SGCC dataset) and the web app
(staff-uploaded files covering any period).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DAYS = 90            # minimum observed days needed to score a customer
MAX_INTERP_GAP = 3       # missing gaps up to this many days are interpolated
ID_COL = "CONS_NO"
LABEL_COL = "FLAG"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a wide table (CONS_NO, [FLAG], one column per date) into a
    customers x dates matrix with chronologically sorted date columns."""
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    meta = [c for c in (ID_COL, LABEL_COL) if c in df.columns]
    date_cols = [c for c in df.columns if c not in meta]
    dates = pd.to_datetime(pd.Index(date_cols).astype(str), errors="coerce")
    keep = ~dates.isna()
    values = df[np.array(date_cols)[keep]].apply(pd.to_numeric, errors="coerce")
    values.columns = dates[keep]
    values = values.T.groupby(level=0).mean().T        # merge duplicate dates
    values = values.sort_index(axis=1)
    values = values.reindex(columns=pd.date_range(values.columns.min(), values.columns.max(), freq="D"))
    values.index = df[ID_COL].values
    return values


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a long table (CONS_NO, DATE, KWH) into a customers x dates matrix."""
    cols = {c.upper().strip(): c for c in df.columns}
    missing = {ID_COL, "DATE", "KWH"} - set(cols)
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}")
    d = pd.DataFrame({
        ID_COL: df[cols[ID_COL]].astype(str).str.strip(),
        "DATE": pd.to_datetime(df[cols["DATE"]], errors="coerce"),
        "KWH": pd.to_numeric(df[cols["KWH"]], errors="coerce"),
    }).dropna(subset=["DATE"])
    wide = d.pivot_table(index=ID_COL, columns="DATE", values="KWH", aggfunc="mean")
    full_range = pd.date_range(wide.columns.min(), wide.columns.max(), freq="D")
    return wide.reindex(columns=full_range)


def read_any(df: pd.DataFrame) -> pd.DataFrame:
    """Detect wide vs long layout and return a customers x dates matrix."""
    upper = {c.upper().strip() for c in df.columns}
    if {"DATE", "KWH"} <= upper:
        return long_to_wide(df)
    if ID_COL not in df.columns:
        raise ValueError(f"File must contain a '{ID_COL}' column.")
    return load_wide(df)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_series(s: pd.Series) -> tuple[pd.Series, dict]:
    """Clean one customer's daily series.

    1. Trim to the observed period (first to last reading).
    2. Negative readings are treated as missing.
    3. Short gaps (<= MAX_INTERP_GAP days) are linearly interpolated,
       longer gaps are set to 0 (no recorded consumption).
    4. Extreme spikes are capped at mean + 3 * std (three-sigma rule).
    Returns the cleaned series plus quality stats used as features.
    """
    s = s.astype(float).copy()
    s[s < 0] = np.nan
    valid = s.dropna()
    if valid.empty:
        return s.iloc[:0], {"n_days": 0, "missing_frac": 1.0, "spike_frac": 0.0}
    s = s.loc[valid.index[0]: valid.index[-1]]
    s = s.asfreq("D") if isinstance(s.index, pd.DatetimeIndex) else s
    missing_frac = float(s.isna().mean())

    s = s.interpolate(limit=MAX_INTERP_GAP, limit_area="inside").fillna(0.0)

    mu, sd = s.mean(), s.std()
    cap = mu + 3 * sd if sd > 0 else mu
    spike_frac = float((s > cap).mean())
    s = s.clip(upper=cap)
    return s, {"n_days": int(len(s)), "missing_frac": missing_frac, "spike_frac": spike_frac}


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
FEATURE_DESCRIPTIONS = {
    "n_days": "Days of history available",
    "missing_frac": "Share of days with no meter reading",
    "spike_frac": "Share of days with extreme spikes (capped)",
    "mean": "Average daily consumption (kWh)",
    "std": "Standard deviation of daily consumption",
    "cv": "Variability: std / mean",
    "median": "Median daily consumption",
    "p10_ratio": "10th percentile / mean",
    "p90_ratio": "90th percentile / mean",
    "max_ratio": "Max day / mean",
    "skew": "Skewness of daily consumption",
    "kurtosis": "Kurtosis (tail heaviness)",
    "zero_frac": "Share of zero-consumption days",
    "longest_zero_run": "Longest run of zero days (fraction of history)",
    "low_frac": "Share of days below 10% of the median",
    "n_sudden_drops": "Sudden drops to <20% of the previous 30-day median (per 100 days)",
    "acf1": "Day-to-day autocorrelation",
    "acf7": "Week-to-week autocorrelation",
    "weekday_cv": "Variation across days of the week",
    "weekend_ratio": "Weekend / weekday average",
    "trend_slope": "Overall trend (relative change per year)",
    "monthly_cv": "Variability of monthly averages",
    "max_month_drop": "Largest month-to-month drop (relative)",
    "min_month_ratio": "Lowest month / average month",
    "sustained_drop": "Largest lasting step-down (after / before)",
    "recent_ratio": "Last 90 days / previous 365 days",
    "last_vs_first_year": "Last 12 months / first 12 months",
    "summer_winter_ratio": "Summer (Jun-Aug) / winter (Dec-Feb) average",
}
FEATURES = list(FEATURE_DESCRIPTIONS)


def _safe_div(a: float, b: float, default: float = 1.0) -> float:
    return float(a / b) if b and np.isfinite(b) and abs(b) > 1e-9 else default


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1 or np.std(x) == 0:
        return 0.0
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def series_features(s: pd.Series, quality: dict) -> dict:
    x = s.to_numpy(dtype=float)
    n = len(x)
    mean = x.mean()
    med = np.median(x)
    f = dict(quality)
    f.update({
        "mean": mean,
        "std": x.std(),
        "cv": _safe_div(x.std(), mean, 0.0),
        "median": med,
        "p10_ratio": _safe_div(np.percentile(x, 10), mean, 0.0),
        "p90_ratio": _safe_div(np.percentile(x, 90), mean, 0.0),
        "max_ratio": _safe_div(x.max(), mean, 0.0),
        "skew": float(pd.Series(x).skew()) if n > 2 else 0.0,
        "kurtosis": float(pd.Series(x).kurt()) if n > 3 else 0.0,
        "zero_frac": float((x == 0).mean()),
        "longest_zero_run": _longest_run(x == 0) / n,
        "low_frac": float((x < 0.1 * med).mean()) if med > 0 else float((x == 0).mean()),
        "acf1": _acf(x, 1),
        "acf7": _acf(x, 7),
    })

    roll_med = pd.Series(x).rolling(30, min_periods=10).median().shift(1).to_numpy()
    drops = (x < 0.2 * roll_med) & (roll_med > 0)
    starts = drops & ~np.r_[False, drops[:-1]]
    f["n_sudden_drops"] = 100.0 * starts.sum() / n

    t = np.arange(n) / 365.0
    slope = np.polyfit(t, x, 1)[0] if n > 1 else 0.0
    f["trend_slope"] = _safe_div(slope, mean, 0.0)

    idx = s.index
    if isinstance(idx, pd.DatetimeIndex):
        dow = pd.Series(x, index=idx).groupby(idx.dayofweek).mean()
        f["weekday_cv"] = _safe_div(dow.std(), dow.mean(), 0.0)
        f["weekend_ratio"] = _safe_div(dow.loc[dow.index >= 5].mean(), dow.loc[dow.index < 5].mean())
        monthly = pd.Series(x, index=idx).resample("MS").mean()
        summer = x[idx.month.isin([6, 7, 8])]
        winter = x[idx.month.isin([12, 1, 2])]
        f["summer_winter_ratio"] = (
            _safe_div(summer.mean(), winter.mean()) if len(summer) and len(winter) else 1.0
        )
    else:
        f["weekday_cv"], f["weekend_ratio"], f["summer_winter_ratio"] = 0.0, 1.0, 1.0
        monthly = pd.Series(x).groupby(np.arange(n) // 30).mean()

    m = monthly.to_numpy()
    m_mean = m.mean()
    f["monthly_cv"] = _safe_div(m.std(), m_mean, 0.0)
    f["min_month_ratio"] = _safe_div(m.min(), m_mean, 0.0)
    if len(m) > 1:
        prev = m[:-1]
        rel = np.where(prev > 0, (prev - m[1:]) / np.where(prev > 0, prev, 1), 0.0)
        f["max_month_drop"] = float(rel.max())
        ratios = [
            _safe_div(m[k:].mean(), m[:k].mean())
            for k in range(2, len(m) - 1)
        ] or [1.0]
        f["sustained_drop"] = float(min(ratios))
    else:
        f["max_month_drop"], f["sustained_drop"] = 0.0, 1.0

    f["recent_ratio"] = (
        _safe_div(x[-90:].mean(), x[-455:-90].mean()) if n >= 180 else 1.0
    )
    f["last_vs_first_year"] = (
        _safe_div(x[-365:].mean(), x[:365].mean()) if n >= 540 else 1.0
    )
    return {k: float(np.clip(f[k], -1e6, 1e6)) for k in FEATURES}


def build_features(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute features for every customer in a customers x dates matrix.

    Returns (features, skipped) where skipped lists customers that could not
    be scored and why.
    """
    rows, skipped = {}, []
    for cust, s in wide.iterrows():
        clean, quality = clean_series(s)
        observed = quality["n_days"] * (1 - quality["missing_frac"])
        if observed < MIN_DAYS:
            skipped.append({ID_COL: cust, "reason": f"Only {int(observed)} day{'s' if int(observed) != 1 else ''} of readings (need {MIN_DAYS})"})
            continue
        if clean.sum() == 0:
            skipped.append({ID_COL: cust, "reason": "All readings are zero - check the meter manually"})
            continue
        rows[cust] = series_features(clean, quality)
    feats = pd.DataFrame.from_dict(rows, orient="index", columns=FEATURES)
    feats.index.name = ID_COL
    return feats, pd.DataFrame(skipped, columns=[ID_COL, "reason"])
