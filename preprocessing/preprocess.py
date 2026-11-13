"""Step 2: data preprocessing for handwriting signals.

Devices differ (tablet, smart pen, iPad-like screens), so every recording is
brought to a common form before features are computed:
  1. sort by time, drop repeated timestamps and empty rows
  2. resample to a fixed rate (TARGET_HZ) by linear interpolation
  3. normalise x, y to zero mean and unit scale (removes device units / page size)
  4. scale pressure to [0, 1] per recording
  5. reject recordings that are too short to be meaningful
"""
import numpy as np
import pandas as pd

TARGET_HZ = 100
MIN_SECONDS = 0.5


def clean(df):
    df = df.sort_values("t").drop_duplicates("t")
    df = df.dropna(subset=["t"])
    return df.reset_index(drop=True)


def resample(df, hz=TARGET_HZ):
    t0, t1 = df["t"].iloc[0], df["t"].iloc[-1]
    t_new = np.arange(t0, t1, 1.0 / hz)
    out = {"t": t_new - t0}
    for col in df.columns:
        if col == "t" or not (np.issubdtype(df[col].dtype, np.number) or df[col].dtype == bool):
            continue
        y = df[col].to_numpy(dtype=float)
        if np.isnan(y).all():
            out[col] = np.full(len(t_new), np.nan)  # channel missing on this device: keep it missing
        elif col == "pen_down":
            idx = np.clip(np.searchsorted(df["t"].to_numpy(), t_new), 0, len(df) - 1)
            out[col] = y[idx]  # nearest sample, stays 0/1
        else:
            ok = ~np.isnan(y)
            out[col] = np.interp(t_new, df["t"].to_numpy()[ok], y[ok])
    return pd.DataFrame(out)


def normalise(df):
    df = df.copy()
    if not df[["x", "y"]].isna().all().all():
        xy = df[["x", "y"]].to_numpy()
        xy = xy - np.nanmean(xy, axis=0)
        scale = np.nanstd(xy) or 1.0
        df[["x", "y"]] = xy / scale
    p = df["pressure"]
    if p.notna().any() and p.max() > p.min():
        df["pressure"] = (p - p.min()) / (p.max() - p.min())
    return df


def preprocess(df):
    """Full preprocessing for one recording. Returns None if it is unusable."""
    df = clean(df)
    if len(df) < 10 or df["t"].iloc[-1] - df["t"].iloc[0] < MIN_SECONDS:
        return None
    df = resample(df)
    return normalise(df)