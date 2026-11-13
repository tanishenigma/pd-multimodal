"""Step 3: kinematic feature extraction (handwriting branch).

One row of features per recording. Features follow the PD handwriting
literature (Drotar et al. 2015/2016, PaHaW):
  - timing: duration, pen-down ratio, in-air time, number of strokes
  - kinematics (needs x, y): path length, velocity, acceleration, jerk,
    number of velocity / acceleration sign changes (NCV, NCA)
  - pressure: mean, std, coefficient of variation
  - tremor: share of signal power in the 4-12 Hz band
NewHandPD's smart pen has no x, y, so its kinematic columns stay NaN and the
tremor feature is taken from the pen's accelerometer instead.
"""
import numpy as np

HZ = 100  # must match preprocessing.TARGET_HZ


def _stats(name, v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {f"{name}_mean": np.nan, f"{name}_std": np.nan, f"{name}_max": np.nan}
    return {f"{name}_mean": v.mean(), f"{name}_std": v.std(), f"{name}_max": v.max()}


def _sign_changes(v):
    v = v[~np.isnan(v)]
    return int(np.sum(np.diff(np.sign(v)) != 0)) if len(v) > 1 else np.nan


def tremor_ratio(signal, hz=HZ, band=(4.0, 12.0)):
    s = signal[~np.isnan(signal)]
    if len(s) < hz:  # under 1 s: not enough for a spectrum
        return np.nan
    s = s - s.mean()
    power = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s), 1.0 / hz)
    total = power[freqs > 0.5].sum()
    return power[(freqs >= band[0]) & (freqs <= band[1])].sum() / total if total else np.nan


def extract(df, hz=HZ):
    dt = 1.0 / hz
    pen = df["pen_down"].fillna(0).to_numpy() > 0
    f = {
        "duration_s": len(df) * dt,
        "pen_down_ratio": pen.mean(),
        "in_air_time_s": (~pen).sum() * dt,
        "n_strokes": int(np.sum(np.diff(pen.astype(int)) == 1) + pen[0]),
    }

    has_xy = not df[["x", "y"]].isna().all().all()
    if has_xy:
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        vx, vy = np.gradient(x, dt), np.gradient(y, dt)
        speed = np.hypot(vx, vy)
        acc = np.gradient(speed, dt)
        jerk = np.gradient(acc, dt)
        f["path_length"] = np.nansum(np.hypot(np.diff(x), np.diff(y)))
        f.update(_stats("velocity", speed))
        f.update(_stats("acceleration", np.abs(acc)))
        f.update(_stats("jerk", np.abs(jerk)))
        f["ncv"] = _sign_changes(vy)
        f["nca"] = _sign_changes(acc)
        f["tremor_ratio"] = tremor_ratio(speed, hz)
    else:
        for k in ["path_length", "ncv", "nca"]:
            f[k] = np.nan
        for k in ["velocity", "acceleration", "jerk"]:
            f.update(_stats(k, np.array([np.nan])))
        acc_cols = [c for c in ("tilt_accel_x", "tilt_accel_y", "tilt_accel_z") if c in df]
        mag = np.linalg.norm(df[acc_cols].to_numpy(), axis=1) if acc_cols else np.array([np.nan])
        f["tremor_ratio"] = tremor_ratio(mag, hz)

    p = df["pressure"].to_numpy(dtype=float)[pen] if pen.any() else np.array([np.nan])
    f["pressure_mean"] = np.nanmean(p) if len(p) else np.nan
    f["pressure_std"] = np.nanstd(p) if len(p) else np.nan
    f["pressure_cv"] = f["pressure_std"] / f["pressure_mean"] if f["pressure_mean"] else np.nan
    f["has_xy"] = int(has_xy)  # explicit missing-channel flag (no silent imputation)
    return f
