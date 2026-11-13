# preprocessing: handwriting signal preprocessing

Owner: Agniprabha Ganguly (231030206)

Flowchart step: **Data preprocessing** (handwriting signals).

`preprocess.py` → `preprocess(df)`:
1. Sort by time, drop repeated timestamps.
2. Resample to 100 Hz (devices record at different rates).
3. Normalise x, y to zero mean and unit scale (removes device units).
4. Scale pressure to [0, 1] per recording.
5. Reject recordings shorter than 0.5 s.

Missing channels (NewHandPD has no x, y) stay NaN; they are never filled in.
