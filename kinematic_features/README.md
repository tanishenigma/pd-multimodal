# kinematic_features: kinematic feature extraction

Owner: Bhumika Chadha (231030036)

Flowchart step: **Kinematic Feature Extraction** (handwriting branch).

- `features.py` → `extract(df)`: one feature row per recording.
  - Timing: duration, pen-down ratio, in-air time, stroke count.
  - Kinematics: path length, velocity, acceleration, jerk (mean/std/max), NCV, NCA.
  - Pressure: mean, std, coefficient of variation.
  - Tremor: power share in the 4–12 Hz band.
  - `has_xy` flag: NewHandPD smart pen has no x, y, so its kinematic columns are NaN and tremor comes from the pen accelerometer.
- `run_pipeline.py`: runs data_input → preprocessing → features for all handwriting recordings.

Run: `.venv/bin/python kinematic_features/run_pipeline.py` → `kinematic_features/output/kinematic_features.csv` (not committed).
