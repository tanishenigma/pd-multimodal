"""Runs the handwriting branch end to end:
input datasets -> modality routing -> preprocessing -> kinematic features.

Usage (from repo root):  .venv/bin/python kinematic_features/run_pipeline.py
Output: kinematic_features/output/kinematic_features.csv
"""
import sys
from pathlib import Path

import warnings

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data_input"), str(ROOT / "preprocessing"), str(ROOT / "kinematic_features")]

from load_data import load_manifests, load_signal, route_by_modality  # noqa: E402
from preprocess import preprocess  # noqa: E402
from features import extract  # noqa: E402

META = ["sample_id", "source_dataset", "task", "label", "subject_id", "subject_group"]


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty-slice warnings on missing channels
    signals, images = load_manifests()
    handwriting, _ = route_by_modality(signals, images)
    rows, skipped = [], 0
    for _, r in handwriting.iterrows():
        df = preprocess(load_signal(r["path"]))
        if df is None:
            skipped += 1
            continue
        rows.append({**{k: r[k] for k in META}, **extract(df)})
    out = pd.DataFrame(rows)
    path = Path(__file__).parent / "output" / "kinematic_features.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{len(out)} recordings -> {path.relative_to(ROOT)} ({skipped} skipped as too short)")
    print(out.groupby(["source_dataset", "label"]).size().unstack(fill_value=0).to_string())
    print("\nmean of key features by label (0=HC, 1=PD):")
    print(out.groupby("label")[["velocity_mean", "pressure_mean", "in_air_time_s", "tremor_ratio"]].mean().to_string())


if __name__ == "__main__":
    main()
