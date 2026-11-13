"""Step 1: input datasets and modality routing.

Reads the merged manifests built by scripts/build_dataset.py and splits the
samples by modality, matching the flowchart's "Modality?" decision:
  - handwriting data (pen/tablet signals) -> kinematic branch
  - spiral images (scans)                  -> image branch (not implemented yet)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load_manifests():
    """Return (signals, images) manifests as DataFrames."""
    signals = pd.read_csv(PROCESSED / "signals_manifest.csv")
    images = pd.read_csv(PROCESSED / "images_manifest.csv")
    return signals, images


def route_by_modality(signals, images):
    """Split samples into the two branches of the flowchart."""
    handwriting = signals[signals["modality"] == "signal"].copy()
    spiral_images = images[(images["modality"] == "scan") & images["task"].str.contains("spiral")].copy()
    return handwriting, spiral_images


def load_signal(path):
    """Load one pen/tablet recording (columns: t, x, y, pressure, pen_down, plus source extras)."""
    return pd.read_parquet(ROOT / path)


def summary(df, name):
    by = df.groupby(["source_dataset", "label"])["subject_group"].nunique().unstack(fill_value=0)
    by.columns = ["HC" if c == 0 else "PD" for c in by.columns]
    print(f"\n{name}: {len(df)} samples, {df['subject_group'].nunique()} subject groups")
    print(by.to_string())


if __name__ == "__main__":
    sig, img = load_manifests()
    hw, sp = route_by_modality(sig, img)
    summary(hw, "Handwriting signals (kinematic branch)")
    summary(sp, "Spiral images (image branch)")
