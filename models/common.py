"""Shared helpers for the signal model: subject-grouped folds and metrics.

The model uses a fixed fold file (splits/folds.csv), keyed by subject_group,
so a person never appears in both train and test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "splits" / "folds.csv"
OUT = ROOT / "models" / "output"
SEED = 42
N_FOLDS = 5


def make_folds():
    """Assign each subject_group to one of N_FOLDS folds, stratified by label."""
    img = pd.read_csv(PROCESSED / "images_manifest.csv")
    sig = pd.read_csv(PROCESSED / "signals_manifest.csv")
    subj = (pd.concat([img, sig])[["subject_group", "label"]]
            .groupby("subject_group")["label"].max().reset_index())
    folds = np.zeros(len(subj), dtype=int)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(skf.split(subj, subj["label"], subj["subject_group"])):
        folds[te] = k
    subj["fold"] = folds
    SPLITS.parent.mkdir(exist_ok=True)
    subj.to_csv(SPLITS, index=False)
    return subj


def load_folds():
    return pd.read_csv(SPLITS) if SPLITS.exists() else make_folds()


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1),
        "specificity": recall_score(y, pred, pos_label=0),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
    }


def subject_level(oof):
    """Mean of sample probabilities per subject_group (one decision per person)."""
    return oof.groupby("subject_group").agg(label=("label", "max"), prob=("prob", "mean"),
                                            fold=("fold", "first")).reset_index()


def report(name, oof):
    """Print and save sample- and subject-level metrics, overall and per fold."""
    OUT.mkdir(parents=True, exist_ok=True)
    subj = subject_level(oof)
    rows = [{"model": name, "level": "sample", **metrics(oof["label"], oof["prob"])},
            {"model": name, "level": "subject", **metrics(subj["label"], subj["prob"])}]
    per_fold = [metrics(g["label"], g["prob"]) for _, g in subj.groupby("fold")]
    auc = np.array([m["roc_auc"] for m in per_fold])
    bacc = np.array([m["balanced_accuracy"] for m in per_fold])
    rows[1]["roc_auc_fold_std"] = np.nanstd(auc)
    rows[1]["bacc_fold_std"] = bacc.std()
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{name}_metrics.csv", index=False)
    oof.to_csv(OUT / f"{name}_oof.csv", index=False)
    print(f"\n== {name} ==")
    print(res.round(3).to_string(index=False))
    return res
