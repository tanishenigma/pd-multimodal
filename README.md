# pd-multimodal

**SHAP-explainable multimodal Parkinson's disease (PD) detection from handwriting and spiral drawings.**

This project combines scanned drawings (images) with pen/tablet trajectories (signals) to detect PD. Every model is validated at the subject level and across datasets, and every prediction comes with a SHAP or Grad-CAM explanation.

> Major Project I, B.Tech CSE & IT, Jaypee University of Information Technology (AY 2026-27).
> Supervisor: Dr. Hari Singh.
> Title: *SHAP-driven Multimodal Framework for Early Parkinson's Disease Detection via Handwriting and Spiral Analysis.*

**Status:** the merged dataset builder is complete. Features, models, explainability and the demo are planned (see [Roadmap](#roadmap)).

> This is a research project. It is not a medical device and must not be used for diagnosis.

---

## Contents

- [Dataset at a glance](#dataset-at-a-glance)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Data format](#data-format)
- [Sources](#sources)
- [Validation rules](#validation-rules)
- [Gated and manual sources](#gated-and-manual-sources)
- [Roadmap](#roadmap)
- [Known dead ends](#known-dead-ends)
- [License and citation](#license-and-citation)

---

## Dataset at a glance

| Modality | Samples | Subject groups | PD / HC groups |
|---|---|---|---|
| Images (scan + rendered) | 2963 (1450 scan, 1513 rendered) | 376 | 239 / 137 |
| Signals (pen/tablet) | 2385 | 260 | 148 / 112 |

Label convention: `1` = PD, `0` = healthy control (HC).

Tasks include spirals (static and dynamic), meanders, circles, waves, straight lines, diadochokinesis, letters, words and sentences (PaHaW, HandyDiag Thai handwriting).

Full per-source, per-task counts are in `data/processed/counts_images.csv` and `data/processed/counts_signals.csv`.

## Quick start

```bash
git clone https://github.com/tanishenigma/pd-multimodal.git
cd pd-multimodal
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pillow pyarrow matplotlib
python -I scripts/build_dataset.py
```

The build is idempotent. To reuse raw data you already downloaded, run:

```bash
python -I scripts/build_dataset.py --skip-download
```

What the builder does:

1. Downloads archives into `data/raw/dl/`, extracts them into `data/raw/<source>/`, and shallow-clones the data-bearing GitHub repos into `data/raw/repos/`.
2. Normalizes every scanned drawing to a 256x256 grayscale PNG (aspect kept, white padding).
3. Converts tablet/pen signals to one Parquet file per sample.
4. Renders the x,y pen-down trajectory of every signal to a 256x256 PNG (`modality=rendered`).
5. Deduplicates across sources (MD5 plus 32x32 thumbnail correlation >= 0.95), merges subjects that share images, and writes the manifests and dataset card.

Downloaded content is treated as data only. Nothing from it is executed.

## Repository layout

```
pd-multimodal/
├── scripts/
│   └── build_dataset.py          # download, normalize, render, dedupe, label
├── docs/
│   └── data_requests.md          # email drafts and request log for gated datasets
└── data/
    ├── raw/                      # downloads (git-ignored)
    │   └── manual/               # drop login-gated files here
    └── processed/
        ├── images/               # scan/ and rendered/ PNGs (git-ignored)
        ├── signals/              # *.parquet (git-ignored)
        ├── images_manifest.csv
        ├── signals_manifest.csv
        ├── counts_images.csv
        ├── counts_signals.csv
        ├── images_dropped_duplicates.csv
        ├── signals_skipped_duplicates.csv
        ├── subject_groups_merged.csv
        ├── DATASET_CARD.md       # per-source counts, licenses, caveats
        └── MANUAL_DOWNLOADS.md   # gated sources and where to put them
```

The manifests, counts and audit files are committed. The images and signals themselves are not, so run the builder to regenerate them.

## Data format

**`images_manifest.csv`**

`sample_id, path, source_dataset, task, label, subject_id, subject_group, original_path, modality (scan|rendered), split_hint, overlap_flag, note, src_md5`

**`signals_manifest.csv`**

The same columns with `modality=signal`, plus `n_samples` and `sampling`.

**Signal Parquet files** hold one sample each, with columns `t` (seconds), `x`, `y`, `pressure`, `pen_down`, plus source-specific extras such as tilt.

NewHandPD smart-pen signals have no x,y coordinates. They therefore have no rendered image.

## Sources

| Source | Modality | Notes |
|---|---|---|
| HandPD | scans | spiral, meander |
| NewHandPD | scans + smart-pen signals | spiral, meander, circle, diadochokinesis; shares subjects with HandPD |
| UCI 395 | tablet signals (+ some scans) | static/dynamic spiral, circle |
| UCI 358 | tablet signals | only files that differ from UCI 395 are kept |
| kmader spiral/wave | scans | via GitHub mirror `widushan/...` |
| PaHaW examples | tablet signals | 4 example subjects from `BDALab/handwriting-features` |
| HandyDiag v1.0 | tablet signals | Thai handwriting + drawings, 60 HC / 59 PD |

Each source keeps its own license. See `data/processed/DATASET_CARD.md` for licenses, citations and caveats.

## Validation rules

These rules exist to prevent inflated accuracy, which is common in PD handwriting papers.

- **Split on `subject_group`, never on `subject_id` or sample.** HandPD and NewHandPD re-ship 30 subjects and 288 byte-identical scans. `subject_group` merges them.
- **Ignore `split_hint` from kmader.** Its original train/test split leaks subjects.
- **Treat HandPD + NewHandPD as one cohort, and UCI 395 + 358 as one cohort.** They are not independent datasets.
- **Within-cohort:** grouped stratified 5-fold CV, repeated 5 times with fixed seeds.
- **Cross-dataset:** leave-one-cohort-out (LOCO). Train on all cohorts but one, test on the held-out one.
- **Class imbalance:** use class weights and report balanced accuracy, with 95% confidence intervals over repeats.
- Tune hyperparameters inside the training folds only (nested CV). Touch the final held-out fold once.

## Gated and manual sources

Some datasets need a login or a signed license. `data/processed/MANUAL_DOWNLOADS.md` lists where to put each file under `data/raw/manual/`. The builder picks them up on the next run.

`docs/data_requests.md` contains request emails and a send/reply log for:

- full PaHaW (BDALab, 37 PD / 38 HC, signed license),
- HandPDMultiMC (21 PD / 21 HC),
- DraWritePD, Cc-PhD, PARKINSON BRASIL spirals, NeuroDiag,
- IEEE DataPort iPad spiral (130 PD / 60 HC) and Iraqi smartphone drawings (30 PD / 30 HC), which only need a free IEEE account.

## Roadmap

- [x] Merged, deduplicated, subject-grouped dataset builder
- [ ] Repo hygiene: LICENSE, pinned `requirements.txt`, `Makefile` (`data`, `features`, `train`, `explain`), committed `splits/*.csv`
- [ ] More data: full PaHaW, IEEE DataPort sets, HandPDMultiMC
- [ ] Features
  - Image: spiral geometry (radius growth, loop spacing, Archimedean fit error), tremor residual, HOG/LBP, frozen ResNet-18/EfficientNet-B0 embeddings
  - Signal: velocity, acceleration, jerk, in-air time, pressure and tilt stats, 4-12 Hz tremor band power, entropy
  - Missing channels stay NaN with a missing flag (no silent imputation); documented in `features/feature_dictionary.md`
- [ ] Models: RF, SVM, XGBoost baselines; task-aware ResNet-18; early and late fusion; subject-level aggregation
- [ ] Explainability: SHAP (global, per-modality, per-subject), Grad-CAM on spirals, rank stability across folds and cohorts
- [ ] Evaluation: accuracy, balanced accuracy, sensitivity, specificity, F1, ROC-AUC with 95% CI; within-cohort vs LOCO tables
- [ ] Streamlit demo: upload a spiral image or signal file, see prediction, confidence and top SHAP features (screening aid only)

## Known dead ends

These were checked and rejected, so you do not need to repeat them:

- UCI 358 is mostly a copy of UCI 395 (35 of 40 files byte-identical).
- `NewSpiral.csv` `ID_PATIENT` values are inconsistent and are not used for subject mapping.
- `musaru/PD_PaHaW`, `khareyash05/...`, `IoannisKansizoglou/DrawingAwarePDdetection`, `ManasaGadepalli/...` and `sayakpaul/Parkinson-s-Disease-Classifier` hold no new raw data.
- PADS (PhysioNet) is smartwatch motion data, not handwriting.
- Zenodo 7967603 holds PD-only outcome scores. `seensiravit/Parkinson-SpiralDrawing` re-uploads HandPD and kmader.

## License and citation

No license has been chosen for the code yet. MIT is planned.

The datasets are **not** covered by this repository's license. Each source keeps its own terms, which are listed in `data/processed/DATASET_CARD.md`. Do not redistribute the processed images or signals. Cite every original dataset you use.


this is an edit!# copy-pd
