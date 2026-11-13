# Manual downloads (login- or license-gated)

`scripts/build_dataset.py` cannot fetch these sources automatically. Download them yourself, put them at the path given, and rerun `.venv/bin/python -I scripts/build_dataset.py --skip-download`. Exact duplicates of data already in the set are dropped by md5 and listed in `images_dropped_duplicates.csv` or `signals_skipped_duplicates.csv`. Near-duplicates are flagged in `overlap_flag`.

There are two hooks:
- **Images**: `data/raw/manual/images/<source_name>/**/<label_dir>/*.{png,jpg,jpeg,bmp,tif}`.
  - The label comes from the nearest folder named `healthy`, `control(s)`, `hc` or `normal` (HC), or `parkinson(s)`, `pd`, `patient(s)` or `pwp` (PD).
  - The first folder under `<source_name>` becomes the task, unless it is a train/test folder.
  - The subject is unknown, so each image gets its own `subject_id`. Fix this by hand if the source documents subjects.
- **Full PaHaW signals**: `data/raw/manual/pahaw/`. Expects `*/<ID>__<task>_<rep>.svc` plus `corpus_PaHaW.xlsx`, or a CSV with `ID` and `Disease` columns. Reading the xlsx needs `pip install openpyxl`.

| Source | URL | Access | What it adds | Drop location |
|---|---|---|---|---|
| PaHaW (full) | https://bdalab.utko.feec.vutbr.cz/ (license form, e-mail BDALab) | Signed license | 37 PD and 38 HC, Czech. Spiral plus letters `l`, `le`, `les`, words and a sentence, as tablet signals. **Real handwriting with controls.** | Unzip into `data/raw/manual/pahaw/` |
| Kaggle kmader/parkinsons-drawings | https://www.kaggle.com/datasets/kmader/parkinsons-drawings | Kaggle login | Nothing new; it is already in the set via the GitHub mirror. | Not needed |
| Kaggle team-ai/parkinson-disease-spiral-drawings | https://www.kaggle.com/datasets/team-ai/parkinson-disease-spiral-drawings | Kaggle login | Nothing new; it is UCI 395. | Not needed |
| Kaggle HandPD mirrors | https://www.kaggle.com/search?q=handpd | Kaggle login | Expected to be duplicates. Drop them in to have them checked. | `data/raw/manual/images/<name>/` |
| Other Kaggle PD drawing sets (search "parkinson spiral", "parkinson handwriting") | https://www.kaggle.com/search?q=parkinson+handwriting | Kaggle login | Unknown. Many are kmader/HandPD re-uploads or augmented copies. Do not add augmented folders. | `data/raw/manual/images/<name>/<task>/<healthy\|parkinson>/` |
| iPad spiral drawings (IEEE DataPort) | https://ieee-dataport.org/ (DOI 10.21227/tydt-1h34) | IEEE account or subscription | iPad spiral drawings from PD and HC. No open mirror found. | Images: `data/raw/manual/images/ipad_spiral/spiral/<healthy\|parkinson>/`. Raw x,y needs a parser added (see `manual_pahaw_signal_rows` as a template). |
| Cc-PhD (PLOS ONE 2025) | https://github.com/dreamhcy/MLforPD (application form) | Application, academic use | Chinese PD/ET/HC tablet handwriting: spirals, characters, sentences. | Needs a new parser; put the raw files in `data/raw/manual/ccphd/` |
| PARKINSON BRASIL spirals (NIATS-UFU) | https://zenodo.org/records/15831979 | Restricted (request on Zenodo) | Paper spirals, 15 HC / 24 PD, 5 spirals per session. | `data/raw/manual/images/niats_ufu/spiral/<healthy\|parkinson>/` |
| HandPDMultiMC / PDMultiMC | Contact the authors (Institut Polytechnique de Paris; see "Detection of Parkinson's disease from handwriting using deep learning") | On request | Online handwriting, 21 PD / 21 HC. | Needs a new parser; put the raw files in `data/raw/manual/handpdmultimc/` |
| DraWritePD | Contact the Tallinn University of Technology group (S. Nõmm) | On request | Drawing and writing tablet tasks, PD and HC. | Needs a new parser; put the raw files in `data/raw/manual/drawritepd/` |
