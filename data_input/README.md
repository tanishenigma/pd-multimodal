# data_input: input datasets and modality routing

Owner: Tanish Sharma (231030199)

Flowchart steps: **Start → Input datasets → Modality?**

- `load_data.py`
  - `load_manifests()` reads `data/processed/signals_manifest.csv` and `images_manifest.csv` (built by `scripts/build_dataset.py`).
  - `route_by_modality()` sends pen/tablet signals to the handwriting (kinematic) branch and spiral scans to the image branch.
  - `load_signal(path)` loads one recording (`t, x, y, pressure, pen_down` + device extras).

Run alone: `.venv/bin/python data_input/load_data.py` (prints samples and subject groups per source and label).
