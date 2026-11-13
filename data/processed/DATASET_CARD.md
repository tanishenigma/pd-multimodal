# PD Multimodal Handwriting/Spiral Dataset: Dataset Card

Built by `scripts/build_dataset.py` (run `.venv/bin/python -I scripts/build_dataset.py`; `--skip-download` reuses raw data).
Label convention: `1` = Parkinson's disease (PD), `0` = healthy control (HC).
Full count tables are in `counts_images.csv` and `counts_signals.csv`.

## Files
- `images/scan/*.png`: scanned or exported drawings, grayscale, 256x256, aspect kept, white padding.
- `images/rendered/*.png`: x,y pen-down trajectories rendered from the tablet signals (256x256, y axis up).
- `images_manifest.csv` columns: `sample_id, path, source_dataset, task, label, subject_id, subject_group, original_path, modality (scan|rendered), split_hint, overlap_flag, note, src_md5`.
- `signals/*.parquet`: one file per sample, columns `t` (seconds), `x`, `y`, `pressure`, `pen_down`, plus source-specific extras.
- `signals_manifest.csv` columns: the same as the image manifest, with `modality=signal`, plus `n_samples` and `sampling`.
- `images_dropped_duplicates.csv`, `signals_skipped_duplicates.csv`, `subject_groups_merged.csv`: the deduplication audit trail.
- **Use `subject_group`, not `subject_id`, for train/test splits.** It merges subject IDs that share identical images (see the caveats below).

## Totals
- Images: 2809 (1296 scan + 1513 rendered), 374 subject groups (238 PD, 136 HC).
- Signals: 2385, 260 subject groups (148 PD, 112 HC).
- New in this revision: HandyDiag (Thai handwriting + drawing tablet signals, 60 HC / 59 PD), 1252 signals and 1172 rendered images. It is the first source with real word and sentence handwriting at scale.

## Scanned images after deduplication (samples / subjects)
| source | task | HC | PD |
|---|---|---|---|
| handpd | spiral | 44 / 11 | 200 / 50 |
| handpd | meander | 44 / 11 | 204 / 51 |
| newhandpd | spiral | 140 / 35 | 104 / 26 |
| newhandpd | meander | 140 / 35 | 100 / 25 |
| newhandpd | circle | 35 / 35 | 31 / 31 |
| kmader_drawings | spiral | 51 / 13 | 51 / 15 |
| kmader_drawings | wave | 51 / 13 | 51 / 15 |
| uci395 | static_spiral | 0 | 25 / 25 |
| uci395 | dynamic_spiral | 0 | 25 / 25 |

## Signals (samples / subjects). Every signal with x,y also has a `<task>_rendered` image.
| source | task | HC | PD |
|---|---|---|---|
| uci395 | static_spiral | 15 / 15 | 61 / 61 |
| uci395 | dynamic_spiral | 15 / 15 | 57 / 57 |
| uci395 | circle (STCP) | 7 / 7 | 47 / 47 |
| uci358 (non-identical files only) | static + dynamic spiral | 5+5 / 5 | 0 |
| newhandpd | spiral, meander | 140 / 35 each | 124 / 31 each |
| newhandpd | circle, circle_air, diadochokinesis_a, diadochokinesis_b | 35 / 35 each | 31 / 31 each |
| pahaw_examples | spiral + handwriting_t2 to t14 | 65 / 2 | 64 / 2 |
| handydiag | spiral | 105 / 60 | 106 / 59 |
| handydiag | line_vertical, line_horizontal | 105 / 60 each | 106 / 59 each |
| handydiag | handwriting_letters (Thai words) | 105 / 60 | 105 / 59 |
| handydiag | handwriting_letters2 | 105 / 60 | 98 / 58 |
| handydiag | handwriting_th_sentence (Thai sentence) | 105 / 60 | 101 / 59 |

HandyDiag counts include both hands; the hand is in `note`. 80 HandyDiag signals have no x,y (see below), so 1172 of 1252 are rendered.

## Per source

### UCI 395: Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet
- Data: `hw_dataset/control` (15 HC), `hw_dataset/parkinson` (25 PD), `new_dataset/parkinson` (37 PD).
- Format: `X;Y;Z;Pressure;GripAngle;Timestamp;TestID`. TestID 0 is the static spiral, 1 the dynamic spiral, 2 the circular-motion (STCP) test. Each file is split into one sample per test.
- `t` is the timestamp in ms divided by 1000. Extra columns: `z`, `grip_angle`. `pen_down` means pressure > 0.
- `hw_drawings`: 25 static and 25 dynamic PNGs. **They are labelled PD by inference:** the count matches the 25 PWP and UCI describes them as patient drawings. They cannot be linked to the signal files, so they get their own subject IDs (`uci395_drawing_NN`).
- Labels come from the folder names.
- License: CC BY 4.0. Cite: Isenkul, Sakar, Kursun, "Improved spiral test using digitized graphics tablet for monitoring Parkinson's disease", ICEHTM 2014.
- Caveat: the STCP circle renders look staircase-like because the coordinates are coarse integers over a small area.

### UCI 358: Improved Spiral Test (same authors)
- Its 40 files (15 Healthy, 25 PWP) are the same cohort as UCI 395 `hw_dataset`.
- 35 files are byte-identical copies and were skipped (see `signals_skipped_duplicates.csv`).
- The other 5 Healthy files start with the same first sample as UCI395 C_0001, C_0002, C_0005, C_0006 and C_0007. They are kept, mapped to those subject IDs, and flagged in `note`.
- Its drawings are byte-identical to UCI 395's, so they were not added again.
- License: CC BY 4.0, same citation as UCI 395.

### HandPD (original, 2016): Spiral_HandPD and Meander_HandPD
- 92 subjects (18 HC, 74 PD), 4 spirals and 4 meanders each, from Botucatu Medical School (UNESP).
- Labels come from the `*Control` / `*Patients` folders. The subject is the leading number of the filename (`0044-4.jpg`).
- Cite: Pereira et al., "A new computer vision-based approach to aid the diagnosis of Parkinson's disease", Computer Methods and Programs in Biomedicine, 2016.
- License: none stated. Academic use, with citation.

### NewHandPD (2016)
- 35 HC and 31 PD subjects. Each subject has 4 spirals, 4 meanders and 1 circle as images, plus 12 BiSP smart-pen signals.
- Labels come from the archive (Healthy* or Patient*), checked against the H/P tag in the filename.
- Filename anomalies:
  - The HealthyCircle archive names its files `circA-P<n>.jpg`. They are treated as HC subject H<n>.
  - `HealthySignal/sigMea1-28.txt` has no H/P tag. It is treated as H28.
- Signals have 6 channels, assigned in the order described in the literature (Pereira et al., SIBGRAPI 2016): microphone, finger_grip, axial_pressure (stored as `pressure`), tilt_accel_x, tilt_accel_y, tilt_accel_z. Sampling is 1000 Hz, read from the header.
- **The pen gives no x,y position**, so `x` and `y` are NaN and these signals have no rendered images.
- Task mapping:
  - circA: circle on paper (the matching image exists).
  - circB: circle in the air.
  - sigDiaA/B: diadochokinesis.
- Cite: Pereira et al., "Deep learning-aided Parkinson's disease diagnosis from handwritten dynamics", SIBGRAPI 2016.

### Kaggle kmader/parkinsons-drawings (spiral + wave)
- Taken from the GitHub mirror `widushan/Parkinson-Disease-Detection-from-Spiral-and-Wave-Drawings` (`data/raw/repos/widushan`). The Kaggle CLI was not installed and needs a login.
- Filename `V<nn><H|P><E|O><rep>`: E = spiral, O = wave. The H/P tag was asserted against the folder label.
- The subject is `V<nn>` plus the H/P tag (28 subjects).
- `split_hint` holds the original train/test split. **That split leaks subjects**, so do not use it.
- Origin: Zham et al., "Distinguishing different stages of Parkinson's disease using composite index of speed and pen-pressure of sketching a spiral", Front. Neurol. 2017.

### PaHaW: only the 4 public example subjects
- Taken from `BDALab/handwriting-features/examples/data` (MIT license). There are 2 HC and 2 PD subjects; labels come from the folder names.
- SVC columns, per the bundled `info.json`: Y, X, timestamp, button, azimuth, altitude, pressure. Extras kept: `azimuth`, `altitude`, `button`. `pen_down` means button = 1.
- Task 1 is the Archimedean spiral. Tasks 2 to 14 are kept as `handwriting_tN`.
- The full PaHaW database (75 subjects) needs a signed license request to BDALab (https://bdalab.utko.feec.vutbr.cz/), so it was not downloaded.
- Cite: Drotar et al., "Evaluation of handwriting kinematics and pressure for differential diagnosis of Parkinson's disease", Artif. Intell. Med. 2016.

### HandyDiag (Thai digital handwriting, 2026)
- Source: GitHub `suchadat/HandyDiag_v1.0` (`Data_HC.zip`, `Data_PD.zip`), cloned into `data/raw/repos/handydiag` and extracted to `data/raw/handydiag`.
- 119 subjects: 60 HC and 59 PD. Labels come from the `Data_HC` / `Data_PD` folders. One JSON per subject, with age, sex, education and dominant hand (kept in `note`).
- Tasks, each with the right (R) and left (L) hand: Archimedean spiral, vertical lines, horizontal lines, `Letters` (word writing), `Letters 2`, and `TH Letters` (Thai sentence). Names come from the authors' MATLAB code (`extract_letters_feature.m`: Letters = words, THLetters = sentences).
- Arrays: `dx`, `dy`, `PenPressure`, `Timestamp` (seconds). `null` entries mark pen lifts, so `pen_down` is false there. The coordinates are screen pixels with y pointing down, so `y = -y_screen` (the raw value is kept in `y_screen`).
- Key spellings differ between files (`Spiral R dx` vs `SpiralRDx`). They are normalized by dropping spaces and lowercasing.
- Data defects:
  - In 89 subject-hand recordings the `Letters 2` dx/dy arrays are copies of `Letters`. In 7 of these the pressure is identical too, so the sample was skipped. In the other 82, `x` and `y` are NaN and noted. 80 of them were kept; 2 had empty arrays.
  - 37 task/hand arrays are empty (mostly the left hand). They were skipped and are listed in `signals_skipped_duplicates.csv`.
- License: non-commercial research and education only. Cite Chanwimalueang, Methawasin, Saetee, Rueangrong, Rung-ruang, Tantisatirapong, "Digital Thai Handwriting for Parkinson's Disease Screening", Scientific Reports, 2026.

### Manual-drop hooks (empty by default)
- `data/raw/manual/images/<source>/...`: login-gated image sets such as Kaggle zips. The label comes from a folder name (healthy/control/hc/normal or parkinson/pd/patient/pwp). Exact md5 duplicates of existing images are dropped, and near-duplicates are flagged in `overlap_flag`. `source_dataset` is `manual_<source>`.
- `data/raw/manual/pahaw/`: the full licensed PaHaW. Files identical to `pahaw_examples` are skipped. `source_dataset` is `pahaw_full`.
- See `MANUAL_DOWNLOADS.md`.

## Subject-overlap caveats (important)
- **HandPD and NewHandPD overlap heavily.** 288 NewHandPD/HandPD scans are byte-identical:
  - 30 HandPD subjects are re-shipped as NewHandPD subjects, for example handpd_0098 as newhandpd_H11 and handpd_0261 as newhandpd_P10.
  - Inside NewHandPD, some patient IDs share identical images, for example P1, P4 and P25, or P2 and P26.
  - The labels always agree.
- How this was handled:
  - Exact duplicate images were dropped, keeping the NewHandPD copy (list in `images_dropped_duplicates.csv`).
  - Linked IDs were merged into `subject_group` (`subject_groups_merged.csv`).
- 8 further near-duplicates (32x32 thumbnail correlation of at least 0.95) are flagged in `overlap_flag`, but not dropped.
- The `ID_PATIENT` column of NewSpiral.csv is inconsistent (for example, healthy subjects carry IDs that HandPD uses for patients), so it was not used.
- The remaining HandPD and NewHandPD subjects may still be the same people. Treat the two as one cohort when splitting.
- UCI 395 and UCI 358 are one cohort. The UCI drawings may be the same people as the UCI PD signal subjects.

## Candidates checked and rejected as duplicates (Nov 2026)
- Mendeley Data `fd5wd6wmdj` ("Parkinson's Disease Detection Using Spiral Images", public download): 254 images. All 254 are byte-identical (md5) to kmader spiral/wave plus the UCI 395 drawings.
- GitHub `shalusingh-tech/Spiral_pd_dataset`: 310 images, all md5-identical to existing NewHandPD/HandPD spirals.
- GitHub `seensiravit/Parkinson-SpiralDrawing`: 447 images re-encoded from HandPD/NewHandPD (`sp1-H32`) and kmader (`V55HE12`), as its README says. They are not md5-identical, and its `degrade` folders are augmented. Rejected on provenance.
- Already present (UCI 395 copies): `irenemao/Handwriting-Analysis`, `MichaelNormandyGavin/parkinsons`, `Kashish-jain/Parkinsons-training-data` on HuggingFace, and Kaggle `team-ai/parkinson-disease-spiral-drawings`.

## Candidates checked and rejected for other reasons
- HuggingFace `cycool29/spiral-nd-drawings`: 41 spirals with no labels.
- GitHub `souvikchakraborty98/WACOM`: a few ad-hoc Wacom recordings (3-4 PD patients; the controls are the developers). There is no license and the task protocol is unclear.
- Zenodo 15831979 (PARKINSON BRASIL spirals, NIATS-UFU, 15 HC / 24 PD): restricted access.
- PLOS figshare 1034789 (Smits et al., standardized handwriting): only figures and methods, no raw data.
- Cc-PhD (PLOS ONE 2025, `dreamhcy/MLforPD`): available only through an application form.
- HandPDMultiMC / PDMultiMC: no public download link found.
- DraWritePD: only segmentation code (`SvenNomm/drawritepd_segmentation`); the data is not public.

## Failed or not obtained
- Full PaHaW: needs a license or login.
- Kaggle kmader via the Kaggle CLI: not installed and needs a login. The GitHub mirror was used instead.
- `sayakpaul/Parkinson-s-Disease-Classifier`: only the kmader wave images, which are redundant.
- `musaru/PD_PaHaW`: only derived feature CSVs, no raw data.
- `khareyash05/...`: only models and notebooks, no data.
- `IoannisKansizoglou/DrawingAwarePDdetection`: only figures.
- `ManasaGadepalli/...`: only a manifest of NewHandPD.
- All the NewHandPD sibling archives (Meander, Circle, Signal for both groups) downloaded fine.
