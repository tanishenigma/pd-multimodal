#!/usr/bin/env python3
"""Build merged Parkinson's disease handwriting/spiral dataset.

Usage (from project root):  .venv/bin/python -I scripts/build_dataset.py [--skip-download]

Steps (idempotent):
  1. download archives into data/raw/dl/ (skip if present), extract into data/raw/<source>/,
     shallow-clone data-bearing GitHub repos into data/raw/repos/.
  2. normalize all scanned drawing images -> data/processed/images/ (grayscale, 256x256 PNG, white pad)
  3. convert dynamic tablet/pen signals -> data/processed/signals/*.parquet (t, x, y, pressure, extras)
  4. render x,y trajectories of signals -> 256x256 PNG (modality=rendered)
  5. write images_manifest.csv, signals_manifest.csv, stats.json
Downloaded content is treated as data only: nothing from it is executed.
"""
import argparse, hashlib, json, os, re, subprocess, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DL = RAW / "dl"
OUT = ROOT / "data/processed"
IMG = OUT / "images"
SIG = OUT / "signals"
SIZE = 256
NEARDUP_CORR = 0.95  # 32x32 thumbnail correlation threshold for cross-dataset near-duplicates

UCI395 = "https://archive.ics.uci.edu/static/public/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip"
UCI358 = "https://archive.ics.uci.edu/static/public/358/improved+spiral+test+using+digitized+graphics+tablet+for+monitoring+parkinson+s+disease.zip"
HPD = "https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd"
ARCHIVES = {  # extract dir -> url
    "uci395": UCI395, "uci358": UCI358,
    "Spiral_HandPD": f"{HPD}/Spiral_HandPD.zip", "Meander_HandPD": f"{HPD}/Meander_HandPD.zip",
    "HealthySpiral": f"{HPD}/NewHealthy/HealthySpiral.zip", "HealthyMeander": f"{HPD}/NewHealthy/HealthyMeander.zip",
    "HealthyCircle": f"{HPD}/NewHealthy/HealthyCircle.zip", "HealthySignal": f"{HPD}/NewHealthy/HealthySignal.zip",
    "PatientSpiral": f"{HPD}/NewPatients/PatientSpiral.zip", "PatientMeander": f"{HPD}/NewPatients/PatientMeander.zip",
    "PatientCircle": f"{HPD}/NewPatients/PatientCircle.zip", "PatientSignal": f"{HPD}/NewPatients/PatientSignal.zip",
}
EXTRA_FILES = [f"{HPD}/{f}" for f in ("Spiral_HandPD.csv", "Meander_HandPD.csv", "NewSpiral.csv", "NewMeander.csv")]
REPOS = {  # only data-bearing repos are used; others were checked and hold no raw data
    "handwriting-features": "https://github.com/BDALab/handwriting-features.git",  # 4 PaHaW example subjects
    "widushan": "https://github.com/widushan/Parkinson-Disease-Detection-from-Spiral-and-Wave-Drawings.git",  # kmader mirror
    "handydiag": "https://github.com/suchadat/HandyDiag_v1.0.git",  # Thai handwriting + drawing tablet signals, 60 HC / 59 PD
}
REPO_ZIPS = {"handydiag": ("handydiag", ["Data_HC.zip", "Data_PD.zip"])}  # repo -> (extract dir, data zips inside repo)
# Login-gated / license-gated sources: drop files here by hand (see data/processed/MANUAL_DOWNLOADS.md).
MANUAL = RAW / "manual"
MANUAL_LABEL_DIRS = {"healthy": 0, "control": 0, "controls": 0, "hc": 0, "normal": 0,
                     "parkinson": 1, "parkinsons": 1, "pd": 1, "patient": 1, "patients": 1, "pwp": 1}


# ---------------------------------------------------------------- download
def sh(*a):
    subprocess.run(a, check=True)


def safe_extract(zpath, dest):
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():  # guard against path traversal
            if m.startswith("/") or ".." in Path(m).parts:
                raise RuntimeError(f"unsafe member {m}")
        zf.extractall(dest)


def download():
    DL.mkdir(parents=True, exist_ok=True)
    for name, url in list(ARCHIVES.items()):
        z = DL / Path(url).name
        if not z.exists() or z.stat().st_size == 0:
            sh("curl", "-sSL", "--retry", "5", "--retry-delay", "3", "-o", str(z), url)
        dest = RAW / name
        if not dest.exists():
            dest.mkdir(parents=True)
            safe_extract(z, dest)
    for url in EXTRA_FILES:
        f = DL / Path(url).name
        if not f.exists():
            sh("curl", "-sSL", "--retry", "5", "-o", str(f), url)
    (RAW / "repos").mkdir(exist_ok=True)
    for name, url in REPOS.items():
        d = RAW / "repos" / name
        if not d.exists():
            env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
            subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(d)], check=True, env=env)
    for repo, (name, zips) in REPO_ZIPS.items():
        dest = RAW / name
        if not dest.exists():
            dest.mkdir(parents=True)
            for z in zips:
                safe_extract(RAW / "repos" / repo / z, dest)
    MANUAL.mkdir(exist_ok=True)


# ---------------------------------------------------------------- helpers
def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def norm_image(args):
    src, dst = args
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("L")
    im.thumbnail((SIZE, SIZE), Image.LANCZOS)
    canvas = Image.new("L", (SIZE, SIZE), 255)
    canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    canvas.save(dst, optimize=True)
    v = np.asarray(ImageOps.autocontrast(canvas).resize((32, 32), Image.BOX), dtype=np.float32).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)  # unit vector for near-duplicate correlation


def render(args):
    pq, dst = args
    df = pd.read_parquet(pq, columns=["x", "y", "pen_down"])
    x, y, down = df.x.to_numpy(float), df.y.to_numpy(float), df.pen_down.to_numpy(bool)
    if len(x) < 2 or np.all(np.isnan(x)):
        return False
    xs, ys = x[down], y[down]
    if len(xs) < 2:
        xs, ys, down = x, y, np.ones_like(down)
    mnx, mxx, mny, mxy = np.nanmin(xs), np.nanmax(xs), np.nanmin(ys), np.nanmax(ys)
    span = max(mxx - mnx, mxy - mny, 1e-9)
    pad = 12
    s = (SIZE - 2 * pad) / span
    px = pad + (x - mnx) * s + ((SIZE - 2 * pad) - (mxx - mnx) * s) / 2
    py = SIZE - (pad + (y - mny) * s + ((SIZE - 2 * pad) - (mxy - mny) * s) / 2)  # y up
    sc = 4  # supersample
    im = Image.new("L", (SIZE * sc, SIZE * sc), 255)
    dr = ImageDraw.Draw(im)
    start = None
    for i in range(len(px) + 1):
        if i < len(px) and down[i]:
            start = i if start is None else start
        elif start is not None:
            pts = list(zip(px[start:i] * sc, py[start:i] * sc))
            if len(pts) > 1:
                dr.line(pts, fill=0, width=2 * sc, joint="curve")
            start = None
    im.resize((SIZE, SIZE), Image.LANCZOS).save(dst, optimize=True)
    return True


def rel(p):
    return str(Path(p).relative_to(ROOT))


# ---------------------------------------------------------------- image sources
def collect_images():
    rows = []

    def add(src, source, task, label, subject, sid, **kw):
        rows.append(dict(sample_id=sid, src=src, source_dataset=source, task=task, label=label,
                         subject_id=subject, original_path=rel(src), modality="scan", **kw))

    # HandPD (original, 2016): label from folder, subject = leading number
    for task, d in (("spiral", "Spiral_HandPD"), ("meander", "Meander_HandPD")):
        for sub, lab in ((f"{task.title()}Control", 0), (f"{task.title()}Patients", 1)):
            for f in sorted((RAW / d / sub).glob("*.jpg")):
                sj, rep = f.stem.split("-")
                add(f, "handpd", task, lab, f"handpd_{int(sj):04d}", f"handpd_{task}_{int(sj):04d}_{rep}")
    # NewHandPD images: label from archive (Healthy*/Patient*), subject from H/P number in filename.
    # HealthyCircle files are misnamed circA-P<n>.jpg inside the Healthy archive -> HC, subject H<n>.
    for arch, task, lab in (("HealthySpiral", "spiral", 0), ("HealthyMeander", "meander", 0), ("HealthyCircle", "circle", 0),
                            ("PatientSpiral", "spiral", 1), ("PatientMeander", "meander", 1), ("PatientCircle", "circle", 1)):
        for f in sorted((RAW / arch).rglob("*.jpg")):
            if "__MACOSX" in f.parts:
                continue
            m = re.match(r"([a-zA-Z]+)(\d*)-[HPhp](\d+)$", f.stem)
            g = "H" if lab == 0 else "P"
            sj = f"newhandpd_{g}{int(m.group(3))}"
            add(f, "newhandpd", task, lab, sj, f"newhandpd_{task}_{g}{int(m.group(3))}_{m.group(2) or 'A'}")
    # UCI 395 drawings (identical copies ship in UCI 358). 25 static + 25 dynamic PNGs; count matches the 25 PWP of
    # hw_dataset and UCI docs describe them as PWP drawings -> label 1 (subject linkage to signal files unknown).
    for task, d, p in (("static_spiral", "Static Spiral Test", "s"), ("dynamic_spiral", "Dynamic Spiral Test", "d")):
        for f in sorted((RAW / "uci395/hw_drawings" / d).glob("*.png")):
            n = int(f.stem[1:])
            add(f, "uci395", task, 1, f"uci395_drawing_{n:02d}", f"uci395_{task}_drawing_{n:02d}")
    # Kaggle kmader/parkinsons-drawings (GitHub mirror). Name V<nn><H|P><E|O><rep>; E=spiral, O=wave.
    base = RAW / "repos/widushan/dataset"
    for f in sorted(base.rglob("*.png")):
        task, split, cls = f.relative_to(base).parts[:3]
        m = re.match(r"V(\d+)([HP])([EO])(\d+)$", f.stem)
        lab = 1 if cls == "parkinson" else 0
        assert (m.group(2) == "P") == bool(lab), f
        add(f, "kmader_drawings", task, lab, f"kmader_V{m.group(1)}{m.group(2)}",
            f"kmader_{task}_V{m.group(1)}{m.group(2)}_{m.group(4)}_{split}", split_hint=split)
    # Manual drops (login-gated Kaggle etc.): data/raw/manual/images/<source>/[<task>/...]<label dir>/*.png|jpg.
    # Label from the nearest folder named in MANUAL_LABEL_DIRS; task = first folder under <source> unless it is a
    # label/split folder. Exact duplicates of existing images are dropped later by md5 (lowest priority).
    for srcdir in sorted(p for p in (MANUAL / "images").glob("*") if p.is_dir()):
        for f in sorted(srcdir.rglob("*")):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff") or "__MACOSX" in f.parts:
                continue
            parts = [x.lower() for x in f.relative_to(srcdir).parts[:-1]]
            labs = [MANUAL_LABEL_DIRS[x] for x in parts if x in MANUAL_LABEL_DIRS]
            if not labs:
                continue
            task = parts[0] if parts and parts[0] not in MANUAL_LABEL_DIRS and parts[0] not in (
                "train", "training", "test", "testing", "val") else "unknown"
            name = re.sub(r"\W+", "_", srcdir.name).strip("_")
            stem = re.sub(r"\W+", "_", "_".join(f.relative_to(srcdir).with_suffix("").parts)).strip("_")
            add(f, f"manual_{name}", task, labs[-1], f"manual_{name}_{stem}", f"manual_{name}_{stem}",
                note="manual drop; subject unknown (one subject_id per image)")
    return rows


# ---------------------------------------------------------------- signal sources
UCI_TASK = {0: "static_spiral", 1: "dynamic_spiral", 2: "circle"}


def read_uci(path):
    df = pd.read_csv(path, sep=";", header=None, names=["x", "y", "z", "pressure", "grip_angle", "ts", "test_id"],
                     engine="python", on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def uci_signal_rows(sig_rows, writes):
    first_line = {}
    files = [(RAW / "uci395/hw_dataset/control", 0, "uci395_"), (RAW / "uci395/hw_dataset/parkinson", 1, "uci395_"),
             (RAW / "uci395/new_dataset/parkinson", 1, "uci395_new_")]
    hashes = {}
    for d, lab, pre in files:
        for f in sorted(d.glob("*.txt")):
            subj = pre + f.stem
            first_line[f.read_text(errors="ignore").splitlines()[0].strip()] = subj
            hashes[md5(f)] = subj
            emit_uci(f, "uci395", lab, subj, sig_rows, writes)
    base = next((RAW / "uci358").glob("*/data"))
    for sub, lab in (("Healthy", 0), ("PWP", 1)):
        for f in sorted((base / sub).glob("*.txt")):
            h = md5(f)
            if h in hashes:
                sig_rows.append(dict(sample_id=None, skipped=f"exact duplicate of {hashes[h]}", original_path=rel(f)))
                continue
            fl = f.read_text(errors="ignore").splitlines()[0].strip()
            subj = first_line.get(fl)
            note = f"same subject as UCI395 {subj} (first sample identical), content differs" if subj else ""
            subj = subj or "uci358_" + re.sub(r"\W+", "_", f.stem).strip("_")
            emit_uci(f, "uci358", lab, subj, sig_rows, writes, note=note)


def emit_uci(f, source, lab, subj, sig_rows, writes, note=""):
    df = read_uci(f)
    for tid, g in df.groupby("test_id"):
        task = UCI_TASK.get(int(tid), f"test{int(tid)}")
        g = g.sort_values("ts")
        out = pd.DataFrame(dict(t=(g.ts - g.ts.iloc[0]) / 1000.0, x=g.x, y=g.y, pressure=g.pressure,
                                z=g.z, grip_angle=g.grip_angle, pen_down=g.pressure > 0))
        sid = f"{source}_{task}_{re.sub(r'^uci\d+_', '', subj)}" + ("_358" if source == "uci358" else "")
        writes.append((sid, out))
        sig_rows.append(dict(sample_id=sid, source_dataset=source, task=task, label=lab, subject_id=subj,
                             original_path=rel(f), n_samples=len(out), sampling="~110-140 Hz (timestamps ms)", note=note))


NHPD_CH = ["microphone", "finger_grip", "axial_pressure", "tilt_accel_x", "tilt_accel_y", "tilt_accel_z"]


def nhpd_task(prefix):
    if prefix.startswith("sigSp"):
        return "spiral"
    if prefix.startswith("sigMea"):
        return "meander"
    return {"circA": "circle", "circB": "circle_air", "sigDiaA": "diadochokinesis_a", "sigDiaB": "diadochokinesis_b"}[prefix]


def nhpd_signal_rows(sig_rows, writes):
    for arch, lab, g in (("HealthySignal", 0, "H"), ("PatientSignal", 1, "P")):
        for f in sorted((RAW / arch).rglob("*.txt")):
            if "__MACOSX" in f.parts:
                continue
            m = re.match(r"([a-zA-Z]+)(\d*)-([HPhp]?)(\d+)$", f.stem)
            pre, rep, n = m.group(1) + m.group(2), m.group(2), int(m.group(4))
            base = m.group(1) if m.group(1) in ("circA", "circB", "sigDiaA", "sigDiaB") else re.sub(r"\d+$", "", pre)
            task = nhpd_task(m.group(1) if base == m.group(1) else base)
            note = "" if m.group(3) else "filename lacks H/P tag; group taken from archive"
            meta = {}
            data = []
            for line in f.read_text(errors="ignore").splitlines():
                if line.startswith("#"):
                    mm = re.match(r"#<(\w+)>(.*)</\1>", line)
                    if mm:
                        meta[mm.group(1)] = mm.group(2)
                elif line.strip():
                    data.append(line.split())
            arr = pd.DataFrame(data, columns=NHPD_CH[: len(data[0])]).apply(pd.to_numeric, errors="coerce")
            sr = float(meta.get("Samplerate", 1000) or 1000)
            out = pd.DataFrame(dict(t=np.arange(len(arr)) / sr, x=np.nan, y=np.nan, pressure=arr["axial_pressure"]))
            out = pd.concat([out, arr.drop(columns=["axial_pressure"])], axis=1)
            out["pen_down"] = True
            subj = f"newhandpd_{g}{n}"
            sid = f"newhandpd_{task}_{g}{n}_{rep or 'A'}"
            writes.append((sid, out))
            sig_rows.append(dict(sample_id=sid, source_dataset="newhandpd", task=task, label=lab, subject_id=subj,
                                 original_path=rel(f), n_samples=len(out), sampling=f"{sr:g} Hz",
                                 note=(note + f" person_id={meta.get('Person_ID_Number','')}").strip()))


def pahaw_signal_rows(sig_rows, writes):
    # BDALab handwriting-features example data: 4 PaHaW-protocol subjects. SVC: 1st line count; then
    # Y, X, timestamp, button, azimuth, altitude, pressure (per bundled info.json). Task 1 = Archimedean spiral.
    base = RAW / "repos/handwriting-features/examples/data"
    for f in sorted(base.glob("*/*.svc")):
        lab = 1 if f.parent.name.startswith("PD") else 0
        m = re.match(r"(\d+)_w\.cz\.fnusa\.(\d+)_(\d+)$", f.stem)
        subj, tno, rep = f"pahaw_{m.group(1)}", int(m.group(2)), m.group(3)
        a = np.loadtxt(f, skiprows=1)
        task = "spiral" if tno == 1 else f"handwriting_t{tno}"
        out = pd.DataFrame(dict(t=(a[:, 2] - a[:, 2][0]) / 1000.0, x=a[:, 1], y=a[:, 0], pressure=a[:, 6],
                                azimuth=a[:, 4], altitude=a[:, 5], button=a[:, 3], pen_down=a[:, 3] > 0))
        sid = f"pahaw_{task}_{m.group(1)}_{rep}"
        writes.append((sid, out))
        sig_rows.append(dict(sample_id=sid, source_dataset="pahaw_examples", task=task, label=lab, subject_id=subj,
                             original_path=rel(f), n_samples=len(out), sampling="~150-200 Hz", note=f.parent.name))


HANDYDIAG_TASKS = {"spiral": "spiral", "vertical": "line_vertical", "horizontal": "line_horizontal",
                   "letters": "handwriting_letters", "letters2": "handwriting_letters2",
                   "thletters": "handwriting_th_sentence"}


def _hd_tok(s):
    return [np.nan if t.strip() == "null" else float(t) for t in s.strip().strip("[]").split(",") if t.strip()]


def handydiag_signal_rows(sig_rows, writes):
    # HandyDiag (Chanwimalueang et al., Sci Rep 2026): Thai tablet handwriting, one JSON per subject, label from
    # Data_HC / Data_PD folder. Keys come in two spellings ("Spiral R dx" / "SpiralRDx"): normalized by dropping
    # spaces + lowercasing. "null" entries in dx/dy/pressure mark pen lifts. Screen coords (y down) -> y negated.
    # Data bug: in many files the Letters 2 dx/dy arrays are copies of Letters; then x,y are set NaN (or the
    # whole sample is skipped when pressure is also identical).
    for grp, lab in (("Data_HC", 0), ("Data_PD", 1)):
        for f in sorted((RAW / "handydiag" / grp).glob("*_rawdata.json")):
            d = {re.sub(r"\s+", "", k).lower(): v for k, v in json.loads(f.read_text()).items()}
            sid0 = re.sub(r"\W+", "_", str(d["id"])).strip("_")
            subj = f"handydiag_{sid0}"
            dom = str(d.get("dominanthand", d.get("dominant", ""))).strip().upper()[:1]
            meta = f"age={d.get('age','')} sex={d.get('sex','')} dominant={dom}"
            for key, task in HANDYDIAG_TASKS.items():
                for h in "RL":
                    k = key + h.lower()
                    if not d.get(k + "dx"):
                        continue
                    note = f"hand={h} ({'dominant' if h == dom else 'non-dominant'}) {meta}"
                    if key == "letters2" and d[k + "dx"] == d.get("letters" + h.lower() + "dx"):
                        if d[k + "penpressure"] == d.get("letters" + h.lower() + "penpressure"):
                            sig_rows.append(dict(sample_id=None, skipped="handydiag letters2 identical to letters",
                                                 original_path=rel(f) + f"#{k}"))
                            continue
                        badxy = True
                        note += "; x,y NaN: source dx/dy copied from Letters task"
                    else:
                        badxy = False
                    x, y, p, t = (np.array(_hd_tok(d[k + c])) for c in ("dx", "dy", "penpressure", "timestamp"))
                    n = len(p)
                    if badxy or len(x) != n:
                        x = y = np.full(n, np.nan)
                    m = min(n, len(t))
                    x, y, p, t = x[:m], y[:m], p[:m], t[:m]
                    if m < 2:
                        sig_rows.append(dict(sample_id=None, skipped="handydiag empty timestamp/pressure array",
                                             original_path=rel(f) + f"#{k}"))
                        continue
                    out = pd.DataFrame(dict(t=t - np.nanmin(t), x=x, y=-y, pressure=p, y_screen=y,
                                            pen_down=~np.isnan(p)))
                    sid = f"handydiag_{task}_{sid0}_{h}"
                    writes.append((sid, out))
                    sig_rows.append(dict(sample_id=sid, source_dataset="handydiag", task=task, label=lab,
                                         subject_id=subj, original_path=rel(f) + f"#{k}", n_samples=len(out),
                                         sampling="tablet events (timestamps s)", note=note))


def manual_pahaw_signal_rows(sig_rows, writes):
    # Full PaHaW (license from BDALab): unzip into data/raw/manual/pahaw/. Expects <ID>/<ID>__<task>_<rep>.svc and
    # corpus_PaHaW.xlsx (or a CSV with ID + Disease columns). Same SVC layout as the example subjects.
    base = MANUAL / "pahaw"
    svcs = sorted(base.rglob("*.svc")) if base.exists() else []
    if not svcs:
        return
    lab_files = sorted(base.rglob("*.csv")) + sorted(base.rglob("*.xls*"))
    if not lab_files:
        raise RuntimeError("manual/pahaw: no corpus_PaHaW.xlsx / labels CSV found")
    lf = lab_files[0]
    meta = pd.read_csv(lf) if lf.suffix == ".csv" else pd.read_excel(lf)  # xlsx needs openpyxl
    meta.columns = [str(c).strip().lower() for c in meta.columns]
    labels = {int(i): (1 if str(dz).strip().upper().startswith("PD") else 0) for i, dz in zip(meta["id"], meta["disease"])}
    ex = {md5(f) for f in (RAW / "repos/handwriting-features/examples/data").glob("*/*.svc")}
    for f in svcs:
        m = re.match(r"(\d+)__(\d+)_(\d+)$", f.stem)
        if not m:
            continue
        if md5(f) in ex:
            sig_rows.append(dict(sample_id=None, skipped="identical to pahaw_examples file", original_path=rel(f)))
            continue
        sj, tno, rep = int(m.group(1)), int(m.group(2)), m.group(3)
        a = np.loadtxt(f, skiprows=1, ndmin=2)
        task = "spiral" if tno == 1 else f"handwriting_t{tno}"
        out = pd.DataFrame(dict(t=(a[:, 2] - a[:, 2][0]) / 1000.0, x=a[:, 1], y=a[:, 0], pressure=a[:, 6],
                                azimuth=a[:, 4], altitude=a[:, 5], button=a[:, 3], pen_down=a[:, 3] > 0))
        sid = f"pahaw_full_{task}_{sj:05d}_{rep}"
        writes.append((sid, out))
        sig_rows.append(dict(sample_id=sid, source_dataset="pahaw_full", task=task, label=labels[sj],
                             subject_id=f"pahaw_full_{sj:05d}", original_path=rel(f), n_samples=len(out),
                             sampling="~150-200 Hz", note="may include the 4 pahaw_examples subjects"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    if not args.skip_download:
        download()
    for d in (IMG / "scan", IMG / "rendered", SIG):
        d.mkdir(parents=True, exist_ok=True)

    # images
    rows = collect_images()
    jobs = [(r["src"], IMG / "scan" / f"{r['sample_id']}.png") for r in rows]
    assert len({j[1] for j in jobs}) == len(jobs), "duplicate sample ids"
    with ProcessPoolExecutor() as ex:
        dh = list(ex.map(norm_image, jobs, chunksize=16))
    for r, (src, dst), h in zip(rows, jobs, dh):
        r["path"] = rel(dst)
        r["src_md5"] = md5(src)
        r["_vec"] = h
    img = pd.DataFrame(rows)

    # signals
    sig_rows, writes = [], []
    uci_signal_rows(sig_rows, writes)
    nhpd_signal_rows(sig_rows, writes)
    pahaw_signal_rows(sig_rows, writes)
    handydiag_signal_rows(sig_rows, writes)
    manual_pahaw_signal_rows(sig_rows, writes)
    for sid, df in writes:
        df.reset_index(drop=True).to_parquet(SIG / f"{sid}.parquet", index=False)
    sig_all = pd.DataFrame(sig_rows)
    skipped = sig_all[sig_all.sample_id.isna()]
    sig = sig_all[sig_all.sample_id.notna()].drop(columns=["skipped"], errors="ignore").copy()
    sig["path"] = [rel(SIG / f"{s}.parquet") for s in sig.sample_id]
    sig["label"] = sig["label"].astype(int)
    sig["modality"] = "signal"

    # rendered trajectories
    rjobs = [(SIG / f"{s}.parquet", IMG / "rendered" / f"{s}.png") for s in sig.sample_id]
    with ProcessPoolExecutor() as ex:
        ok = list(ex.map(render, rjobs, chunksize=8))
    rend = sig[ok].copy()
    rend["original_path"] = rend["path"]
    rend["path"] = [rel(IMG / "rendered" / f"{s}.png") for s in rend.sample_id]
    rend["task"] = rend["task"] + "_rendered"
    rend["sample_id"] = rend["sample_id"] + "_rendered"
    rend["modality"] = "rendered"

    # ---- duplicate handling. NewHandPD re-ships many original HandPD scans byte-identically, and some
    # NewHandPD subject IDs (e.g. P1/P4/P25) share identical files. Link subjects that share any exact image
    # into one `subject_group` (union-find), drop exact-duplicate images (prefer NewHandPD copy, which links to
    # signals), and flag near-duplicates (32x32 thumbnail correlation >= NEARDUP_CORR) across HandPD/NewHandPD.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in img.groupby("src_md5"):
        subs = sorted(set(grp.subject_id))
        for s2 in subs[1:]:
            ra, rb = find(subs[0]), find(s2)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    img["overlap_flag"] = ""
    a = img[img.source_dataset == "handpd"]
    b = img[img.source_dataset == "newhandpd"]
    if len(a) and len(b):
        A, B = np.stack(a._vec), np.stack(b._vec)
        for i, j in np.argwhere(A @ B.T >= NEARDUP_CORR):
            if a.src_md5.iloc[i] != b.src_md5.iloc[j]:
                img.loc[a.index[i], "overlap_flag"] = f"near-dup of {b.sample_id.iloc[j]}"
                img.loc[b.index[j], "overlap_flag"] = f"near-dup of {a.sample_id.iloc[i]}"
    man = img.source_dataset.str.startswith("manual_")
    if man.any() and (~man).any():
        A, B = np.stack(img[man]._vec), np.stack(img[~man]._vec)
        S = A @ B.T
        for i, j in zip(range(len(A)), S.argmax(1)):
            if S[i, j] >= NEARDUP_CORR and img[man].src_md5.iloc[i] != img[~man].src_md5.iloc[j]:
                img.loc[img[man].index[i], "overlap_flag"] = f"near-dup of {img[~man].sample_id.iloc[j]}"
    pri = {"newhandpd": 0, "handpd": 1}
    img["_pri"] = img.source_dataset.map(pri).fillna(2).where(~man, 3)
    img = img.sort_values(["_pri", "sample_id"])
    dup = img.duplicated("src_md5", keep="first")
    keeper = img[~dup].set_index("src_md5").sample_id
    dropped = img[dup].assign(duplicate_of=lambda d: d.src_md5.map(keeper))
    img = img[~dup].drop(columns="_pri").sort_index()
    conflicts = dropped.merge(img[["sample_id", "label"]], left_on="duplicate_of", right_on="sample_id")
    assert (conflicts.label_x == conflicts.label_y).all(), "label conflict among duplicate images"
    dropped[["sample_id", "source_dataset", "task", "label", "subject_id", "original_path", "duplicate_of"]] \
        .to_csv(OUT / "images_dropped_duplicates.csv", index=False)
    for d in dropped.sample_id:
        (IMG / "scan" / f"{d}.png").unlink(missing_ok=True)
    img.loc[img.source_dataset == "kmader_drawings", "overlap_flag"] = "kmader subjects shared across train/test"

    cols = ["sample_id", "path", "source_dataset", "task", "label", "subject_id", "subject_group", "original_path",
            "modality"]
    img_out = pd.concat([img.drop(columns=["src", "_vec"]), rend], ignore_index=True)
    img_out["subject_group"] = img_out.subject_id.map(find)
    sig["subject_group"] = sig.subject_id.map(find)
    extra = [c for c in ["split_hint", "overlap_flag", "note", "src_md5"] if c in img_out]
    img_out[cols + extra].to_csv(OUT / "images_manifest.csv", index=False)
    sig[cols + ["n_samples", "sampling", "note"]].to_csv(OUT / "signals_manifest.csv", index=False)
    skipped.to_csv(OUT / "signals_skipped_duplicates.csv", index=False)
    groups = pd.Series({k: find(k) for k in parent})
    groups = groups[groups.groupby(groups).transform("size") > 1]
    groups.rename("subject_group").rename_axis("subject_id").to_csv(OUT / "subject_groups_merged.csv")

    def counts(df):
        g = df.groupby(["source_dataset", "modality", "task", "label"])
        return g.agg(samples=("sample_id", "size"), subjects=("subject_id", "nunique"),
                     subject_groups=("subject_group", "nunique")).reset_index()

    counts(img_out).to_csv(OUT / "counts_images.csv", index=False)
    counts(sig).to_csv(OUT / "counts_signals.csv", index=False)
    print(counts(img_out).to_string())
    print(counts(sig).to_string())
    print("images", len(img_out), "signals", len(sig), "skipped dup signals", len(skipped),
          "dropped dup images", len(dropped), "near-dup flagged", int(img.overlap_flag.str.startswith("near-dup").sum()),
          "merged subject groups", groups.nunique())


if __name__ == "__main__":
    sys.exit(main())
