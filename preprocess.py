"""Build the project dataset from one or more ZIP files / folders.

Examples
  python preprocess.py --zip archive.zip
  python preprocess.py --zip kaggle_a.zip kaggle_b.zip --classes car bicycle motorbike truck bus
  python preprocess.py --source D:\\raw_images --max-per-class 800

What it does: extract -> find class folders (aliases such as 'motorcycle' and
'motor bike' map to 'motorbike'; bicycle and motorbike stay separate) -> skip
unreadable files -> drop exact (MD5) and near duplicates (perceptual hash) ->
stratified 70/15/15 split -> write data/{train,validation,test}/<class>/.
Duplicates are removed BEFORE splitting so no picture can leak between splits.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from utils.helpers import ALLOWED_EXTENSIONS, DATA_DIR, OUTPUTS_DIR, ensure_dirs, write_json

SEED = 42
NEAR_DUP_HAMMING = 4
MAX_SIDE = 512  # copies are down-scaled to this size to keep the repo/training light

ALIASES = {
    "car": "car", "cars": "car", "automobile": "car", "sedan": "car",
    "bicycle": "bicycle", "bicycles": "bicycle", "cycle": "bicycle",
    "motorbike": "motorbike", "motorbikes": "motorbike", "motor bike": "motorbike",
    "motorcycle": "motorbike", "motorcycles": "motorbike", "bike": "motorbike", "bikes": "motorbike",
    "truck": "truck", "trucks": "truck", "lorry": "truck",
    "van": "van", "vans": "van", "minivan": "van",
    "bus": "bus", "buses": "bus",
    "auto rickshaw": "auto_rickshaw", "autorickshaw": "auto_rickshaw",
    "rickshaw": "auto_rickshaw", "tuk tuk": "auto_rickshaw",
    "train": "train", "trains": "train",
    "airplane": "airplane", "aeroplane": "airplane", "plane": "airplane",
    "ship": "ship", "ships": "ship", "boat": "ship",
}


def normalise(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def extract_zip(zip_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="vehicle_raw_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    print(f"[extract] {zip_path.name} -> {tmp}")
    return tmp


def print_tree(root: Path, depth_limit: int = 3) -> None:
    print(f"[structure] {root}")
    def walk(p: Path, depth: int, prefix: str = ""):
        if depth > depth_limit:
            return
        for d in sorted(x for x in p.iterdir() if x.is_dir() and "__MACOSX" not in x.name):
            n = sum(1 for f in d.iterdir() if f.is_file())
            print(f"{prefix}|-- {d.name}/  ({n} files)")
            walk(d, depth + 1, prefix + "|   ")
    walk(root, 1)


def find_class_folders(root: Path, restrict: set[str] | None) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    ignored = set()
    for d in root.rglob("*"):
        if not d.is_dir() or "__MACOSX" in d.parts:
            continue
        if not any(f.is_file() for f in d.iterdir()):
            continue
        canon = ALIASES.get(normalise(d.name))
        if canon is None and restrict and normalise(d.name).replace(" ", "_") in restrict:
            canon = normalise(d.name).replace(" ", "_")
        if canon is None:
            ignored.add(d.name)
        elif restrict is None or canon in restrict:
            found[canon].append(d)
        else:
            ignored.add(d.name)
    if ignored:
        print("[info] folders not used (not a requested/known class):", sorted(ignored))
    return found


POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def dhash64(gray: np.ndarray) -> np.uint64:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, nargs="*", default=[], help="dataset ZIP file(s)")
    ap.add_argument("--source", type=Path, nargs="*", default=[], help="already extracted folder(s)")
    ap.add_argument("--classes", nargs="*", help="only use these classes, e.g. car bicycle motorbike truck bus")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap images per class; the FIRST source is always kept in full")
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--source-labels", nargs="*", help="friendly name per source, in the order zips then folders")
    ap.add_argument("--dry-run", action="store_true", help="inspect + clean + report only; do not write data/")
    args = ap.parse_args()

    roots = [extract_zip(z) for z in args.zip] + list(args.source)
    if not roots:
        sys.exit("Provide --zip <file.zip> and/or --source <folder>.")
    restrict = {normalise(c).replace(" ", "_") for c in args.classes} if args.classes else None

    labels_src = args.source_labels or [r.name for r in roots]
    assert len(labels_src) == len(roots), "--source-labels must match the number of sources"
    class_files: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for ri, root in enumerate(roots):
        print_tree(root)
        for cls, dirs in find_class_folders(root, restrict).items():
            for d in dirs:
                class_files[cls] += [(f, ri) for f in sorted(d.iterdir()) if f.is_file()]

    classes = sorted(restrict) if restrict else sorted(class_files)
    missing = [c for c in classes if c not in class_files]
    if missing:
        sys.exit(f"[error] class folder(s) not found: {missing}. Found: {sorted(class_files)}")
    if len(classes) < 2:
        sys.exit("[error] need at least two classes.")
    print("[classes]", classes)

    report = {"classes": classes, "source_labels": labels_src, "sources": [str(r) for r in roots],
              "raw_counts": {c: len(class_files[c]) for c in classes},
              "corrupted": [], "exact_duplicates": [], "near_duplicates": [],
              "cross_class_duplicates": [], "unsupported_files": []}

    total = sum(len(class_files[c]) for c in classes)
    hashes = np.zeros(total, dtype=np.uint64)
    owner: list[tuple[str, Path]] = []
    seen_md5: dict[str, tuple[str, Path]] = {}
    clean: dict[str, list[tuple[Path, int]]] = {c: [] for c in classes}

    for cls in classes:
        for f, ri in class_files[cls]:
            if f.suffix.lower() not in ALLOWED_EXTENSIONS:
                report["unsupported_files"].append(f"{cls}/{f.name}")
                continue
            img = cv2.imread(str(f), cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if img is None or img.size == 0:
                report["corrupted"].append(f"{cls}/{f.name}")
                continue
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            if md5 in seen_md5:
                prev_cls, prev = seen_md5[md5]
                (report["cross_class_duplicates"] if prev_cls != cls else report["exact_duplicates"]).append(
                    f"{cls}/{f.name} == {prev_cls}/{prev.name}")
                continue
            seen_md5[md5] = (cls, f)
            h = dhash64(img)
            k = len(owner)
            if k:
                x = np.bitwise_xor(hashes[:k], h)
                dist = POP[x.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.int32)
                j = int(dist.argmin())
                if dist[j] <= NEAR_DUP_HAMMING:
                    pc, pf = owner[j]
                    (report["cross_class_duplicates"] if pc != cls else report["near_duplicates"]).append(
                        f"{cls}/{f.name} ~ {pc}/{pf.name}")
                    continue
            hashes[k] = h
            owner.append((cls, f))
            clean[cls].append((f, ri))

    rng = random.Random(SEED)
    if args.max_per_class:
        for c in classes:
            items = clean[c]
            if len(items) > args.max_per_class:
                first = [t for t in items if t[1] == 0]
                rest = [t for t in items if t[1] != 0]
                if len(first) >= args.max_per_class:
                    clean[c] = rng.sample(first, args.max_per_class)
                else:
                    clean[c] = first + rng.sample(rest, args.max_per_class - len(first))
    for c in classes:
        if len(clean[c]) < args.min_per_class:
            sys.exit(f"[error] class '{c}' has only {len(clean[c])} usable images (min {args.min_per_class}).")

    report["clean_counts"] = {c: len(v) for c, v in clean.items()}
    report["clean_counts_by_source"] = {c: {labels_src[i]: sum(1 for t in v if t[1] == i) for i in range(len(roots))}
                                        for c, v in clean.items()}
    report["total_raw"] = sum(report["raw_counts"].values())
    report["total_clean"] = sum(report["clean_counts"].values())
    for k in ("raw_counts", "corrupted", "exact_duplicates", "near_duplicates", "cross_class_duplicates"):
        v = report[k]
        print(f"[report] {k:<22}:", v if isinstance(v, dict) else len(v))
    print("[report] clean_counts          :", report["clean_counts"])
    print("[report] by source             :", report["clean_counts_by_source"])
    if args.dry_run:
        ensure_dirs()
        write_json(OUTPUTS_DIR / "dataset_report_dryrun.json", report)
        print("[dry-run] nothing written to data/")
        return

    paths, labels, srcs = [], [], []
    for c in classes:
        for f, ri in clean[c]:
            paths.append(f); labels.append(c); srcs.append(ri)
    strata = [f"{l}|{s}" for l, s in zip(labels, srcs)]   # keeps class AND source proportions in every split
    idx = np.arange(len(paths))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=strata, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=[strata[i] for i in tmp], random_state=SEED)
    splits = {"train": tr, "validation": va, "test": te}

    for s in splits:
        shutil.rmtree(DATA_DIR / s, ignore_errors=True)
        for c in classes:
            (DATA_DIR / s / c).mkdir(parents=True, exist_ok=True)

    counter = defaultdict(int)
    split_counts = {s: {c: 0 for c in classes} for s in splits}
    for s, ids in splits.items():
        for i in ids:
            c = labels[i]
            im = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
            if im is None:
                report["corrupted"].append(f"{c}/{paths[i].name}")
                continue
            m = max(im.shape[:2])
            if m > MAX_SIDE:
                sc = MAX_SIDE / m
                im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
            counter[c] += 1
            cv2.imwrite(str(DATA_DIR / s / c / f"{c}_s{srcs[i]}_{counter[c]:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            split_counts[s][c] += 1
    report["split_counts"] = split_counts
    report["split_totals"] = {s: sum(v.values()) for s, v in split_counts.items()}

    seen: dict[str, str] = {}
    for s in splits:
        for f in (DATA_DIR / s).rglob("*.jpg"):
            h = hashlib.md5(f.read_bytes()).hexdigest()
            assert h not in seen, f"LEAK: {f} duplicates a file in split '{seen[h]}'"
            seen[h] = s
    print("[check] no identical image appears in more than one split")

    ensure_dirs()
    write_json(OUTPUTS_DIR / "dataset_report.json", report)
    print("[split]", report["split_totals"])
    for s, cnt in split_counts.items():
        print(f"   {s:<11}", cnt)


if __name__ == "__main__":
    main()
