"""Read-only inventory of the official IDRiD A/B packages for CP2.

Run from C:\\CP2: python idrid_audit_v1.py
Only NumPy and Pillow are required. No training, inference, preprocessing,
automatic exclusions, new split, or missing-mask imputation is performed.
Image IDs are scoped by package AND official partition. Cross-package links require
identical decoded RGB pixels, including dimensions; filenames alone never join.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from uuid import uuid4

VERSION = "idrid_audit_v1_20261030_r3"
PROJECT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LESIONS = ("MA", "HE", "EX", "SE", "OD")
EXPECTED_SIZE = (4288, 2848)
EXPECTED_IMAGES = {"segmentation": {"training": 54, "testing": 27},
                   "grading": {"training": 413, "testing": 103}}
EXPECTED_MASKS = {"training": {"MA": 54, "HE": 53, "EX": 54, "SE": 26, "OD": 54},
                  "testing": {"MA": 27, "HE": 27, "EX": 27, "SE": 14, "OD": 27}}
SOURCES = [
    {"title": "IDRiD official data description", "role": "Image and available-mask counts; dataset roles",
     "url": "https://idrid.grand-challenge.org/Data/"},
    {"title": "Porwal et al. (2018), Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research",
     "role": "Dataset description, DR versus DME labels, official grading split",
     "doi": "10.3390/data3030025", "url": "https://www.mdpi.com/2306-5729/3/3/25"},
    {"title": "Porwal et al. (2020), IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge",
     "role": "Official partition and lesion-mask availability counts (Table 3)",
     "doi": "10.1016/j.media.2019.101561"},
    {"title": "Original IDRiD data deposit", "doi": "10.21227/H25W98",
     "url": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"},
    {"title": "Pillow concepts: image modes and palettes", "role": "Technical reference: P pixels are palette indices; RGB/RGBA store colour channels and optional alpha. Actual binary colour mappings are checked from the supplied masks.",
     "url": "https://pillow.readthedocs.io/en/stable/handbook/concepts.html"},
]


def normal(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def identifier(value):
    # Canonicalise only within (package, official_split); never join on ID alone.
    value = str(value).strip()
    value = re.sub(r"\.(?:jpg|jpeg|png|tif|tiff)$", "", value, flags=re.I)
    m = re.fullmatch(r"IDRiD[_ -]?(\d+)", value, re.I)
    if not m or int(m.group(1)) < 1:
        raise ValueError(f"Unexpected IDRiD image ID: {value!r}")
    return f"IDRiD_{int(m.group(1))}"


def partition(parts):
    hits = set()
    for part in parts:
        p = normal(part)
        if "train" in p:
            hits.add("training")
        if "test" in p:
            hits.add("testing")
    if len(hits) != 1:
        raise ValueError(f"Cannot resolve one official training/testing partition: {list(parts)}")
    return next(iter(hits))


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path, data):
    """New run outputs; retry transient Windows sharing/replace failures."""
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        start = time.monotonic()
        delay = 0.1
        while True:
            try:
                os.replace(temp, path)
                break
            except OSError as e:
                if getattr(e, "winerror", None) not in (5, 32, 33) or time.monotonic() - start >= 30:
                    raise
                time.sleep(delay)
                delay = min(1.0, delay * 2)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def save_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def save_csv(path, rows, fields):
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    atomic_bytes(path, s.getvalue().encode("utf-8-sig"))


def clock(seconds):
    v = max(0, int(seconds))
    return f"{v // 3600:02d}:{v % 3600 // 60:02d}:{v % 60:02d}"


def binary_mask_array(im):
    """Validate binary values before deriving support; preserve indexed labels.

    The supplied original IDRiD_01_MA.tif is P mode: 0 -> black, 1 -> red.
    IDRiD_81_EX.tif is RGBA: only black/red pixels and alpha=255 everywhere.
    RGB conversion is display rendering and must not precede label decoding.
    Unknown palettes, intermediate values and ambiguous RGB colours fail closed.
    """
    arr = np.asarray(im)
    encoding, palette_used = "scalar_binary", {}
    if im.mode == "P":
        if "transparency" in im.info or getattr(im.palette, "mode", "RGB") != "RGB":
            raise ValueError("Indexed mask transparency requires explicit interpretation.")
        palette = im.getpalette("RGB")
        if palette is None or palette[:3] != [0, 0, 0]:
            raise ValueError("Indexed mask requires palette index 0 to be black background.")
        values = [int(x) for x in np.unique(arr)]
        foreground_values = [x for x in values if x != 0]
        if len(foreground_values) > 1 or any(x not in (1, 255) for x in foreground_values):
            raise ValueError(f"Unsupported indexed-mask labels {values}; no thresholding or palette guessing.")
        for value in sorted({0, *foreground_values}):
            colour = palette[3 * value:3 * value + 3]
            if len(colour) != 3 or (value != 0 and colour == [0, 0, 0]):
                raise ValueError(f"Palette index {value} has no distinct foreground colour.")
            palette_used[str(value)] = colour
        encoding = "palette_indices_zero_background"
    elif arr.ndim == 3:
        if im.mode not in ("RGB", "RGBA") or arr.dtype != np.uint8 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported multichannel mask: mode={im.mode}, shape={arr.shape}, dtype={arr.dtype}.")
        if arr.shape[2] == 4 and not np.all(arr[..., 3] == 255):
            raise ValueError("Mask has non-opaque alpha; requires explicit interpretation.")
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
            encoding = "equal_rgb_channels_binary"
        elif (np.all(arr[..., 1:3] == 0)
              and np.all((arr[..., 0] == 0) | (arr[..., 0] == 255))):
            # Check every pixel before using R as the label value. This is an
            # explicit black/red codebook, never an arbitrary channel selection.
            # Alpha was checked above and must never count as lesion foreground.
            encoding = "black_red_rgb_binary"
        else:
            raise ValueError("Unsupported RGB mask colours; expected equal-channel binary values or exact black/red with opaque alpha.")
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.dtype.kind not in "uib":
        raise ValueError(f"Expected integer/bool 2D mask, got {arr.shape}, {arr.dtype}.")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi not in (0, 1, 255, 65535) or np.any((arr != 0) & (arr != hi)):
        raise ValueError(f"Non-binary mask values (min={lo}, max={hi}); not silently thresholded.")
    return arr, encoding, palette_used


class Progress:
    def __init__(self):
        self.start = self.phase_start = time.monotonic()
        self.phase, self.done, self.total = "starting", 0, 0
        self.images = self.masks = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def begin(self, phase, total):
        with self.lock:
            self.phase, self.total, self.done = phase, total, 0
            self.phase_start = time.monotonic()

    def step(self, kind=""):
        with self.lock:
            self.done += 1
            self.images += kind == "image"
            self.masks += kind == "mask"

    def render(self):
        with self.lock:
            now = time.monotonic()
            eta = clock((now - self.phase_start) * (self.total - self.done) / self.done) if self.done else "--"
            line = (f"IDRiD audit | {self.phase} | Files {self.done}/{self.total} | "
                    f"Images {self.images} | Masks {self.masks} | Elapsed {clock(now - self.start)} | Phase ETA {eta}")
        print("\r" + line.ljust(190), end="", flush=True)

    def run(self):
        while not self.stop.wait(1):
            self.render()

    def __enter__(self):
        self.render()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.render()
        print()


class Audit:
    def __init__(self, raw, output, progress):
        self.raw, self.output, self.progress = raw, output, progress
        self.issues, self.files, self.images, self.masks, self.labels = [], [], [], [], []
        self.mask_files = []  # Physical presence is retained even when decoding fails.
        self.partition_local_id_reuse_counts = {}
        self.coverage, self.crosswalk, self.duplicates = [], [], []
        self.csv_schemas = []
        self.roots = {}
        self.input_paths = []
        self.unchanged = False

    def issue(self, severity, code, path, detail):
        self.issues.append(dict(severity=severity, code=code, path=str(path), detail=str(detail)))

    def rel(self, path):
        return path.relative_to(self.raw).as_posix()

    def discover(self):
        if not self.raw.is_dir():
            raise FileNotFoundError(f"Raw directory not found: {self.raw}")
        for package, wanted in [("segmentation", "asegmentation"), ("grading", "bdiseasegrading")]:
            candidates = [x for x in self.raw.iterdir() if x.is_dir() and normal(x.name) == wanted]
            if len(candidates) != 1:
                raise ValueError(f"Expected one {package} folder directly inside {self.raw}; found {len(candidates)}")
            self.roots[package] = candidates[0]
        for package, root in self.roots.items():
            for p in sorted(root.rglob("*")):
                if p.is_symlink():
                    self.issue("ERROR", "SYMLINK_INPUT", self.rel(p), "Use ordinary extracted files; link not followed.")
                    continue
                if p.is_file():
                    self.input_paths.append((package, p))
        case_paths = defaultdict(list)
        for _, p in self.input_paths:
            case_paths[self.rel(p).casefold()].append(self.rel(p))
        for paths in case_paths.values():
            if len(paths) > 1:
                self.issue("ERROR", "CASE_COLLISION", paths[0], json.dumps(paths))

    def kind(self, package, p):
        parts = [normal(x) for x in p.relative_to(self.roots[package]).parts[:-1]]
        if p.suffix.lower() == ".csv" and package == "grading":
            return "labels"
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            if any(x.endswith("originalimages") for x in parts):
                return "image"
            if package == "segmentation" and any("groundtruth" in x for x in parts):
                return "mask"
            return "unknown_raster"
        return "metadata"

    def read_inputs(self):
        self.progress.begin("read and decode raw files", len(self.input_paths))
        for package, p in self.input_paths:
            kind = self.kind(package, p)
            rel = self.rel(p)
            row = dict(path=rel, package=package, kind=kind, bytes="", sha256="", read_status="ERROR", final_hash_matches=False)
            mask_file = None
            if kind == "mask":
                mask_file = dict(official_split="", canonical_id="", lesion="", path=rel,
                                 file_sha256="", read_status="ERROR", decode_error="")
                self.mask_files.append(mask_file)
            try:
                before = p.stat()
                blob = p.read_bytes()
                after = p.stat()
                row.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), read_status="OK")
                if mask_file is not None:
                    mask_file["file_sha256"] = row["sha256"]
                    split, ident, lesion = self.mask_identity(p)
                    mask_file.update(official_split=split, canonical_id=ident, lesion=lesion)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    self.issue("ERROR", "INPUT_CHANGED_DURING_READ", rel, "File stat changed during read.")
                if kind == "image":
                    self.read_image(package, p, blob, row["sha256"])
                elif kind == "mask":
                    self.read_mask(p, blob, row["sha256"])
                    mask_file["read_status"] = "OK"
                elif kind == "labels":
                    self.read_labels(p, blob)
                elif kind == "unknown_raster":
                    self.issue("ERROR", "UNRECOGNISED_RASTER_LOCATION", rel, "Expected Original Images or Segmentation Groundtruths folder.")
            except Exception as e:
                row["read_status"] = "ERROR"
                if mask_file is not None:
                    mask_file["decode_error"] = f"{type(e).__name__}: {e}"
                self.issue("ERROR", "FILE_READ_OR_SCHEMA_ERROR", rel, f"{type(e).__name__}: {e}")
            self.files.append(row)
            self.progress.step(kind)

    def image_info(self, blob):
        with Image.open(io.BytesIO(blob)) as im:
            if getattr(im, "n_frames", 1) != 1:
                raise ValueError("Expected a single image frame.")
            im.load()  # Full decoding; truncated-image loading is disabled.
            orientation = int(im.getexif().get(274, 1))
            return im.copy(), im.format, orientation

    def read_image(self, package, p, blob, file_hash):
        split = partition(p.relative_to(self.roots[package]).parts)
        ident = identifier(p.stem)
        im, fmt, orientation = self.image_info(blob)
        width, height = im.size
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        pixel = hashlib.sha256(f"IDRID_RGB8_V1\0{width}\0{height}\0".encode())
        pixel.update(rgb.tobytes(order="C"))
        self.images.append(dict(package=package, official_split=split, canonical_id=ident,
                                source_id=p.stem, path=self.rel(p), width=width, height=height,
                                mode=im.mode, format=fmt, exif_orientation=orientation,
                                file_sha256=file_hash, rgb_sha256=pixel.hexdigest(),
                                dr_grade="", dme_grade="", label_status="not_applicable" if package == "segmentation" else "unmatched"))
        if im.size != EXPECTED_SIZE:
            self.issue("REVIEW", "NONSTANDARD_RAW_SIZE", self.rel(p), f"{im.size}; published native size {EXPECTED_SIZE}.")
        if im.mode != "RGB" or orientation != 1 or fmt != "JPEG":
            self.issue("REVIEW", "IMAGE_ENCODING_OR_ORIENTATION", self.rel(p), f"mode={im.mode}, format={fmt}, orientation={orientation}; no automatic rotation or file rewrite.")

    def mask_identity(self, p):
        split = partition(p.relative_to(self.roots["segmentation"]).parts)
        match = re.fullmatch(r"(IDRiD[_ -]?\d+)[_ -](MA|HE|EX|SE|OD)", p.stem, re.I)
        if not match:
            raise ValueError("Mask filename must identify its image and MA/HE/EX/SE/OD suffix.")
        ident, lesion = identifier(match.group(1)), match.group(2).upper()
        return split, ident, lesion

    def read_mask(self, p, blob, file_hash):
        split, ident, lesion = self.mask_identity(p)
        folder_names = {"MA": ("microaneurysms", "microaneurysm"),
                        "HE": ("haemorrhages", "hemorrhages", "haemorrhage", "hemorrhage"),
                        "EX": ("hardexudates", "hardexudate"), "SE": ("softexudates", "softexudate"),
                        "OD": ("opticdisc", "opticdisk")}
        folder_types = {k for k, names in folder_names.items() for part in p.parts[:-1]
                        if any(normal(part).endswith(x) for x in names)}
        if folder_types and folder_types != {lesion}:
            raise ValueError(f"Mask suffix {lesion} contradicts folder type {sorted(folder_types)}.")
        im, fmt, orientation = self.image_info(blob)
        arr, encoding, palette_used = binary_mask_array(im)
        lo, hi = int(arr.min()), int(arr.max())
        support = arr != 0
        positive = int(np.count_nonzero(support))
        width, height = im.size
        h = hashlib.sha256(f"IDRID_BINARY_SUPPORT_V1\0{width}\0{height}\0".encode())
        h.update(support.astype(np.uint8).tobytes(order="C"))
        self.masks.append(dict(official_split=split, canonical_id=ident, lesion=lesion,
                               role="anatomy" if lesion == "OD" else "lesion", path=self.rel(p),
                               width=width, height=height, mode=im.mode, format=fmt, exif_orientation=orientation,
                               value_encoding=encoding, palette_used_rgb_json=json.dumps(palette_used),
                               value_min=lo, value_max=hi, positive_pixels=positive,
                               positive_fraction=positive / (width * height), file_sha256=file_hash,
                               binary_support_sha256=h.hexdigest(), alignment_status="unchecked"))
        if orientation != 1:
            self.issue("REVIEW", "MASK_ORIENTATION", self.rel(p), f"EXIF orientation {orientation}; no automatic rotation.")
        if positive == width * height:
            self.issue("REVIEW", "FULL_MASK", self.rel(p), "Every pixel is foreground; review annotation.")
        if positive == 0:
            self.issue("NOTE", "PRESENT_EMPTY_MASK", self.rel(p), "Existing valid all-zero mask, distinct from an absent file.")

    def read_labels(self, p, blob):
        split = partition(p.relative_to(self.roots["grading"]).parts)
        reader = csv.reader(io.StringIO(blob.decode("utf-8-sig"), newline=""))
        header = next(reader)
        names = [normal(x) for x in header]
        aliases = {"id": {"imagename", "imageno", "imagenumber", "imageid", "idcode"},
                   "dr": {"retinopathygrade", "drgrade", "diabeticretinopathygrade"},
                   "dme": {"riskofmacularedema", "riskofdme", "macularedemagrade", "dmegrade"}}
        indices = {}
        for role, allowed in aliases.items():
            found = [i for i, name in enumerate(names) if name in allowed]
            if len(found) != 1:
                raise ValueError(f"Expected one explicit {role} column, found {len(found)}; columns={header!r}")
            indices[role] = found[0]
        if len({x for x in names if x}) != len([x for x in names if x]):
            raise ValueError("Duplicate CSV column names.")
        self.csv_schemas.append(dict(path=self.rel(p), official_split=split, headers=header,
                                     id_column=header[indices["id"]], dr_column=header[indices["dr"]],
                                     dme_column=header[indices["dme"]]))
        for line, cells in enumerate(reader, 2):
            if not any(x.strip() for x in cells):
                continue
            try:
                if len(cells) > len(header) and any(x.strip() for x in cells[len(header):]):
                    raise ValueError("Extra nonempty CSV cells.")
                if any(cells[i].strip() for i in range(min(len(cells), len(header))) if i not in indices.values()):
                    raise ValueError("Unexpected populated column; inspect schema before use.")
                ident = identifier(cells[indices["id"]])
                dr_text, dme_text = cells[indices["dr"]].strip(), cells[indices["dme"]].strip()
                if not re.fullmatch(r"[0-4]", dr_text) or not re.fullmatch(r"[0-2]", dme_text):
                    raise ValueError(f"Invalid DR/DME grade: {dr_text!r}, {dme_text!r}")
                self.labels.append(dict(official_split=split, canonical_id=ident,
                                        source_id=cells[indices["id"]].strip(), dr_grade=int(dr_text),
                                        dme_grade=int(dme_text), csv_path=self.rel(p), csv_line=line))
            except (ValueError, IndexError) as e:
                self.issue("ERROR", "LABEL_ROW_ERROR", self.rel(p), f"line {line}: {e}")

    def check_counts_and_links(self):
        images_by_key = defaultdict(list)
        for x in self.images:
            images_by_key[(x["package"], x["official_split"], x["canonical_id"])].append(x)
        labels_by_key = defaultdict(list)
        for x in self.labels:
            labels_by_key[(x["official_split"], x["canonical_id"])].append(x)
        for key, values in images_by_key.items():
            if len(values) != 1:
                self.issue("ERROR", "DUPLICATE_IMAGE_ID", key, [x["path"] for x in values])
        # Official grading IDs 001..103 are reused across partitions. Equal local
        # IDs do not establish equal photographs; content hashes check that below.
        ids = defaultdict(set)
        for x in self.images:
            ids[(x["package"], x["canonical_id"])].add(x["official_split"])
        self.partition_local_id_reuse_counts = {
            package: sum(p == package and len(splits) > 1 for (p, _), splits in ids.items())
            for package in EXPECTED_IMAGES
        }
        for key, labels in labels_by_key.items():
            if len(labels) != 1:
                self.issue("ERROR", "DUPLICATE_LABEL_ID", key, "Repeated label row; not automatically deduplicated.")
            if ("grading", *key) not in images_by_key:
                self.issue("ERROR", "LABEL_WITHOUT_IMAGE", key, "No matching original in the same official partition.")
        for x in self.images:
            if x["package"] == "grading":
                y = labels_by_key.get((x["official_split"], x["canonical_id"]), [])
                if len(y) == 1:
                    x.update(dr_grade=y[0]["dr_grade"], dme_grade=y[0]["dme_grade"], label_status="matched")
                else:
                    self.issue("ERROR", "IMAGE_LABEL_AMBIGUOUS_OR_MISSING", x["path"], f"Matching label rows: {len(y)}")
        for package, splits in EXPECTED_IMAGES.items():
            for split, expected in splits.items():
                n = sum(x["package"] == package and x["official_split"] == split for x in self.images)
                if n != expected:
                    self.issue("ERROR", "IMAGE_COUNT", f"{package}/{split}", f"Decoded {n}, published expected {expected}.")
        for split, expected in EXPECTED_IMAGES["grading"].items():
            count = sum(x["official_split"] == split for x in self.labels)
            sources = sum(x["official_split"] == split for x in self.csv_schemas)
            if count != expected or sources != 1:
                self.issue("ERROR", "LABEL_COUNT_OR_SOURCE", split, f"Rows={count} (expected {expected}), CSVs={sources} (expected 1).")
        masks_by_key, mask_files_by_key = defaultdict(list), defaultdict(list)
        for m in self.mask_files:
            if m["official_split"] and m["canonical_id"] and m["lesion"]:
                mask_files_by_key[(m["official_split"], m["canonical_id"], m["lesion"])].append(m)
        for m in self.masks:
            key = (m["official_split"], m["canonical_id"], m["lesion"])
            masks_by_key[key].append(m)
            owners = images_by_key.get(("segmentation", m["official_split"], m["canonical_id"]), [])
            if len(owners) != 1:
                m["alignment_status"] = "owner_missing_or_ambiguous"
                self.issue("ERROR", "MASK_WITHOUT_UNIQUE_IMAGE", m["path"], len(owners))
            elif (m["width"], m["height"]) != (owners[0]["width"], owners[0]["height"]):
                m["alignment_status"] = "dimension_mismatch"
                self.issue("ERROR", "MASK_IMAGE_DIMENSIONS", m["path"], "Raw image/mask dimensions differ.")
            else:
                m["alignment_status"] = "id_split_dimensions_match_not_visual_validation"
        for key, values in mask_files_by_key.items():
            if len(values) != 1:
                self.issue("ERROR", "DUPLICATE_MASK", key, [x["path"] for x in values])
        for split, expected in EXPECTED_MASKS.items():
            for lesion, count in expected.items():
                actual = sum(x["official_split"] == split and x["lesion"] == lesion for x in self.masks)
                present = sum(x["official_split"] == split and x["lesion"] == lesion for x in self.mask_files)
                if present != count:
                    self.issue("ERROR", "MASK_FILE_COUNT", f"{split}/{lesion}", f"Identified files present {present}, published available masks {count}; decoded {actual}.")
                if actual != present:
                    self.issue("ERROR", "MASK_DECODE_INCOMPLETE", f"{split}/{lesion}", f"Identified files present {present}, decoded {actual}; parse failures are not absent files.")
        for im in self.images:
            if im["package"] != "segmentation":
                continue
            for lesion in LESIONS:
                key = (im["official_split"], im["canonical_id"], lesion)
                ms, fs = masks_by_key.get(key, []), mask_files_by_key.get(key, [])
                if not fs:
                    status = "absent_not_inferred"
                elif len(fs) != 1:
                    status = "ambiguous"
                elif fs[0]["read_status"] != "OK" or len(ms) != 1:
                    status = "present_unreadable_or_invalid"
                elif ms[0]["alignment_status"] != "id_split_dimensions_match_not_visual_validation":
                    status = "present_alignment_error"
                else:
                    status = "present"
                if not fs and lesion in ("MA", "EX", "OD"):
                    self.issue("ERROR", "REQUIRED_MASK_MISSING", im["path"], lesion)
                self.coverage.append(dict(official_split=im["official_split"], canonical_id=im["canonical_id"],
                                          image_path=im["path"], lesion=lesion, role="anatomy" if lesion == "OD" else "lesion",
                                          availability=status, mask_path=fs[0]["path"] if len(fs) == 1 else "",
                                          mask_read_status=fs[0]["read_status"] if len(fs) == 1 else "",
                                          alignment_status=ms[0]["alignment_status"] if len(ms) == 1 else "",
                                          positive_pixels=ms[0]["positive_pixels"] if len(ms) == 1 and len(fs) == 1 else ""))
        self.build_crosswalk_and_duplicates()

    def build_crosswalk_and_duplicates(self):
        by_hash = defaultdict(list)
        for im in self.images:
            by_hash[(im["package"], im["rgb_sha256"])].append(im)
        for (package, digest), values in by_hash.items():
            if len(values) < 2:
                continue
            grades = sorted({str(x["dr_grade"]) for x in values if x["dr_grade"] != ""})
            dme_grades = sorted({str(x["dme_grade"]) for x in values if x["dme_grade"] != ""})
            splits = sorted({x["official_split"] for x in values})
            members = [{k: x[k] for k in ("official_split", "source_id", "path", "dr_grade", "dme_grade", "file_sha256")}
                       for x in values]
            self.duplicates.append(dict(package=package, rgb_sha256=digest, image_count=len(values),
                                        paths_json=json.dumps([x["path"] for x in values]),
                                        partitions_json=json.dumps(splits), cross_partition=len(splits) > 1,
                                        dr_grades_json=json.dumps(grades), conflicting_dr_labels=len(grades) > 1,
                                        dme_grades_json=json.dumps(dme_grades), conflicting_dme_labels=len(dme_grades) > 1,
                                        encoded_bytes_also_equal=len({x["file_sha256"] for x in values}) == 1,
                                        members_json=json.dumps(members)))
            self.issue("ERROR" if len(grades) > 1 else "REVIEW", "WITHIN_PACKAGE_EXACT_DUPLICATES", values[0]["path"],
                       f"{len(values)} identical decoded RGB images; conflicting DR labels={len(grades) > 1}; "
                       f"cross-partition={len(splits) > 1}; no removal. Members: {json.dumps(members)}")
        for im in self.images:
            if im["package"] != "segmentation":
                continue
            matches = by_hash.get(("grading", im["rgb_sha256"]), [])
            status = "unique_exact_rgb" if len(matches) == 1 else "no_exact_rgb_match" if not matches else "ambiguous_exact_rgb"
            if len(matches) != 1:
                self.issue("REVIEW", "PACKAGE_CROSSWALK_UNRESOLVED", im["path"], status + "; no filename-based grade assignment.")
            m = matches[0] if len(matches) == 1 else {}
            self.crosswalk.append(dict(segmentation_id=im["source_id"], segmentation_split=im["official_split"],
                                       segmentation_path=im["path"], rgb_sha256=im["rgb_sha256"], match_status=status,
                                       grading_id=m.get("source_id", ""), grading_split=m.get("official_split", ""),
                                       grading_path=m.get("path", ""), dr_grade=m.get("dr_grade", ""), dme_grade=m.get("dme_grade", ""),
                                       encoded_bytes_also_equal=m.get("file_sha256") == im["file_sha256"],
                                       candidate_paths_json=json.dumps([x["path"] for x in matches])))

    def verify_inputs(self):
        self.progress.begin("verify original file hashes", len(self.files))
        ok = True
        for row in self.files:
            try:
                row["final_hash_matches"] = bool(row["sha256"]) and sha_file(self.raw / row["path"]) == row["sha256"]
            except OSError:
                row["final_hash_matches"] = False
            if not row["final_hash_matches"]:
                ok = False
                self.issue("ERROR", "SOURCE_HASH_CHANGED_OR_UNREADABLE", row["path"], "Original file cannot be verified unchanged.")
            self.progress.step()
        current = {self.rel(p) for root in self.roots.values() for p in root.rglob("*") if p.is_file()}
        expected = {self.rel(p) for _, p in self.input_paths}
        if current != expected:
            ok = False
            self.issue("ERROR", "INPUT_FILE_SET_CHANGED", self.raw, f"Added={sorted(current-expected)}, removed={sorted(expected-current)}")
        self.unchanged = ok

    def save(self, elapsed, stopped=None):
        severity = Counter(x["severity"] for x in self.issues)
        status = "STOPPED" if stopped else "FAIL" if severity["ERROR"] else "REVIEW_REQUIRED" if severity["REVIEW"] else "PASS"
        counts = {p: {s: sum(x["package"] == p and x["official_split"] == s for x in self.images)
                       for s in ("training", "testing")} for p in ("segmentation", "grading")}
        mask_counts = {s: {k: sum(x["official_split"] == s and x["lesion"] == k for x in self.masks)
                          for k in LESIONS} for s in ("training", "testing")}
        present_mask_counts = {s: {k: sum(x["official_split"] == s and x["lesion"] == k for x in self.mask_files)
                                  for k in LESIONS} for s in ("training", "testing")}
        dr_counts = {s: dict(sorted(Counter(str(x["dr_grade"]) for x in self.labels if x["official_split"] == s).items()))
                     for s in ("training", "testing")}
        outputs = [
            ("idrid_raw_files_v1.csv", self.files, "path package kind bytes sha256 read_status final_hash_matches"),
            ("idrid_image_manifest_v1.csv", self.images, "package official_split canonical_id source_id path width height mode format exif_orientation file_sha256 rgb_sha256 dr_grade dme_grade label_status"),
            ("idrid_mask_manifest_v1.csv", self.masks, "official_split canonical_id lesion role path width height mode format exif_orientation value_encoding palette_used_rgb_json value_min value_max positive_pixels positive_fraction file_sha256 binary_support_sha256 alignment_status"),
            ("idrid_mask_files_v1.csv", self.mask_files, "official_split canonical_id lesion path file_sha256 read_status decode_error"),
            ("idrid_grading_labels_v1.csv", self.labels, "official_split canonical_id source_id dr_grade dme_grade csv_path csv_line"),
            ("idrid_mask_coverage_v1.csv", self.coverage, "official_split canonical_id image_path lesion role availability mask_path mask_read_status alignment_status positive_pixels"),
            ("idrid_package_crosswalk_v1.csv", self.crosswalk, "segmentation_id segmentation_split segmentation_path rgb_sha256 match_status grading_id grading_split grading_path dr_grade dme_grade encoded_bytes_also_equal candidate_paths_json"),
            ("idrid_duplicate_groups_v1.csv", self.duplicates, "package rgb_sha256 image_count paths_json partitions_json cross_partition dr_grades_json conflicting_dr_labels dme_grades_json conflicting_dme_labels encoded_bytes_also_equal members_json"),
            ("idrid_audit_issues_v1.csv", self.issues, "severity code path detail"),
        ]
        self.progress.begin("save audit evidence", len(outputs) + 3)
        hashes = {}
        for name, rows, field_string in outputs:
            save_csv(self.output / name, rows, field_string.split())
            hashes[name] = sha_file(self.output / name)
            self.progress.step()
        config = {"script_version": VERSION, "script_sha256": sha_file(Path(__file__)),
                  "raw_root": str(self.raw), "output": str(self.output), "created_at": datetime.now().astimezone().isoformat(),
                  "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "Pillow": PIL.__version__,
                  "expected_image_counts": EXPECTED_IMAGES, "expected_available_mask_counts": EXPECTED_MASKS,
                  "expected_native_size": EXPECTED_SIZE, "label_columns": self.csv_schemas,
                  "pixel_hash_definition": "SHA256(UTF8('IDRID_RGB8_V1\\0{width}\\0{height}\\0') + decoded RGB uint8 C-order bytes); no EXIF rotation/ICC transform",
                  "mask_hash_definition": "SHA256(UTF8('IDRID_BINARY_SUPPORT_V1\\0{width}\\0{height}\\0') + foreground support uint8 C-order bytes)",
                  "mask_decoder_policy": "Preserve P-mode indices: 0 must map to black; a single nonzero label 1 or 255 must map to a distinct colour. Scalar/equal-RGB binary values also supported. RGB/RGBA black/red masks require every pixel to be exactly (0,0,0) or (255,0,0), and RGBA alpha=255 everywhere. Alpha never contributes foreground. No arbitrary thresholding, resizing or file rewriting.",
                  "mask_decoder_evidence": "Uploaded original IDRiD_01_MA.tif: SHA256 0b0d689e49e3b83006fb8cefffa3c4d6251ac3bfab2379c13dd80aa84bdbf2ff; TIFF P mode, BitsPerSample=1; palette 0=black, 1=red; 6941 foreground pixels. Uploaded original IDRiD_81_EX.tif: SHA256 205e6604aa79e4d450cd61f8fb6c9d1bfabcbd4ff0016c7fc05cee6f1b24bd1a; RGBA uint8, exact black/red RGB, alpha=255 everywhere; 312918 foreground pixels. All raw masks are checked individually on each run.",
                  "missing_mask_policy": "Physical presence and decode status recorded separately; absent_not_inferred is never used for an identified present-but-invalid file; never manufacture empty masks or infer absence of disease",
                  "image_identity_key": ["package", "official_split", "canonical_id"],
                  "cross_package_policy": "A/B namespaces distinct; exact dimension-aware RGB matching only; unresolved links require review",
                  "official_partitions": "Preserved metadata, not a new CP2 training assignment",
                  "project_choices": "Read-only audit and hash/provenance rules are project engineering choices, not a clinical validation method"}
        save_json(self.output / "idrid_audit_config_v1.json", config)
        self.progress.step()
        save_json(self.output / "method_sources_v1.json", SOURCES)
        self.progress.step()
        for name in ("idrid_audit_config_v1.json", "method_sources_v1.json"):
            hashes[name] = sha_file(self.output / name)
        summary = {"status": status, "audit_scope": "Read-only IDRiD A/B inventory, labels, binary masks and exact-pixel crosswalk",
                   "script_version": VERSION, "raw_root": str(self.raw), "saved": str(self.output),
                   "elapsed_seconds_before_output_export": elapsed, "stopped_reason": stopped,
                   "decoded_image_counts": counts, "decoded_images": len(self.images), "decoded_mask_counts": mask_counts,
                   "decoded_masks": len(self.masks), "grading_label_rows": len(self.labels), "dr_grade_counts": dr_counts,
                   "present_mask_files": len(self.mask_files), "identified_mask_file_counts": present_mask_counts,
                   "failed_mask_files": sum(x["read_status"] != "OK" for x in self.mask_files),
                   "mask_value_encodings": dict(Counter(x["value_encoding"] for x in self.masks)),
                   "partition_local_id_reuse_counts": self.partition_local_id_reuse_counts,
                   "mask_coverage_status": dict(Counter(x["availability"] for x in self.coverage)),
                   "package_crosswalk_status": dict(Counter(x["match_status"] for x in self.crosswalk)),
                   "within_package_duplicate_groups": len(self.duplicates),
                   "duplicate_groups_with_conflicting_dr_labels": sum(x["conflicting_dr_labels"] for x in self.duplicates),
                   "duplicate_groups_with_conflicting_dme_labels": sum(x["conflicting_dme_labels"] for x in self.duplicates),
                   "cross_partition_duplicate_groups": sum(x["cross_partition"] for x in self.duplicates),
                   "unique_rgb_hashes_across_both_packages": len({x["rgb_sha256"] for x in self.images}),
                   "input_file_count": len(self.files), "input_bytes": sum(int(x["bytes"] or 0) for x in self.files),
                   "input_files_unchanged": self.unchanged, "issue_counts": {x: severity[x] for x in ("ERROR", "REVIEW", "NOTE")},
                   "issues": self.issues, "output_sha256": hashes,
                   "optimizer_steps": 0, "model_forwards": 0, "preprocessed_images_created": 0,
                   "limits": ["The user's original archive checksums were not supplied; this records local extracted-file hashes, not download authenticity.",
                              "Equal IDs/dimensions do not prove anatomical annotation accuracy; no visual registration review performed.",
                              "Exact decoded RGB matching does not detect recompressed, resized, near-duplicate or same-patient images.",
                              "Cross-dataset overlap with APTOS and patient-level independence have not been checked.",
                              "Classification and localisation evaluation cohorts/metrics are not finalised by this inventory.",
                              "A/B package overlaps are annotation reuse; 597 file entries need not be 597 distinct photographs."],
                   "next_step": "Review this summary and crosswalk before preprocessing; model evaluation is a separate stage."}
        save_json(self.output / "idrid_audit_summary_v1.json", summary)
        self.progress.step()
        return summary


def load_libraries():
    global np, PIL, Image
    import numpy as np
    import PIL
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=PROJECT / "IDRiD" / "raw")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "environment" / "data_audit" / "idrid_audit_v1")
    args = parser.parse_args()
    raw, output_root = args.raw_root.resolve(), args.output_root.resolve()
    if output_root == raw or raw in output_root.parents:
        parser.error("Output must be outside the raw input directory.")
    print("Loading IDRiD audit libraries...", flush=True)
    try:
        load_libraries()
    except ImportError as e:
        print(f"Missing dependency: {e}. Use the existing CP2 venv with NumPy and Pillow.")
        return 1
    out = output_root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with Progress() as progress:
        audit = Audit(raw, out, progress)
        stopped = None
        try:
            audit.discover()
            audit.read_inputs()
            progress.begin("check labels, masks and package links", 1)
            audit.check_counts_and_links()
            progress.step()
            audit.verify_inputs()
        except (Exception, KeyboardInterrupt) as e:
            stopped = f"{type(e).__name__}: {e}"
            audit.issue("ERROR", "AUDIT_STOPPED", raw, stopped)
        try:
            summary = audit.save(time.monotonic() - started, stopped)
        except (Exception, KeyboardInterrupt) as e:
            print(f"\nAUDIT OUTPUT SAVE STOPPED | {type(e).__name__}: {e}")
            print(f"Partial output: {out}; no PASS is claimed. Rerun after resolving the error.")
            return 1
    print(f"\nIDRID DATA AUDIT: {summary['status']} (data inventory only)")
    print(f"Grading images: {summary['decoded_image_counts']['grading']} | DR label rows: {summary['grading_label_rows']}")
    print(f"Segmentation images: {summary['decoded_image_counts']['segmentation']} | Mask files present: {summary['present_mask_files']} | Decoded: {summary['decoded_masks']} | Failed: {summary['failed_mask_files']}")
    print(f"Mask availability: {summary['mask_coverage_status']}")
    print(f"A/B exact-pixel crosswalk: {summary['package_crosswalk_status']}")
    print(f"Duplicate groups: {summary['within_package_duplicate_groups']} | Conflicting DR: {summary['duplicate_groups_with_conflicting_dr_labels']} | Cross-partition: {summary['cross_partition_duplicate_groups']}")
    print(f"Issues: {summary['issue_counts']} | Original files unchanged: {summary['input_files_unchanged']}")
    print("Preprocessing: 0 | Model forward: 0 | Optimizer steps: 0")
    print(f"Total elapsed: {clock(time.monotonic() - started)}\nSaved: {out}")
    print("Return idrid_audit_summary_v1.json and idrid_mask_manifest_v1.csv.")
    print("If FAIL or REVIEW_REQUIRED, also return idrid_audit_issues_v1.csv.")
    print("After review passes: remember to commit idrid_audit_v1.py; keep raw data out of Git.")
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
