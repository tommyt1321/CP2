"""Collect existing CP2 validation evidence; no mask generation or model inference.

From C:/CP2, in the existing venv:
    python -B ~/Downloads/collect_aptos_fov_v2_review.py
Creates a new ZIP in Downloads. Does not modify CP2 or send any files.
"""
import csv
import hashlib
import io
import json
import math
import shutil
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image


AUDIT = "2ec1c77e9504d7a0510d13f5bb1948a765a0914f5879405267e35b72c58b35b7"
SUMMARY = "5a3ddbacb8aaf93921f40760d207a635fd069bb55e872953717b50fe0fa8d88a"
CONTROL = "4704dbb59536"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(obj):
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def locate(root):
    runs = root / "experiments" / "aptos_fov_corner_review_v2" / "runs"
    require(runs.is_dir(), "Run from the CP2 root containing the completed v2 geometry review.")
    for d in sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True):
        try:
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8-sig"))
            if fingerprint(summary) != SUMMARY:
                continue
            rows = sorted(read_csv(d / "mask_audit_v2.csv"), key=lambda r: r["id_code"])
            if fingerprint(rows) == AUDIT:
                return d, rows
        except (FileNotFoundError, KeyError, ValueError, csv.Error):
            continue
    raise ValueError("No completed v2 run matches the summary and 527-row audit that were reviewed.")


def select(rows):
    flagged = {r["id_code"] for r in rows if r["review"] == "1"}
    high = set()
    for r in rows:
        # p95 values are estimated from accepted camera-corner tiles, not anatomical labels.
        if r["action"] != "refined":
            continue
        levels = [float(x) for x in r["corner_p95"].split("|") if x]
        if levels and statistics.median(levels) > float(r["threshold"]):
            high.add(r["id_code"])
    require((len(flagged), len(high), len(flagged & high), CONTROL in (flagged | high))
            == (61, 24, 1, False), "Selection does not match the reviewed run.")
    return flagged, high


def make_packet(root, run, rows):
    flagged, high = select(rows)
    chosen = sorted(flagged | high | {CONTROL})
    audit = {r["id_code"]: r for r in rows}
    manifest = root / "environment" / "data_audit" / "preprocessing_v1" / "preprocessing_manifest_v1.csv"
    rgb_hash = {r["id_code"]: r["output_sha256"] for r in read_csv(manifest)
                if r["split"] == "validation" and r["status"] == "ok"}
    require(set(rgb_hash) == set(audit), "The preprocessing manifest and validation audit differ.")
    buf = io.BytesIO()
    index = io.StringIO(newline="")
    writer = csv.writer(index)
    writer.writerow(["id_code", "v2_review_flag", "corner_brightness_check", "fragment_control",
                     "rgb_sha256", "candidate_mask_sha256"])
    with zipfile.ZipFile(buf, "w") as z:
        for name in ("summary.json", "mask_audit_v2.csv"):
            z.write(run / name, "v2/" + name, compress_type=zipfile.ZIP_DEFLATED)
        for page in range(1, math.ceil(len(flagged) / 12) + 1):
            name = f"flagged_{page:02d}.png"
            z.write(run / name, "sheets/" + name)
        for k in chosen:
            rgb = (root / "processed_images_384_v1" / (k + ".png")).read_bytes()
            require(sha(rgb) == rgb_hash[k], k + ": RGB hash differs from preprocessing manifest.")
            mask = (run / "candidate_masks" / (k + ".png")).read_bytes()
            with Image.open(io.BytesIO(mask)) as im:
                require(im.format == "PNG" and im.mode == "L" and im.size == (384, 384),
                        k + ": unexpected candidate mask format.")
                hist = im.histogram()
                require(hist[0] + hist[255] == 384**2 and hist[255] == int(audit[k]["candidate_pixels"]),
                        k + ": mask is not binary or its area differs from the reviewed audit.")
            writer.writerow([k, int(k in flagged), int(k in high), int(k == CONTROL), rgb_hash[k], sha(mask)])
            z.writestr("rgb/" + k + ".png", rgb)
            z.writestr("candidate_masks/" + k + ".png", mask)
        z.writestr("review_index.csv", index.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("README.txt",
                   "Existing APTOS validation outputs for FOV geometry review only.\n"
                   "61 v2 flags + 24 candidate corner-floor cases - 1 overlap = 84 review images.\n"
                   "One extra, unchanged-mask control (4704dbb59536) is included for clipped-region analysis.\n"
                   "The corner-floor test compares the median accepted tile p95 to the actual threshold; "
                   "it is a reason to inspect, not proof of a wrong mask.\n"
                   "RGB and candidate masks are the original saved PNG bytes. "
                   "No grades, performance measures, new masks or model outputs were computed.\n")
    buf.seek(0)
    return buf


def main():
    root = Path.cwd().resolve()
    run, rows = locate(root)
    print("Reading existing 384x384 RGBs, masks and review sheets; no model is loaded.", flush=True)
    buf = make_packet(root, run, rows)
    dest = Path.home() / "Downloads"
    dest.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = dest / f"CP2_FOV_v2_visual_review_{stamp}_{uuid4().hex[:8]}.zip"
    with buf, target.open("xb") as f:
        shutil.copyfileobj(buf, f)
    print("Saved:", target)
    print("84 review cases + 1 control; 6 existing review sheets. No CP2 files changed.")


if __name__ == "__main__":
    main()
