"""Embed and search your own videos and photos: make check-data DIR=... and make load DIR=...

Expected folder layout:

    my_media/
      videos/   .mp4 .avi .mov    food preparation clips
      images/   .jpg .jpeg .png   photos of finished dishes (optional)

The file name, without its extension, becomes the ID you search by, so every
name must be unique across both folders and the sample data.

--check-only reads every file and writes nothing. Without it, once every check
passes, it runs the embedding generator on the folder (writing to
data/embeddings/, next to the sample embeddings) and lists the closest matches
for each new file.
"""

import argparse
import json
import math
import os
import subprocess
import sys

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # unreadable videos are reported below
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS = os.path.join(ROOT, "data", "embeddings")
# What src/embeddings/generator.py reads (it matches the extension case-sensitively)
TYPES = {"videos": (".mp4", ".avi", ".mov"), "images": (".jpg", ".jpeg", ".png")}
FRAME_SAMPLE_RATE = 5  # generator.py keeps every 5th frame in memory
MAX_FRAME_BYTES = 2 * 1024 ** 3
MIN_SIDE = 32


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(100).startswith(b"version https://git-lfs")


def check_video(path, rel, problems):
    with open(path, "rb") as f:
        head = f.read(12)
    if (head[:4] != b"RIFF") if path.endswith(".avi") else (head[4:8] != b"ftyp"):
        problems.append(f"{rel}: not a {os.path.splitext(path)[1]} video (no file header)")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        problems.append(f"{rel}: can't open as a video")
        return
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    cap.release()
    if not ok or frames <= 0:
        problems.append(f"{rel}: no readable video frames")
    elif not fps or fps <= 0 or math.isnan(fps):
        problems.append(f"{rel}: no frame rate in the file (the generator divides by it)")
    else:
        held = math.ceil(frames / FRAME_SAMPLE_RATE) * frame.nbytes
        if held > MAX_FRAME_BYTES:
            problems.append(f"{rel}: too long: {frames} frames at {frame.shape[1]}x{frame.shape[0]}; the generator "
                            f"would hold {held / 1024 ** 3:.1f} GB of frames in memory (limit 2 GB). Cut it into clips")


def check_image(path, rel, problems):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError):
        problems.append(f"{rel}: can't open as an image")
        return
    if min(width, height) < MIN_SIDE:
        problems.append(f"{rel}: too small ({width}x{height}), needs at least {MIN_SIDE}x{MIN_SIDE}")


def existing_sources():
    """media_id -> original_file for the embeddings already in data/embeddings."""
    sources = {}
    if os.path.isdir(EMBEDDINGS):
        for f in os.listdir(EMBEDDINGS):
            if f.endswith("_metadata.json"):
                try:
                    with open(os.path.join(EMBEDDINGS, f)) as fh:
                        meta = json.load(fh)
                    sources[meta["media_id"]] = meta.get("original_file", "")
                except (OSError, ValueError, KeyError):
                    pass
    return sources


def scan(folder):
    """Check a folder. Returns ({media_id: (kind, path)}, problems). Writes nothing."""
    if not os.path.isdir(folder):
        return {}, [f"{folder}: not a folder"]
    problems, media = [], {}
    if "," in os.path.abspath(folder):
        problems.append(f"{folder}: the path contains a comma; the generator splits its --input on commas")
    present = [k for k in TYPES if os.path.isdir(os.path.join(folder, k))]
    if not present:
        return {}, problems + ["no videos/ or images/ folder"]
    sources = existing_sources()
    for kind in present:
        for name in sorted(os.listdir(os.path.join(folder, kind))):
            if name.startswith("."):
                continue
            rel, path = f"{kind}/{name}", os.path.join(folder, kind, name)
            stem, ext = os.path.splitext(name)
            if not os.path.isfile(path):
                problems.append(f"{rel}: a folder; the generator reads only files directly in {kind}/")
                continue
            if ext.lower() in TYPES[kind] and ext not in TYPES[kind]:
                problems.append(f"{rel}: upper-case extension; the generator skips it. Rename it to {stem}{ext.lower()}")
                continue
            if ext not in TYPES[kind]:
                other = next((k for k in TYPES if ext.lower() in TYPES[k]), None)
                where = f"; put it in {other}/" if other else ""
                problems.append(f"{rel}: not one of {' '.join(TYPES[kind])}{where}")
                continue
            if os.path.getsize(path) == 0:
                problems.append(f"{rel}: empty file")
                continue
            if is_lfs_pointer(path):
                problems.append(f"{rel}: a Git LFS pointer, not the file (run git lfs pull)")
                continue
            before = len(problems)
            (check_video if kind == "videos" else check_image)(path, rel, problems)
            if len(problems) > before:
                continue
            if stem in media:
                problems.append(f"{rel}: same name as {media[stem][2]}; their embeddings would overwrite each other")
                continue
            src = sources.get(stem)
            if src and os.path.abspath(os.path.join(ROOT, src)) != os.path.abspath(path):
                problems.append(f"{rel}: data/embeddings already has '{stem}' (from {src}); rename the file")
                continue
            media[stem] = (kind, path, rel)
    if not media and not problems:
        problems.append("no videos or images found")
    return media, problems


def nearest(media):
    ids = sorted(f[:-len("_embedding.npy")] for f in os.listdir(EMBEDDINGS) if f.endswith("_embedding.npy"))
    vecs = np.stack([np.load(os.path.join(EMBEDDINGS, f"{i}_embedding.npy")) for i in ids])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    print(f"\nClosest matches among {len(ids)} embeddings (cosine similarity; 1.0 is identical):")
    for stem in sorted(media):
        i = ids.index(stem)
        order = [j for j in np.argsort(-(vecs @ vecs[i])) if j != i][:3]
        matches = ", ".join(f"{ids[j]} {vecs[i] @ vecs[j]:.3f}" for j in order)
        print(f"  {stem:28} {matches}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="folder with videos/ and/or images/")
    parser.add_argument("--check-only", action="store_true", help="check the files and write nothing")
    args = parser.parse_args()
    folder = os.path.abspath(args.dir)

    media, problems = scan(folder)
    if problems:
        print(f"{len(problems)} problem(s) in {args.dir}; nothing was written:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    counts = {k: sum(1 for v in media.values() if v[0] == k) for k in TYPES}
    print(f"{args.dir}: {counts['videos']} videos and {counts['images']} images. All checks passed.", flush=True)
    if args.check_only:
        return
    inputs = ",".join(os.path.join(folder, k) for k in TYPES if counts[k])
    # The generator runs in its own process: PyTorch and FAISS clash on macOS (see Troubleshooting)
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "embeddings", "generator.py"),
                    "--input", inputs, "--output", EMBEDDINGS], check=True, cwd=ROOT)
    nearest(media)


if __name__ == "__main__":
    main()
