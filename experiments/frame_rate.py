"""Try this 1: how many frames does a video embedding need? (make try-frame-rate)

The generator (src/embeddings/generator.py) runs every 5th frame through
ResNet-18 and averages the results: 24 frames for a 5-second clip at 24 fps.
This re-embeds the 32 sample videos keeping every 10th, 30th, 60th and 120th
frame, and compares with the embeddings `make run` saved (every 5th): time,
similarity to the saved vector, and whether each reference photo's three
closest videos stay the same. Writes nothing.

Change RATES and run it again.
"""

import contextlib
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # hide harmless h264 decoder messages
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "embeddings"))
import generator  # noqa: E402

EMBEDDINGS = os.path.join(ROOT, "data", "embeddings")
VIDEOS = os.path.join(ROOT, "data", "videos")
IMAGES = os.path.join(ROOT, "data", "images")
RATES = [10, 30, 60, 120]


def top3(video_vecs, image_vecs):
    return [tuple(np.argsort(-(video_vecs @ img))[:3]) for img in image_vecs]


def main():
    videos = sorted(os.path.splitext(f)[0] for f in os.listdir(VIDEOS) if f.endswith(".mp4"))
    images = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    try:
        saved = np.stack([np.load(os.path.join(EMBEDDINGS, f"{v}_embedding.npy")) for v in videos])
        img = np.stack([np.load(os.path.join(EMBEDDINGS, f"{i}_embedding.npy")) for i in images])
    except FileNotFoundError:
        sys.exit("No embeddings in data/embeddings. Run `make run` first.")
    meta = [json.load(open(os.path.join(EMBEDDINGS, f"{v}_metadata.json"))) for v in videos]
    saved_time = sum(m["processing_time_seconds"] for m in meta)
    saved_frames = int(np.median([m["num_frames_processed"] for m in meta]))
    base_top = top3(saved, img)

    with contextlib.redirect_stdout(io.StringIO()):
        gen = generator.MediaEmbeddingGenerator()
    print(f"{len(videos)} videos. Compared with every 5th frame (what make run saved):\n")
    print(f"{'every Nth frame':>15} {'frames':>7} {'time':>7} {'similarity to saved':>19} {'lowest':>7} {'same top 3 per photo':>21}")
    print(f"{5:>15} {saved_frames:>7} {saved_time:>6.0f}s {1.0:>19.4f} {1.0:>7.3f} {len(images):>16} of {len(images)}")
    for rate in RATES:
        generator.FRAME_SAMPLE_RATE = rate
        start = time.time()
        out = [gen.generate_video_embedding(os.path.join(VIDEOS, v + ".mp4")) for v in videos]
        vecs = np.stack([o[0] for o in out])
        frames = int(np.median([len(o[1]) for o in out]))
        elapsed = time.time() - start
        sim = np.sum(vecs * saved, axis=1)
        same = sum(a == b for a, b in zip(top3(vecs, img), base_top))
        print(f"{rate:>15} {frames:>7} {elapsed:>6.0f}s {sim.mean():>19.4f} {sim.min():>7.3f}"
              f" {same:>16} of {len(images)}")


if __name__ == "__main__":
    main()
