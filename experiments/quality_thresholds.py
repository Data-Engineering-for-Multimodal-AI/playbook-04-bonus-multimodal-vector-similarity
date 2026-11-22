"""Try this 3: can the quality check tell one dish from another? (make try-quality-thresholds)

The quality check (src/utils/quality_checker.py) samples 8 frames from a
video, compares each with a reference photo, and passes the video if the best
frame's cosine similarity reaches the threshold (0.75 by default; the guide
also uses 0.65 and 0.80). Each video shows one dish, so it should pass against
at most one of the 7 reference photos. This scores every video against every
photo, the same way, and counts how many photos each video passes against.

Change THRESHOLDS and run it again.
"""

import contextlib
import io
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # hide harmless h264 decoder messages
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))
from quality_checker import QualityChecker  # noqa: E402

VIDEOS = os.path.join(ROOT, "data", "videos")
IMAGES = os.path.join(ROOT, "data", "images")
THRESHOLDS = [0.65, 0.75, 0.80, 0.85]


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        checker = QualityChecker()
    videos = sorted(f for f in os.listdir(VIDEOS) if f.endswith(".mp4"))
    images = sorted(f for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    start = time.time()
    refs = np.stack([checker.generate_embedding(checker.load_reference_image(os.path.join(IMAGES, f)))
                     for f in images])
    scores = np.zeros((len(videos), len(images)))
    for i, name in enumerate(videos):
        frames, _ = checker.extract_frames(os.path.join(VIDEOS, name))
        emb = np.stack([checker.generate_embedding(fr) for fr in frames])
        scores[i] = (emb @ refs.T).max(axis=0)  # best frame per photo, as assess_quality does
    print(f"Scored {len(videos)} videos against {len(images)} reference photos in {time.time() - start:.0f} s.\n")

    print(f"{'threshold':>9} {'pairs passed':>14} {'videos passing 2+ photos':>25} {'photos per video: mean (max)':>29}")
    for t in THRESHOLDS:
        per_video = (scores >= t).sum(axis=1)
        print(f"{t:>9.2f} {int(per_video.sum()):>6} of {scores.size:<6} {int((per_video >= 2).sum()):>14} of {len(videos):<8}"
              f" {per_video.mean():>18.1f} ({per_video.max()})")

    top2 = np.sort(scores, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    print(f"\nScores range from {scores.min():.3f} to {scores.max():.3f}. Gap between a video's best and second-best "
          f"photo: median {np.median(gap):.3f}, largest {gap.max():.3f}.")
    j = images.index("Salmon_Edamame.jpg")
    i = videos.index("download (25).mp4")
    print(f"Check against the guide: download (25) vs Salmon_Edamame.jpg = {scores[i, j]:.4f}")


if __name__ == "__main__":
    main()
