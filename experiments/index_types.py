"""Try this 2: does the FAISS index type change the results? (make try-index-types)

src/similarity/search.py builds one of three FAISS indexes over the embeddings:
Flat (exact; the API uses it), IVF or HNSW. This builds each one over the same
39 embeddings from `make run`, asks every embedding for its 5 nearest
neighbors, and counts how many of Flat's neighbors each index also returns
(recall@5). For IVF it also varies nprobe, the number of clusters searched,
which search.py leaves at FAISS's default of 1. FAISS only; no PyTorch.

Change NPROBES, or nlist in search.py, and run it again.
"""

import contextlib
import io
import os
import sys
import time

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "similarity"))
from search import SimilaritySearch  # noqa: E402

EMBEDDINGS = os.path.join(ROOT, "data", "embeddings")
K = 5
NPROBES = [1, 2, 4, 8]


def neighbors(search):
    start = time.perf_counter()
    _, idx = search.index.search(search.embeddings, K + 1)
    elapsed = (time.perf_counter() - start) * 1000
    return [set(row[row >= 0].tolist()) - {i} for i, row in enumerate(idx)], elapsed


def build(kind):
    s = SimilaritySearch(EMBEDDINGS)
    with contextlib.redirect_stdout(io.StringIO()):
        s.load_embeddings()
        s.build_index(kind)
    return s


def main():
    if not os.path.isdir(EMBEDDINGS) or not any(f.endswith("_embedding.npy") for f in os.listdir(EMBEDDINGS)):
        sys.exit("No embeddings in data/embeddings. Run `make run` first.")
    flat = build("Flat")
    truth, _ = neighbors(flat)
    n = len(flat.embeddings)
    print(f"{n} embeddings, {K} nearest neighbors for each, compared with Flat (exact search):\n")
    print(f"{'index':27} {'recall@5':>9} {'search time (all queries)':>26}")

    def row(label, search):
        found, ms = neighbors(search)
        recall = np.mean([len(f & t) / len(t) for f, t in zip(found, truth)])
        print(f"{label:27} {recall:>8.0%} {ms:>23.2f} ms")

    row("Flat", flat)
    ivf = build("IVF")
    for nprobe in NPROBES:
        ivf.index.nprobe = nprobe
        default = " (search.py)" if nprobe == 1 else ""
        row(f"IVF, nprobe {nprobe}{default}", ivf)
    row("HNSW (M=32)", build("HNSW"))
    print(f"\nIVF has nlist = {ivf.index.nlist} clusters for {n} vectors.")


if __name__ == "__main__":
    main()
