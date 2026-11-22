"""Ask the running similarity search API (make similar, make quality). Start it with `make serve`.

    similar ID         GET  /api/similar/{ID}?k=5
    quality VIDEO IMG  POST /api/quality/check {"video_id", "reference_image", "threshold"}
"""

import argparse
import json
import os
import sys
import urllib.parse

import requests

API = f"http://localhost:{os.environ.get('API_PORT', '8085')}"


def call(method, path, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=120, **kwargs)
    except requests.ConnectionError:
        sys.exit(f"No API at {API}. Start it in another terminal with: make serve")
    if response.status_code != 200:
        sys.exit(f"{method} {path} -> HTTP {response.status_code} {response.text}")
    return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("similar")
    s.add_argument("id")
    s.add_argument("-k", type=int, default=5)
    q = sub.add_parser("quality")
    q.add_argument("video")
    q.add_argument("image")
    q.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    if args.command == "similar":
        path = f"/api/similar/{urllib.parse.quote(args.id)}?k={args.k}"
        data = call("GET", path)
        print(f"GET {path}")
        for r in data["results"]:
            kind = r.get("metadata", {}).get("media_type", "?")
            print(f"  {r['rank']}. {r['video_id']:20} {kind:6} L2 distance {r['distance']:.4f}  "
                  f"similarity_score {r['similarity_score']:.4f}")
    else:
        body = {"video_id": args.video, "reference_image": args.image, "threshold": args.threshold}
        data = call("POST", "/api/quality/check", json=body)
        print(f"POST /api/quality/check {json.dumps(body)}")
        result = "PASSED" if data["quality_passed"] else "FAILED"
        print(f"  {data['video_id']} vs {data['reference_image']}: best frame {data['max_similarity_score']:.4f}, "
              f"threshold {data['threshold']} -> {result} ({data['assessment']}), "
              f"{data['processing_time_seconds']:.1f} s")


if __name__ == "__main__":
    main()
