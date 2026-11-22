"""Checks for the "Use your own data" adapter (scripts/own_media.py). CPU only, no network."""

import json
import os
import shutil
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import own_media  # noqa: E402
from own_media import scan  # noqa: E402

VIDEO = os.path.join(ROOT, "data", "videos", "download (25).mp4")
IMAGE = os.path.join(ROOT, "data", "images", "Tuna_Lover.jpg")


@pytest.fixture(autouse=True)
def no_sample_embeddings(tmp_path, monkeypatch):
    """Scan against an empty embeddings folder unless a test sets one up."""
    monkeypatch.setattr(own_media, "EMBEDDINGS", str(tmp_path / "embeddings"))


def media(folder, videos=("clip.mp4",), images=("dish.jpg",)):
    for kind, names, src in (("videos", videos, VIDEO), ("images", images, IMAGE)):
        os.makedirs(os.path.join(folder, kind), exist_ok=True)
        for name in names:
            shutil.copy(src, os.path.join(folder, kind, name))
    return str(folder)


def write(folder, rel, data):
    path = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def test_good_folder_passes_and_writes_nothing(tmp_path):
    folder = media(tmp_path / "mine")
    write(folder, "videos/.DS_Store", b"\0")
    before = sorted(os.listdir(ROOT))
    found, problems = scan(folder)
    assert problems == []
    assert {k: v[0] for k, v in found.items()} == {"clip": "videos", "dish": "images"}
    assert sorted(os.listdir(ROOT)) == before


def test_images_only_is_fine(tmp_path):
    folder = media(tmp_path / "mine", videos=())
    shutil.rmtree(os.path.join(folder, "videos"))
    assert scan(folder)[1] == []


@pytest.mark.parametrize("rel,data,expected", [
    ("videos/clip2.mp4", b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
    ("videos/broken.mp4", b"not a video", "not a .mp4 video (no file header)"),
    ("videos/broken.avi", b"not a video", "not a .avi video (no file header)"),
    ("videos/fake.mov", b"\0\0\0\x14ftypqt  " + b"\0" * 64, "can't open as a video"),
    ("videos/empty.mp4", b"", "empty file"),
    ("videos/notes.txt", b"hello", "not one of .mp4 .avi .mov"),
    ("videos/photo.jpg", b"x", "put it in images/"),
    ("images/broken.jpg", b"not an image", "can't open as an image"),
    ("images/clip.webm", b"x", "not one of .jpg .jpeg .png"),
])
def test_bad_files_are_reported(tmp_path, rel, data, expected):
    folder = media(tmp_path / "mine")
    write(folder, rel, data)
    _, problems = scan(folder)
    assert len(problems) == 1 and problems[0].startswith(rel) and expected in problems[0], problems


def test_upper_case_extension_is_reported(tmp_path):
    folder = media(tmp_path / "mine", videos=("BIG.MP4",), images=("Dish.JPG",))
    _, problems = scan(folder)
    assert problems == ["videos/BIG.MP4: upper-case extension; the generator skips it. Rename it to BIG.mp4",
                        "images/Dish.JPG: upper-case extension; the generator skips it. Rename it to Dish.jpg"]


def test_tiny_image_is_reported(tmp_path):
    folder = media(tmp_path / "mine")
    Image.new("RGB", (16, 16)).save(os.path.join(folder, "images", "tiny.png"))
    assert scan(folder)[1] == ["images/tiny.png: too small (16x16), needs at least 32x32"]


def test_name_clashes_are_reported(tmp_path):
    folder = media(tmp_path / "mine", videos=("bowl.mp4", "bowl.mov"), images=("bowl.jpg",))
    _, problems = scan(folder)
    assert problems == ["videos/bowl.mp4: same name as videos/bowl.mov; their embeddings would overwrite each other",
                        "images/bowl.jpg: same name as videos/bowl.mov; their embeddings would overwrite each other"]


def test_clash_with_an_existing_embedding(tmp_path):
    folder = media(tmp_path / "mine", videos=("download (25).mp4",))
    os.makedirs(own_media.EMBEDDINGS)
    with open(os.path.join(own_media.EMBEDDINGS, "download (25)_metadata.json"), "w") as f:
        json.dump({"media_id": "download (25)", "original_file": "data/videos/download (25).mp4"}, f)
    _, problems = scan(folder)
    assert problems == ["videos/download (25).mp4: data/embeddings already has 'download (25)' "
                        "(from data/videos/download (25).mp4); rename the file"]
    # Loading the same folder again is fine: the embedding came from this file
    with open(os.path.join(own_media.EMBEDDINGS, "download (25)_metadata.json"), "w") as f:
        json.dump({"media_id": "download (25)",
                   "original_file": os.path.join(folder, "videos", "download (25).mp4")}, f)
    assert scan(folder)[1] == []


def test_long_video_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(own_media, "MAX_FRAME_BYTES", 10 * 1024 ** 2)
    folder = media(tmp_path / "mine")
    _, problems = scan(folder)
    assert len(problems) == 1 and "too long" in problems[0] and "GB of frames in memory" in problems[0]


def test_comma_in_path_and_empty_folder(tmp_path):
    folder = media(tmp_path / "a,b")
    assert "the path contains a comma" in scan(folder)[1][0]
    os.makedirs(tmp_path / "empty")
    assert scan(str(tmp_path / "empty"))[1] == ["no videos/ or images/ folder"]
