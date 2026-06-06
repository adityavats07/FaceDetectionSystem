"""
Download DNN model weights for the accurate face detection mode.
Files: OpenCV's pre-trained SSD ResNet-10 Caffe model.
"""

import urllib.request
import os
from pathlib import Path

MODELS_DIR = "models"

FILES = [
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "dest": "models/deploy.prototxt",
        "desc": "Network architecture (prototxt)",
    },
    {
        "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "dest": "models/res10_300x300_ssd_iter_140000.caffemodel",
        "desc": "Pre-trained weights (caffemodel, ~10MB)",
    },
]


def download_file(url: str, dest: str, desc: str):
    if os.path.exists(dest):
        print(f"[SKIP] Already exists: {dest}")
        return

    print(f"[DOWNLOAD] {desc}")
    print(f"  URL:  {url}")
    print(f"  Dest: {dest}")

    os.makedirs(Path(dest).parent, exist_ok=True)

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print(f"\n  [OK] Saved {dest}")
    except Exception as e:
        print(f"\n  [ERROR] Failed: {e}")


if __name__ == "__main__":
    print("Face Detection — DNN Model Downloader\n" + "=" * 40)
    for file_info in FILES:
        download_file(**file_info)
    print("\n[DONE] All model files ready. You can now use --mode dnn")
