"""
Batch Face Detection — processes an entire folder of images/videos.
Usage: python batch_detect.py --input ./samples --mode dnn
"""

import cv2
import os
import json
import time
import argparse
from pathlib import Path
from face_detector import FaceDetector, CONFIG


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def batch_process(input_dir: str, output_dir: str, mode: str = "haar",
                  detect_eyes: bool = False) -> dict:
    """
    Process all media files in input_dir.
    Returns a summary report dict.
    """
    detector = FaceDetector(mode=mode, detect_eyes=detect_eyes)
    os.makedirs(output_dir, exist_ok=True)

    files = list(Path(input_dir).iterdir())
    media_files = [f for f in files if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]

    if not media_files:
        print(f"[WARN] No supported media files found in '{input_dir}'")
        return {}

    print(f"[INFO] Found {len(media_files)} file(s) to process.\n")

    report = {"files": [], "total_faces_detected": 0, "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for i, filepath in enumerate(media_files, 1):
        ext = filepath.suffix.lower()
        file_report = {"file": filepath.name, "type": None, "faces": 0, "status": "ok"}

        print(f"[{i}/{len(media_files)}] Processing: {filepath.name}")

        try:
            if ext in IMAGE_EXTS:
                file_report["type"] = "image"
                frame = cv2.imread(str(filepath))
                if frame is None:
                    file_report["status"] = "read_error"
                    continue
                faces = detector.detect(frame)
                annotated = detector.annotate(frame.copy(), faces)
                out_path = Path(output_dir) / f"detected_{filepath.name}"
                cv2.imwrite(str(out_path), annotated)
                file_report["faces"] = len(faces)
                file_report["output"] = str(out_path)

            elif ext in VIDEO_EXTS:
                file_report["type"] = "video"
                cap = cv2.VideoCapture(str(filepath))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 30
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = Path(output_dir) / f"detected_{filepath.stem}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

                total_faces, frame_count = 0, 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    faces = detector.detect(frame)
                    total_faces += len(faces)
                    annotated = detector.annotate(frame.copy(), faces)
                    writer.write(annotated)

                cap.release()
                writer.release()
                file_report["faces"] = total_faces
                file_report["frames"] = frame_count
                file_report["output"] = str(out_path)

        except Exception as e:
            file_report["status"] = f"error: {e}"
            print(f"  [ERROR] {e}")

        report["files"].append(file_report)
        report["total_faces_detected"] += file_report.get("faces", 0)
        print(f"  → {file_report['faces']} face(s) detected. Saved: {file_report.get('output', 'N/A')}\n")

    # Save JSON report
    report_path = Path(output_dir) / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"[DONE] Processed {len(report['files'])} files.")
    print(f"       Total faces detected: {report['total_faces_detected']}")
    print(f"       Report saved: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Face Detection")
    parser.add_argument("--input",  default="samples", help="Input folder")
    parser.add_argument("--output", default="output",  help="Output folder")
    parser.add_argument("--mode",   choices=["haar", "dnn"], default="haar")
    parser.add_argument("--eyes",   action="store_true", help="Detect eyes too")
    args = parser.parse_args()

    batch_process(args.input, args.output, args.mode, args.eyes)
