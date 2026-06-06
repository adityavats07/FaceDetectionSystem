"""
Face Detection System
Detects faces in images and videos using OpenCV and optional deep learning.
Supports: webcam live feed, image files, video files.
"""

import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "scale_factor": 1.1,       # How much image size is reduced at each scale
    "min_neighbors": 5,        # How many neighbors each candidate must have
    "min_face_size": (30, 30), # Minimum face size to detect
    "box_color": (0, 255, 0),  # BGR Green for bounding boxes
    "text_color": (255, 255, 255),
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "confidence_threshold": 0.5,  # For DNN model
}

HAAR_CASCADES = {
    "frontalface": "haarcascade_frontalface_default.xml",
    "profileface":  "haarcascade_profileface.xml",
    "eye":          "haarcascade_eye.xml",
    "smile":        "haarcascade_smile.xml",
}

# DNN model files (download separately — see README.md)
DNN_PROTO  = "models/deploy.prototxt"
DNN_WEIGHTS = "models/res10_300x300_ssd_iter_140000.caffemodel"


# ─── Detector Class ───────────────────────────────────────────────────────────

class FaceDetector:
    """
    Multi-mode face detector.
    Modes:
      'haar'  — Haar Cascade (fast, CPU-friendly, lower accuracy)
      'dnn'   — OpenCV DNN SSD (accurate, works on GPU too)
    """

    def __init__(self, mode: str = "haar", detect_eyes: bool = False):
        self.mode = mode
        self.detect_eyes = detect_eyes
        self.face_count_history = []

        if mode == "haar":
            cascade_path = cv2.data.haarcascades + HAAR_CASCADES["frontalface"]
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if detect_eyes:
                eye_path = cv2.data.haarcascades + HAAR_CASCADES["eye"]
                self.eye_cascade = cv2.CascadeClassifier(eye_path)
            print(f"[INFO] Haar Cascade loaded: {cascade_path}")

        elif mode == "dnn":
            if not os.path.exists(DNN_PROTO) or not os.path.exists(DNN_WEIGHTS):
                raise FileNotFoundError(
                    "DNN model files missing. Run: python download_models.py"
                )
            self.net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_WEIGHTS)
            print("[INFO] DNN SSD model loaded.")

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'haar' or 'dnn'.")

    # ── Core Detection ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run face detection on a single frame.
        Returns list of dicts: [{x, y, w, h, confidence}, ...]
        """
        if self.mode == "haar":
            return self._detect_haar(frame)
        elif self.mode == "dnn":
            return self._detect_dnn(frame)

    def _detect_haar(self, frame: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improve contrast

        faces_rect = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=CONFIG["scale_factor"],
            minNeighbors=CONFIG["min_neighbors"],
            minSize=CONFIG["min_face_size"],
        )

        results = []
        for (x, y, w, h) in (faces_rect if len(faces_rect) > 0 else []):
            face_data = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "confidence": None}

            if self.detect_eyes:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
                face_data["eyes"] = [{"x": int(ex), "y": int(ey), "w": int(ew), "h": int(eh)} for (ex, ey, ew, eh) in eyes]

            results.append(face_data)

        return results

    def _detect_dnn(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONFIG["confidence_threshold"]:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            results.append({
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
                "confidence": confidence,
            })

        return results

    # ── Annotate Frame ─────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, faces: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto frame."""
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            conf = face.get("confidence")

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), CONFIG["box_color"], 2)

            # Label
            label = f"Face {f'{conf:.0%}' if conf else ''}"
            label_y = y - 10 if y - 10 > 10 else y + 20
            cv2.putText(frame, label, (x, label_y),
                        CONFIG["font"], 0.6, CONFIG["text_color"], 2)

            # Eye sub-boxes
            if "eyes" in face:
                for eye in face["eyes"]:
                    ex, ey, ew, eh = eye["x"], eye["y"], eye["w"], eye["h"]
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 1)

        # Face count overlay
        count_label = f"Faces: {len(faces)}"
        cv2.putText(frame, count_label, (10, 30), CONFIG["font"], 0.9, (0, 200, 255), 2)

        return frame


# ─── Image Processing ──────────────────────────────────────────────────────────

def process_image(image_path: str, detector: FaceDetector, output_dir: str = "output"):
    """Detect faces in a single image and save annotated result."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    faces = detector.detect(frame)
    annotated = detector.annotate(frame.copy(), faces)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"detected_{Path(image_path).name}")
    cv2.imwrite(out_path, annotated)

    print(f"[RESULT] {len(faces)} face(s) detected in '{image_path}'")
    print(f"[SAVED]  {out_path}")

    # Show result
    cv2.imshow("Face Detection — Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return faces


# ─── Video Processing ──────────────────────────────────────────────────────────

def process_video(source, detector: FaceDetector, output_dir: str = "output",
                  save_output: bool = True, skip_frames: int = 0):
    """
    Detect faces in a video file or live webcam stream.
    source: path to video file (str) or webcam index (int, usually 0)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    # Video properties
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label  = "Webcam" if isinstance(source, int) else Path(str(source)).name

    print(f"[INFO] Source: {label} | {width}x{height} @ {fps:.1f}fps | {total or '∞'} frames")

    writer = None
    if save_output and not isinstance(source, int):
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"detected_{label}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {out_path}")

    frame_idx = 0
    fps_counter, fps_start = 0, time.time()
    total_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for performance
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            continue

        # Detect
        faces = detector.detect(frame)
        total_faces += len(faces)
        annotated = detector.annotate(frame.copy(), faces)

        # FPS overlay
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            live_fps = fps_counter / elapsed
            fps_counter, fps_start = 0, time.time()
        else:
            live_fps = fps_counter / max(elapsed, 0.001)

        cv2.putText(annotated, f"FPS: {live_fps:.1f}", (10, 60),
                    CONFIG["font"], 0.7, (0, 200, 255), 2)

        if writer:
            writer.write(annotated)

        cv2.imshow(f"Face Detection — {label}", annotated)

        # 'q' to quit, 's' to save screenshot
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] User quit.")
            break
        elif key == ord("s"):
            shot_path = os.path.join(output_dir, f"screenshot_{frame_idx}.jpg")
            cv2.imwrite(shot_path, annotated)
            print(f"[SAVED] Screenshot → {shot_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_faces = total_faces / max(frame_idx, 1)
    print(f"\n[SUMMARY] Processed {frame_idx} frames | Avg faces/frame: {avg_faces:.2f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection System")
    parser.add_argument("--source", default="0",
                        help="Path to image/video file, or '0' for webcam")
    parser.add_argument("--mode", choices=["haar", "dnn"], default="haar",
                        help="Detection mode: haar (fast) or dnn (accurate)")
    parser.add_argument("--output", default="output",
                        help="Output directory for results")
    parser.add_argument("--eyes", action="store_true",
                        help="Also detect eyes within each face (Haar mode only)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip N frames between detections (speeds up video)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output video (display only)")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = FaceDetector(mode=args.mode, detect_eyes=args.eyes)

    source = args.source
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    # Auto-detect input type
    if source.isdigit():
        print("[MODE] Live webcam")
        process_video(int(source), detector, args.output, not args.no_save, args.skip)
    elif Path(source).suffix.lower() in img_exts:
        print("[MODE] Image file")
        process_image(source, detector, args.output)
    elif Path(source).suffix.lower() in vid_exts:
        print("[MODE] Video file")
        process_video(source, detector, args.output, not args.no_save, args.skip)
    else:
        print(f"[ERROR] Unrecognized source type: {source}")


if __name__ == "__main__":
    main()
