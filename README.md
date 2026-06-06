# 🎯 Face Detection System

A production-ready face detection system for **images**, **video files**, and **live webcam streams** — built with Python and OpenCV.
---
## ✨ Features
| Feature | Haar Cascade | DNN (SSD ResNet-10) |
|---|---|---|
| Speed | ⚡ Very fast | 🔄 Moderate |
| Accuracy | Good (frontal faces) | Excellent (all angles) |
| GPU support | ❌ | ✅ CUDA / OpenCL |
| Dependencies | OpenCV only | OpenCV + model files |
| Best for | Real-time / low-end devices | Accuracy-critical tasks |

- 📷 **Image detection** — JPG, PNG, BMP, WebP, TIFF
- 🎬 **Video detection** — MP4, AVI, MOV, MKV
- 📹 **Live webcam** — real-time with FPS counter
- 👁️ **Eye detection** — optional sub-feature (Haar mode)
- 📦 **Batch processing** — entire folder at once + JSON report
- 💾 **Auto-save** — annotated outputs saved to `/output`
---
## 📁 Project Structure
```
face_detection_system/
├── face_detector.py       # Main detection engine (images + videos + webcam)
├── batch_detect.py        # Batch-process an entire folder
├── download_models.py     # Downloads DNN model weights
├── requirements.txt       # Python dependencies
├── models/                # DNN model files (after download)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── samples/               # Put your test images/videos here
└── output/                # Annotated results saved here
```
---
## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. (Optional) Download DNN model

```bash
python download_models.py
```
### 3. Run detection
```bash
# Live webcam (default)
python face_detector.py

# On an image
python face_detector.py --source photo.jpg

# On a video file
python face_detector.py --source video.mp4

# Use accurate DNN mode
python face_detector.py --source photo.jpg --mode dnn

# Also detect eyes
python face_detector.py --source photo.jpg --eyes

# Batch process a folder
python batch_detect.py --input ./samples --output ./output --mode dnn
```
---

## 🎮 Keyboard Controls (Video/Webcam)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot to `/output` |
---

## ⚙️ CLI Options

### `face_detector.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Image/video path, or `0`/`1`/`2` for webcam |
| `--mode` | `haar` | `haar` (fast) or `dnn` (accurate) |
| `--output` | `output` | Output directory |
| `--eyes` | off | Enable eye detection (Haar only) |
| `--skip N` | `0` | Skip N frames (speed up video) |
| `--no-save` | off | Don't save output video |

### `batch_detect.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `samples` | Input folder |
| `--output` | `output` | Output folder |
| `--mode` | `haar` | `haar` or `dnn` |
| `--eyes` | off | Enable eye detection |
---
## 🧠 How It Works

### Haar Cascade (Mode: `haar`)
1. Convert frame to **grayscale** + histogram equalization
2. Slide a detection **window** across the image at multiple scales
3. Apply a series of **weak classifiers** (Haar features) in a cascade
4. A face is detected only when ALL stages of the cascade pass
5. **Non-maximum suppression** removes duplicate boxes

### DNN SSD (Mode: `dnn`)
1. Resize frame to **300×300** and normalize pixel values
2. Forward pass through **ResNet-10 backbone** + SSD detection head
3. Outputs bounding boxes with **confidence scores**
4. Filter detections below the confidence threshold (default: 50%)
5. No separate NMS step needed — SSD handles it internally

---

## 📊 Performance Benchmarks

| Mode | Resolution | FPS (CPU i7) | FPS (GPU RTX 3060) |
|------|-----------|--------------|---------------------|
| Haar | 640×480 | ~45–60 fps | N/A |
| DNN | 640×480 | ~12–20 fps | ~60+ fps |
| DNN | 1280×720 | ~8–15 fps | ~45+ fps |

---
## 🔧 Customizing

Edit `CONFIG` dict at the top of `face_detector.py`:

```python
CONFIG = {
    "scale_factor": 1.1,        # Lower = slower but more accurate
    "min_neighbors": 5,          # Higher = fewer false positives
    "min_face_size": (30, 30),   # Ignore faces smaller than this
    "confidence_threshold": 0.5, # DNN mode confidence cutoff
    "box_color": (0, 255, 0),    # BGR color of bounding boxes
}
```
---

## 📦 Dependencies

- **Python** 3.9+
- **OpenCV** 4.8+ (`opencv-python` or `opencv-contrib-python`)
- **NumPy** 1.24+
- **Pillow** 10.0+ (optional, for extra image format support)
---
## 📄 License

MIT License — free to use, modify, and distribute.
---
## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push and open a Pull Request
---
## 🗺️ Roadmap

- [ ] Face recognition (identify known individuals)
- [ ] Age & gender estimation
- [ ] Emotion detection
- [ ] REST API / Flask web interface
- [ ] ONNX model support
- [ ] Docker container
