# 🧭 Video Processing – Scoreboard Detection & OCR

A Python pipeline that detects scoreboards in video frames using YOLO and extracts team names and scores via TrOCR (Transformer-based OCR).

---

## 📚 Table of Contents

[About](#-about)  
[Features](#-features)  
[Tech Stack](#-tech-stack)  
[Installation](#-installation)  
[Usage](#-usage)  
[Configuration](#-configuration)  
[Screenshots](#-screenshots)  
[API Documentation](#-api-documentation)  
[Contact](#-contact)  
[Acknowledgements](#-acknowledgements)

---

## 🧩 About

This project automates score extraction from sports or broadcast videos. It uses a custom-trained YOLO model to locate scoreboard regions in each frame, then runs Microsoft TrOCR on those crops to read team names and scores. The goal is to provide a reusable pipeline for video-based scoreboard analysis without manual cropping or typing.

---

## ✨ Features

- **YOLO scoreboard detection** – Localizes scoreboard regions in video frames using a trained Ultralytics YOLO model.
- **TrOCR-based OCR** – Extracts text from detected scoreboard crops using the Transformer-based TrOCR model for robust reading.
- **Team & score parsing** – Regex-based extraction of team abbreviations and numeric scores from OCR output.
- **Video processing** – Processes `video.mp4` with configurable frame sampling (every 150 frames) for efficient analysis.
- **Single-image testing** – `app.py` allows quick testing on a single frame (`./frames/frame_75750.jpg`).
- **Custom model training** – `train.py` supports training YOLO on your own dataset via `data.yaml` for different scoreboard layouts.

---

## 🧠 Tech Stack

| Category   | Technologies |
|-----------|--------------|
| **Languages** | Python |
| **Frameworks / Libraries** | OpenCV (cv2), PyTorch, Ultralytics YOLO, Hugging Face Transformers |
| **Models** | YOLO11n (training), custom `.pt` weights (inference), Microsoft TrOCR (OCR) |
| **Tools** | PIL/Pillow, pytesseract (optional), CUDA (optional, for GPU) |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Winner960111/Real-time-Video-analytics-YOLO.git

# Navigate to the project directory
cd Real-time-Video-analytics-YOLO

# Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install torch torchvision opencv-python ultralytics transformers pillow pytesseract
```

**Note:** For training, ensure `data.yaml` exists and points to your dataset (under `dataset/`). For OCR, TrOCR will download model weights on first run.

---

## 🚀 Usage

**Process a full video**

```bash
python main.py
```

Place your video as `video.mp4` in the project root. The script samples frames (every 150th), runs detection + OCR, and prints extracted team scores.

**Test on a single image:**

```bash
python app.py
```

Update `image_path` in `app.py` (e.g. `./frames/frame_75750.jpg`) to your frame path. A window will show the image with the detected scoreboard bounding box.

**Train a custom YOLO model:**

```bash
python train.py
```

Uses `data.yaml` and saves/uses weights (`best.pt`, `best2.pt`) for inference in `main.py` and `app.py`.

---

## 🧾 Configuration

- **Video path:** Set `video_path` and `output_folder` in `main.py` (default: `video.mp4`, `frames`).
- **Frame sampling:** In `main.py`, change `if frame_count % 150 == 0` to sample more or fewer frames.
- **Model weights:** Point `YOLO("best2.pt")` (or `best.pt` / `best1.pt`) to your trained model in `main.py` and `app.py`.
- **Image path:** In `app.py`, set `image_path` to the frame you want to test.
- **Training:** Edit `train.py` to change base model (`yolo11n.pt`), and ensure `data.yaml` exists with correct `train`/`valid` paths (under `dataset/`).

No `.env` file is required by default; TrOCR and YOLO use local/cached weights.

---

## 📜 API Documentation

This project is script-based and does not expose an HTTP API. The main entry points are:

| Script      | Purpose |
|------------|---------|
| `main.py`  | Run scoreboard detection + OCR on `video.mp4`. |
| `app.py`   | Run detection + visualization on a single image. |
| `train.py` | Train YOLO on custom scoreboard data using `data.yaml`. |

For integration, import and call `getScore(frame)` from `main.py` (with `yolo_model` and TrOCR loaded) to get scores for a single frame.

---

## 🌟 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection.
- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-base-stage1) (Hugging Face) for Transformer-based OCR.
- [OpenCV](https://opencv.org/) for video and image I/O and drawing.
- [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model inference and training.
