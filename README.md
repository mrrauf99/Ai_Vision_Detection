# Ai_Vision_Detection

Lightweight AI vision detection project using YOLOv8 for object detection on camera input.

## Overview

- Uses YOLOv8 models (`yolov8n.pt`, `yolov8s.pt`) for real-time detection.
- Modular code under `modules/` for camera, detector, display, and voice.
- `utils/` contains logging utilities.

## Files

- `main.py` — application entrypoint.
- `config.py` — configuration values.
- `requirements.txt` — Python dependencies.
- `yolov8n.pt`, `yolov8s.pt` — pretrained model files (included).
- `modules/`:
  - `camera.py` — camera capture and frame handling.
  - `detector.py` — model loading and inference logic.
  - `display.py` — rendering detections to screen.
  - `voice.py` — optional voice alerts.

## Quickstart

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Run the app:

```bash
python main.py
```

3. If you have multiple models, configure `config.py` to choose `yolov8n.pt` or `yolov8s.pt`.

## Notes

- Adjust camera index/settings in `modules/camera.py` if needed.
- This repository contains model weight files; ensure you have sufficient disk space.

## License

Add your preferred license to the repository if desired.
