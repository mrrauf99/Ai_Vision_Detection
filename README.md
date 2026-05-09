# Ai_Vision_Detection

AI vision detection using YOLOv8 for real-time object detection.

## Overview

- Real-time object detection using YOLOv8 (`yolov8n.pt`, `yolov8s.pt`).
- Modular implementation under `modules/` for camera, detection, display, and voice.
- Utility functions and logging are provided in `utils/`.

## Repository structure

- `main.py` — application entry point
- `config.py` — configuration settings
- `requirements.txt` — Python dependencies
- `yolov8n.pt`, `yolov8s.pt` — pretrained model weights
- `modules/` — `camera.py`, `detector.py`, `display.py`, `voice.py`
- `utils/` — helper utilities and logging

## Quickstart

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

2. Run the application:

```bash
python main.py
```

3. To select a model, edit `config.py` and set the desired weight file.

## Notes

- Adjust camera index or parameters in `modules/camera.py` as needed.
- Model weight files are large; ensure adequate disk space.

## License

This project is licensed under the MIT License. See the `LICENSE` file for full terms.
