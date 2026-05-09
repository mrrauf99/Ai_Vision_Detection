"""Configuration settings for AI Vision Assistant."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Centralized configuration for AI Vision Assistant.

    Groups all tunable parameters for camera, model, voice, and display.
    """

    # ========== Camera Settings ==========
    CAMERA_INDEX: int = 0  # Webcam device index (0 = default)
    FRAME_WIDTH: int = 1280  # Capture width in pixels
    FRAME_HEIGHT: int = 720  # Capture height in pixels
    FRAME_SKIP: int = 3  # Process every Nth frame to save CPU

    # ========== Model Settings ==========
    MODEL_NAME: str = "yolov8s.pt"  # YOLOv8 small model (better accuracy, still fast)
    CONFIDENCE_THRESHOLD: float = 0.40  # Minimum detection confidence (slightly stricter to reduce false positives)
    IOU_THRESHOLD: float = 0.45  # Non-max suppression IOU threshold
    MAX_DETECTIONS: int = 30  # Max objects to detect per frame
    DEVICE: str = "cpu"  # "cpu" or "cuda" for GPU

    # ========== Voice Settings ==========
    TTS_RATE: int = 150  # Speech rate (words per minute)
    TTS_VOLUME: float = 1.0  # Volume level (0.0–1.0)
    SPEECH_COOLDOWN: float = 3.0  # Seconds before re-announcing same object
    MAX_ANNOUNCE_OBJECTS: int = 3  # Max objects announced per speech cycle

    # ========== Display Settings ==========
    WINDOW_NAME: str = "AI Vision Assistant"
    BOX_THICKNESS: int = 3  # Bounding box line thickness
    FONT_SCALE: float = 0.8  # Label text size
    FONT_THICKNESS: int = 2  # Label text thickness
    SHOW_CONFIDENCE: bool = True  # Show confidence % on label
    SHOW_FPS: bool = True  # Show FPS counter on screen

    # ========== Color Palette ==========
    # BGR format for OpenCV - bright, vibrant colors
    BOX_COLORS: list = field(default_factory=lambda: [
        (0, 255, 0),      # Bright Green
        (255, 100, 0),    # Bright Blue
        (0, 100, 255),    # Bright Orange
        (0, 255, 255),    # Bright Yellow
        (255, 0, 255),    # Bright Magenta
        (255, 255, 0),    # Bright Cyan
        (100, 255, 100),  # Light Green
        (255, 150, 150),  # Light Blue
    ])


# Singleton configuration instance
config = Config()
