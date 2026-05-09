"""Object detection module using YOLOv8."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from config import Config
from utils.logger import get_logger


@dataclass
class DetectionResult:
    """
    Single object detection result.

    Attributes:
        label: Object class name
        confidence: Detection confidence score (0.0-1.0)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        color: BGR color for visualization
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    color: Tuple[int, int, int]


class ObjectDetector:
    """
    YOLOv8-based object detection engine.

    Features:
    - Auto-downloads YOLOv8 model on first run
    - Class-based color assignment
    - Inference time tracking
    - Model warm-up for first-frame optimization
    """

    def __init__(self, config: Config):
        """
        Initialize object detector.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self._color_map: Dict[int, Tuple[int, int, int]] = {}
        self.last_inference_ms: float = 0.0

    def load_model(self) -> bool:
        """
        Load YOLOv8 model from file or download if needed.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading model: {self.config.MODEL_NAME}")
            self.model = YOLO(self.config.MODEL_NAME)
            self.class_names = self.model.names
            self.logger.info(f"Model loaded: {len(self.class_names)} classes available")
            self.logger.info(f"Device: {self.config.DEVICE}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def warm_up(self) -> None:
        """
        Warm up model with dummy inference to eliminate first-frame latency.
        """
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detect(dummy_frame)
        self.logger.info("Model warmed up")

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Get cached or assign new color for object class.

        Args:
            class_id: YOLO class ID

        Returns:
            BGR color tuple
        """
        if class_id not in self._color_map:
            color_index = class_id % len(self.config.BOX_COLORS)
            self._color_map[class_id] = self.config.BOX_COLORS[color_index]

        return self._color_map[class_id]

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run object detection on frame.

        Args:
            frame: Input image frame

        Returns:
            List of detection results
        """
        if self.model is None:
            self.logger.warning("Model not loaded, skipping detection")
            return []

        try:
            # Run inference on the original frame so YOLO preserves the camera aspect ratio.
            start_time = time.time()
            results = self.model.predict(
                source=frame,
                imgsz=512,
                conf=self.config.CONFIDENCE_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                max_det=self.config.MAX_DETECTIONS,
                device=self.config.DEVICE,
                verbose=False
            )
            self.last_inference_ms = (time.time() - start_time) * 1000
            self.logger.debug(f"Inference: {self.last_inference_ms:.1f}ms")

            # Parse detections
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes

                for box in boxes:
                    # Extract box data
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    # Create detection result
                    detections.append(DetectionResult(
                        label=self.class_names[cls],
                        confidence=conf,
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        color=self._get_class_color(cls)
                    ))

            return detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def get_unique_labels(self, detections: List[DetectionResult]) -> List[str]:
        """
        Extract unique object labels from detections.

        Args:
            detections: List of detection results

        Returns:
            Deduplicated list of label strings (preserves order)
        """
        seen = set()
        unique_labels = []
        for detection in detections:
            if detection.label not in seen:
                seen.add(detection.label)
                unique_labels.append(detection.label)
        return unique_labels
