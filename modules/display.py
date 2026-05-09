"""Display module for rendering detections and UI overlay."""

import time
from typing import List

import cv2
import numpy as np

from config import Config
from modules.detector import DetectionResult
from utils.logger import get_logger


class DisplayManager:
    """
    Manages on-screen rendering of bounding boxes and UI elements.

    Features:
    - Color-coded bounding boxes with labels
    - FPS counter with rolling average
    - Semi-transparent overlay bar
    - Confidence score display
    """

    def __init__(self, config: Config):
        """
        Initialize display manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._fps_start_time: float = time.time()
        self._fps_frame_count: int = 0
        self._current_fps: float = 0.0
        self._window_initialized: bool = False

    def _calculate_fps(self):
        """Update FPS calculation every 30 frames."""
        self._fps_frame_count += 1

        if self._fps_frame_count >= 30:
            elapsed = time.time() - self._fps_start_time
            self._current_fps = 30 / elapsed if elapsed > 0 else 0.0
            self._fps_start_time = time.time()
            self._fps_frame_count = 0

    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of detection results

        Returns:
            Annotated frame
        """
        output = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Draw bounding box with bright color
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                detection.color,
                self.config.BOX_THICKNESS
            )

            # Build label text
            if self.config.SHOW_CONFIDENCE:
                label_text = f"{detection.label.upper()} {int(detection.confidence * 100)}%"
            else:
                label_text = detection.label.upper()

            # Calculate label background size with padding
            font = cv2.FONT_HERSHEY_DUPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                font,
                self.config.FONT_SCALE,
                self.config.FONT_THICKNESS
            )

            # Add padding around text
            padding = 8
            text_width += padding * 2
            text_height += padding

            # Position label above box, or below if not enough space
            if y1 - text_height - baseline - 10 > 0:
                label_y_top = y1 - text_height - baseline - 5
                label_y_bottom = y1 - 5
                text_y = y1 - baseline - 8
            else:
                label_y_top = y1 + 5
                label_y_bottom = y1 + text_height + baseline + 5
                text_y = y1 + text_height

            # Draw dark semi-transparent background for better readability
            overlay = output.copy()
            cv2.rectangle(
                overlay,
                (x1, label_y_top),
                (x1 + text_width, label_y_bottom),
                (0, 0, 0),  # Black background
                -1
            )
            cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

            # Draw colored border around label
            cv2.rectangle(
                output,
                (x1, label_y_top),
                (x1 + text_width, label_y_bottom),
                detection.color,
                2
            )

            # Draw label text in bright color for maximum visibility
            cv2.putText(
                output,
                label_text,
                (x1 + padding, text_y),
                font,
                self.config.FONT_SCALE,
                (255, 255, 255),  # White text
                self.config.FONT_THICKNESS,
                cv2.LINE_AA
            )

        return output

    def draw_overlay(self, frame: np.ndarray, detection_count: int = 0) -> np.ndarray:
        """
        Draw UI overlay with title, FPS, and object count.

        Args:
            frame: Input frame
            detection_count: Number of detected objects

        Returns:
            Frame with overlay
        """
        output = frame.copy()

        # Calculate FPS
        if self.config.SHOW_FPS:
            self._calculate_fps()

        # Draw semi-transparent dark bar at top
        overlay = output.copy()
        bar_height = 40
        cv2.rectangle(overlay, (0, 0), (output.shape[1], bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

        # Draw title (top-left)
        cv2.putText(
            output,
            self.config.WINDOW_NAME,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green
            2,
            cv2.LINE_AA
        )

        # Draw FPS (top-right)
        if self.config.SHOW_FPS:
            fps_text = f"FPS: {self._current_fps:.1f}"
            (text_width, _), _ = cv2.getTextSize(
                fps_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                1
            )
            cv2.putText(
                output,
                fps_text,
                (output.shape[1] - text_width - 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),  # Cyan
                1,
                cv2.LINE_AA
            )

        # Draw object count (bottom-left)
        count_text = f"Objects: {detection_count}"
        cv2.putText(
            output,
            count_text,
            (10, output.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White
            1,
            cv2.LINE_AA
        )

        return output

    def show(self, frame: np.ndarray):
        """
        Display frame in window.

        Args:
            frame: Frame to display
        """
        if not self._window_initialized:
            cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.config.WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            self._window_initialized = True

        cv2.imshow(self.config.WINDOW_NAME, frame)

    def check_exit(self) -> bool:
        """
        Check if user pressed exit key.

        Returns:
            True if 'q' or ESC pressed
        """
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q') or key == 27  # 'q' or ESC

    def destroy(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
        self.logger.info("Display windows closed")
