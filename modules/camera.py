"""Camera module for webcam capture and frame preprocessing."""

import cv2
import numpy as np
from typing import Optional, Tuple

from config import Config
from utils.logger import get_logger


class CameraModule:
    """
    Manages webcam capture with frame skipping and preprocessing.

    Features:
    - Auto-fallback camera index detection (0, 1, 2)
    - Low-latency buffer configuration
    - Frame skipping for CPU efficiency
    - Mirror effect preprocessing
    """

    def __init__(self, config: Config):
        """
        Initialize camera module.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0

    def initialize(self) -> bool:
        """
        Open and configure webcam.

        Returns:
            True if camera opened successfully, False otherwise
        """
        # Try camera indices 0, 1, 2 automatically
        for camera_index in range(3):
            self.logger.info(f"Attempting to open camera at index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index)

            if self.cap.isOpened():
                # Configure camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

                self.logger.info(f"Camera opened successfully at index {camera_index}")
                return True

            self.cap.release()

        self.logger.error("Failed to open camera on any index (0-2)")
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None

        success, frame = self.cap.read()
        if success:
            self.frame_count += 1

        return success, frame

    def should_process_frame(self) -> bool:
        """
        Check if current frame should be processed (frame skipping logic).

        Returns:
            True if frame should be processed
        """
        return self.frame_count % self.config.FRAME_SKIP == 0

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for display (mirror effect).

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Flip horizontally for mirror effect (more natural for users)
        return cv2.flip(frame, 1)

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.release()
