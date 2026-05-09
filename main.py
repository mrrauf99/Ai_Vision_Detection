"""
AI Vision Assistant - Real-time Object Detection & Voice Assistance System
Entry point for the application.
"""

import sys
import time

from config import config
from modules.camera import CameraModule
from modules.detector import ObjectDetector
from modules.display import DisplayManager
from modules.voice import VoiceAssistant
from utils.logger import get_logger


def run_assistant() -> None:
    """Main application loop."""
    logger = get_logger("main")

    # Startup banner
    logger.info("=" * 60)
    logger.info("AI VISION ASSISTANT - STARTING")
    logger.info("=" * 60)

    # Initialize modules
    detector = ObjectDetector(config)
    voice = VoiceAssistant(config)
    display = DisplayManager(config)

    # Load YOLOv8 model
    logger.info("Loading YOLOv8 model...")
    if not detector.load_model():
        logger.critical("Failed to load detection model. Exiting.")
        sys.exit(1)

    # Warm up model for optimal first-frame performance
    logger.info("Warming up model...")
    detector.warm_up()

    # Initialize voice assistant (optional - continues if fails)
    logger.info("Initializing voice assistant...")
    if not voice.initialize():
        logger.warning("Voice assistant unavailable (continuing without voice)")

    # Log configuration
    logger.info("-" * 60)
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    logger.info(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    logger.info(f"Speech cooldown: {config.SPEECH_COOLDOWN}s")
    logger.info(f"Press Q or ESC to quit")
    logger.info("-" * 60)

    # Main loop with camera context manager
    start_time = time.time()
    frame_counter = 0
    last_detections = []
    tracked_labels = {}
    disappearance_grace_seconds = 1.0

    try:
        with CameraModule(config) as cam:
            if not cam.initialize():
                logger.critical("Failed to open camera. Exiting.")
                sys.exit(1)

            logger.info("Starting main detection loop...")

            while True:
                # Read frame
                success, frame = cam.read_frame()
                if not success:
                    logger.warning("Failed to read frame, skipping...")
                    continue

                # Preprocess frame (mirror effect)
                frame = cam.preprocess_frame(frame)

                # Run detection on selected frames only
                if cam.should_process_frame():
                    detections = detector.detect(frame)
                    last_detections = detections  # Cache for display

                    # Speak labels only when they transition from absent to visible.
                    current_time = time.time()
                    unique_labels = detector.get_unique_labels(detections)
                    labels_to_speak = []

                    # Refresh visibility state for labels seen in this frame.
                    for label in unique_labels:
                        state = tracked_labels.get(label)
                        if state is None or not state["visible"]:
                            labels_to_speak.append(label)
                            tracked_labels[label] = {
                                "visible": True,
                                "last_seen": current_time,
                            }
                        else:
                            state["last_seen"] = current_time

                    # Mark labels as not visible only after they have been absent long enough.
                    for label, state in list(tracked_labels.items()):
                        if label not in unique_labels:
                            if (current_time - state["last_seen"]) >= disappearance_grace_seconds:
                                state["visible"] = False

                    if labels_to_speak:
                        voice.announce_objects(labels_to_speak, apply_cooldown=False)

                # Always draw last detections (even on skipped frames)
                frame = display.draw_detections(frame, last_detections)

                # Draw UI overlay
                frame = display.draw_overlay(frame, len(last_detections))

                # Display frame
                display.show(frame)

                # Check for exit key
                if display.check_exit():
                    logger.info("Exit key pressed")
                    break

                # Periodic stats logging
                frame_counter += 1
                if frame_counter % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Stats: {frame_counter} frames | "
                        f"FPS: {display._current_fps:.1f} | "
                        f"Objects: {len(last_detections)}"
                    )

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    finally:
        # Cleanup
        voice.shutdown()
        display.destroy()

        # Final stats
        total_runtime = time.time() - start_time
        logger.info("-" * 60)
        logger.info(f"Total runtime: {total_runtime:.1f}s")
        logger.info(f"Total frames processed: {frame_counter}")
        logger.info("AI Vision Assistant shut down cleanly")
        logger.info("=" * 60)


if __name__ == "__main__":
    run_assistant()
