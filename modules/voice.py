"""Voice assistant module for object announcements."""

import queue
import subprocess
import threading
import time
from typing import Dict, List, Optional

from config import Config
from utils.logger import get_logger


class VoiceAssistant:
    """
    Text-to-Speech voice assistant with cooldown management.

    Features:
    - Threaded speech for non-blocking operation
    - Per-object cooldown tracking
    - Automatic driver fallback (SAPI5 on Windows)
    - Concurrent speech prevention

    Important Notes:
    - pyttsx3 must run in main thread context or its own dedicated thread
    - Uses daemon threads so app exits cleanly
    - Never call engine.runAndWait() from multiple threads simultaneously
    """

    def __init__(self, config: Config):
        """
        Initialize voice assistant.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._last_spoken: Dict[str, float] = {}  # label -> timestamp
        self._speech_lock = threading.Lock()
        self._speech_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._speech_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    def initialize(self) -> bool:
        """
        Initialize TTS engine with optimal settings.

        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            if self._speech_thread is None or not self._speech_thread.is_alive():
                self._shutdown_event.clear()
                self._speech_thread = threading.Thread(
                    target=self._speech_worker,
                    daemon=True,
                )
                self._speech_thread.start()

            self.logger.info("Windows speech worker started")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            return False

    def _speech_worker(self) -> None:
        """Serialize speech requests through Windows' built-in speech engine."""
        while not self._shutdown_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:
                break

            try:
                self._speak_windows(text)
            except Exception as e:
                self.logger.error(f"Speech error: {e}")

    def _speak_windows(self, text: str) -> None:
        """Speak text through PowerShell's System.Speech API."""
        safe_text = text.replace("'", "''")
        volume = max(0, min(100, int(self.config.TTS_VOLUME * 100)))
        rate = 0
        if self.config.TTS_RATE < 130:
            rate = -2
        elif self.config.TTS_RATE > 170:
            rate = 2

        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Volume = {volume}; "
            f"$s.Rate = {rate}; "
            f"$s.Speak('{safe_text}')"
        )

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False,
            creationflags=creationflags,
        )

    def _should_announce(self, label: str) -> bool:
        """
        Check if object should be announced (cooldown check).

        Args:
            label: Object label to check

        Returns:
            True if cooldown has expired or label never spoken
        """
        current_time = time.time()
        last_time = self._last_spoken.get(label, 0)
        return (current_time - last_time) >= self.config.SPEECH_COOLDOWN

    def _speak_threaded(self, text: str):
        """Queue speech so it is played in order without overlapping."""
        self._speech_queue.put(text)

    def announce_objects(self, labels: List[str], apply_cooldown: bool = True):
        """
        Announce detected objects with cooldown filtering.

        Args:
            labels: List of object labels to announce
            apply_cooldown: When False, announce immediately for new appearances
        """
        current_time = time.time()
        if apply_cooldown:
            labels_to_announce = [
                label for label in labels
                if self._should_announce(label)
            ]
        else:
            labels_to_announce = list(labels)

        # Limit announcement count
        labels_to_announce = labels_to_announce[:self.config.MAX_ANNOUNCE_OBJECTS]

        if not labels_to_announce:
            return

        # Build announcement text with only the object names.
        if len(labels_to_announce) == 1:
            announcement = labels_to_announce[0]
        else:
            announcement = ', '.join(labels_to_announce)

        # Update last spoken timestamps
        if apply_cooldown:
            for label in labels_to_announce:
                self._last_spoken[label] = current_time

        # Queue speech so new detections are not dropped when frames arrive quickly.
        self._speak_threaded(announcement)

        self.logger.info(f"Announced: {labels_to_announce}")

    def shutdown(self):
        """Stop TTS engine and cleanup."""
        self._shutdown_event.set()
        self._speech_queue.put(None)

        if self._speech_thread is not None and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=1.0)

        self.logger.info("TTS worker stopped")
