"""
tts_engine.py — Offline text-to-speech for VisionAid.

Wraps pyttsx3, which drives the OS's native speech engine (SAPI5 on
Windows) entirely offline — no network call, no API key, minimal latency.
"""

from __future__ import annotations

import queue
import threading

import pyttsx3


class SpeechOutput:
    def __init__(self, rate: int = 185, volume: float = 1.0):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        while True:
            text = self._queue.get()
            if text is None:  # sentinel for shutdown
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, text: str):
        """Non-blocking: queues text to be spoken without freezing the video loop."""
        if text:
            self._queue.put(text)

    def clear_queue(self):
        with self._queue.mutex:
            self._queue.queue.clear()

    def shutdown(self):
        self._queue.put(None)
        self._thread.join(timeout=2)
