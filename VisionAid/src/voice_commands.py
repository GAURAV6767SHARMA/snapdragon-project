"""
voice_commands.py — Offline hands-free voice control for VisionAid.

Uses Vosk for fully local speech recognition (no cloud STT), so a user can
say "read this", "what's around me", or "stop" without touching the
keyboard — important for a tool designed for users with vision impairment.

Download a small Vosk model once (e.g. vosk-model-small-en-us-0.15, ~40MB)
and point VOSK_MODEL_PATH at the extracted folder.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable

import sounddevice as sd
from vosk import KaldiRecognizer, Model

COMMANDS = {
    "read": ["read this", "read", "what does this say"],
    "scene": ["what's around me", "describe scene", "scan"],
    "stop": ["stop", "quiet", "pause"],
}


def _match_command(text: str) -> str | None:
    text = text.lower().strip()
    for action, phrases in COMMANDS.items():
        if any(phrase in text for phrase in phrases):
            return action
    return None


class VoiceCommandListener:
    def __init__(self, model_path: str, on_command: Callable[[str], None], sample_rate: int = 16000):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self.on_command = on_command
        self._audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):  # noqa: ANN001
        self._audio_queue.put(bytes(indata))

    def _listen_loop(self):
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while self._running:
                data = self._audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    action = _match_command(text)
                    if action:
                        self.on_command(action)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
