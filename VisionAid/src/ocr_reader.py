"""
ocr_reader.py — On-device text reading for VisionAid ("Read Mode").

Uses EasyOCR (fully local, no cloud call) to detect and read printed text
from the webcam frame — signs, menus, mail, labels — so it can be handed
to the offline TTS engine.
"""

from __future__ import annotations

import numpy as np
from importlib import import_module


class TextReader:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool = False):
        # Imported lazily: EasyOCR pulls in torch, which is heavy and only
        # needed when Read Mode is actually used.
        try:
            easyocr = import_module("easyocr")
        except ImportError as exc:
            raise ImportError(
                "EasyOCR is required for Read Mode. Install it with: pip install easyocr"
            ) from exc

        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)

    def read_frame(self, frame: np.ndarray, min_confidence: float = 0.4) -> str:
        """Run OCR on a frame and return the detected text, reading-order joined."""
        results = self.reader.readtext(frame)
        lines = [text for (_box, text, conf) in results if conf >= min_confidence]
        return " ".join(lines).strip()
