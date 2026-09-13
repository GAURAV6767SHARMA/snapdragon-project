"""
app.py — VisionAid main application (GUI).

Ties together the NPU object detector, on-device OCR, offline TTS, and
optional offline voice commands into one live loop with a simple Tkinter
window showing the camera feed and current mode.

Run:
    python src/app.py
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import Counter

import cv2
from PIL import Image, ImageTk

from detector import NPUObjectDetector
from ocr_reader import TextReader
from tts_engine import SpeechOutput

NARRATION_COOLDOWN_SECONDS = 3.0


def summarize_detections(detections) -> str:
    if not detections:
        return ""
    counts: Counter[str] = Counter()
    positions: dict[str, list[str]] = {}
    for d in detections:
        counts[d.label] += 1
        positions.setdefault(d.label, []).append(d.position)

    phrases = []
    for label, count in counts.most_common(4):
        pos_summary = Counter(positions[label]).most_common(1)[0][0]
        noun = label if count == 1 else f"{count} {label}s"
        phrases.append(f"{noun} {pos_summary}")
    return ", ".join(phrases)


class VisionAidApp:
    def __init__(self, model_path: str, camera_index: int = 0):
        self.detector = NPUObjectDetector(model_path=model_path)
        self.speech = SpeechOutput()
        self.ocr: TextReader | None = None  # lazy-loaded on first Read Mode use
        self.mode = "scene"  # "scene" | "read" | "off"
        self.last_narration_time = 0.0

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera_index / permissions.")

        self.root = tk.Tk()
        self.root.title(f"VisionAid — running on {self.detector.active_provider}")
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.status_var = tk.StringVar(value="Mode: Scene | Press S=Scene R=Read Space=Read-once Q=Quit")
        tk.Label(self.root, textvariable=self.status_var, font=("Calibri", 12)).pack(pady=6)

        self.root.bind("<Key>", self._on_key)
        self._update_frame()

    def _on_key(self, event):
        key = event.char.lower()
        if key == "s":
            self.mode = "scene"
            self.status_var.set("Mode: Scene")
        elif key == "r":
            self.mode = "read"
            self.status_var.set("Mode: Read (continuous)")
        elif key == " ":
            self._read_once()
        elif key == "q":
            self.shutdown()

    def _read_once(self):
        if self.ocr is None:
            self.ocr = TextReader()
        ret, frame = self.cap.read()
        if not ret:
            return
        text = self.ocr.read_frame(frame)
        self.speech.say(text if text else "No text detected.")

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.mode == "scene":
                detections, inference_ms = self.detector.detect(frame)
                for d in detections:
                    x1, y1, x2, y2 = d.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, d.label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

                now = time.time()
                if now - self.last_narration_time > NARRATION_COOLDOWN_SECONDS:
                    summary = summarize_detections(detections)
                    if summary:
                        self.speech.say(summary)
                    self.last_narration_time = now

                cv2.putText(frame, f"NPU inference: {inference_ms:.1f} ms", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.root.after(30, self._update_frame)

    def shutdown(self):
        self.speech.shutdown()
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAid — offline AI vision assistant")
    parser.add_argument("--model", default="models/yolov8n_int8.onnx")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    app = VisionAidApp(model_path=args.model, camera_index=args.camera)
    app.run()
