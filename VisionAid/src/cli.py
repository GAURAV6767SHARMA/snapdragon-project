"""
cli.py — Headless VisionAid runner, ideal for a quick stage demo.

Example:
    python src/cli.py --mode scene
    python src/cli.py --mode read
"""

from __future__ import annotations

import argparse
import time

import cv2

from app import summarize_detections
from detector import NPUObjectDetector
from ocr_reader import TextReader
from tts_engine import SpeechOutput

NARRATION_COOLDOWN_SECONDS = 3.0


def run_scene_mode(model_path: str, camera_index: int):
    detector = NPUObjectDetector(model_path=model_path)
    speech = SpeechOutput()
    cap = cv2.VideoCapture(camera_index)
    last_time = 0.0

    print(f"[VisionAid] Scene Mode running on {detector.active_provider}. Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            detections, inference_ms = detector.detect(frame)
            now = time.time()
            if now - last_time > NARRATION_COOLDOWN_SECONDS:
                summary = summarize_detections(detections)
                if summary:
                    print(f"[{inference_ms:.1f} ms] {summary}")
                    speech.say(summary)
                last_time = now
    except KeyboardInterrupt:
        pass
    finally:
        speech.shutdown()
        cap.release()


def run_read_mode(camera_index: int):
    reader = TextReader()
    speech = SpeechOutput()
    cap = cv2.VideoCapture(camera_index)

    print("[VisionAid] Read Mode. Press Enter to capture + read, Ctrl+C to stop.")
    try:
        while True:
            input()
            ret, frame = cap.read()
            if not ret:
                continue
            text = reader.read_frame(frame)
            print(f"Read: {text!r}")
            speech.say(text if text else "No text detected.")
    except KeyboardInterrupt:
        pass
    finally:
        speech.shutdown()
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAid headless demo runner")
    parser.add_argument("--mode", choices=["scene", "read"], default="scene")
    parser.add_argument("--model", default="models/yolov8n_int8.onnx")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "scene":
        run_scene_mode(args.model, args.camera)
    else:
        run_read_mode(args.camera)
