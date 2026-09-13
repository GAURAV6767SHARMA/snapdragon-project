"""
detector.py — NPU-accelerated object detector for VisionAid.

Runs a YOLOv8n ONNX model through ONNX Runtime, preferring the QNN
(Qualcomm Neural Network) Execution Provider so inference is offloaded to
the Hexagon NPU on Snapdragon-powered HP PCs. Falls back cleanly to CPU on
any other machine, so the same code works for development and for the
on-stage demo.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    position: str  # "left" / "center" / "right"


class NPUObjectDetector:
    """Wraps an ONNX YOLOv8n model, preferring the Snapdragon NPU (QNN)."""

    def __init__(
        self,
        model_path: str = "models/yolov8n_int8.onnx",
        input_size: int = 640,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session, self.active_provider = self._load_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _load_session(self, model_path: str) -> tuple[ort.InferenceSession, str]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. See docs/model_export_guide.md "
                "to export & quantize the model, then place it under models/."
            )

        # Try providers in priority order: Snapdragon NPU -> GPU (DirectML) -> CPU.
        # QNNExecutionProvider is what routes inference to the Hexagon NPU.
        candidate_providers = [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            ("DmlExecutionProvider", {}),
            ("CPUExecutionProvider", {}),
        ]
        available = ort.get_available_providers()

        for name, options in candidate_providers:
            if name in available:
                try:
                    session = ort.InferenceSession(
                        model_path,
                        providers=[(name, options)] if options else [name],
                    )
                    print(f"[VisionAid] Inference running on: {name}")
                    return session, name
                except Exception as exc:  # noqa: BLE001 - fall through to next provider
                    print(f"[VisionAid] Could not init {name} ({exc}); trying next provider.")
                    continue

        raise RuntimeError("No usable ONNX Runtime execution provider found.")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]  # NCHW
        return np.ascontiguousarray(blob), scale, scale

    def _postprocess(self, outputs: np.ndarray, scale: float, frame_w: int) -> list[Detection]:
        # outputs shape: (1, 84, 8400) for YOLOv8 -> transpose to (8400, 84)
        preds = outputs[0].transpose(1, 0)
        boxes, scores, class_ids = [], [], []

        for row in preds:
            cls_scores = row[4:]
            class_id = int(np.argmax(cls_scores))
            confidence = float(cls_scores[class_id])
            if confidence < self.conf_threshold:
                continue
            cx, cy, w, h = row[0:4]
            x1 = (cx - w / 2) / scale
            y1 = (cy - h / 2) / scale
            x2 = (cx + w / 2) / scale
            y2 = (cy + h / 2) / scale
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        detections = []
        for i in np.array(indices).flatten() if len(indices) else []:
            x, y, w, h = boxes[i]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            center_x = (x1 + x2) / 2
            if center_x < frame_w / 3:
                position = "left"
            elif center_x > 2 * frame_w / 3:
                position = "right"
            else:
                position = "center"

            label = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else "object"
            detections.append(Detection(label, scores[i], (x1, y1, x2, y2), position))

        return detections

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """Run one detection pass. Returns (detections, inference_ms)."""
        blob, scale, _ = self._preprocess(frame)
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})[0]
        inference_ms = (time.perf_counter() - start) * 1000
        detections = self._postprocess(outputs, scale, frame.shape[1])
        return detections, inference_ms
