# VisionAid — Offline AI Vision Assistant for Snapdragon-Powered HP PCs

**Snapdragon® AI Lab Build & Present Challenge — Qualcomm**

VisionAid is an always-on, fully offline AI assistant that helps blind and
low-vision users understand their surroundings using only the webcam and
speaker on a Snapdragon-powered HP PC. Everything — object detection, text
reading, and voice commands — runs **entirely on-device** on the Snapdragon
NPU, so it works with no internet connection, keeps all camera data private,
and sips battery instead of draining it.

## The problem

Over 2.2 billion people worldwide live with some form of vision impairment
(WHO). Most existing "AI vision assistant" apps require a live cloud
connection to GPT-4V-style APIs — meaning they stop working without Wi-Fi,
send personal video streams to a third-party server, and add 1-3 seconds of
network latency to a task where fast feedback (e.g. "step — curb ahead")
genuinely matters. A privacy-preserving, offline, low-latency assistant is
only realistically possible with dedicated on-device NPU acceleration —
exactly what Snapdragon-powered HP PCs are built for.

## What it does

| Mode | What it does | How |
|---|---|---|
| **Scene Mode** | Continuously narrates what's in front of the camera ("person, 2 meters, center; chair, left") | YOLOv8n ONNX model, INT8-quantized, run on the Snapdragon NPU via the QNN Execution Provider |
| **Read Mode** | Reads any printed text out loud — menus, signs, mail, labels | On-device OCR (EasyOCR / Tesseract) + offline TTS |
| **Voice Control** | Hands-free mode switching — "read this", "what's around me", "stop" | Offline speech recognition (Vosk), no cloud STT |

All three modes share one NPU-accelerated inference pipeline and run
simultaneously without a network connection.

## Why this fits the Challenge

- **Optimized for Snapdragon-powered HP PCs**: inference runs through
  `onnxruntime` with the **QNN Execution Provider**, targeting the Hexagon
  NPU on Snapdragon X Elite/Plus devices, with automatic CPU fallback for
  development on non-Snapdragon machines.
- **Uses Qualcomm AI Hub models**: the detection model is exported and
  quantized following the Qualcomm AI Hub YOLOv8 recipe (see
  `docs/model_export_guide.md`) so it is pre-optimized for Hexagon NPU
  execution rather than a generic ONNX export.
- **New AI use case**: combines detection + OCR + offline STT/TTS into a
  single always-available accessibility pipeline, not a single-model demo.
- **Deployment & Accessibility**: this *is* the use case — the product only
  makes sense as a low-latency, private, on-device experience, which is the
  core value proposition of Snapdragon NPUs over cloud inference.

## Project structure

```
VisionAid/
├── README.md
├── requirements.txt
├── src/
│   ├── detector.py         # ONNX Runtime + QNN EP object detector
│   ├── ocr_reader.py        # On-device text reading
│   ├── tts_engine.py        # Offline text-to-speech
│   ├── voice_commands.py    # Offline speech recognition (Vosk)
│   ├── app.py                # Tkinter GUI + orchestration loop
│   └── cli.py                 # Headless CLI runner (for a fast live demo)
├── models/
│   └── README.md            # Where to drop the exported .onnx files
└── docs/
    ├── model_export_guide.md   # Export & quantize YOLOv8n for QNN
    └── SUBMISSION.md            # Ready-to-paste hackathon submission write-up
```

## Quickstart

```bash
# 1. Install dependencies (Windows, on the Snapdragon HP PC)
pip install -r requirements.txt

# 2. Get the NPU-optimized model (see docs/model_export_guide.md)
#    Drop the resulting yolov8n_int8.onnx into models/

# 3. Run the GUI app
python src/app.py

# --- or, for a fast on-stage demo with no window management ---
python src/cli.py --mode scene
```

On first run, `detector.py` will try the QNN Execution Provider first (NPU),
and automatically fall back to `CPUExecutionProvider` if it isn't available
(e.g. while developing on a non-Snapdragon laptop) — so the exact same code
runs in dev and at demo time, just faster on the real hardware.

## Demo script (for the "Present" part of the challenge)

1. Open with the problem stat (2.2B people, cloud-dependent competitors).
2. Launch Scene Mode, walk in front of the camera — narration should update
   in under ~150ms on the Snapdragon NPU (call out the low latency).
3. Hold up a printed page → say "read this" (voice command) → it reads the
   text aloud, hands-free.
4. Turn off Wi-Fi on stage — everything keeps working. This is the single
   biggest applause line: cloud competitors go dark, VisionAid doesn't.
5. Close with the architecture slide showing the NPU pipeline and the
   Qualcomm AI Hub model export flow.

See `docs/SUBMISSION.md` for the full write-up mapped to each evaluation
criterion (Technical Implementation, Use Case & Innovation, Deployment &
Accessibility, Presentation & Documentation).
