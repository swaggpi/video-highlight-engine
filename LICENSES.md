# Licence register

| Component | Used for | Licence | Commercial | Notes |
|---|---|---|---|---|
| SigLIP2 (`google/siglip2-base-patch16-256`) | frame/text embeddings, tags | Apache-2.0 | yes | ship LICENSE + NOTICE |
| faster-whisper + Whisper weights | transcript | MIT | yes | |
| Silero VAD | voice activity | MIT | yes | |
| RapidOCR (`rapidocr-onnxruntime`) + PP-OCRv4 det/rec ONNX models (PaddleOCR) | on-screen text | Apache-2.0 (code and models) | yes | models ship inside the wheel |
| ONNX Runtime, pyclipper, Shapely | OCR runtime | MIT / MIT / BSD-3-Clause | yes | |
| YuNet `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | faces | MIT | yes | fetched once to `~/.cache/vhe/models`, SHA-256 checked |
| librosa, soundfile, OpenCV, numpy, torch, transformers | signals, runtime | ISC / BSD / Apache-2.0 | yes | |
| FFmpeg (system binary, not bundled) | decode, export | LGPL/GPL depending on build | check build | ship an LGPL build if bundling |
| Lighthouse code | highlight model wrapper | Apache-2.0 | yes | |
| **CG-DETR checkpoint `clip_cg_detr_qvhighlight.ckpt`** | query-conditioned highlight scores | trained on QVHighlights, **CC BY-NC-SA 4.0** | **no** | prototype only; disable with `--no-nc` |
| OpenAI CLIP ViT-B/32 (via Lighthouse) | features for the checkpoint above | MIT | yes | only used by the NC component |
