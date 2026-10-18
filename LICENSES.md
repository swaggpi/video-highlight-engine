# Licence register

| Component | Used for | Licence | Commercial | Notes |
|---|---|---|---|---|
| SigLIP2 (`google/siglip2-base-patch16-256`) | frame/text embeddings, tags | Apache-2.0 | yes | ship LICENSE + NOTICE |
| faster-whisper + Whisper weights | transcript | MIT | yes | |
| CTranslate2 | faster-whisper's inference runtime | MIT | yes | |
| Silero VAD | voice activity | MIT | yes | |
| RapidOCR (`rapidocr-onnxruntime`) + PP-OCRv4 det/rec ONNX models (PaddleOCR) | on-screen text | Apache-2.0 (code and models) | yes | models ship inside the wheel |
| ONNX Runtime, pyclipper, Shapely | OCR runtime | MIT / MIT / BSD-3-Clause | yes | |
| YuNet `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | faces | MIT | yes | fetched once to `~/.cache/vhe/models`, SHA-256 checked |
| librosa, soundfile, OpenCV, numpy, torch, transformers | signals, runtime | ISC / BSD / Apache-2.0 | yes | |
| sentencepiece, protobuf | SigLIP2 tokenizer (transformers) | Apache-2.0 / BSD-3-Clause | yes | |
| Pillow | frame loading for SigLIP2 | MIT-CMU (HPND-style) | yes | |
| Ollama (local server, not bundled) | brief parsing and topic embeddings over HTTP | MIT | yes | optional: without it the rules parse and topics fall back to keywords |
| Qwen3-4B (`qwen3:4b` via Ollama) | default brief parser | Apache-2.0 | yes | pulled by the user (`ollama pull qwen3:4b`) |
| BGE-M3 (`bge-m3` via Ollama) | spoken-topic embeddings | MIT | yes | pulled by the user |
| FFmpeg (system binary, not bundled) | decode, export | LGPL/GPL depending on build | check build | ship an LGPL build if bundling |
| Lighthouse code | highlight model wrapper | Apache-2.0 | yes | |
| **CG-DETR checkpoint `clip_cg_detr_qvhighlight.ckpt`** | query-conditioned highlight scores | trained on QVHighlights, **CC BY-NC-SA 4.0** | **no** | prototype only; disable with `--no-nc` |
| OpenAI CLIP ViT-B/32 (via Lighthouse) | features for the checkpoint above | MIT | yes | only used by the NC component |
| msclap, torchvision, torchaudio | imported by Lighthouse / CLIP (the `nc` extra) | MIT / BSD-3-Clause / BSD-2-Clause | yes | vhe runs no CLAP model; only installed for the NC component |
