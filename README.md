# vhe: local video highlight and search engine

Pipeline A, v1. Index a video once, then search it by text and get ranked clips for a query such as
"chill scenic view" or "short high-energy scenes". Runs locally (CPU, CUDA or Apple Silicon).

> **Non-commercial component:** query-conditioned highlight scores come from Lighthouse CG-DETR, whose
> checkpoint is trained on QVHighlights (CC BY-NC-SA). It is for prototyping only and is switched off with
> `--no-nc`. Everything else uses commercially usable licences; see `LICENSES.md`.

## Setup

Requires Python 3.11 (Lighthouse pins old numpy) and ffmpeg.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA/MPS build
uv pip install --python .venv/bin/python -e ".[nc,dev]"
uv pip install --python .venv/bin/python --no-deps "git+https://github.com/line/lighthouse.git"
mkdir -p ~/.cache/vhe/models
curl -L -o ~/.cache/vhe/models/clip_cg_detr_qvhighlight.ckpt \
  https://zenodo.org/records/13960580/files/clip_cg_detr_qvhighlight.ckpt
```

## Use

```bash
vhe index video.mp4                 # writes video.mp4.vhe/ next to the video, prints per-stage timings
vhe search video.mp4.vhe "dog on the beach"
vhe clips video.mp4.vhe "3 short high-energy scenes" --export out/
vhe clips video.mp4.vhe "ruhige Landschaft ohne Sprache" --llm qwen3.5:9b   # optional LLM query parsing via Ollama
vhe info video.mp4.vhe
vhe ui                              # http://127.0.0.1:8765: path, Index, query, Find clips, Search, Export
```

## How it works

```
index (once per video)
  decode ........ one ffmpeg pass: 3 fps frames, tiny full-fps grey stream, 16 kHz audio (proxy files, HDR tone-map)
  cuts .......... adaptive frame-difference detector on the full-fps stream, flash suppression
  motion ........ phase correlation per frame pair -> camera speed, shake, subject motion
  audio ......... loudness, onset rate, Silero VAD, faster-whisper transcript with word times
  frames ........ SigLIP2 embeddings (search index), zero-shot tags, sharpness/exposure
  change points . content changes inside long shots (raw footage)
  nc highlight .. Lighthouse features per 148 s window with overlap (NON-COMMERCIAL)

query (per request)
  parse ......... mood -> signal weights, length/count limits, content text (rules; optional Ollama LLM)
  candidates .... sub-shots, multi-length windows, peak-centred windows, model proposals
  score ......... per-video robust z-scores; relevance and quality kept separate, weighted per query
  select ........ MMR: no overlap, near-duplicate penalty from embedding similarity
  refine ........ snap edges to cuts, word boundaries, speech pauses, steady frames; enforce lengths
  export ........ re-encoded cuts (stream copy can only cut on keyframes)
```

## Layout

```
vhe/
  config.py, timing.py, pipeline.py, engine.py, cli.py
  ingest/decode.py
  signals/{shots,motion,audio,speech,frames,tags,quality,highlight_nc}.py
  index/store.py
  query/{parser,candidates,score,select,refine}.py
  export/clips.py
  ui/{server.py,index.html}
tests/
```

## Not in v1

- Captioner + LLM rerank of the top 30 (Qwen3.5-4B / 9B) is not wired yet; ranking is deterministic.
- Parakeet ASR, TransNetV2/OmniShotCut cuts, PE-AV audio events, LanceDB index: planned swaps.
- The non-commercial highlight model must be replaced by a head retrained on owned data before shipping.
