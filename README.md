# vhe: local video highlight and search engine

Pipeline A, v1. Index a video or a folder of raw event footage once, then search it by text and get the best
clips across all files, in edit order, for an editor's brief such as "60-90 s energetic recap, open with an
establishing shot" or "every moment the host talks to camera". Runs locally (CPU, CUDA or Apple Silicon).

> **Non-commercial component:** query-conditioned highlight scores come from Lighthouse CG-DETR, whose
> checkpoint is trained on QVHighlights (CC BY-NC-SA). It is for prototyping only and is switched off with
> `--no-nc`. Everything else uses commercially usable licences; see `LICENSES.md`.

## Setup

Requires Python 3.11 (Lighthouse pins old numpy) and ffmpeg.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA/MPS build
uv pip install --python .venv/bin/python -e ".[nc,dev]"
uv pip install --python .venv/bin/python --no-deps "git+https://github.com/line/lighthouse.git" msclap
mkdir -p ~/.cache/vhe/models
curl -L -o ~/.cache/vhe/models/clip_cg_detr_qvhighlight.ckpt \
  https://zenodo.org/records/13960580/files/clip_cg_detr_qvhighlight.ckpt
```

NVIDIA GPU (e.g. GTX 1660 Super, Turing): use the CUDA build of torch instead of the CPU one; `--device auto`
then picks CUDA and runs SigLIP2 and Whisper in fp16.

```bash
uv pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

On Windows the venv paths are `.venv\Scripts\python.exe` and `.venv\Scripts\vhe.exe`; ffmpeg must be on PATH.

## Use

```bash
vhe index video.mp4                 # writes video.mp4.vhe/ next to the video, prints per-stage timings
vhe index footage/                  # a folder of raw footage: every video (recursive) into footage/.vhe/,
                                    # incremental (only new/changed files), one speech language (--lang or majority)
vhe search footage/ "food stalls"  # PATH: project folder, video file or video.mp4.vhe; prints file + local times
vhe clips footage/ "60-90 s energetic recap, open with an establishing shot" --json edit.json --export out/
vhe clips footage/ "5 calm B-roll shots 3-6 s each" --count 5 --budget 20-30   # flags override the brief
vhe clips video.mp4 "ruhige Landschaft ohne Sprache" --llm qwen3.5:9b   # optional LLM query parsing via Ollama
vhe info footage/                   # project files and status (or a video's index timings)
vhe ui                              # http://127.0.0.1:8765: folder or file, Index, brief, Find clips, Search, Export
```

Briefs: a total runtime ("60-90 s recap") or per-clip lengths ("3-6 s each"), a count, a mood (energetic/calm),
speech ("talks to camera", "every moment ... speaks"), "every/all ..." for exhaustive lists, and "open with ..." /
"end with ..." for the first and last shot. Content words are matched visually (and in the transcript for search).

Output: `--json` writes `{"query", "parsed", "clips": [{"file", "start", "end", "duration", "score", "reason",
"source"}]}` in edit order with file-local seconds. `--export DIR` writes one preview file per clip, `clips.csv`
(order, file, start, end, duration) and `edit.edl` (CMX3600, references the original files for relinking in
Resolve/Premiere); `--rough-cut` adds `rough_cut.mp4`, all clips joined at 1280x720.

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

query (per request; a folder is one corpus: all files on a virtual timeline, a single video is a corpus of one)
  parse ......... mood -> signal weights, runtime budget vs clip length, count, opener/closer, content text
  candidates .... per file: utterances, sub-shots, multi-length windows, peak windows, model proposals
  gate .......... drop candidates with > 20 % unusable steps (covered, blown out, blurry, very shaky), any mood
  score ......... relevance = SigLIP sigmoid probability (absolute across files); other signals z-scored over the
                  pooled corpus (loudness centred per file first); quality from blur/blown/shake, not brightness
  refine ........ per file: snap edges to cuts, word boundaries, speech pauses, steady frames; enforce lengths
  select ........ opener/closer first, then MMR with no overlap, runtime budget, per-file cap for variety
  order ......... opener, then files in recording order and time within each file, then closer
  export ........ re-encoded preview cuts (downscaled before HDR tone-map), clips.csv, CMX3600 EDL, rough cut
```

## Layout

```
vhe/
  config.py, timing.py, models.py, pipeline.py, project.py, corpus.py, engine.py, cli.py
  ingest/decode.py
  signals/{shots,motion,audio,speech,frames,tags,quality,changes,highlight_nc}.py
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
