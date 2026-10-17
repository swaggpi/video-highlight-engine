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
# run from the repo: [tool.uv] in pyproject.toml drops opencv-python, which rapidocr-onnxruntime (OCR) requires
# although opencv-python-headless already provides cv2 (plain pip installs both; they write the same cv2 package)
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
vhe clips video.mp4 "ruhige Landschaft ohne Sprache" --no-llm   # rules only (default: local LLM, see below)
vhe info footage/                   # project files and status (or a video's index timings)
vhe ui                              # http://127.0.0.1:8765: folder or file, Index, brief, Find clips, Search, Export
```

Briefs: a total runtime ("60-90 s recap") or per-clip lengths ("3-6 s each"), a count, a mood (energetic/calm),
speech ("talks to camera", "every moment ... speaks"), "every/all ..." for exhaustive lists, and "open with ..." /
"end with ..." for the first and last shot. A multi-sentence brief becomes a plan: a list of wanted shots ("build
through the mounted police, the band, dancers ..."; relevance is the best item, and selection takes every item once
before any twice; "every float: A, B, C" gives the best clip per named item and skips items with no footage), things
to leave out ("leave out shots where ...", "no whip-pans"; a bounded relevance penalty), spoken topics ("talks
about ...", "explains ...", "sums up ..."; transcript sentences matched with Ollama bge-m3 embeddings, or keyword
overlap without it), live sound ("with natural sound (not muted)": muted stretches are dropped) and a soundbite next
to the visual clips ("the best 5-8 s soundbite where ..."). The brief is parsed by a local Ollama model (`qwen3:4b`,
or `--llm MODEL` / `$VHE_LLM`; answers cached in `~/.cache/vhe/parse/`) when the server is reachable, else by the
rules; numbers the model returns that are not in the brief are ignored. `--no-llm` forces the rules.

Output: `--json` writes `{"query", "parser", "parsed", "clips": [{"file", "start", "end", "duration", "score",
"reason", "source", "role", "item"}]}` in edit order with file-local seconds; `item` is the shot item (or spoken
topic) a clip serves, `role` is opener/closer/soundbite. `--export DIR` writes one preview file per clip, `clips.csv`
(order, file, start, end, duration) and `edit.edl` (CMX3600, references the original files for relinking in
Resolve/Premiere); `--rough-cut` adds `rough_cut.mp4`, all clips joined at 1280x720.

Ranking data: `--dump-candidates OUT.json` writes every candidate the run scored (`vhe-candidates/1`: the run, the
parsed brief and its query vector, `feature_names`, and per candidate file, file-local start/end as in the clips,
source, stage, score, features, selected, rank, role, item); `--dump-all` adds every gated candidate before
refinement. `vhe features footage/ --brief TEXT --spans spans.json --out OUT.json` featurises spans picked elsewhere
(`[{"file", "start", "end"}]` or a `--json` edit list) the same way. `--persona ID` records who the edit is for;
`--scorer model.json` ranks with a learned model (`vhe-linear-rank/1`) instead of the hand-weighted total.

## How it works

```
index (once per video)
  decode ........ one ffmpeg pass: 3 fps frames, tiny full-fps grey stream, 16 kHz audio (proxy files, HDR tone-map)
  cuts .......... adaptive frame-difference detector on the full-fps stream, flash suppression
  motion ........ phase correlation per frame pair -> camera speed, shake, subject motion
  audio ......... loudness, onset rate, Silero VAD, faster-whisper transcript with word times
  frames ........ SigLIP2 embeddings (search index), zero-shot tags, sharpness/exposure, shot scale
                  (scale_close / scale_wide margins)
  text .......... RapidOCR on 1 fps of the analysis frames: text_area, text_conf per step, ocr.json text store
  faces ......... YuNet on every analysis frame: face_n, face_area_max, face_center; approach (subject growing
                  toward the lens). OCR and faces are skipped with a warning when unavailable
  change points . content changes inside long shots (raw footage)
  nc highlight .. Lighthouse features per 148 s window with overlap (NON-COMMERCIAL)

query (per request; a folder is one corpus: all files on a virtual timeline, a single video is a corpus of one)
  parse ......... LLM (cached, grounded) or rules: mood -> signal weights, runtime budget vs clip length, count,
                  opener/closer, content text, plan (shots, avoid, topics, live sound, soundbite length)
  candidates .... per file: utterances, sub-shots, multi-length windows, peak windows, model proposals
  gate .......... drop candidates with > 20 % unusable steps (covered, blown out, blurry, very shaky), any mood;
                  with live sound asked for, > 10 % muted steps
  signals ....... relevance = SigLIP sigmoid probability (absolute across files), the best shot item minus the
                  avoid penalty; highlight saliency (NON-COMMERCIAL) as its own signal; other signals z-scored over
                  the pooled corpus (loudness centred per file first); quality from blur/blown/shake, not brightness
  score ......... per candidate a versioned feature vector (rank/features.py) and a Scorer (rank/scorer.py): the
                  hand-weighted total (relevance averaged with the saliency, length-bucket centring, length and
                  cut priors, speech bonus) or a learned model file; refined spans are featurised and rescored
  refine ........ per file: snap edges to cuts, word boundaries, speech pauses, steady frames; enforce lengths
  select ........ opener/closer first, then shot items round robin, MMR with no overlap, runtime budget, per-file
                  cap for variety; spoken topics: transcript sentences above a similarity threshold, time order
  order ......... opener, then files in recording order and time within each file, then closer
  export ........ re-encoded preview cuts (downscaled before HDR tone-map), clips.csv, CMX3600 EDL, rough cut
```

## Layout

```
vhe/
  config.py, timing.py, models.py, pipeline.py, project.py, corpus.py, engine.py, cli.py
  ingest/decode.py
  signals/{shots,motion,audio,speech,frames,tags,quality,changes,ocr,faces,highlight_nc}.py
  index/store.py
  query/{parser,plan,candidates,refine}.py
  rank/{signals,features,scorer,select,dump}.py
  export/clips.py
  ui/{server.py,index.html}
tests/
```

## Not in v1

- Captioner + LLM rerank of the top 30 (Qwen3.5-4B / 9B) is not wired yet; ranking is deterministic.
- Parakeet ASR, TransNetV2/OmniShotCut cuts, PE-AV audio events, LanceDB index: planned swaps.
- The non-commercial highlight model must be replaced by a head retrained on owned data before shipping.
