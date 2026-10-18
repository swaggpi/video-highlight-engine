# vhe: local video highlight and search engine

Pipeline A, v1. Index a video or a folder of raw event footage once, then search it by text and get the best
clips across all files, in edit order, for an editor's brief such as "60-90 s energetic recap, open with an
establishing shot" or "every moment the host talks to camera". Runs locally (CPU, CUDA or Apple Silicon).

> **Non-commercial component:** query-conditioned highlight scores come from Lighthouse CG-DETR, whose
> checkpoint is trained on QVHighlights (CC BY-NC-SA). It is for prototyping only and is switched off with
> `--no-nc`. Everything else uses commercially usable licences; see `LICENSES.md`.

Module map, data flow and where a learned ranker plugs in: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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

torchvision, torchaudio and msclap are only needed by the non-commercial component (OpenAI CLIP and Lighthouse
import them); vhe itself does not. Optional: a local [Ollama](https://ollama.com) server with `qwen3:4b` (brief
parsing) and `bge-m3` (spoken-topic matching); without it the rules parse the brief and topics match by keywords.

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

The UI plays each result from a preview of exactly its span, transcoded on first view to WebM (VP9 + Opus, 640 px
wide; about 0.1 s per second of clip) and cached in the project's `.vhe/previews/`, so it plays in browsers without
an H.264 decoder (some Firefox builds); this needs an ffmpeg with libvpx and libopus. `vhe clips` prints the time
per stage (parse, signals, candidates, slots, shortlist, select, ...); spoken-topic embeddings are cached in
`~/.cache/vhe/embed.sqlite`, parses in `~/.cache/vhe/parse/`.

What a brief can say (English or German):

- **Length and count**: a total runtime ("60-90 s recap") or per-clip lengths ("3-6 s each"), a count ("5 clips").
- **Mood and speech**: energetic / calm; "talks to camera", "every moment ... speaks"; "no talking".
- **Exhaustive lists**: "every / all ...". Visual ones keep only strong, distinct matches of each item (at most 20
  without a count); "every float: A, B, C" gives the best clip per named item and skips items with no footage.
- **First and last shot**: "open with ..." / "end with ...".
- **A shot list**: "build through the mounted police, the band, dancers ...": relevance is the best item, and
  selection takes every item once before any twice.
- **Things to leave out**: "avoid ..." is a bounded relevance penalty; "no / never / leave out ..." drops the clip.
  Camera defects ("no whip-pans", "camera at the ground", "something blocks the lens") use the index's defect
  signals; an excluded item about text or politics ("nothing with 'executed' on it", "no protest signs", "nothing
  political") also drops every clip whose on-screen text (OCR) holds one of its quoted words or a political term.
- **Spoken topics**: "talks about ...", "explains ...", "sums up ...": transcript sentences matched with bge-m3
  embeddings (keyword overlap without Ollama); "the best 5-8 s soundbite where ..." adds one next to the visual clips.
- **Sound**: "with natural sound (not muted)" drops muted stretches; a montage whose sound comes later ("music bed
  to be added", "it goes under music") gets speech-free clips only.
- **Visual cues**: framing close ("close-ups", "faces", "fills the frame", "right at the camera", "vertical",
  "9:16", "POV") or wide ("establishing", "wide view", "the street as the main character") weights the index's shot
  scale, largest face and approach (wide also penalises close-ups); "centred" adds face centring; "hook" / "the first
  clip has to grab in the first second" makes the first clip the best hook candidate; "names / text readable" adds an
  OCR legibility prior and prefers spans where an item's name is read. Cues inside the opener/closer description or
  an avoid clause do not count for the whole edit.

Every clip starts and ends on clean frames (visual) or holds whole sentences with the first and last word whole
(speech: lead-in 0.25 s, tail 0.4 s, at most to the middle of the pause); no moment is picked twice. The brief is
parsed by a local Ollama model (`qwen3:4b`, or `--llm MODEL` / `$VHE_LLM`; answers cached in `~/.cache/vhe/parse/`)
when the server is reachable, else by the rules; numbers the model returns that are not in the brief are ignored,
and its spoken topics and "every ..." flag count only where the brief has the words for them.
`--no-llm` forces the rules.

Output: `--json` writes `{"query", "parser", "parsed", "clips": [{"file", "start", "end", "duration", "score",
"reason", "source", "role", "item"}], "warnings"}`, clips in edit order with file-local seconds; `item` is the shot
item (or spoken topic) a clip serves, `role` is opener/closer/soundbite, and `warnings` says what the edit could not
deliver (a runtime below the budget, fewer clips than asked for, a model moved to the CPU after a GPU out-of-memory
error). `--export DIR` writes one preview file per clip, `clips.csv` (order, file, start, end, duration) and
`edit.edl` (CMX3600, references the original files for relinking in Resolve/Premiere); `--rough-cut` adds
`rough_cut.mp4`, all clips joined at 1280x720.

Ranking data: `--dump-candidates OUT.json` writes every candidate the run scored (`vhe-candidates/1`: the run, the
parsed brief and its query vector, `feature_names`, and per candidate file, file-local start/end as in the clips,
source, stage, score, features, selected, rank, role, item); `--dump-all` adds every gated candidate before
refinement. `vhe features footage/ --brief TEXT --spans spans.json --out OUT.json` featurises spans picked elsewhere
(`[{"file", "start", "end"}]` or a `--json` edit list) the same way. `--persona ID` records who the edit is for;
`--scorer model.json` ranks with a learned model (`vhe-linear-rank/1`, trained with `tools/train`) instead of the
hand-weighted total. Features v2 (`feature_version`/`query_version` "v2"): 82 columns, v1's 71 then shot scale,
face size/centring/count, approach, OCR text area/confidence and `ocr_item_match`; the query vector adds the cue
one-hots (framing none/close/wide, want_faces, hook, legible_text, centered) before the personas. v1 model files
are refused (retrain on v2 dumps).

## How it works

```
index (once per video; vhe/pipeline.py, vhe/project.py for folders)
  decode ........ one ffmpeg pass: 3 fps frames, tiny full-fps grey stream, 16 kHz audio (proxy files, HDR tone-map)
  cuts .......... adaptive frame-difference detector on the full-fps stream, flash suppression
  motion ........ phase correlation per frame pair -> camera speed, shake, subject motion
  audio ......... loudness, onset rate, Silero VAD, faster-whisper transcript with word times
  frames ........ SigLIP2 embeddings (search index), zero-shot tags, defect margins (obstructed, bad framing),
                  shot scale (scale_close / scale_wide margins)
  quality ....... re-blur sharpness ratio, luma and contrast, blown highlights; the usable mask and its reason bits
                  (covered, uniform, blown out, blurry, shaky, obstructed; bad framing as a soft reason)
                  stored; the query applies only covered, uniform, blown out, blurry (config.ACTIVE_REASONS)
  text .......... RapidOCR on 1 fps of the analysis frames: text_area, text_conf per step, ocr.json text store
  faces ......... YuNet on every analysis frame: face_n, face_area_max, face_center; approach (subject growing
                  toward the lens). OCR and faces are skipped with a warning when unavailable
  change points . content changes inside long shots (raw footage)
  nc highlight .. Lighthouse features per 148 s window with overlap (NON-COMMERCIAL)

query (per request; a folder is one corpus: all files on a virtual timeline, a single video is a corpus of one)
  parse ......... LLM (cached, grounded) or rules -> ParsedQuery: signal weights, runtime budget vs clip length,
                  count, opener/closer, content text, plan (shots, avoid/exclude, topics, live sound, soundbite),
                  visual cues
  signals ....... per grid step: relevance = SigLIP sigmoid probability (absolute across files), the best shot item
                  minus the avoid penalty, averaged with the highlight saliency (NON-COMMERCIAL, also kept as its
                  own signal); quality from blur/blown/shake, not brightness; energy, steadiness, speech z-scored
                  over the pooled corpus (loudness centred per file first); the cue bonus; the exclusion masks
  candidates .... per file: utterances, sub-shots, multi-length windows, windows on signal peaks, model proposals
  gate .......... drop candidates with > 20 % unusable steps, any mood; with live sound asked for > 10 % muted;
                  excluded items, defects, forbidden words on screen, speech when the brief wants none
  score ......... a versioned feature vector per candidate and a Scorer: the hand-weighted total (length-bucket
                  centring, length and cut priors, speech bonus, visual-cue bonus) or a learned model file
  refine ........ the best candidates: speech to whole sentences, visual edges onto clean frames, cuts, word gaps;
                  gated again, featurised again and rescored
  select ........ opener (or the hook) / closer from their own pools first, then shot items round robin, MMR with
                  no overlap or near-duplicates, runtime budget (lengthening picks when the pool runs out), per-file
                  cap; spoken topics: transcript sentences above a similarity threshold
  order ......... opener, then files in recording order and time within each file, then closer, then a soundbite
  export ........ re-encoded preview cuts (downscaled before HDR tone-map), clips.csv, CMX3600 EDL, rough cut
```

Streamer VODs and other talk-heavy footage: briefs about reactions ("the 5 funniest moments", "every jump scare,
death and clutch play", "a cold open with the biggest reaction", "a 10 min highlights video") rank by an audio
reaction curve (loudness/onset bursts over the file's own running level) and the transcript's excited speech
(exclamations, laughs, strong words), so the named moments do not have to be visible to SigLIP; quoted lines
("the 'I made it into Harvard' bit") and the named moments are looked up in the transcript; "cut all dead air: the
menus, ..." is a list to avoid, not shots. Dark frames that still show something (a black game-over screen with a
facecam) are usable: `covered` needs a uniform frame.

## Layout

```
vhe/
  cli.py, ui/{server.py,index.html}    commands and the local web UI
  engine.py                            the query entry point: models, caches, clips() and search() orchestration
  results.py                           result clips (times, scores, labels, reason, transcript) and warnings
  models.py                            model cache, device choice, inference lock, GPU out-of-memory fallback,
                                       lending the GPU memory to the parser LLM
  config.py, timing.py                 thresholds and Config; stage timings
  pipeline.py, project.py              index one video; index a folder incrementally (manifest)
  ingest/decode.py                     ffmpeg decode and probe
  signals/                             index-time signals (shots, motion, audio, speech, frames, tags, quality,
                                       changes, ocr, faces, highlight_nc)
  index/store.py, corpus.py            the on-disk index; many indexes on one virtual timeline
  query/                               brief -> ParsedQuery (parser, rules, lexicon, llm, model) and what the
                                       brief matches (text_match, exclusions, topics); candidates, refine
  rank/                                brief_signals, gate, features, scorer, pool, select, items, signals, dump
  export/clips.py, export/preview.py   preview cuts, CSV, EDL, rough cut; the UI's WebM clip previews
tools/train/                           learned ranker: build_dataset, train_linear, eval_rank, DATASET_CARD.md
tests/                                 pytest; tests/data/heuristic_clips.json pins Engine.clips on synthetic cases
docs/ARCHITECTURE.md                   module map, data flow, the learnable seam, the training loop
```

## Not in v1

- Captioner + LLM rerank of the top 30 (Qwen3.5-4B / 9B) is not wired yet; ranking is deterministic.
- Parakeet ASR, TransNetV2/OmniShotCut cuts, PE-AV audio events, LanceDB index: planned swaps.
- The non-commercial highlight model must be replaced by a head retrained on owned data before shipping.
