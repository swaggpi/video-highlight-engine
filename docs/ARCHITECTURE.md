# Architecture

## Layers

Imports only point down this list; nothing in `query/` imports `rank/`, nothing in `rank/` imports the parser
(`tests/test_layers.py` checks both):

```
signals/, index/, ingest/, models, config     per-video signals on the 3 fps grid, the on-disk index, model cache
project, pipeline, corpus                     indexing a folder; many indexes on one virtual timeline
query/                                        the brief: parse it, and what it matches (text, exclusions, topics)
rank/                                         per-step signals of a brief, gates, features, scorer, pools, selection
results, engine                               result clips; orchestration and the models' runtime state
cli, ui                                       entry points
```

## Module map

| module | job |
|---|---|
| `models.py` | process-wide model cache, `resolve_device`, `MODEL_LOCK`, `on_device` (CPU fallback after a GPU OOM) |
| `config.py` | index-time `Config` and calibrated thresholds; `CANDIDATE_LENGTHS` (candidate windows, scorer buckets) |
| `pipeline.py`, `project.py` | index one video; index a folder incrementally into `<DIR>/.vhe/` |
| `signals/*` | index-time signals: cuts, motion/shake, audio/VAD, transcript, SigLIP2 frames and tags, quality and the usable mask (`quality.reason_mask` decodes its reason bits), OCR, faces, CG-DETR |
| `index/store.py` | `Index` (meta.json, grid.npz, speech.json, ocr.json); refuses another index or text-store version |
| `corpus.py` | `Corpus`: indexes concatenated on one virtual timeline; virtual <-> file-local times (`locate`), transcript words in a span |
| `query/model.py` | `ParsedQuery`, clip/count limits, signal-weight presets, slot weights (`EDGE_WEIGHTS`, `HOOK_WEIGHTS`, `HOOK_LIST_WEIGHTS`) |
| `query/parser.py` | `parse_brief`: the LLM when reachable, else the rules |
| `query/lexicon.py` | every word list and regex the rules match |
| `query/rules.py` | deterministic parse (`parse`), and `finalise`, shared with the LLM path |
| `query/llm.py` | Ollama parse with a JSON schema, disk cache, grounding against the rules |
| `query/text_match.py` | the one `STOPWORDS`, word tokens, squashing; shot-item words read on screen; legibility prior |
| `query/exclusions.py` | hard-exclusion masks: SigLIP thresholds, camera defects, forbidden words on screen |
| `query/topics.py` | spoken topics vs transcript sentences (bge-m3 embeddings or keywords) |
| `query/candidates.py` | candidate spans per file (sub-shots, windows, peaks, proposals, utterances) |
| `query/refine.py` | clip edges: whole sentences and words, clean frames, cuts |
| `rank/signals.py` | per-step signals over the corpus grid (relevance scale, quality, energy, steadiness, muted), span statistics |
| `rank/brief_signals.py` | a brief -> its per-step signal dict (item relevance, avoid penalty, nc saliency, cue bonus, highlight prior, exclusion masks) |
| `rank/items.py` | shot-list relevance: combine with avoid, item evidence (SigLIP + names read on screen), which item a span serves |
| `rank/gate.py` | the hard gates (unusable, muted, excluded, defects, forbidden text, speech) |
| `rank/features.py` | `featurize` (95 raw columns, `FEATURE_VERSION` v3), `item_features` (per shot item), `pool_stats` (length-bucket statistics), `query_vector` (`QUERY_VERSION` v3) |
| `rank/scorer.py` | the `Scorer` protocol, `HeuristicScorer` (every hand-set weight and bonus), `LinearRankScorer`, `load_scorer` |
| `rank/pool.py` | stage 1/2 shortlist, the opener/closer/hook slot pools, spoken-topic picks |
| `rank/select.py` | selection: MMR, near-duplicates, item round robin, budgets, exhaustive lists, budget fill; `ScoreScale` (thresholds in pool spreads) |
| `rank/dump.py` | candidate dumps (`vhe-candidates/1`) for training, and features for external spans |
| `results.py` | the result clip dict (the scorer's total, signal summary, labels, reason from the scorer's contributions, transcript) and shortfall warnings |
| `engine.py` | `Engine`: SigLIP / CG-DETR runtime and caches; `clips()` and `search()` orchestration |

## Data flow

Index time (`pipeline.build_index`, per file): ffmpeg decode -> cuts, motion, audio, transcript -> SigLIP2
embeddings, tags, defect and shot-scale margins -> quality and the usable mask -> OCR, faces, approach -> change
points -> optional CG-DETR features. Everything per step lands in `grid.npz` on a 3 fps grid.

Query time (`Engine.clips`):

1. `parse_brief` -> `ParsedQuery` (weights, lengths, budget, count, plan lists, visual cues).
2. `brief_signals` -> per-step signals: corpus signals plus SigLIP relevance per shot item (absolute log-odds
   scale), the avoid penalty, the nc saliency, legibility, the highlight prior, the clean-edge mask, the cue bonus and
   the exclusion masks; `relevance`, the per-step curve candidate peaks follow, is composed with the heuristic's
   constants (no scorer reads it).
3. `corpus_candidates` -> spans per file; `gate` drops unusable / muted spans, then the forbidden ones.
4. `pool.shortlist`: `featurize` -> `pool_stats` -> `Scorer.score` on every candidate (with a shot list also
   `Scorer.score_items` for each item's own best); refine the best; gate, featurise (with the stage-1 pool
   statistics) and `Scorer.rescore` the refined spans, `Scorer.score_items` per item. The opener/closer/hook slots run
   the same on their own relevance and weights.
5. `select_for` on the shortlist with the brief's `ScoreScale` (stage-1 scores); `fill_budget`, and lengthened clips
   featurised and rescored as returned; edit order; `results.clip` per pick; `results.shortfall`.
6. Optionally `dump.write_run`: every scored candidate with its features, score and selection; a selected row is the
   clip as returned.

Spoken-topic briefs skip 2-5 for transcript sentences (`pool.topic_picks`).

## The learnable seam

`featurize(corpus, sig, spans) -> X (n, 95)` and `query_vector(p, persona) -> q (31 with the six default personas)`
are the only inputs of a scorer. A `Scorer` has `name`, `feature_version`, `personas`, `score(X, q)` (shortlist),
`rescore(s, X_pre, X_post, q)` (final totals after refinement) and `score_items(X_items, q, final)` (each span's
score per shot item, from `item_features`: the same spans as if the brief named that item alone); optionally
`explain(X, q, final)` (contributions summing to the score; the clip's `reason`). A clip's `score` is the scorer's
total.

Features carry raw signals; every hand-set weight, bonus and penalty is in `HeuristicScorer` (legibility, the
highlight-prior blend, on-screen names, cue bonuses, unclean edges, length and cut priors). The one pool statistic is
`pool_stats`: per candidate-length bucket, its size and the medians of a few signal means over the brief's stage-1
candidates (`bucket_*` columns), attached to any span by its length, so every scorer is a function of a span's row.

Selection never reads a score's absolute value: the exhaustive floor and margin and the MMR penalty are in units of
the brief's `ScoreScale` (median and robust spread of its stage-1 scores; constants calibrated so one spread is two
heuristic units, `select.HEURISTIC_SCALE`). Any scorer that is an increasing affine function of another picks the
same clips (`tests/test_rank.py`: the pinned cases with the heuristic x7 + 100 and x0.2 - 5), and the exhaustive floor
never empties a list that has matches. Which item a span serves and the exhaustive relevance gates are
scorer-independent matching (SigLIP item relevance plus names read on screen, `items.evidence`).

- `HeuristicScorer` is the hand-weighted total, an exact function of each row (it reads `h_rel_*`, the per-step blend
  of SigLIP and nc saliency, and the `bucket_*` columns); `tests/data/heuristic_clips.json` pins its output. A dumped
  row's score is the scorer on that row's features (score() for stage-1 candidates, the final total otherwise).
- `LinearRankScorer` (`vhe-linear-rank/1` JSON: mu, sd, w over `[z, outer(z, q_small)]`, a feature mask) is the
  first learned model; `vhe clips --scorer models/<name>.json`. A model of another feature or query version (v1, v2)
  is refused.
- A fine-tuned model plugs in the same way: implement the protocol (e.g. a GBDT or a small MLP over X and q) and add
  its file format to `load_scorer`. Learned models should not read `h_rel_*` or `src_*` (masked in training).

Feature v3 (2026-10-18) after v2's 82 columns (unchanged order): `edges_clean`, `legibility_mean`,
`highlight_prior_mean`, `detail_z_mean`, `framing_frac`, `bucket_size`, `bucket_h_rel`, `bucket_quality`,
`bucket_energy`, `bucket_steady`, `bucket_speech`, `bucket_legibility`, `bucket_highlight`; `h_rel_*` lost the
legibility bonus and the highlight blend (now composed by the heuristic), `ocr_item_match` has the one OCR ignore
rule. Query v3 appends `highlight` before the persona one-hot.
- An owned highlight head replaces the non-commercial CG-DETR saliency in its slot: `Engine._nc_scores(text)` returns
  per-step saliency on the corpus grid (NaN where a file has none) and proposals per file; `brief_signals` z-scores it
  into `nc_sal`, which feeds `nc_sal_mean`, `nc_sal_top60`, `nc_present` and the heuristic's blend. A head with
  another meaning or scale is a new `FEATURE_VERSION`.
- Changing, adding or reordering a feature or query entry means a new version (models record theirs).

## Dataset and training loop

1. Runs write dumps: `vhe clips ... --persona ID --dump-candidates run.json [--dump-all]`; spans picked elsewhere
   get the same features with `vhe features`.
2. The field-test harness, outside this repo (`vhe-field-test/eval/`: `app_pick.py`, `opus_pick.py`, `judge.py`,
   `featurize_opus.py`, `labels.py`), has a model judge score both systems' picks and joins labels onto dumped spans:
   `eval/dataset/rows.jsonl` (schema and biases: `tools/train/DATASET_CARD.md`).
3. `tools/train/build_dataset.py` -> arrays and pairs; `train_linear.py` -> `models/<name>.json` (RankNet or ridge,
   L2 by leave-one-brief-out CV); `eval_rank.py` -> pairwise accuracy and NDCG@5 against the heuristic, on the
   held-out set (festival-commons) when it has judged rows.
4. The model file goes back into step 1 via `--scorer`.
