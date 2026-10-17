# Dataset card: vhe ranking rows (features v1)

Training data for the learned candidate scorer (`vhe/rank/scorer.py`, `LinearRankScorer`). Built outside this
repository by the field-test harness (`vhe-field-test/eval/`), consumed by `tools/train/`. Not shipped with the
package.

## How a row comes about

1. `app_pick.py` runs `vhe clips <footage> <brief> --persona <id> --dump-candidates <run>/app/<brief>.cands.json`.
   The dump (`vhe-candidates/1`, `vhe/rank/dump.py`) holds every scored candidate of the run: the selected clips and
   the unselected shortlist pool (refined edges), opener/closer picks, spoken-topic clips; with `--dump-all` also
   every gated candidate before refinement.
2. `opus_pick.py` asks Claude Opus for its own clip selection for the same brief (an independent baseline).
3. `judge.py` shows both selections, shuffled and anonymised (K01..Knn), to a model judge, which scores every clip
   (fit 0-3, usable 0-2, redundant_with) and prefers one of the two sets.
4. `featurize_opus.py` runs `vhe features` on the Opus picks (and on the app's picks when the app wrote no dump) with
   the same app command, so both systems' clips get features from one code path (`stage: "external"`).
5. `labels.py` joins features and judge labels and writes `eval/dataset/rows.jsonl`. Rows of a (set, run) are
   replaced on rerun.

## Schema (one JSON object per line)

| field | meaning |
|---|---|
| `row_id` | `set/run_id/brief_id/origin/index` |
| `set`, `run_id`, `brief_id` | the query: rows are only ever compared within one (set, run, brief) |
| `persona`, `brief_kind`, `brief_text`, `parsed`, `q` | who the edit is for, `"reject"` briefs, the brief, the app's parse of it, the query vector (`query_names()`, v1) |
| `file`, `start`, `end` | the span, file-local seconds (3 dp) |
| `source`, `stage` | candidate generator (`window`, `peak`, ..., `external`) and dump stage |
| `origin` | `app_dump` (the app's own dump), `app_picks` (app picks featurised afterwards), `opus` |
| `selected_by` | `app`, `opus` or `none` (an unselected pool row) |
| `app_rank`, `role` | selection order / opener-closer-soundbite in the app's run |
| `dump_score` | the score recorded with the features (the run's scorer; for `external` rows: `score()` over that file's spans only) |
| `f_<name>` | the 71 features v1 (`vhe/rank/features.py: FEATURES`) |
| `judge_clip`, `label_iou` | the judged clip the label came from, and the overlap (1.0 for an exact match) |
| `judge_fit`, `judge_usable`, `judge_redundant` | the judge's clip scores; redundant = repeats an earlier clip of the same system's set |
| `set_pref`, `set_margin` | +1 / -1 if the judge preferred / rejected the set this row was selected into; margin slight/clear/strong/forfeit |
| `label_kind` | `judged`, `label_transfer`, `pool_negative`, `unjudged` |
| `y_point`, `weight` | the target and row weight (below) |
| `split` | `heldout` for set `festival-commons`, else `train` |
| `feature_version`, `scorer_at_dump`, `app_sha` | provenance; rows of another feature version are refused by `build_dataset.py` |

## Labels

- `y_point = fit/3` if `usable > 0`, else 0; for briefs of kind `reject` `fit/3` regardless of usability; minus 1/3
  for a clip that repeats an earlier clip of its own set; floored at 0. Values: 0, 1/3, 2/3, 1.
- `judged` (weight 1): the row's span equals a judged clip (every endpoint within 0.0051 s: the judge's times were
  rounded to 2 dp by the harness, the dump's to 3 dp).
- `label_transfer` (weight 0.5): no exact match, but IoU >= 0.7 with a judged clip of the same file; that clip's label.
- `pool_negative` (weight 0.3, `y_point` null): an unselected pool row with no judged neighbour. Not a measured
  negative: used only as the lower side of a pair against a positive (y >= 2/3) of the same query, at most 5 per
  positive.
- `unjudged` (weight 0): a selected clip the judge did not score; kept for bookkeeping, dropped by `build_dataset.py`.

Training pairs: labelled rows of one query whose y differ by >= 1/3 (weight = the smaller row weight), plus the
pool pairs above. Learned features drop `src_*` (external spans have no generator source, so the column would tell
the model who picked a clip) and `h_rel_*` (the hand-weighted relevance, kept only so the heuristic is an exact
function of the features); the mask travels in the model file.

## Splits

- `train`: every set except festival-commons (today: irl-event). Model selection (L2) is leave-one-brief-out
  cross-validation inside train; a brief judged in several runs is one fold.
- `heldout`: festival-commons (Brussels Pride 2022). Never read by `train_linear.py`; `eval_rank.py --split heldout`
  reports pairwise accuracy and NDCG@5 against the heuristic. No held-out rows exist until a festival-commons run is
  judged with dumps.

## Known biases and limits

- **The judge is a model**, not an editor. Its fit/usable scores are one sample from one prompt version
  (`judge-v2-persona`), through a persona text; no human calibration yet. A learned ranker learns the judge.
- **Selection bias.** Labels exist only for clips one of the two systems selected (plus IoU transfers). Pool rows
  are unlabelled; treating them as weak negatives assumes the systems' picks are better than what they left out.
  The app's own picks are scored by the app's current heuristic, so app rows are pre-filtered by the scorer the
  model is meant to replace; Opus picks come from a different distribution (and are featurised as `external`).
- **Few, correlated sources.** Each footage set is one shoot by one person (irl-event: one handheld livestream;
  festival-commons: one uploader's phone), so camera, location and style are shared by every row of a set. Briefs
  were written for the set by one author; two runs of the same brief are not independent (folds are by brief).
- **Small.** Tens of briefs, hundreds of labelled rows: the 426-weight interaction model (`phi = [z, outer(z,
  q_small)]`) can overfit; CV picks the L2 and `--no-interactions` fits 71 weights.
- **Parse noise.** The brief is parsed by a local LLM; app and Opus rows of one brief can carry different `q` if the
  parse differs between calls (`eval_rank.py` reports how many queries are affected).
- **Code-version mix.** `vhe features` should run the same app code as the run; for runs whose app snapshot predates
  `vhe features`, features come from a newer checkout (see the run's `featurize.json`).

## Footage licences

- irl-event: "i Need To Travel Now" YouTube livestream (Persian Day Parade, New York, 2026-04-19), CC BY 3.0
  (YouTube Creative Commons). Attribution: "i Need To Travel Now - youtube.com/watch?v=NESYHTmVclQ - CC BY". Parade
  music in the audio may be third-party copyrighted; people filmed in public.
- festival-commons: Wikimedia Commons, Brussels Pride 2022 by C.Suthorn, CC BY-SA 4.0
  (`C.Suthorn / cc-by-sa-4.0 / commons.wikimedia.org`). Share-alike applies to derivatives; personality rights of the
  people shown apply.
- The rows contain features and timestamps, not pixels or audio; brief texts and the judge's labels are ours. Local,
  private test use; see `vhe-field-test/sets/<set>/SOURCES.md` for per-file provenance.
