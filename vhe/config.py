from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "vhe" / "models"

# Tiny grey stream rate: the source's average rate (resampling 25 fps to 30 duplicates every 5th frame, which
# reads as shake), clamped; VFR files without a usable average fall back to r_frame_rate, then to 30.
TINY_FPS_MIN, TINY_FPS_MAX, TINY_FPS_FALLBACK = 10.0, 60.0, 30.0

# Projects (`vhe index DIR`).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".mpg", ".mpeg", ".webm", ".3gp",
                    ".mxf", ".wmv"}
MIN_DURATION_S = 2.0  # shorter files (accidental taps) are recorded as too_short and not indexed

# Speech. Whisper gets VAD spans merged across short gaps, padded and capped, so edge words are not cut off.
ASR_MERGE_GAP_S, ASR_PAD_S, ASR_MAX_SPAN_S = 1.0, 0.3, 30.0
# A segment is dropped from the saved transcript when Whisper thinks it is probably not speech AND is unsure of
# the text (crowd noise and music produce such hallucinations).
ASR_DROP_NO_SPEECH_PROB, ASR_DROP_AVG_LOGPROB = 0.6, -1.0
LANG_MIN_PROB = 0.7  # only confident language detections vote for the project language
LANG_DETECT_S = 30.0  # seconds of speech per file used for language detection

# Usable mask: ABSOLUTE thresholds per analysis step. BLUR_RATIO is calibrated on real footage (below); the others
# are still initial guesses.
# Darkness alone is not a defect (a well exposed concert can average 0.1), so only these count.
# Covered = dark AND uniform (mean luma below COVERED_LUMA and luma std below UNIFORM_STD): a black frame that still
# shows something (a game's black "Game Over" / "Day 2" screen with a facecam and chat overlay, std 0.06-0.09) is
# content, not a covered lens. Real covered lenses / pockets in the camera sets read std <= 0.004.
COVERED_LUMA = 0.03  # mean luma (0..1) below this (and uniform): lens covered / in a pocket
UNIFORM_STD = 0.02  # luma std (0..1) below this: near-uniform frame (fabric, lens cap, wall right in front)
BLOWN_FRAC = 0.5  # fraction of pixels >= 250 above this: blown out
# Blurry: var(Lap(g)) / var(Lap(blur(g))) at BLUR_SIDE px below BLUR_RATIO (re-blur test, independent of texture).
# Calibrated 2026-10-17 on 200 frames of real phone footage (vhe-field-test festival-commons, 20 clips, analysis
# JPEGs from the pipeline's own ffmpeg path) plus synthetic defocus/motion blur applied at source resolution:
# 197 visually sharp frames read 12.6..26 (also with noise sigma 3 at 360 px, and from 360-1080p sources);
# defocus gaussian sigma 1.3/2.7 px and disk r 2.7/5.3 px at 360 px read <= 11.8/5.4 and 10.9/4.4. At 10: 0/591
# sharp flagged, 0/1200 dark, deep-dark, haze, strong-haze, sky-filled and flat-gradient frames flagged, 5/200
# textureless real-region crops (4 visibly soft: motion or out-of-focus foreground, 1 borderline); recall 96%
# sigma 1.3, 98% r 2.7, 100% stronger, 94% heavy motion blur. Strong defocus is still caught with noise sigma 10
# at 360 px (the 120 px downscale averages sensor/JPEG noise away). The old 2.5 missed even disk r 5.3 (reads
# ~3.1). Alternatives on the same set: Crete-Roffet 2007 and a noise-floored re-blur lost defocused frames at
# noise sigma >= 6; edge width flagged 26% of dark frames; an FFT high-band ratio flagged more textureless crops.
BLUR_RATIO = 10.0
BLUR_SIDE = 120  # px short side the re-blur test runs at (resolution-independent; averages away JPEG/sensor noise)
# Very shaky: camera jitter (motion.py shake: RMS over 1 s of the camera velocity minus its 0.5 s mean, frame-widths
# per second) above SHAKY. Calibrated 2026-10-17 on 71 visually labelled 1 s strips (festival-commons + irl-event:
# 13 shaky, 37 steady / smooth pan / subject-only motion / covered lens, 21 whip-pan or ambiguous handheld). Value at
# the labelled step: 0.5 -> precision 1.00, recall 0.85 (0.4 -> 0.92 / 0.85); max within +-0.5 s: 0.92 / 0.92. The
# old cv2.phaseCorrelate signal at the old 0.3 had precision 0.35 (a spinning umbrella read 7.3, a covered lens 15.8).
SHAKY = 0.5
# Framing defects from zero-shot SigLIP2 margins (tags.defect_scores: logsumexp of defect prompts minus logsumexp of
# scene prompts, in logits). Calibrated 2026-10-17 on 169 labelled analysis frames (same two sets; candidates from
# the top of several prompt scores, known bad moments, and 60 random steps).
# Obstructed (hard): lens covered by fabric/hand, or a near object (a back, a flag, a bag) filling the frame. At 0.5:
# precision 0.77, recall 0.85 (17/20) against normal/ground/sky frames; 2/97 normal frames flagged; 14/20 partial
# blocks (a back or head over ~40% of the frame) flagged too. 1.4-1.8% of all festival steps, 0.4-0.6% of irl-event.
OBSTRUCTED_MARGIN = 0.5
# Bad framing (soft: its reason bit is set and quality is penalised, but the step stays usable, since a sky shot can
# be intended): camera at the ground or pointed straight up. At 1.0: precision 0.77, recall 0.84 (27/32); the false
# positives are mostly whip-pan blur.
BAD_FRAMING_MARGIN = 1.0
# The usable-mask reasons the query side applies (vhe.signals.quality: query_usable, reason_mask, reason_active).
# shaky, obstructed and bad_framing are off (2026-10-18): they model a physical handheld camera and rejected ~60 %
# of a game livestream VOD (in-game camera turns read as shake, dark game scenes as a covered lens). Their bits are
# still computed and stored at index time, so adding a reason back here needs no re-index. An inactive reason gates
# no candidate, marks no clip edge unclean, costs no quality and answers no exclusion; with "shaky" off the shake
# signal is no defect either (no quality penalty, no clean-edge limit, no exclusion). It still feeds the brief's
# "steady" weight and nudges edges to the steadier frame nearby (refine._steadier), which are preferences.
ACTIVE_REASONS = frozenset({"covered", "uniform", "blown_out", "blurry"})

# On-screen text (signals/ocr.py, RapidOCR PP-OCRv4 on the 360 p analysis frames). Measured 2026-10-18 on irl-event
# (dev set; festival-commons is held out): at the native 360 p detector size recall on the named floats, banners and
# signs matched RapidOCR's default 736 p upscale at 2.5x less time; 4 ONNX threads ran 38 ms/frame vs 93 ms with all
# 16 (contention). Every OCR_EVERY-th step (1 fps) is read and carried to its neighbours.
OCR_EVERY = 3
OCR_MIN_CONF = 0.7  # a detection counts (text_area, text_conf, ocr.json) at or above this recognition confidence
OCR_MIN_LEN = 2  # ... and with at least this many non-space characters
OCR_THREADS = 4
# Faces (signals/faces.py, YuNet). Detector score threshold; frames are run at most FACE_SIDE px on the long side.
FACE_MIN_SCORE = 0.7
FACE_SIDE = 640
# Shot scale: zero-shot SigLIP2 margins (tags.scale_scores, logits, same construction as the defect margins).
# Suggested query-side thresholds from irl-event: close-ups / something filling the frame read >= 1 (giant flag
# C0019 2-12 s: 1.1..5.8; 95th percentile of all steps -0.3); wide establishing views >= 2.5 (C0001 40-45 s: 4.8..6.2,
# C0011 76-80 s: 2.5..3.3; medium shots of people read < 0, median step 1.9 since the set is mostly street wides).
SCALE_CLOSE_MARGIN = 1.0
SCALE_WIDE_MARGIN = 2.5
# Approach / looming (signals/faces.approach): growth rate of the dominant subject over APPROACH_WINDOW_S, in log
# face area per second (doubling per second = 0.69) where the face ends >= APPROACH_FACE_MIN of the frame, or
# APPROACH_CLOSE_WEIGHT x the close-margin rate in logits/s where the shot ends at scale_close >= APPROACH_CLOSE_MIN.
# FACE_AREA_EPS floors the log so a 0.05 %-of-frame face flickering in and out is not growth. Without the two gates
# 19 % of irl-event test steps (C0005, C0017, C0019) read > 0.3 (SigLIP jitter in wide shots); with them 6.6 %, while
# the giant flag coming at the lens (C0019 2-12 s) keeps 0.3..1.2 per second.
APPROACH_WINDOW_S = 1.0
APPROACH_CLOSE_WEIGHT = 0.35
APPROACH_FACE_MIN = 0.003
APPROACH_CLOSE_MIN = 0.0
FACE_AREA_EPS = 0.002

# Query side. Candidate window lengths (s): vhe.query.candidates generates windows of these lengths that fit the
# brief's clip limits, and the hand-weighted scorer centres scores per bucket of them (vhe.rank.scorer).
CANDIDATE_LENGTHS = (3, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90)


@dataclass
class Config:
    # Sampling grid: every per-timestep signal lives on t_i = i / fps.
    fps: float = 3.0
    frame_height: int = 360  # cached analysis frames; encoders resize further
    tiny_fps: float | None = None  # grey stream for cuts, shake and motion; None = source rate (see decode)
    tiny_w: int = 64
    tiny_h: int = 36
    hwaccel: str = "auto"  # auto | none | videotoolbox | cuda

    encoder: str = "google/siglip2-base-patch16-256"
    batch_size: int = 32
    device: str = "auto"  # auto | cpu | cuda | mps

    asr_model: str = "small"  # faster-whisper size; Parakeet planned for v2
    asr_language: str | None = None  # None = detect

    use_ocr: bool = True  # on-screen text (RapidOCR); skipped with a warning when not installed
    use_faces: bool = True  # YuNet faces (model fetched once into MODEL_DIR); skipped with a warning when missing

    # Lighthouse CG-DETR trained on QVHighlights: NON-COMMERCIAL, prototype only.
    use_nc_highlight: bool = True
    nc_checkpoint: str = str(MODEL_DIR / "clip_cg_detr_qvhighlight.ckpt")
    nc_window_s: float = 150.0
    nc_overlap_s: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

