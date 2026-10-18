"""Brief words against on-screen text (vhe.query.text_match): the legibility prior and the OCR item hook."""
from types import SimpleNamespace

import numpy as np

from vhe.query.text_match import legibility_steps, ocr_item_scores


def test_legibility_prior_only_for_readable_text_briefs_with_ocr():
    grid = {"text_conf": np.array([0.0, 0.9, 0.8, np.nan]), "text_area": np.array([0.0, 0.1, 0.0, 0.2])}
    leg = legibility_steps(grid, "every shot where on-screen text is legible")
    np.testing.assert_allclose(leg, [0.0, 0.9, 0.0, 0.0])
    assert legibility_steps(grid, "5 clips of dogs") is None
    assert legibility_steps({}, "every shot where text is legible") is None  # index without OCR


def test_ocr_hook_scores_items_when_the_corpus_has_text_and_is_a_no_op_otherwise():
    S, E, K = np.array([0.0, 10.0]), np.array([5.0, 15.0]), np.array([0, 0])
    assert ocr_item_scores(SimpleNamespace(), ["pasargad float"], S, E, K) is None
    corpus = SimpleNamespace(text_items=[{"t": 2.0, "text": "TOMBof CYRUS PASARGAD", "conf": 0.9, "file": 0},
                                         {"t": 12.0, "text": "RETAILSPACEFORLEASE", "conf": 0.8, "file": 0}])
    s = ocr_item_scores(corpus, ["Pasargad float", "Tomb of Cyrus"], S, E, K)
    assert s.shape == (2, 2) and s[0, 0] > 0 and s[0, 1] > 0.5 and not s[1].any()


def test_ocr_item_scores_read_only_the_spans_own_file_and_count_repeated_words():
    # Reads at the boundary of two files (virtual t = 10.0 starts file 1), an empty read, a repeated item word.
    corpus = SimpleNamespace(text_items=[{"t": 9.5, "text": "GATE", "file": 0},
                                         {"t": 10.0, "text": "NATIONS", "file": 1},
                                         {"t": 12.0, "text": "", "file": 1}, {"t": 30.0, "text": "gate", "file": 1}])
    S, E, K = np.array([8.0, 10.0, 11.5, 29.0]), np.array([10.0, 11.0, 12.5, 31.0]), np.array([0, 1, 1, 1])
    s = ocr_item_scores(corpus, ["gate of the nations", "gate gate band"], S, E, K)
    np.testing.assert_array_equal(s, [[0.5, 2 / 3], [0.5, 0.0], [0.0, 0.0], [0.5, 2 / 3]])
