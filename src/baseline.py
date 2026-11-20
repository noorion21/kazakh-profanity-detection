"""
Baseline WITHOUT machine learning: the word-list (lexicon) rules alone.
Predict 1 if the review contains an obscene word (incl. masked ones) or an insult word.

Important: our labels were CREATED with help of this lexicon (+ hand checks), so on our
own test set this baseline is unfairly strong. Its weakness shows on spellings it has
never seen -> see the robustness test in evaluate.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lexicon import match_insults, match_terms  # noqa: E402


def known_words(train_texts):
    """Vocabulary of obscene words seen in TRAIN (used to un-mask 'бл**')."""
    known = set()
    for t in train_texts:
        known.update(w for _, w in match_terms(t)[0])
    return known


def lexicon_predict(texts, known):
    return np.array([int(bool(match_terms(t, known)[0] or match_insults(t)[0])) for t in texts])
