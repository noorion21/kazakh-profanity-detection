"""Shared helpers: seed, data loading, text cleaning, and chunked prediction."""
import json
import random

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_text  # noqa: E402


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text, normalize=True):
    """Preprocessing before the tokenizer.
    Always: normalise whitespace. We do NOT lower-case or remove punctuation:
    XLM-RoBERTa was pretrained on raw text; its tokenizer handles case, punctuation,
    emoji and '*' itself.
    If normalize=True (our main setting): Latin -> Kazakh Cyrillic, Latin look-alike
    letters fixed, long letter repeats shortened (see normalize.py)."""
    text = " ".join(str(text).split())
    return normalize_text(text) if normalize else text


def load_split(data_dir, name, normalize=True):
    df = pd.read_csv(f"{data_dir}/{name}.csv", keep_default_na=False)
    df["raw_text"] = df["text"]
    df["text"] = df["text"].map(lambda t: clean_text(t, normalize))
    df["label"] = df["label"].astype(int)
    return df


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_length=128, stride=32,
                  batch_size=32, device="cpu", normalize=True):
    """Return P(profane) for every text.

    Long comments: the text is cut into overlapping windows of `max_length`
    tokens (overlap = `stride`). Each window gets a probability and the text
    gets the MAXIMUM, because one profane word anywhere is enough.
    """
    model.eval()
    texts = [clean_text(t, normalize) for t in texts]
    enc = tokenizer(texts, truncation=True, max_length=max_length, stride=stride,
                    return_overflowing_tokens=True, padding=False)
    owner = enc["overflow_to_sample_mapping"]          # window -> original text index
    probs = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(owner), batch_size):
        feats = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
                 for i in range(start, min(start + batch_size, len(owner)))]
        batch = tokenizer.pad(feats, return_tensors="pt").to(device)
        logits = model(**batch).logits.float()
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        for j, pj in enumerate(p):
            k = owner[start + j]
            probs[k] = max(probs[k], pj)
    return probs


# ---------------------------------------------------------------- calibration (Week 5 lecture)
def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_temperature(probs, T):
    """Temperature scaling: p_T = sigmoid(logit(p) / T).
    T > 1 makes the model less over-confident. The ORDER of the probabilities does not change,
    so with a matching threshold the decisions stay the same; only the confidence becomes honest."""
    return 1 / (1 + np.exp(-_logit(probs) / T))


def fit_temperature(probs, y):
    """Choose T on the VALIDATION set by minimising the negative log-likelihood (grid search)."""
    y = np.asarray(y)
    best_T, best_nll = 1.0, np.inf
    for T in np.round(np.arange(0.5, 5.01, 0.05), 2):
        p = np.clip(apply_temperature(probs, T), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


def expected_calibration_error(probs, y, bins=10):
    """ECE: split predictions into 10 confidence bins, compare confidence with accuracy, average the gap."""
    probs, y = np.asarray(probs), np.asarray(y)
    pred = (probs >= 0.5).astype(int)
    conf = np.where(pred == 1, probs, 1 - probs)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)
