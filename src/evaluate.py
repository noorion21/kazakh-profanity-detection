"""
Step 3 – Evaluate ONCE on the held-out test set (after all settings are chosen).

    python src/evaluate.py

Saves to results/: test_metrics.json, confusion_matrix.png,
test_predictions.csv, errors_false_positives.csv, errors_false_negatives.csv,
robustness.json (stress test: the same test reviews written in harder ways)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import random  # noqa: E402

from augment import AUGMENTATIONS  # noqa: E402
from baseline import known_words, lexicon_predict  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from utils import (apply_temperature, expected_calibration_error, load_config,  # noqa: E402
                   load_split, predict_proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_dir = Path(args.model_dir or cfg["output_dir"])
    res = Path(cfg["results_dir"])
    res.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inf = json.load(open(model_dir / "inference_config.json"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32).to(device)

    test = load_split(cfg["data_dir"], "test", normalize=False)   # raw text in
    probs = predict_proba(model, tokenizer, test.text.tolist(), inf["max_length"],
                          inf["stride"], device=device, normalize=inf.get("normalize", True))
    raw_probs = probs
    probs = apply_temperature(probs, inf.get("temperature", 1.0))      # calibrated probabilities
    test["probability"] = probs.round(4)
    test["prediction"] = (probs >= inf["threshold"]).astype(int)

    y, p = test.label.values, test.prediction.values
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "n_test": int(len(test)), "n_profane": int(y.sum()), "n_clean": int((y == 0).sum()),
        "threshold": inf["threshold"],
        "precision_profane": round(float(prec[0]), 4),
        "recall_profane": round(float(rec[0]), 4),
        "f1_profane": round(float(f1[0]), 4),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        # where do the mistakes come from? (source = how the example was sampled)
        "errors_by_source": test[test.label != test.prediction]
        .groupby("source").size().to_dict(),
        "errors_by_domain": test[test.label != test.prediction]
        .groupby("domain").size().to_dict(),
    }
    # recall for each kind of profanity / script (label stays binary; this is analysis only)
    pos = test[test.label == 1]
    metrics["recall_by_profanity_type"] = {
        k: {"n": int(len(g)), "recall": round(float(g.prediction.mean()), 4)}
        for k, g in pos.groupby("profanity_type")}
    metrics["recall_by_script"] = {
        k: {"n": int(len(g)), "recall": round(float(g.prediction.mean()), 4)}
        for k, g in pos.groupby("script")}
    neg = test[test.label == 0]
    metrics["false_positive_rate_by_source"] = {
        k: {"n": int(len(g)), "fp_rate": round(float(g.prediction.mean()), 4)}
        for k, g in neg.groupby("source")}
    # ---- Week 5 lecture: reliability of the result
    # (a) metric uncertainty: 95% bootstrap confidence interval (1000 resamples of the test set)
    rs = np.random.RandomState(0)
    boot = {"precision": [], "recall": [], "f1": []}
    for _ in range(1000):
        idx = rs.randint(0, len(y), len(y))
        bp, br, bf, _ = precision_recall_fscore_support(y[idx], p[idx], labels=[1], zero_division=0)
        boot["precision"].append(bp[0]); boot["recall"].append(br[0]); boot["f1"].append(bf[0])
    metrics["bootstrap_95ci"] = {k: [round(float(np.percentile(v, 2.5)), 4),
                                     round(float(np.percentile(v, 97.5)), 4)] for k, v in boot.items()}
    # (b) calibration: does "90% sure" mean right 90% of the time?
    metrics["calibration"] = {
        "temperature": inf.get("temperature", 1.0),
        "ECE_before_temperature": round(expected_calibration_error(raw_probs, y), 4),
        "ECE_after_temperature": round(expected_calibration_error(probs, y), 4)}
    # (c) abstention: unsure reviews (0.2 < p < 0.8) go to a human moderator
    sure = (probs <= 0.2) | (probs >= 0.8)
    sp, sr, sf, _ = precision_recall_fscore_support(y[sure], p[sure], labels=[1], zero_division=0)
    metrics["abstention_0.2_0.8"] = {
        "coverage": round(float(sure.mean()), 4), "sent_to_human": int((~sure).sum()),
        "precision_auto": round(float(sp[0]), 4), "recall_auto": round(float(sr[0]), 4),
        "f1_auto": round(float(sf[0]), 4)}

    # baseline without ML (word list). NOTE: unfairly strong here, labels were built with it.
    known = known_words(load_split(cfg["data_dir"], "train", normalize=False).text)
    lex = lexicon_predict(test.text, known)
    lp, lr, lf, _ = precision_recall_fscore_support(y, lex, labels=[1], zero_division=0)
    metrics["lexicon_baseline"] = {"precision": round(float(lp[0]), 4), "recall": round(float(lr[0]), 4),
                                   "f1": round(float(lf[0]), 4)}
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(res / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # confusion matrix picture
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    names = ["0 clean", "1 profane"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Test set (n={len(test)}), threshold={inf['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(res / "confusion_matrix.png", dpi=150)

    cols = ["id", "text", "label", "prediction", "probability", "profanity_type", "script",
            "source", "domain", "sentiment"]
    test[cols].to_csv(res / "test_predictions.csv", index=False)
    test[(test.label == 0) & (test.prediction == 1)].sort_values(
        "probability", ascending=False)[cols].to_csv(res / "errors_false_positives.csv", index=False)
    test[(test.label == 1) & (test.prediction == 0)].sort_values(
        "probability")[cols].to_csv(res / "errors_false_negatives.csv", index=False)
    # ---- robustness stress test: rewrite the same test reviews in harder ways
    rng = random.Random(123)
    robust = {}
    for name, fn in AUGMENTATIONS.items():
        new_texts, keep = [], []
        for i, (t, lab) in enumerate(zip(test.text, test.label)):
            nt = fn(t, lab, rng)
            if nt:
                new_texts.append(nt)
                keep.append(i)
        sub_y = test.label.values[keep]
        mp = predict_proba(model, tokenizer, new_texts, inf["max_length"], inf["stride"],
                           device=device, normalize=inf.get("normalize", True))
        robust[name] = {
            "n": len(keep), "n_profane": int(sub_y.sum()),
            "model_f1_original": round(float(f1_score(sub_y, test.prediction.values[keep])), 4),
            "model_f1_changed": round(float(f1_score(sub_y, (mp >= inf["threshold"]).astype(int))), 4),
            "lexicon_f1_changed": round(float(f1_score(sub_y, lexicon_predict(new_texts, known))), 4),
            "example": new_texts[int(next(j for j, k in enumerate(keep) if test.label.values[k] == 1))][:120],
        }
    print("robustness:", json.dumps(robust, indent=2, ensure_ascii=False))
    with open(res / "robustness.json", "w", encoding="utf-8") as f:
        json.dump(robust, f, indent=2, ensure_ascii=False)
    print("saved results to", res)


if __name__ == "__main__":
    main()
