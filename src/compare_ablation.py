"""
Ablation: which parts of the pipeline help?  (compared on the VALIDATION set only;
the test set is used once, for the final chosen model)

    python src/compare_ablation.py

rows:
  lexicon only (no model)                       - word-list rules, no training
  XLM-R, raw text                               - no normalization, no augmentation
  XLM-R, normalized text                        - + Latin->Cyrillic normalization
  XLM-R, normalized + augmentation (full)       - + synthetic masked/symbol/Latin/no-Kazakh copies
"""
import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from baseline import known_words, lexicon_predict  # noqa: E402
from utils import load_split  # noqa: E402

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
DATA = "data/processed"
RUNS = {"XLM-R, raw text": f"{RESULTS}/ablation_raw/training_log.json",
        "XLM-R, normalized text": f"{RESULTS}/ablation_norm/training_log.json",
        "XLM-R, normalized + augmentation (full)": f"{RESULTS}/training_log.json"}

val = load_split(DATA, "val", normalize=False)
train = load_split(DATA, "train", normalize=False)
pred = lexicon_predict(val.text, known_words(train.text))
rows = [{"setting": "lexicon only (no model)", "best_epoch": "-", "threshold": "-",
         "val_precision": round(precision_score(val.label, pred), 4),
         "val_recall": round(recall_score(val.label, pred), 4),
         "val_F1": round(f1_score(val.label, pred), 4)}]
for name, path in RUNS.items():
    if not Path(path).exists():
        print(f"(missing {path} - run training for '{name}' first)")
        continue
    log = json.load(open(path))
    rows.append({"setting": name, "best_epoch": log["best_epoch"], "threshold": log["threshold"],
                 "val_precision": log.get("val_precision_at_threshold", "-"),
                 "val_recall": log.get("val_recall_at_threshold", "-"),
                 "val_F1": round(log["val_f1_at_threshold"], 4)})

lines = ["| setting | best epoch | threshold | val precision | val recall | val F1 |",
         "|---|---:|---:|---:|---:|---:|"]
lines += [f"| {r['setting']} | {r['best_epoch']} | {r['threshold']} | {r['val_precision']} | "
          f"{r['val_recall']} | {r['val_F1']} |" for r in rows]
models = [r for r in rows if r["setting"].startswith("XLM-R")]
if models:
    best = max(models, key=lambda r: r["val_F1"])
    lines.append(f"\nChosen on validation: **{best['setting']}**")
    json.dump({"rows": rows, "chosen": best["setting"]}, open(f"{RESULTS}/ablation.json", "w"), indent=2)
print("\n".join(lines))
Path(f"{RESULTS}/ablation.md").write_text("\n".join(lines) + "\n")
