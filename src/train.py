"""
Step 2 – Fine-tune XLM-RoBERTa-base for Kazakh profanity detection.

    python src/train.py                      # uses config.json
    python src/train.py --epochs 2           # override a setting
    # ablation runs (same model, parts of the pipeline switched off):
    python src/train.py --normalize 0 --augment 0 --output_dir model/xlmr-raw --results_dir results/ablation_raw
    python src/train.py --normalize 1 --augment 0 --output_dir model/xlmr-norm --results_dir results/ablation_norm

Objective: cross-entropy over 2 classes, CLASS-WEIGHTED (profane mistakes cost more,
           because profane reviews are the minority class).
Model selection: the epoch with the best F1 (profanity class) on val.csv.
Threshold: chosen on val.csv to maximise F1 (the test set is NOT used here).
Saves to output_dir: model weights, tokenizer files, inference_config.json.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).parent))
from utils import (apply_temperature, fit_temperature, load_config, load_split,  # noqa: E402
                   predict_proba, set_seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    for k, t in [("model_name", str), ("epochs", int), ("batch_size", int),
                 ("learning_rate", float), ("max_length", int), ("output_dir", str),
                 ("max_train_samples", int), ("normalize", int), ("augment", int), ("class_weighted_loss", int),
                 ("results_dir", str)]:
        ap.add_argument(f"--{k}", type=t)
    return ap.parse_args()


def best_threshold(y_true, probs):
    """Try thresholds 0.05 ... 0.95 and keep the one with the best F1 on validation."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best]), list(zip(grid.tolist(), scores))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    cfg["normalize"] = bool(cfg["normalize"])
    cfg["augment"] = bool(cfg["augment"])
    cfg["class_weighted_loss"] = bool(cfg["class_weighted_loss"])
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {cfg['model_name']}")

    # ---- data
    train_df = load_split(cfg["data_dir"], "train", cfg["normalize"])
    val_df = load_split(cfg["data_dir"], "val", cfg["normalize"])
    if cfg["augment"]:                                    # + synthetic copies (train only)
        aug_df = load_split(cfg["data_dir"], "train_augmented", cfg["normalize"])
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
    if cfg.get("max_train_samples"):                      # only for quick tests
        train_df = train_df.sample(cfg["max_train_samples"], random_state=cfg["seed"])
    print(f"train: {len(train_df)} (profane {train_df.label.sum()}) | "
          f"val: {len(val_df)} (profane {val_df.label.sum()}) | normalize={cfg['normalize']} "
          f"| augment={cfg['augment']}")

    # class weights: w_c = N / (2 * N_c)  -> the rarer class (profane) gets a bigger weight
    counts = np.bincount(train_df.label, minlength=2)
    weights = len(train_df) / (2 * counts) if cfg["class_weighted_loss"] else np.ones(2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    print(f"class weights: clean {weights[0]:.3f} | profane {weights[1]:.3f}")

    # ---- tokenizer + model (pretrained on 100 languages incl. Kazakh and Russian)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2,
        id2label={0: "clean", 1: "profane"}, label2id={"clean": 0, "profane": 1},
    ).to(device)

    # text -> token ids (sub-words). Long texts are truncated to max_length for training.
    enc = tokenizer(train_df.text.tolist(), truncation=True, max_length=cfg["max_length"])
    train_items = [{"input_ids": i, "attention_mask": a, "labels": int(y)}
                   for i, a, y in zip(enc["input_ids"], enc["attention_mask"], train_df.label)]
    loader = DataLoader(train_items, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=DataCollatorWithPadding(tokenizer),
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    # ---- optimiser: AdamW + linear warm-up/decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    total_steps = len(loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_epoch = [], -1.0, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits      # shape (batch, 2)
            loss = loss_fn(logits.float(), labels)  # weighted cross-entropy
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-50:]):.4f}")

        val_probs = predict_proba(model, tokenizer, val_df.text.tolist(), cfg["max_length"],
                                  cfg["stride"], device=device, normalize=False)  # already cleaned
        val_f1 = f1_score(val_df.label, (val_probs >= 0.5).astype(int), zero_division=0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_f1_at_0.5": float(val_f1)})
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f} | val F1@0.5 {val_f1:.4f}")
        if val_f1 > best_f1:                        # keep only the best epoch
            best_f1, best_epoch = val_f1, epoch
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  -> saved best model (epoch {epoch})")

    # ---- reload best model, pick the decision threshold on validation
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    val_probs = predict_proba(model, tokenizer, val_df.text.tolist(), cfg["max_length"],
                              cfg["stride"], device=device, normalize=False)
    # temperature scaling on validation (calibration, Week 5 lecture), then the threshold
    temperature = fit_temperature(val_probs, val_df.label.values)
    val_probs = apply_temperature(val_probs, temperature)
    print(f"temperature T = {temperature}")
    thr, thr_f1, curve = best_threshold(val_df.label.values, val_probs)
    val_pred = (val_probs >= thr).astype(int)
    val_p = precision_score(val_df.label, val_pred, zero_division=0)
    val_r = recall_score(val_df.label, val_pred, zero_division=0)
    print(f"best epoch {best_epoch} | chosen threshold {thr:.2f} (val F1 {thr_f1:.4f})")

    # save smaller fp16 copy of the weights (~560 MB instead of ~1.1 GB)
    model.half().save_pretrained(out_dir)
    inference_cfg = {"threshold": thr, "max_length": cfg["max_length"], "stride": cfg["stride"],
                     "normalize": cfg["normalize"], "augment": cfg["augment"],
                     "temperature": temperature,
                     "labels": {"0": "no Kazakh profanity", "1": "Kazakh profanity"}}
    with open(out_dir / "inference_config.json", "w") as f:
        json.dump(inference_cfg, f, indent=2)

    res = Path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    log = {"config": cfg, "device": device, "train_size": len(train_df),
           "class_weights": [float(w) for w in weights], "temperature": temperature,
           "best_epoch": best_epoch, "history": history, "threshold": thr,
           "val_f1_at_threshold": thr_f1, "val_precision_at_threshold": round(float(val_p), 4),
           "val_recall_at_threshold": round(float(val_r), 4), "threshold_curve": curve,
           "training_minutes": round((time.time() - t0) / 60, 1),
           "versions": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}}
    with open(res / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("saved model to", out_dir, "| log to", res / "training_log.json")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
