"""
Step 4 – Use the saved model on new comments (no retraining needed).

    python src/predict.py "Тамаша тауар, жылдам жеткізді"
    python src/predict.py                      # interactive: type comments, empty line = exit

In Python / a notebook:
    from predict import ProfanityDetector
    det = ProfanityDetector("model/xlmr-kaz-profanity")
    det.predict(["Өте жаман қызмет", "..."])
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import apply_temperature, predict_proba  # noqa: E402


class ProfanityDetector:
    def __init__(self, model_dir="model/xlmr-kaz-profanity", device=None):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = json.load(open(self.model_dir / "inference_config.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, dtype=torch.float32).to(self.device)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        probs = predict_proba(self.model, self.tokenizer, texts, self.cfg["max_length"],
                              self.cfg["stride"], device=self.device,
                              normalize=self.cfg.get("normalize", True))
        probs = apply_temperature(probs, self.cfg.get("temperature", 1.0))   # calibrated
        thr = self.cfg["threshold"]
        return [{"text": t, "label": int(p >= thr), "probability_profane": round(float(p), 4),
                 "meaning": self.cfg["labels"][str(int(p >= thr))]}
                for t, p in zip(texts, probs)]


def show(results):
    for r in results:
        print(f"[{r['label']}] p(profane)={r['probability_profane']:.3f}  {r['meaning']:20s} | {r['text']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--model_dir=")]
    model_dir = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model_dir=")),
                     "model/xlmr-kaz-profanity")
    det = ProfanityDetector(model_dir)
    if args:
        show(det.predict(args))
    else:
        print("Type a Kazakh comment (empty line to quit):")
        for line in sys.stdin:
            if not line.strip():
                break
            show(det.predict(line.strip()))
