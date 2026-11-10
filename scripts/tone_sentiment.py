#!/usr/bin/env python3
"""FinBERT-tone sentiment runner.

Loads yiyanghkust/finbert-tone (a FinBERT variant fine-tuned for tone)
via a transformers sentiment pipeline on CPU and scores English
financial texts.

Usage:
    python scripts/tone_sentiment.py --input samples/poul_headlines.txt

Outputs (into results/):
  - sentiment.json: per-item label + probabilities, counts, mean probs
  - report.md: human-readable table
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, json
from collections import Counter
from pathlib import Path

MODEL_ID = "yiyanghkust/finbert-tone"
CANONICAL = ("positive", "negative", "neutral")

def load_texts(path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                raise ValueError(f"Unsupported JSON item: {item!r}")
        return [t for t in texts if t.strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def canonicalise(raw_label):
    low = str(raw_label).lower()
    for c in CANONICAL:
        if c in low:
            return c
    # LABEL_0 etc: cannot map safely
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    from transformers import pipeline, AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID)
    id2label = {int(k): str(v) for k, v in getattr(config, "id2label", {}).items()}
    print(f"Model: {MODEL_ID}")
    print(f"id2label: {id2label}")

    texts = load_texts(Path(args.input))
    if not texts:
        raise SystemExit("No texts found in input.")
    clf = pipeline("sentiment-analysis", model=MODEL_ID, top_k=None, device=-1)
    raw_outputs = clf(texts)

    items = []
    for text, scores in zip(texts, raw_outputs):
        raw = [{"label": str(e["label"]), "score": float(e["score"])} for e in scores]
        probs = {label: 0.0 for label in CANONICAL}
        unmapped = []
        for entry in raw:
            canon = canonicalise(entry["label"])
            if canon:
                probs[canon] = entry["score"]
            else:
                unmapped.append(entry)
        # predicted = highest scoring raw entry, canonicalised if possible
        top = max(raw, key=lambda e: e["score"])
        predicted_canon = canonicalise(top["label"])
        predicted = predicted_canon if predicted_canon else top["label"]
        items.append({
            "text": text,
            "predicted": predicted,
            "predicted_raw_label": top["label"],
            "probabilities": probs,
            "raw_scores": raw,
            "unmapped_labels": unmapped,
        })

    counts = Counter(item["predicted"] for item in items)
    means = {}
    for label in CANONICAL:
        means[label] = sum(item["probabilities"][label] for item in items) / len(items) if items else 0.0

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": args.label,
        "model": MODEL_ID,
        "id2label": id2label,
        "count": len(items),
        "label_counts": {label: counts.get(label, 0) for label in CANONICAL},
        "label_counts_raw": dict(counts),
        "mean_probabilities": means,
        "items": items,
    }
    (results_dir / "sentiment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# FinBERT-tone sentiment",
        "",
        f"- Model: `{MODEL_ID}` (CPU)",
        f"- id2label: `{id2label}`",
        f"- Texts scored: {len(items)}",
        "- Label counts: " + ", ".join(f"{l}={counts.get(l,0)}" for l in CANONICAL),
        "- Mean probabilities: " + ", ".join(f"{l}={means[l]:.4f}" for l in CANONICAL),
        "",
        "| # | Predicted | Raw label | P(pos) | P(neg) | P(neu) | Text |",
        "|---|-----------|-----------|--------|--------|--------|------|",
    ]
    for i, item in enumerate(items, 1):
        p = item["probabilities"]
        text = item["text"].replace("|", "\\|")
        lines.append(f"| {i} | {item['predicted']} | {item['predicted_raw_label']} | {p['positive']:.4f} | {p['negative']:.4f} | {p['neutral']:.4f} | {text} |")
    lines.append("")
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Scored {len(items)} texts | counts={dict(counts)} | means=" + ", ".join(f"{l}={means[l]:.4f}" for l in CANONICAL))

if __name__ == "__main__":
    main()
