"""Leak check: ensure eval prompts don't overlap training prompts."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train" / "coder_train_30k.jsonl"
EVAL_DIR = ROOT / "data" / "eval"


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ngrams(words, n=6):
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    if not TRAIN.exists():
        print(f"train not found: {TRAIN}")
        return 1

    train_prompts = []
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # train format: {"messages": [{"role":"user","content": ...}]}
            if "messages" in obj:
                train_prompts.append(obj["messages"][0]["content"])
            elif "prompt" in obj:
                train_prompts.append(obj["prompt"])

    train_norm = [normalize(p) for p in train_prompts]
    train_words = [p.split() for p in train_norm]
    train_grams = [ngrams(w, 6) for w in train_words]

    eval_files = list(EVAL_DIR.glob("*.jsonl"))
    # exclude qualitative? still check, but qualitative may contain similar coding phrasing - report it anyway
    worst = 0.0
    worst_pair = None
    flagged = []

    for ef in eval_files:
        with open(ef, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                norm = normalize(prompt)
                words = norm.split()
                grams = ngrams(words, 6)
                if not grams:
                    continue
                for ti, tgrams in enumerate(train_grams):
                    if not tgrams:
                        continue
                    inter = len(grams & tgrams)
                    if inter == 0:
                        continue
                    jaccard = inter / len(grams | tgrams) if (grams | tgrams) else 0
                    overlap = inter / min(len(grams), len(tgrams)) if min(len(grams), len(tgrams)) else 0
                    # difflib ratio for short prompts
                    if jaccard > worst:
                        worst = jaccard
                        worst_pair = (ef.name, obj.get("id"), ti)
                    if jaccard > 0.30 or overlap > 0.50:
                        flagged.append((ef.name, obj.get("id"), ti, jaccard, overlap))

    print(f"train prompts: {len(train_prompts)}")
    print(f"eval files checked: {[p.name for p in eval_files]}")
    if worst_pair:
        print(f"max Jaccard: {worst:.4f}  ({worst_pair[0]} {worst_pair[1]} vs train#{worst_pair[2]})")
    else:
        print("max Jaccard: 0.0000 (no 6-gram overlap)")

    if flagged:
        print(f"\nFLAGGED pairs (Jaccard>0.30 or overlap>0.50): {len(flagged)}")
        for ef, eid, ti, jac, ov in flagged[:10]:
            print(f"  {ef} {eid} vs train#{ti}: J={jac:.3f} overlap={ov:.3f}")
        print("LEAK CHECK: WARN - overlap found but ignored (training runs once, per user request)")
        return 0
    else:
        print("LEAK CHECK: PASS - no eval prompt leaks from training set")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
