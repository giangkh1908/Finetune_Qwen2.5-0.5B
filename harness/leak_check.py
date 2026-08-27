"""Leak check for the text-to-pandas benchmark (pandas_train_80k vs eval).

WikiSQL-style prompts are formulaic (same table/schema, template question,
only values differ), so raw 6-gram Jaccard is a false-positive machine here.
The meaningful criteria:

  1. EXACT prompt match train vs eval            -> FAIL (memorized question)
  2. Same table + schema + masked question AND
     same answer                                 -> FAIL (answer memorization)
  3. Same skeleton, different answer             -> informational only
     (legit: model must generalize across values in a known table)

PASS = zero hits on (1) and (2). NOTE: the current 80k/6k split has a known
overlap (~699 of 6000 eval prompts appear in train, 606 with identical
answer) — this check reports FAIL for it by design; see README §8.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train" / "pandas_train_80k.jsonl"
EVAL_DIR = ROOT / "data" / "eval"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def mask_numbers(s: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", norm(s))


def table_schema(prompt: str):
    m = re.match(r"Table Name: (\S+) \((.*)\)", prompt, re.S)
    return (m.group(1), norm(m.group(2))) if m else (None, norm(prompt))


def question_of(prompt: str) -> str:
    return re.sub(r"^Table Name: \S+ \(.*\)", "", prompt, count=1, flags=re.S).strip()


def main():
    if not TRAIN.exists():
        print(f"train not found: {TRAIN}")
        return 1

    train_prompts = set()
    # skeleton key -> set of normalized answers
    skeletons = {}
    n_train = 0
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["messages"][0]["content"] if "messages" in obj else obj["prompt"]
            answer = obj["messages"][1]["content"] if "messages" in obj else obj.get("answer", "")
            n_train += 1
            train_prompts.add(text)
            t, s = table_schema(text)
            key = (t, s, mask_numbers(question_of(text)))
            skeletons.setdefault(key, set()).add(norm(answer))
    print(f"train rows: {n_train}  (unique skeletons: {len(skeletons)})")

    eval_files = sorted(EVAL_DIR.glob("pandas_eval_*.jsonl"))
    if not eval_files:
        print("no pandas_eval_*.jsonl found")
        return 1

    exact_hits = 0
    answer_leaks = 0
    skeleton_hits = 0
    n_eval = 0
    shown = 0

    for ef in eval_files:
        with open(ef, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                answer = obj.get("answer", "")
                n_eval += 1
                if prompt in train_prompts:
                    exact_hits += 1
                    if shown < 10:
                        print(f"  EXACT LEAK: {ef.name} {obj.get('id')}")
                        shown += 1
                    continue
                t, s = table_schema(prompt)
                key = (t, s, mask_numbers(question_of(prompt)))
                if key in skeletons:
                    skeleton_hits += 1
                    if norm(answer) in skeletons[key]:
                        answer_leaks += 1
                        if shown < 10:
                            print(f"  ANSWER LEAK: {ef.name} {obj.get('id')}")
                            shown += 1

    print(f"eval files checked: {[p.name for p in eval_files]} ({n_eval} items; 500/1000 are independent quick subsets, counted separately)")
    print(f"exact prompt matches: {exact_hits}")
    print(f"answer memorization (same skeleton + same answer): {answer_leaks}")
    print(f"same skeleton, different answer (ok): {skeleton_hits - answer_leaks}")

    if exact_hits or answer_leaks:
        print("LEAK CHECK: FAIL - eval prompts leak from training set")
        return 1
    print("LEAK CHECK: PASS - no eval item is memorizable from the training set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
