"""Convert both pandas datasets to messages format, merged.

Sources:
- channudambal 10k (complex, median 329 chars)
- Rahima train 57k + test 19k (short, median 205 chars)

Total ~86k. We shuffle and split: 80k train, 6k eval (held-out).

Outputs:
  data/train/pandas_train_80k.jsonl
  data/eval/pandas_eval_6k.jsonl
"""
import csv, json, os, random, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW1 = os.path.join(ROOT, "data", "raw", "pandas_finetune.csv")
RAW2_TRAIN = os.path.join(ROOT, "data", "raw", "rahima_train.csv")
RAW2_TEST = os.path.join(ROOT, "data", "raw", "rahima_test.csv")
OUT_TRAIN = os.path.join(ROOT, "data", "train", "pandas_train_80k.jsonl")
OUT_EVAL = os.path.join(ROOT, "data", "eval", "pandas_eval_6k.jsonl")
SEED = 20261101

rows=[]
# channudambal 10k
with open(RAW1, encoding="utf-8") as f:
    r=csv.DictReader(f)
    for row in r:
        rows.append({"input": row["input"].strip(), "output": row["output"].strip(), "source": "channudambal"})

# Rahima train + test
for path in [RAW2_TRAIN, RAW2_TEST]:
    with open(path, encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({"input": row["Input"].strip(), "output": row["Pandas Query"].strip(), "source": "rahima"})

print(f"total collected: {len(rows)}")
rng = random.Random(SEED)
rng.shuffle(rows)

# Split
train_rows = rows[:80000]
eval_rows = rows[80000:86000]

def to_msg(r):
    # Keep input as user prompt, output as assistant code
    # Add a short system-like prefix for clarity, but keep chat format
    user = r["input"]
    assistant = r["output"]
    return {"messages":[{"role":"user","content":user},{"role":"assistant","content":assistant}]}

# Ensure output dir
os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
os.makedirs(os.path.dirname(OUT_EVAL), exist_ok=True)

with open(OUT_TRAIN, "w", encoding="utf-8", newline="\n") as f:
    for r in train_rows:
        f.write(json.dumps(to_msg(r), ensure_ascii=False)+"\n")

with open(OUT_EVAL, "w", encoding="utf-8", newline="\n") as f:
    for i, r in enumerate(eval_rows):
        msg = to_msg(r)
        # Add eval wrapper for harness compatibility (prompt/answer/checker)
        f.write(json.dumps({
            "id": f"pandas_{i:04d}",
            "category":"pandas","subcategory":"text2pandas",
            "prompt": msg["messages"][0]["content"],
            "answer": msg["messages"][1]["content"],
            "checker":"exact",
            "messages": msg["messages"],
            "source": r["source"]
        }, ensure_ascii=False)+"\n")

print(f"train: {len(train_rows)} -> {OUT_TRAIN}")
print(f"eval: {len(eval_rows)} -> {OUT_EVAL}")
for p in [OUT_TRAIN, OUT_EVAL]:
    h=hashlib.sha256(open(p,"rb").read()).hexdigest()
    print(f"{os.path.basename(p)} sha {h[:16]}")
