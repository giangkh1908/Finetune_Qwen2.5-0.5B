---
license: apache-2.0
base_model:
- Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- text-to-pandas
- wikisql
- lora
- qwen2.5
- finetune
library_name: transformers
language:
- en
datasets:
- channudambal/pandas-finetune
---

# Qwen2.5-Coder-0.5B — Text-to-Pandas (merged, LoRA r=8)

Fine-tuned **Qwen/Qwen2.5-Coder-0.5B-Instruct** (494M) to convert a table schema + natural-language question into a **pandas query**. This repo is the **merged model** (LoRA already folded in) — load and serve it directly, no PEFT needed.

- Adapter-only version: [`giangkh19/qwen-0.5b-pandas-r8`](https://huggingface.co/giangkh19/qwen-0.5b-pandas-r8) (~17MB, needs PEFT)
- Pipeline & scripts: [`github.com/giangkh1908/Finetune_Qwen2.5-0.5B`](https://github.com/giangkh1908/Finetune_Qwen2.5-0.5B)

## Results

| Model | pandas eval 6k (exact match) |
|---|---|
| Base (Qwen2.5-Coder-0.5B-Instruct) | **0.0%** — answers SQL (46%) or prose (53%), never pandas format |
| **This model (LoRA r=8)** | **79.1%** (4747/6000, greedy, temperature 0) |

## Input / output format

```
Table Name: table_name_44 (power__kw_ (object), location (object))
What is the Davao's power (kW)?
```

→

```python
result = table_name_44[table_name_44['location'] == "davao"][['power__kw_']]
```

## Training

- **Data:** 80,000 (prompt, pandas-query) pairs merged from two WikiSQL-style sources (channudambal ~10k + Rahima ~70k), no dedup, split by row (seed 20261101)
- **Method:** LoRA r=8, alpha=16, all attention+MLP projections, dropout 0.05, lr 2e-4, 1 epoch, effective batch 16, seq 512 (no truncation — samples max ~350 tokens), completion-only masking
- **Hardware:** RTX 3090 24GB, bf16, ~2h10m (5,000 steps)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("giangkh19/qwen-0.5b-pandas-merged")
model = AutoModelForCausalLM.from_pretrained("giangkh19/qwen-0.5b-pandas-merged")

prompt = "Table Name: t (a (int64), b (object))\nWhat is b when a is 3?"
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True)
out = model.generate(ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
# result = t[t['a'] == 3][['b']]
```

Works with vLLM directly (`vllm serve giangkh19/qwen-0.5b-pandas-merged`).

## Limitations

- **Known train/eval overlap:** the training split contains ~699 of the 6,000 eval prompts verbatim (606 with identical answers, from duplicated WikiSQL-style sources). Absolute scores are optimistic by up to ~12 points; relative comparisons on this benchmark remain fair.
- Exact-match scorer is format-tolerant (case/punctuation/whitespace normalized, substring accepted) — scores reflect "correct query present in answer".
- Trained only on single-table WikiSQL-style English questions; multi-step analytics or real pandas DataFrames are out of distribution.
- Small model (0.5B): arithmetic-heavy filters with unusual values can still miss.

## License

Apache-2.0 (inherits from base model Qwen2.5-Coder-0.5B).
