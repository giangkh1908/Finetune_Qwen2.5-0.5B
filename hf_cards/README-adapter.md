---
license: apache-2.0
base_model:
- Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- text-to-pandas
- wikisql
- lora
- peft
- qwen2.5
library_name: peft
language:
- en
---

# Qwen2.5-Coder-0.5B — Text-to-Pandas LoRA adapter (r=8)

LoRA adapter for **Qwen/Qwen2.5-Coder-0.5B-Instruct**, fine-tuned on 80k WikiSQL-style (table schema + question → pandas query) pairs. This is the **adapter-only** repo (~17MB); the ready-to-use merged model is [`giangkh19/qwen-0.5b-pandas-merged`](https://huggingface.co/giangkh19/qwen-0.5b-pandas-merged).

| Model | pandas eval 6k (exact match) |
|---|---|
| Base | 0.0% |
| **+ this adapter** | **79.1%** |

## Training

LoRA r=8, alpha=16, target `q,k,v,o,gate,up,down_proj`, dropout 0.05 · lr 2e-4 · 1 epoch · effective batch 16 · seq 512 · completion-only masking · RTX 3090 bf16 ~2h10m. Full pipeline and dataset builder: [GitHub](https://github.com/giangkh1908/Finetune_Qwen2.5-0.5B).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "giangkh19/qwen-0.5b-pandas-r8")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

prompt = "Table Name: t (a (int64), b (object))\nWhat is b when a is 3?"
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True)
out = model.generate(ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
# result = t[t['a'] == 3][['b']]
```

## Limitations

- ~699/6,000 benchmark prompts appear verbatim in training data (duplicated WikiSQL-style sources) — absolute scores optimistic by up to ~12 pts.
- Exact-match scoring, single-table WikiSQL-style questions only.

## License

Apache-2.0.
