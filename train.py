"""Fine-tune Qwen2.5-0.5B-Instruct on the Claude reasoning dataset (701x distil).

Usage (on GPU, 24GB):
    # LoRA r=8
    python train.py --r 8 --max-seq-length 4096 --tag lora_r8
    # LoRA r=32
    python train.py --r 32 --max-seq-length 4096 --tag lora_r32
    # QLoRA 4-bit r=32 (needs bitsandbytes)
    python train.py --r 32 --qlora --max-seq-length 4096 --tag qlora_r32

Outputs adapter -> outputs/<tag>/ (peft adapter_config.json + weights).
Benchmark it with:
    python harness/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/<tag> --tag <tag>

Design:
- chat template applied to the single user/assistant turn.
- tokenized with truncation to max_seq_length (dataset is long: median ~6.5k tok,
  so 4096 truncates; raising to 8192 keeps more but needs VRAM headroom).
- labels mask out the USER portion so the model learns to generate the ANSWER,
  not to repeat the prompt (completion-only LM).
- standard Trainer: no TRL dependency, version-proof.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "train" / "claude_opus_743_clean.jsonl"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RESPONSE_TEMPLATE = "\nassistant"  # Qwen chat marker before the assistant answer


def tokenize_example(example, tokenizer, max_seq_length):
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )  # up to "assistant\n"
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )  # prompt + answer
    enc_full = tokenizer(full, truncation=True, max_length=max_seq_length)

    input_ids = enc_full["input_ids"]
    # find where the assistant response begins to mask the prompt
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # locate prompt end within full (last "assistant\n" occurrence is robust)
    labels = input_ids.copy()
    # default: mask everything (we only un-mask the answer)
    labels = [-100] * len(input_ids)
    # find the response marker position
    # search for the last occurrence of the prompt's tail (assistant marker)
    tail = tokenizer(RESPONSE_TEMPLATE, add_special_tokens=False)["input_ids"]
    found = None
    if len(tail) >= 1:
        for i in range(len(input_ids) - len(tail), -1, -1):
            if input_ids[i:i + len(tail)] == tail:
                found = i + len(tail)  # start of answer after marker
                break
    if found is None:
        found = max(len(tail), 0)
    for j in range(found, len(input_ids)):
        labels[j] = input_ids[j]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": enc_full["attention_mask"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lora_r8")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if args.qlora:
        load_kwargs = {
            "load_in_4bit": True,
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto", **load_kwargs,
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, target_modules=targets,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"dataset: {len(rows)} rows")

    def map_fn(ex):
        return tokenize_example(ex, tokenizer, args.max_seq_length)

    dataset = Dataset.from_list(rows).map(
        map_fn, remove_columns=["messages"], batched=False
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        seed=args.seed,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved adapter -> {out_dir}")


if __name__ == "__main__":
    main()
