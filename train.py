"""Fine-tune Qwen2.5-Coder-0.5B-Instruct on the Claude coding dataset.

Usage (on GPU, 24GB):
    # LoRA r=8 (code-only)
    python train.py --r 8 --max-seq-length 4096 --epochs 1 --tag coder_r8
    # QLoRA 4-bit (needs bitsandbytes)
    python train.py --r 32 --qlora --max-seq-length 4096 --tag coder_qlora

Outputs adapter -> outputs/<tag>/ (peft adapter_config.json + weights).
Benchmark with:
    python harness/run_eval.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --adapter outputs/<tag> --tag <tag> --suites coding.jsonl
"""
import argparse
import json
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
DATA = ROOT / "data" / "train" / "coder_train_code_743.jsonl"
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
RESPONSE_TEMPLATE = "\nassistant"  # Qwen chat marker before the assistant answer


def tokenize_example(example, tokenizer, max_seq_length):
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc_full = tokenizer(full, truncation=True, max_length=max_seq_length)
    input_ids = enc_full["input_ids"]
    labels = [-100] * len(input_ids)
    tail = tokenizer(RESPONSE_TEMPLATE, add_special_tokens=False)["input_ids"]
    found = None
    if len(tail) >= 1:
        for i in range(len(input_ids) - len(tail), -1, -1):
            if input_ids[i:i + len(tail)] == tail:
                found = i + len(tail)
                break
    if found is None:
        found = max(len(tail), 0)
    for j in range(found, len(input_ids)):
        labels[j] = input_ids[j]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": enc_full["attention_mask"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="coder_r8")
    ap.add_argument("--model", default=MODEL_ID, help="HF model id or local path")
    ap.add_argument("--data", default=str(DATA), help="path to train jsonl")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
        args.model, trust_remote_code=True,
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
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"dataset: {len(rows)} rows")

    def map_fn(ex):
        return tokenize_example(ex, tokenizer, args.max_seq_length)

    dataset = Dataset.from_list(rows).map(map_fn, remove_columns=["messages"], batched=False)

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
