"""Run the frozen benchmark on a model (generate) and/or score existing outputs.

Usage - score only (no model, no torch needed):
    python harness/run_eval.py --score-only results/dummy/outputs.jsonl --tag dummy

Usage - generate + score (needs transformers + torch):
    python harness/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --tag base_before
    python harness/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter path/to/lora --tag lora_r8

Outputs:
    results/<tag>/outputs.jsonl   (raw model responses)
    results/<tag>/result.json     (per-item scores + aggregates)
"""
import argparse
import json
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS_DIR = ROOT / "results"

# scorers imported lazily so --help works without deps
sys.path.insert(0, str(ROOT / "harness"))
import scorers  # noqa: E402


def load_eval_items(eval_dir: Path):
    items = []
    for name in ["math.jsonl", "reasoning.jsonl", "coding.jsonl", "general.jsonl"]:
        path = eval_dir / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                obj["_suite"] = name.replace(".jsonl", "")
                items.append(obj)
    # qualitative is not scored
    return items


def verify_manifest(eval_dir: Path):
    mpath = eval_dir / "manifest.json"
    if not mpath.exists():
        print("WARN: no manifest.json, skipping verify")
        return
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for name, info in manifest.get("files", {}).items():
        path = eval_dir / name
        if not path.exists():
            print(f"WARN: manifest lists {name} but file missing")
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != info["sha256"]:
            print(f"ERROR: manifest mismatch for {name}")
            print(f"  expected {info['sha256']}")
            print(f"  got      {h}")
            print("  -> eval/ was modified after freezing! Restore from git or regenerate with tools/build_eval.py")
        else:
            print(f"manifest ok: {name} ({info['count']} items)")


def score_outputs(items, outputs_by_id):
    results = []
    for it in items:
        resp = outputs_by_id.get(it["id"], "")
        if it["checker"] == "none":
            # qualitative never scored
            continue
        try:
            score, reason = scorers.score_item(it, resp)
        except Exception as e:
            score, reason = 0.0, f"scorer error {e}"
        results.append({
            "id": it["id"],
            "suite": it["_suite"],
            "subcategory": it.get("subcategory", ""),
            "checker": it["checker"],
            "score": score,
            "reason": reason,
            "response_preview": (resp[:200].replace("\n", " ") if resp else ""),
        })
    return results


def aggregate(results):
    from collections import defaultdict
    by_suite = defaultdict(list)
    for r in results:
        by_suite[r["suite"]].append(r["score"])
    agg = {}
    for suite, scores in by_suite.items():
        # coding is 0..1 per item (pass rate); binary exact is 0/1 -> mean is accuracy/pass-rate
        agg[suite] = sum(scores) / len(scores) if scores else 0.0
    # overall = macro average over suites
    if agg:
        agg["overall"] = sum(agg.values()) / len(agg)
    return agg


def generate_with_model(model_id, adapter, items, out_path, max_new_tokens=1024):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"Missing dependency for generation: {e}")
        print("Install: pip install transformers torch accelerate")
        sys.exit(1)

    print(f"loading tokenizer {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"loading model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    if adapter:
        try:
            from peft import PeftModel
        except ImportError:
            print("peft not installed: pip install peft")
            sys.exit(1)
        print(f"loading adapter {adapter} ...")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # resume: skip ids already in outputs
    existing = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"resuming: {len(existing)} already done")

    import random
    # keep order deterministic
    for it in items:
        if it["id"] in existing:
            continue
        prompt = it["prompt"]
        # Qwen chat template: single user turn, no system
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        # per-suite token budget: coding needs more
        budget = 1536 if it["_suite"] == "coding" else max_new_tokens
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                temperature=0,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"id": it["id"], "suite": it["_suite"], "prompt": prompt, "response": gen}, ensure_ascii=False) + "\n")
        print(f"  {it['id']} ({it['_suite']}) -> {len(gen)} chars")
    print(f"done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id or local path, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path (peft)")
    ap.add_argument("--tag", required=True, help="run tag, e.g. base_before / lora_r8")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--score-only", default=None, help="path to outputs.jsonl to score without generation")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    verify_manifest(eval_dir)
    items = load_eval_items(eval_dir)
    print(f"loaded {len(items)} auto items from {eval_dir}")

    if args.score_only:
        outputs_path = Path(args.score_only)
    else:
        if not args.model:
            ap.error("--model is required unless --score-only is given")
        outputs_path = out_dir / "outputs.jsonl"
        generate_with_model(args.model, args.adapter, items, outputs_path, args.max_new_tokens)
        # also copy qualitative prompts' outputs? generate them too as extra (not scored)
        # generate qualitative separately with larger budget (not mixing scores)
        qual_path = eval_dir / "qualitative.jsonl"
        if qual_path.exists():
            with open(qual_path, encoding="utf-8") as f:
                qual_items = [json.loads(l) for l in f]
            for qi in qual_items:
                qi["_suite"] = "qualitative"
            # generate qualitative to same outputs file for convenience (checker=none, not scored)
            qual_out = out_dir / "outputs_qualitative.jsonl"
            generate_with_model(args.model, args.adapter, qual_items, qual_out, 1536)

    # load outputs
    outputs_by_id = {}
    with open(outputs_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            outputs_by_id[obj["id"]] = obj.get("response", "")

    results = score_outputs(items, outputs_by_id)
    agg = aggregate(results)

    # also score qualitative? no - just count
    result_path = out_dir / "result.json"
    payload = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "counts": {k: len([r for r in results if r["suite"] == k]) for k in set(r["suite"] for r in results)},
        "aggregate": agg,
        "results": results,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {result_path}")

    # pretty table
    print("\n=== Benchmark ===")
    header = f"{'suite':<12} {'n':>4}  {'score':>6}"
    print(header)
    print("-" * len(header))
    for suite in ["math", "reasoning", "coding", "general"]:
        if suite in agg:
            n = sum(1 for r in results if r["suite"] == suite)
            print(f"{suite:<12} {n:>4}  {agg[suite]*100:5.1f}%")
    if "overall" in agg:
        print("-" * len(header))
        print(f"{'overall':<12} {len(results):>4}  {agg['overall']*100:5.1f}%")

    # also write a markdown snippet for the comparison table the user sketched
    md = out_dir / "summary.md"
    lines = [f"# {args.tag}", "", "| Model | Math | Coding | Reasoning | General | Avg |", "|---|---|---|---|---|---|"]
    row = f"| {args.tag} | {agg.get('math',0)*100:.0f} | {agg.get('coding',0)*100:.0f} | {agg.get('reasoning',0)*100:.0f} | {agg.get('general',0)*100:.0f} | {agg.get('overall',0)*100:.1f} |"
    lines.append(row)
    md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
