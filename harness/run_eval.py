"""Run the frozen benchmark on a model (generate) and/or score existing outputs.

Modes:
  1) Direct (Colab/GPU thuê load model):
     python harness/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --tag base_before

  2) API serving (GPU serve vLLM, client gọi batch):
     # terminal 1: serve
     python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
     # terminal 2: benchmark qua API (không cần torch, chạy ở laptop cũng được)
     python harness/run_eval.py --api http://localhost:8000/v1 --tag base_before --batch-size 40

  3) Score-only (đã có outputs.jsonl):
     python harness/run_eval.py --score-only results/base_before/outputs.jsonl --tag base_before

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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "harness"))
import scorers  # noqa: E402
def load_eval_items(eval_dir: Path, suites=None):
    items = []
    if suites is None:
        suites = ["math.jsonl", "reasoning.jsonl", "coding.jsonl", "general.jsonl"]
    for name in suites:
        path = eval_dir / name
        if not path.exists():
            print(f"WARN: {name} not found, skip")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                obj["_suite"] = name.replace(".jsonl", "")
                items.append(obj)
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
            print("  -> eval/ was modified after freezing!")
        else:
            print(f"manifest ok: {name} ({info['count']} items)")


def score_outputs(items, outputs_by_id):
    results = []
    for it in items:
        resp = outputs_by_id.get(it["id"], "")
        if it["checker"] == "none":
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
        agg[suite] = sum(scores) / len(scores) if scores else 0.0
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
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto")
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
    existing = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try: existing.add(json.loads(line)["id"])
                except Exception: pass
        print(f"resuming: {len(existing)} already done")
    for it in items:
        if it["id"] in existing:
            continue
        prompt = it["prompt"]
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        budget = 1536 if it["_suite"] == "coding" else max_new_tokens
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=budget, do_sample=False, temperature=0, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"id": it["id"], "suite": it["_suite"], "prompt": prompt, "response": gen}, ensure_ascii=False) + "\n")
        print(f"  {it['id']} ({it['_suite']}) -> {len(gen)} chars")
    print(f"done -> {out_path}")


def generate_via_api(api_base, api_key, model_name, items, out_path, batch_size=40, max_new_tokens=1024, retries=2):
    """Batch 40 câu/lần qua OpenAI-compatible API (vLLM, TGI, Ollama)."""
    try:
        import requests
    except ImportError:
        print("pip install requests")
        sys.exit(1)
    api_base = api_base.rstrip("/")
    # /v1/chat/completions cho Qwen chat, fallback /v1/completions
    chat_url = f"{api_base}/chat/completions"
    comp_url = f"{api_base}/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # detect model name: nếu không truyền thì để server tự chọn
    # với vLLM cần đúng model id đã serve
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try: existing.add(json.loads(line)["id"])
                except Exception: pass
        print(f"resuming API: {len(existing)} already done")

    import concurrent.futures

    def call_one(it):
        if it["id"] in existing:
            return None
        prompt = it["prompt"]
        budget = 1536 if it["_suite"] == "coding" else max_new_tokens
        payload = {
            "model": model_name or "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": budget,
            "stream": False,
        }
        last_err = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(chat_url, headers=headers, json=payload, timeout=120)
                if r.status_code == 404:
                    # fallback completions
                    payload2 = {"model": payload["model"], "prompt": prompt, "temperature": 0, "max_tokens": budget}
                    r = requests.post(comp_url, headers=headers, json=payload2, timeout=120)
                r.raise_for_status()
                data = r.json()
                # chat.completions
                if "choices" in data and data["choices"]:
                    ch = data["choices"][0]
                    if "message" in ch:
                        gen = ch["message"].get("content", "")
                    else:
                        gen = ch.get("text", "")
                else:
                    gen = ""
                return (it, gen)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        print(f"  FAIL {it['id']}: {last_err}")
        return (it, "")

    # batch theo batch_size, mỗi batch chạy concurrent
    pending = [it for it in items if it["id"] not in existing]
    print(f"API batch: {len(pending)} items, batch_size={batch_size}, api={api_base}")
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        print(f" batch {i//batch_size+1}/{(len(pending)+batch_size-1)//batch_size} ({len(batch)} items)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(batch))) as ex:
            futs = {ex.submit(call_one, it): it for it in batch}
            for fut in concurrent.futures.as_completed(futs):
                res = fut.result()
                if res is None:
                    continue
                it, gen = res
                with open(out_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps({"id": it["id"], "suite": it["_suite"], "prompt": it["prompt"], "response": gen}, ensure_ascii=False) + "\n")
                print(f"  {it['id']} -> {len(gen)} chars")
    print(f"done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--model", default=None, help="HF model id (direct mode)")
    g.add_argument("--api", default=None, help="OpenAI-compatible API base, e.g. http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None, help="API key if needed")
    ap.add_argument("--api-model", default=None, help="model name for API (default Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path (direct mode only)")
    ap.add_argument("--tag", required=True, help="run tag, e.g. base_before / lora_r8")
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    ap.add_argument("--score-only", default=None, help="path to outputs.jsonl to score without generation")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=40, help="batch size for --api mode (concurrent requests)")
    ap.add_argument("--suites", default=None, help="comma-separated suite files to eval, e.g. coding.jsonl (default: all 4)")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_manifest(eval_dir)
    suites = [s.strip() + (".jsonl" if not s.strip().endswith(".jsonl") else "") for s in args.suites.split(",")] if args.suites else None
    items = load_eval_items(eval_dir, suites)
    print(f"loaded {len(items)} auto items from {eval_dir}")

    if args.score_only:
        outputs_path = Path(args.score_only)
    else:
        outputs_path = out_dir / "outputs.jsonl"
        if args.api:
            if args.adapter:
                print("WARN: --adapter ignored in --api mode (adapter phải merge ở server)")
            generate_via_api(args.api, args.api_key, args.api_model, items, outputs_path, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
            # qualitative cũng qua API batch
            qual_path = eval_dir / "qualitative.jsonl"
            if qual_path.exists():
                with open(qual_path, encoding="utf-8") as f:
                    qual_items = [json.loads(l) for l in f]
                for qi in qual_items: qi["_suite"] = "qualitative"
                qual_out = out_dir / "outputs_qualitative.jsonl"
                generate_via_api(args.api, args.api_key, args.api_model, qual_items, qual_out, batch_size=args.batch_size, max_new_tokens=1536)
        else:
            if not args.model:
                ap.error("cần --model hoặc --api hoặc --score-only")
            generate_with_model(args.model, args.adapter, items, outputs_path, args.max_new_tokens)
            qual_path = eval_dir / "qualitative.jsonl"
            if qual_path.exists():
                with open(qual_path, encoding="utf-8") as f:
                    qual_items = [json.loads(l) for l in f]
                for qi in qual_items: qi["_suite"] = "qualitative"
                qual_out = out_dir / "outputs_qualitative.jsonl"
                generate_with_model(args.model, args.adapter, qual_items, qual_out, 1536)

    outputs_by_id = {}
    with open(outputs_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            outputs_by_id[obj["id"]] = obj.get("response", "")
    results = score_outputs(items, outputs_by_id)
    agg = aggregate(results)
    result_path = out_dir / "result.json"
    payload = {"tag": args.tag, "model": args.model, "api": args.api, "api_model": args.api_model, "adapter": args.adapter, "counts": {k: len([r for r in results if r["suite"] == k]) for k in set(r["suite"] for r in results)}, "aggregate": agg, "results": results}
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {result_path}")
    print("\n=== Benchmark ===")
    header = f"{'suite':<12} {'n':>4}  {'score':>6}"
    print(header); print("-"*len(header))
    for suite in ["math","reasoning","coding","general"]:
        if suite in agg:
            n = sum(1 for r in results if r["suite"] == suite)
            print(f"{suite:<12} {n:>4}  {agg[suite]*100:5.1f}%")
    if "overall" in agg:
        print("-"*len(header))
        print(f"{'overall':<12} {len(results):>4}  {agg['overall']*100:5.1f}%")
    md = out_dir / "summary.md"
    suite_keys = [s for s in ["math", "reasoning", "coding", "general"] if s in agg]
    lines = [f"# {args.tag}", "", "| Model | " + " | ".join(s.capitalize() for s in suite_keys) + " | Avg |", "|---" + "|---"*len(suite_keys) + "|"]
    row = f"| {args.tag} | " + " | ".join(f"{agg[s]*100:.0f}" for s in suite_keys) + f" | {agg.get('overall',0)*100:.1f} |"
    lines.append(row)
    md.write_text("\n".join(lines), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
