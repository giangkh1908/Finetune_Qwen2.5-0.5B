# Finetune Qwen2.5-Coder-0.5B — Lab Guide

Fine-tune **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** (494M, chuyên code) trên **80,000 text-to-pandas** (10k channudambal + 76k Rahima), đo **trước / sau** bằng benchmark pandas 6k + code 200, so sánh LoRA.

> Model mục tiêu: **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** — base `Qwen2ForCausalLM` + chat template `<|im_start|>`.

---

## 1. Repo structure

```
data/
  train/coder_train_all_chunked.jsonl  # 6,425 code samples (đang dùng) — ≤4000 tok, no truncation
    ├─ claude_opus_743 (production code, chunked tại block boundary)
    ├─ curated knowledge 685 câu (OOP/Big-O/HTTP/SQL/trace, do tác giả soạn)
    └─ Magicoder-OSS-Instruct-75K Python slice 5,000 câu (MIT, lọc trùng eval)
  eval/                                # FROZEN — ĐỪNG SỬA (code-only)
    coding.jsonl        84 items  unit_test (mỗi câu 4-5 assert, ~660 tests)
    manifest.json       sha256 — chống sửa
harness/
  run_eval.py       # generate + score (direct --model hoặc API --api, batch 40)
  scorers.py        # numeric/exact/choice/constraint/unit_test
  leak_check.py     # kiểm tra eval không leak từ train (6-gram Jaccard)
  compare.py        # gộp nhiều run → bảng markdown
serving/server.py   # FastAPI OpenAI-compatible (fallback khi không dùng vLLM)
tools/
  build_eval.py / build_eval_coding.py / build_eval_coding_extra.py  # builder eval
  build_code_knowledge.py / build_magicoder.py / chunk_train.py       # builder train
  shorten_and_augment.py                                              # pipeline train
train.py            # LoRA/QLoRA finetune (standard Trainer, completion-only masking)
results/            # output benchmark (<tag>/result.json, summary.md, outputs.jsonl)
outputs/            # adapter LoRA sau train (<tag>/adapter_config.json + safetensors ~17MB)
requirements.txt
HF model: giangkh19/qwen-0.5b-coder-r8  (adapter LoRA r=8)
```

---

## 2. Chuẩn bị GPU (thuê) — template vLLM OpenAI CUDA 12.9

**Cấu hình chốt:**

* **Image:** `vLLM OpenAI CUDA 12.9 — 9.06 GB` (đã có `torch` + `vllm` + `nvcc` + `ninja`, không cài lại)
* **GPU:** `RTX 3090 24GB` / CPU `Ryzen 5 5500 6C/12T` / RAM `48GB` / SSD `338GB`
* **Model vLLM:** `Qwen/Qwen2.5-Coder-0.5B-Instruct`
* **Tham số vLLM (args):** `--gpu-memory-utilization 0.8 --enable-prefix-caching --served-model-name qwen-coder`
  * `0.8` ≈ 19.2 GB / 24 GB, chừa lại cho hệ thống
  * `--enable-prefix-caching` tái dùng KV cache cho prefix giống nhau (system prompt dài / RAG)
  * Không cần `--tensor-parallel-size`, `--kv-cache-dtype fp8`, `--quantization` ở bước baseline — 0.5B quá nhẹ

Dòng lệnh tương đương:
```bash
vllm serve Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen-coder \
  --gpu-memory-utilization 0.8 --enable-prefix-caching
```

### Boot check (đừng cài gì, chạy đúng 4 lệnh)

```bash
nvidia-smi
df -h
vllm --version
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Gửi 4 output — nếu template đúng thì đi thẳng serve → test API → benchmark, không quay lại vòng `torch`/`nvcc`.

### Clone repo (sau khi boot check OK)

```bash
cd ~
git clone https://github.com/giangkh1908/Finetune_Qwen2.5-0.5B.git
cd Finetune_Qwen2.5-0.5B
pip3 install -r requirements.txt -q  # chỉ harness/tools, KHÔNG cài lại torch/vllm
python3 harness/leak_check.py          # PASS
```

---

## 3. Baseline BEFORE (code-only, 84 câu) — qua vLLM

```bash
# Hệ thống đã tự serve vLLM trên :8000 với model/args ở trên. Test:
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-coder","messages":[{"role":"user","content":"Write a Python function to sort a list."}],"max_tokens":256,"temperature":0.2}'

# Benchmark (batch 40 = 40 request concurrent, vLLM continuous batching)
python3 harness/run_eval.py --api http://localhost:8000/v1 --tag coder_base --batch-size 40
# hoặc chỉ định model name nếu khác served-model-name:
# python3 harness/run_eval.py --api http://localhost:8000/v1 --model qwen-coder --tag coder_base --batch-size 40
```

Kết quả: `results/coder_base/result.json` — **73.9%** (84 code).

> Fallback không vLLM (nếu muốn chạy direct): `python3 harness/run_eval.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tag coder_base` hoặc `python3 serving/server.py` (transformers, không cần vLLM).

Transfer về Windows:
```powershell
mkdir D:\Finetune\results\coder_base 2>$null
scp -P <PORT> <USER>@<GPU_IP>:/root/Finetune_Qwen2.5-0.5B/results/coder_base/result.json D:\Finetune\results\coder_base\result.json
```

---

## 4. Finetune (LoRA / QLoRA)

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `--r` | LoRA rank. Cao = học mạnh, dễ overfit | **8** điểm ngọt cho 6425 rows; 32 chỉ khi data lớn hơn |
| `--alpha` | scaling (thường 2×r) | 16 (r=8) |
| `--max-seq-length` | Giới hạn token/sample. **Nguồn VRAM chính** | 4096 (đã chunk ≤4000 nên **không truncate**, 24GB thoải mái) |
| `--epochs` | Số vòng qua dataset | **1** (6425 rows đa dạng, không cần lặp) |
| `--batch-size` / `--grad-accum` | Batch thực/effective | 1 / 8 |
| `--lr` | Tốc độ học | 2e-4 |
| `--qlora` | Nén 4-bit → giảm 40–60% VRAM | Thêm khi chạm trần |

```bash
# LoRA r=8 — chính
python3 train.py --r 8 --max-seq-length 4096 --epochs 1 --tag coder_r8

# QLoRA 4-bit r=8 (nếu muốn)
python3 train.py --r 8 --qlora --max-seq-length 4096 --epochs 1 --tag coder_qlora

# Thử r=32 (mạnh hơn, dễ overfit)
python3 train.py --r 32 --alpha 64 --max-seq-length 4096 --epochs 1 --tag coder_r32
```

Output: `outputs/<tag>/` — adapter ~17MB (r=8) / ~70MB (r=32). Log `train_loss` giảm dần là đang học; `grad_norm` 0.2–0.5 là ổn.

> Train đã bật `gradient_checkpointing` + `enable_input_require_grads()` — fix OOM khi `seq 4096`.

---

## 5. Benchmark AFTER

```bash
python3 harness/run_eval.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter outputs/coder_r8 --tag coder_r8

# Windows:
mkdir D:\Finetune\results\coder_r8 2>$null
scp -P <PORT> <USER>@<GPU_IP>:/root/Finetune_Qwen2.5-0.5B/results/coder_r8/result.json D:\Finetune\results\coder_r8\result.json
```

---

## 6. So sánh

```bash
python3 harness/compare.py coder_base coder_r8
# | Model | Coding | Avg |
# | coder_base | 74 | 73.9 |
# | coder_r8   | 80 | 80.5 |
```

---

## 7. Kết quả thực nghiệm

| Model | Data | Eval | Kết quả | Δ vs base |
|-------|------|------|---------|-----------|
| Qwen-Coder 0.5B base | — | 84 code | **73.9%** | — |
| **+ LoRA r=8** | 6,425 code (743 prod chunked + 685 curated + 5,000 Magicoder Python) | 84 code | **80.5%** | **+6.6 pts** |

**Chẩn đoán:** Không còn catastrophic forgetting như lần finetune đầu (trước: 60.8% → 43.5%, math −36, coding −23). Đổi sang **code-only + Coder model + dataset đa dạng + chunk không truncate + 1 epoch** đã đảo chiều.

**Nguyên nhân thành công:** dataset đa dạng (prod + knowledge + Magicoder), chunk ≤4000 thay vì truncate 40%, epochs 1 chống overfit, Coder model hợp domain.

---

## 8. Dataset (đang dùng)

`data/train/coder_train_all_chunked.jsonl` — **6,425 rows**, ≤4000 tok, **0 sample vượt ngưỡng** (không truncate):

| Nguồn | Rows | Mô tả |
|-------|------|-------|
| Claude production code (chunked) | 743 → chunked 1,336 | Queue/file-watcher/migration helper — chia tại block boundary |
| Curated knowledge (tác giả soạn) | 685 | OOP/Big-O/HTTP/SQL/trace, ngắn (<70 tok) |
| Magicoder-OSS-Instruct-75K Python | 5,000 | Python thực tế (uploader, config, API...), MIT, lọc trùng eval (0 collision) |

Build lại:
```bash
python3 tools/build_magicoder.py  # tạo 5000 Python slice
python3 tools/chunk_train.py --input data/train/coder_train_all.jsonl --output data/train/coder_train_all_chunked.jsonl --max-tokens 4000
```

Eval: `data/eval/coding.jsonl` — 84 hàm Python (24 gốc + 60 mở rộng thuật toán: string/list/dict/math/parsing), mỗi câu 4–5 assert, reference 100% pass qua `scorers`, **leak PASS** (Jaccard 0.009).

---

## 9. Push model lên Hugging Face

Adapter **giangkh19/qwen-0.5b-coder-r8** (HF namespace là `giangkh19`, không phải `giangkh1908`):

```powershell
# Windows — tạo Classic Write token tại https://huggingface.co/settings/tokens
hf auth logout
hf auth login   # chọn Paste, dán Classic Write token

hf upload giangkh19/qwen-0.5b-coder-r8 outputs/coder_r8 --repo-type model
# README trong outputs/coder_r8/README.md đã ghi sẵn dataset + metrics
```

Dùng lại:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "giangkh19/qwen-0.5b-coder-r8")
```

---

## 10. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| Template vLLM báo `model not found` | Điền ô **Model vLLM**: `Qwen/Qwen2.5-Coder-0.5B-Instruct`, Args như §2 |
| Muốn đổi 80% VRAM / cache | Sửa ô **Tham số vLLM**: `--gpu-memory-utilization 0.8 --enable-prefix-caching --served-model-name qwen-coder` rồi restart instance |
| `No space left` / `triton 197MB` | Template 12.9 đã có sẵn, **đừng** `pip install vllm` lại; chỉ `pip install -r requirements.txt` |
| `libtorch_cuda.so: ncclCommResume` | Sai `torch` — template 12.9 đã đúng, đừng `pip install torch` đè lên; nếu tự cài bare Ubuntu thì dùng `cu124` khớp driver |
| `python: command not found` | `python3` |
| VRAM OOM khi train | 24GB dư dả với 0.5B; nếu OOM check `nvidia-smi` process cũ, `kill -9 <PID>` |
| `git pull` báo file đè | `rm -rf results/<tag>` rồi `git pull` |
| scp hỏi password | Chạy riêng trong PowerShell/CMD, nhập thủ công |
| `403 Forbidden` khi push HF | Token phải **Classic Write** (không phải read), `hf auth whoami` phải ra `giangkh19` |

---

## 11. Bước tiếp theo

1. Thử `coder_r32` / `coder_qlora` so với `coder_r8`.
2. Thêm data lớn hơn (pull thêm Magicoder, giữ `max-seq-length 4000` + 1 epoch).
3. Serve: `python serving/server.py --model giangkh19/qwen-0.5b-coder-r8` (merge adapter trước nếu cần).
