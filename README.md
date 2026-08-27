# Finetune Qwen2.5-Coder-0.5B — Text-to-Pandas Lab Guide

Fine-tune **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** (494M) trên **~70.7k text-to-pandas** (WikiSQL-style: bảng + câu hỏi → `result = table[...]`), đo **trước / sau** bằng benchmark pandas **6,000 câu held-out** (exact match), so sánh LoRA.

> Model mục tiêu: **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** — base `Qwen2ForCausalLM` + chat template `<|im_start|>`, hợp domain vì output là code pandas.

---

## 1. Repo structure

```
data/
  train/pandas_train_80k.jsonl # 80,000 rows text-to-pandas (channudambal + Rahima, chưa dedup)
  raw/                       # CSV nguồn (gitignored): channudambal 10k + Rahima 76k
  eval/                      # FROZEN — ĐỪNG SỬA
    pandas_eval_6k.jsonl     # 6,000 items, checker=exact (benchmark chính)
    pandas_eval_1000.jsonl   # 1,000 items — quick loop (mẫu độc lập, KHÔNG phải prefix của 6k)
    pandas_eval_500.jsonl    # 500 items — smoke test (mẫu độc lập)
    manifest.json            # sha256 — chống sửa
harness/
  run_eval.py       # generate + score (direct --model hoặc API --api, batch 40)
  scorers.py        # exact (text-to-pandas), + numeric/choice/constraint/unit_test (legacy)
  leak_check.py     # eval không memorize từ train (exact prompt + skeleton/answer)
  compare.py        # gộp nhiều run → bảng markdown (cột suite động)
serving/server.py   # FastAPI OpenAI-compatible (fallback khi không dùng vLLM)
tools/
  convert_pandas_all.py      # builder: raw CSVs → train 80k + eval 6k + manifest
train.py            # LoRA/QLoRA finetune (standard Trainer, completion-only masking)
results/            # output benchmark (<tag>/result.json, summary.md, outputs.jsonl)
outputs/            # adapter LoRA sau train (<tag>/adapter_config.json + safetensors ~17MB)
requirements.txt
```

---

## 2. Chuẩn bị GPU — 2 nhánh theo driver

**Nhánh A — máy RTX 4070 driver 550 (CUDA 12.4 max):**

| Thành phần | Giá trị | Lý do |
|------------|---------|-------|
| GPU | RTX 4070 16GB | máy hiện có |
| Driver | 550.142 | `nvidia-smi` báo "CUDA Version: 12.4" = mức driver này hỗ trợ |
| CUDA runtime | **12.4** | driver 550 chưa đạt chuẩn CUDA 12.8 (cần ≥570) / 12.9 (≥575) / 13.0 (≥580) |
| Python | 3.10 | |
| PyTorch | **2.6.0+cu124** | bản cuối có wheel cu124 |
| vLLM | **0.8.5** | khớp torch 2.6.0+cu124 |

**Không lên vLLM 0.27/0.28 với driver 550:** wheel vLLM mới kéo torch 2.13 → CUDA 13 → `ncclCommResume`/driver fail trên 550. `cuda-compat` workaround chỉ áp dụng GPU datacenter, không áp dụng cho 4070 (consumer). Muốn vLLM mới thì nâng driver ≥580 — nhưng **không cần** cho Qwen2.5-Coder-0.5B: 0.8.5 serve đủ, model quá nhỏ để cần feature mới.

### Nhánh B — máy thuê driver ≥ 580 (vd 3090, driver 590 / CUDA 13.1): dùng vLLM MỚI NHẤT

Không cần pin gì cả:
```bash
vllm --version   # image đã bundle sẵn? → dùng luôn, KHÔNG pip install lại
# nếu chưa có:
pip3 install -U vllm   # torch cu13 tự kéo theo, driver 590 hỗ trợ native
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Serve giống hệt dưới, giữ nguyên args. `pip install -r requirements.txt` sau đó không được đè torch (`torch>=2.1` đã thỏa).

**Không cần `nvcc`:** serving bằng torch/vLLM binary không cần CUDA Toolkit; chỉ JIT kernel (flashinfer build v.v.) mới cần, 0.8.5 không bắt.

### Cài môi trường (làm đúng thứ tự)

```bash
rm -rf .venv
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip

# 1) PyTorch cu124 TRƯỚC
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 2) Verify torch OK rồi mới cài vLLM
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# kỳ vọng: 2.6.0+cu124  12.4  True  NVIDIA GeForce RTX 4070

pip install vllm==0.8.5
python3 -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
# kỳ vọng: 2.6.0+cu124  12.4  0.8.5
```

> Nếu `pip install vllm==0.8.5` kéo torch bản khác về đè: cài thêm `--extra-index-url https://download.pytorch.org/whl/cu124` và kiểm tra lại `torch.__version__` phải còn `+cu124`.

### Serve

```bash
vllm serve Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen-coder \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enable-prefix-caching
```
* `0.8` ≈ 12.8 GB / 16 GB — model 0.5B (~1GB bf16) nên phần lớn là KV cache, rất dư
* `--max-model-len 8192` đủ cho prompt pandas (~350 tok) + headroom
* Không cần tensor-parallel / quantization

### Clone repo (sau khi serve OK)

```bash
cd ~
git clone https://github.com/giangkh1908/Finetune_Qwen2.5-0.5B.git
cd Finetune_Qwen2.5-0.5B
pip3 install -r requirements.txt -q  # harness/tools; torch+vllm đã cài ở trên, KHÔNG pip install lại vllm
python3 harness/leak_check.py          # xem ghi chú overlap §8 (báo FAIL có chủ đích với split hiện tại)
```

> Train (`§4`) cần thêm `peft transformers datasets accelerate bitsandbytes` — đã nằm trong `requirements.txt`. Nếu train chung venv với vLLM 0.8.5 bị xung đột phiên bản, tạo venv thứ hai cho train (train chỉ cần torch cu124 + HF stack, không cần vLLM).

---

## 3. Baseline BEFORE (6k pandas, exact match) — qua vLLM

```bash
# Sau khi `vllm serve` ở §2 đã chạy trên :8000. Test:
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-coder","messages":[{"role":"user","content":"Table Name: t (a (int64), b (object))\nWhat is b when a is 3?"}],"max_tokens":64,"temperature":0}'

# Smoke test trước (500 câu, ~1 phút):
python3 harness/run_eval.py --api http://localhost:8000/v1 --api-model qwen-coder --tag pandas_base_smoke --suites pandas_eval_500.jsonl --batch-size 40

# Benchmark chính (6,000 câu, batch 40 = 40 request concurrent)
python3 harness/run_eval.py --api http://localhost:8000/v1 --api-model qwen-coder --tag pandas_base --batch-size 40
```

Kết quả: `results/pandas_base/result.json` + `summary.md`.

> Fallback không vLLM (direct): `python3 harness/run_eval.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tag pandas_base` (chậm hơn nhiều — không continuous batching) hoặc `python3 serving/server.py` (transformers, serialize generation).

Transfer về Windows:
```powershell
mkdir D:\Finetune\results\pandas_base 2>$null
scp -P <PORT> <USER>@<GPU_IP>:/root/Finetune_Qwen2.5-0.5B/results/pandas_base/result.json D:\Finetune\results\pandas_base\result.json
```

---

## 4. Finetune (LoRA / QLoRA)

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `--r` | LoRA rank. Cao = học mạnh, dễ overfit | **8** điểm ngọt cho 80k rows |
| `--alpha` | scaling (thường 2×r) | 16 (r=8) |
| `--max-seq-length` | Giới hạn token/sample. **Nguồn VRAM chính** | **512** (pandas samples max ~350 tok → **không truncate**) |
| `--epochs` | Số vòng qua dataset | **1** (80k rows, không cần lặp) |
| `--batch-size` / `--grad-accum` | Batch thực/effective | 2 / 8 (seq ngắn, nhét được 2) |
| `--lr` | Tốc độ học | 2e-4 |
| `--qlora` | Nén 4-bit → giảm 40–60% VRAM | Không cần ở 0.5B + seq 512 |

```bash
# LoRA r=8 — chính
python3 train.py --r 8 --epochs 1 --tag pandas_r8

# Smoke 200 bước trước khi commit cả epoch (optional): thêm --epochs 1 rồi kill sau ~200 step, hoặc test nhanh bằng:
python3 train.py --r 8 --epochs 1 --tag pandas_r8_smoke --max-seq-length 512

# Thử r=32 (mạnh hơn, dễ overfit pattern có sẵn)
python3 train.py --r 32 --alpha 64 --epochs 1 --tag pandas_r32
```

Output: `outputs/<tag>/` — adapter ~17MB (r=8) / ~70MB (r=32). Log `train_loss` giảm nhanh trong vài trăm step đầu là đang học; `grad_norm` 0.2–1.0 là ổn.

> Train đã bật `gradient_checkpointing` + `enable_input_require_grads()`.
> 80k rows × seq 512 trên 4070 16GB: ước **1.5–3 giờ** cho 1 epoch (batch 2). Không phải 20 phút như 6.4k rows code cũ — tính trước thời gian nếu thuê GPU theo giờ.

---

## 5. Benchmark AFTER

Adapter LoRA phải **merge trước khi serve qua vLLM** (vLLM API mode không nhận `--adapter`):

```bash
# 1) Merge adapter vào base → saved model
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base='Qwen/Qwen2.5-Coder-0.5B-Instruct'
m=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype='auto'), 'outputs/pandas_r8').merge_and_unload()
m.save_pretrained('merged_pandas_r8'); AutoTokenizer.from_pretrained(base).save_pretrained('merged_pandas_r8')"

# 2) Serve model đã merge
vllm serve merged_pandas_r8 --host 0.0.0.0 --port 8000 --served-model-name qwen-coder \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enable-prefix-caching

# 3) Benchmark
python3 harness/run_eval.py --api http://localhost:8000/v1 --api-model qwen-coder --tag pandas_r8 --batch-size 40
```

Hoặc chạy direct không vLLM (không cần merge, PEFT tự load):
```bash
python3 harness/run_eval.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter outputs/pandas_r8 --tag pandas_r8
```

Windows:
```powershell
mkdir D:\Finetune\results\pandas_r8 2>$null
scp -P <PORT> <USER>@<GPU_IP>:/root/Finetune_Qwen2.5-0.5B/results/pandas_r8/result.json D:\Finetune\results\pandas_r8\result.json
```

---

## 6. So sánh

```bash
python3 harness/compare.py pandas_base pandas_r8
# | Model | pandas_eval_6k | Avg |
# |---|---|---|
# | pandas_base | .. | .. |
# | pandas_r8   | .. | .. |
```

---

## 7. Kết quả thực nghiệm

| Model | Data | Eval | Kết quả | Δ vs base |
|-------|------|------|---------|-----------|
| Qwen-Coder 0.5B base | — | 6k pandas | *(chưa chạy — điền sau §3)* | — |
| **+ LoRA r=8** | 80k pandas (lưu ý overlap §8) | 6k pandas | *(chưa chạy — điền sau §5)* | — |

> Kinh nghiệm từ thí nghiệm code cũ (đã archive trong git history): catastrophic forgetting xảy ra khi data lệch domain + truncate 40% sample; sửa bằng data đúng domain, chunk/không truncate, 1 epoch. Ở pandas: seq 512 phủ 100% samples (max ~350 tok) nên không có rủi ro truncate.

**Lưu ý scoring:** `checker=exact` so answer `result = table...` với response (normalize thường + substring word-boundary). Model trả lời dài dòng nhưng chứa đúng query vẫn được tính đúng — đó là chủ ý (tolerate format), nhưng cũng nghĩa là điểm hơi dễ dãi nếu model "đoán" query ngắn.

---

## 8. Dataset

**Train:** `data/train/pandas_train_80k.jsonl` — **80,000 rows**, format `{"messages":[user, assistant]}`. Gộp 2 nguồn, chưa dedup.

**Eval:** `data/eval/pandas_eval_6k.jsonl` — **6,000 items** `{id, prompt, answer, checker:"exact", ...}`, FROZEN theo `manifest.json` (sha256 content-hash, line-ending agnostic; `run_eval.py` tự verify mỗi lần chạy). `pandas_eval_1000/500.jsonl` là **mẫu nhanh độc lập** (id trùng nhưng nội dung không phải prefix của 6k) — dùng cho vòng lặp debug, **không dùng làm benchmark chính**.

Nguồn (gộp 2 dataset, `data/raw/`):

| Nguồn | Rows raw | Mô tả |
|-------|----------|-------|
| channudambal text-to-pandas | ~9.9k | phức tạp hơn (median ~329 chars) |
| Rahima train + test | ~76.7k | ngắn (median ~205 chars) |

`tools/convert_pandas_all.py` (seed 20261101): shuffle toàn bộ 86.7k rows → `rows[:80000]` train, `rows[80000:86000]` eval. **Split theo dòng, không theo prompt.**

### ⚠️ Overlap đã biết giữa train 80k và eval 6k

Vì raw 2 nguồn trùng nhau nhiều, split-theo-dóng để lọt:

| Chỉ số | Giá trị |
|--------|---------|
| Eval prompt xuất hiện nguyên văn trong train | **699 / 6,000 (11.7%)** |
| Trong đó (prompt, answer) trùng nguyên cặp (memorization) | **606** |
| Eval items có ≥2 phiên bản answer khác nhau trong chính train (prompt conflict) | ~307 prompt overlap, một phần do answer mâu thuẫn giữa 2 nguồn |
| Trùng nội bộ train: 80,000 rows chỉ ~73.8k unique pairs | ~6,157 dòng trùng |

Hệ quả: điểm AFTER trên ~12% eval items là **điểm nhớ lại**, báo cáo cao hơn thực lực tổng quát hóa. `harness/leak_check.py` vì vậy báo **FAIL (872 exact matches tính cả 500/1000 subsets)** — đó là chẩn đoán đúng, không phải bug tool.

2 hướng xử lý (chọn 1, chưa làm):
1. **Giữ nguyên, chấp nhận caveat** — so sánh base vs r8 trên cùng 6k vẫn công bằng tương đối (cùng benefit), chỉ tuyệt đối hóa cao. Ghi chú vào mọi báo cáo.
2. **Rebuild leak-free** — dedup (prompt,answer), drop ~491 prompt conflict, split theo prompt: ~70.7k train + 6k eval prompt-disjoint, leak_check PASS. (Đã proof-of-concept; cần regenerate + chạy lại baseline — chưa commit vì dataset hiện tại là của bạn, đang giữ nguyên.)

Build lại như hiện tại (deterministic, seed 20261101):
```bash
python3 tools/convert_pandas_all.py
python3 harness/leak_check.py   # FAIL = overlap 699 prompts, xem bảng trên
```

---

## 9. Push model lên Hugging Face

Tạo repo mới **`giangkh19/qwen-0.5b-pandas-r8`** (HF namespace là `giangkh19`, không phải `giangkh1908`):

```powershell
# Windows — tạo Classic Write token tại https://huggingface.co/settings/tokens
hf auth logout
hf auth login   # chọn Paste, dán Classic Write token

hf repo-create giangkh19/qwen-0.5b-pandas-r8 --repo-type model 2>$null  # repo đã có thì bỏ lệnh này
hf upload giangkh19/qwen-0.5b-pandas-r8 outputs/pandas_r8 --repo-type model
```

Dùng lại:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "giangkh19/qwen-0.5b-pandas-r8")
```

Adapter code cũ `giangkh19/qwen-0.5b-coder-r8` vẫn còn trên HF (thí nghiệm trước, không liên quan pipeline pandas này).

---

## 10. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| vLLM startup fail `ncclCommResume` / undefined symbol | torch bị kéo bản cu12.9/cu13 (mặc định PyPI) trong khi driver chỉ 550 — cài lại đúng §2: torch `2.6.0+cu124` + `vllm==0.8.5`. Đừng `pip install -U vllm` |
| `pip install vllm` kéo torch mới đè cu124 | Dùng `pip install vllm==0.8.5 --extra-index-url https://download.pytorch.org/whl/cu124`, kiểm tra `torch.__version__` còn `+cu124` |
| Muốn dùng vLLM 0.27/0.28 mới | Bắt buộc nâng driver ≥580 (CUDA 13) — provider cho đổi image thì đổi, không có workaround cho 4070 |
| `ERROR: manifest mismatch` | `data/eval/` đã bị sửa sau freeze — `git checkout data/eval/` khôi phục, đừng tự vá hash |
| `No space left` | torch cu124 + vllm ~8GB; dọn `~/.cache/pip` hoặc clone model bằng `HF_HOME` chỉ định ổ còn trống |
| `python: command not found` | `python3` |
| VRAM OOM khi train | 0.5B + seq 512 cần <10GB; nếu OOM check `nvidia-smi` process cũ, `kill -9 <PID>` |
| `run_eval` API mode chậm | Tăng `--batch-size` (vLLM continuous batching chịu được 64–128 với 0.5B) |
| Kết quả `--api` bị 401/404 | Đúng path phải là `.../v1`; `--api-model` phải khớp `--served-model-name` |
| `git pull` báo file đè | `rm -rf results/<tag>` rồi `git pull` |
| scp hỏi password | Chạy riêng trong PowerShell/CMD, nhập thủ công |
| `403 Forbidden` khi push HF | Token phải **Classic Write** (không phải read), `hf auth whoami` phải ra `giangkh19` |

---

## 11. Bước tiếp theo

1. Chạy §3 → §6 với data pandas mới, điền kết quả vào §7.
2. So `pandas_r8` vs `pandas_r32` (pattern hẹp hơn code — r nhỏ có thể đã đủ).
3. Cân nhắc scorer nghiêm ngặt hơn: `unit_test`-style execute query trên dataframe dựng từ schema trong prompt (exact-match hiện tại tolerate format).
4. Serve: merge adapter (§5 bước 1) rồi `vllm serve merged_pandas_r8`, hoặc `MODEL=merged_pandas_r8 python serving/server.py`.
