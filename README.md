# Finetune Qwen2.5-0.5B-Instruct — Lab Guide

Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` trên tập dữ liệu suy luận của Claude, đo
chất lượng **trước / sau** bằng benchmark frozen, và so sánh nhiều cấu hình LoRA.

> Mô hình mục tiêu: **`Qwen/Qwen2.5-0.5B-Instruct`** (bản chat — đúng template cho SFT).

---

## 1. Repo structure

```
data/
  train/claude_opus_743_clean.jsonl    # 743 sample (đã loại 15 dòng assistant rỗng)
  eval/                                # FROZEN benchmark — ĐỪNG SỬA
    math.jsonl          50 item  numeric (answer computed)
    reasoning.jsonl     40 item  exact  (ground truth verified bằng brute-force)
    coding.jsonl        24 item  unit_test (chạy Python VM)
    general.jsonl       36 item  choice (24 MCQ) + constraint (12 checker)
    qualitative.jsonl   24 prompt  (no GT — cho human diff)
    manifest.json       sha256 + mô tả — chống sửa khi experiment
harness/
  run_eval.py       # generate + score (direct hoặc API)
  scorers.py        # numeric/exact/choice/constraint/unit_test
  leak_check.py     # kiểm tra eval không leak từ train
  compare.py        # gộp nhiều run → bảng markdown
serving/server.py  # FastAPI OpenAI-compatible (thay vLLM)
tools/
  build_eval.py     # builder benchmark (deterministic, SEED 20260826)
  gen_dummy.py      # sinh output giả để smoke-test
train.py            # LoRA/QLoRA finetune
results/            # output của từng lần benchmark (<tag>/result.json,...)
outputs/            # adapter LoRA sau khi train (<tag>/)
requirements.txt
```

---

## 2. Chuẩn bị máy GPU (thuê)

Yêu cầu: GPU ≥ 8GB (24GB là thoải mái). Ví dụ SSH: `ssh -p 1757 root@n3.ckey.vn`

```bash
# 1. Clone repo
cd ~
git clone https://github.com/giangkh1908/Finetune_Qwen2.5-0.5B.git
cd Finetune_Qwen2.5-0.5B

# 2. Cài Python 3.10+ (vLLM cần; nếu muốn dùng serving/server.py thì chỉ cần 3.8+)
# Ubuntu 20.04 mặc định 3.8 → dùng conda:
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh
bash miniforge.sh -b -p ~/miniforge
source ~/miniforge/etc/profile.d/conda.sh
conda create -y -n llm python=3.10 && conda activate llm

# 3. Cài deps
pip install -r requirements.txt
pip install bitsandbytes huggingface_hub   # nếu muốn QLoRA / push HF
```

---

## 3. Bước 0 — Chạy baseline BEFORE (trên GPU)

Đo model **gốc chưa finetune**. Có 2 cách:

### 3a. Direct (đơn giản, chạy thẳng trên GPU)

```bash
python3 harness/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --tag base_before
```

### 3b. Serve API (nếu muốn máy khác gọi)

```bash
# Terminal 1 — mở server (không cần vLLM/nvcc)
python3 serving/server.py          # lắng nghe http://0.0.0.0:8000

# Terminal 2 — benchmark qua API
python3 harness/run_eval.py --api http://localhost:8000/v1 --tag base_before --batch-size 40
```

> ⚠️ **Đừng dùng vLLM** trên máy này: vLLM 0.28 cần FlashInfer JIT-compile CUDA
> kernel → đòi `nvcc` + `ninja`, máy không có. `serving/server.py` (transformers
> thuần) là thay thế nhẹ, cùng endpoint OpenAI-compatible.

Kết quả ghi `results/base_before/result.json`. Transfer về máy Windows:

```powershell
mkdir D:\Finetune\results\base_before
scp -P 1757 root@n3.ckey.vn:/root/Finetune_Qwen2.5-0.5B/results/base_before/result.json D:\Finetune\results\base_before\result.json
```

---

## 4. Bước 1 — Finetune (LoRA / QLoRA)

### Các tham số quan trọng

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `--r` | LoRA rank. Cao = học mạnh nhưng dễ overfit | r=8 điểm ngọt cho dataset nhỏ; r=32 mạnh hơn |
| `--alpha` | scaling (thường = 2× r) | 16 (khi r=8) |
| `--max-seq-length` | cắt sample xuống tối đa N token. **Nguồn VRAM chính** | 4096 (24GB); 8192 giữ nhiều hơn |
| `--epochs` | số vòng qua toàn bộ dataset | 1–3 (dataset nhỏ: 3 là an toàn) |
| `--batch-size` | số sample/GPU mỗi bước | 1 |
| `--grad-accum` | gộp N bước trước khi update | 8 (≈ effective batch 8) |
| `--lr` | tốc độ học | 2e-4 |
| `--qlora` | nén 4-bit → giảm 40–60% VRAM | thêm khi bị chạm trần VRAM |

### Chạy

```bash
# LoRA r=8
python3 train.py --r 8 --max-seq-length 4096 --tag lora_r8

# LoRA r=32
python3 train.py --r 32 --max-seq-length 4096 --tag lora_r32

# QLoRA 4-bit (cần bitsandbytes)
python3 train.py --r 32 --qlora --max-seq-length 4096 --tag qlora_r32
```

Kết quả: `outputs/<tag>/` (adapter LoRA nhỏ ~10–20MB, **không phải** model 1GB).

**Cách đọc log:** `loss` giảm dần = đang học. `grad_norm` ổn định ≈ 0.2–0.5.
`epoch` = tiến độ (1.0 = qua 1 vòng). **Loss thấp ≠ giỏi** — phải benchmark mới biết.

---

## 5. Bước 2 — Benchmark AFTER

```bash
python3 harness/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/lora_r8 --tag lora_r8
```

(hoặc qua API: `--adapter` phải được merge ở server; direct là đơn giản nhất)

Transfer về Windows:

```powershell
mkdir D:\Finetune\results\lora_r8 2>$null
scp -P 1757 root@n3.ckey.vn:/root/Finetune_Qwen2.5-0.5B/results/lora_r8/result.json D:\Finetune\results\lora_r8\result.json
```

---

## 6. Bước 3 — So sánh

Trên Windows (hoặc GPU đã pull hết results về):

```bash
python3 harness/compare.py base_before lora_r8 lora_r32
# → bảng:
# | Model | Math | Coding | Reasoning | General | Avg |
# | base_before | 58 | 71 | 50 | 64 | 60.8 |
# | lora_r8 | 22 | 48 | 40 | 64 | 43.5 |
```

---

## 7. Kết quả thực nghiệm (lora_r8 vs base)

| Suite | base_before | lora_r8 | Δ |
|-------|-------------|---------|---|
| math | 58.0 | 22.0 | **−36** |
| reasoning | 50.0 | 40.0 | **−10** |
| coding | 71.2 | 48.3 | **−23** |
| general | 63.9 | 63.9 | 0 |
| **overall** | **60.8** | **43.5** | **−17** |

**Chẩn đoán: catastrophic forgetting / overfit.** Tất cả mảng trừ general đều tụt.

- Math sai toàn bộ (model đoán số, ví dụ dự đoán `900` thay vì `-261`).
- Coding lỗi syntax — model giờ **in text kiểu Claude** ("Tie-breaking rule")
  thay vì code thuần. Tức là đã thuộc **style lời văn** của dataset, mất kỹ năng
  **output code**.

**Nguyên nhân chính:**
1. Dataset train 100% code template (production-style) — khác hẳn eval (Python function ngắn).
2. 743 sample nhỏ + 3 epochs + LoRA r=8 → học vẹt.
3. `max-seq-length 4096` cắt ~40% dữ liệu khi sample median 6.5K token.

**Thử cải thiện (theo thứ tự):**

```bash
# giảm epochs chống overfit
python3 train.py --r 8 --max-seq-length 4096 --epochs 1 --tag lora_r8_e1
# giữ hơn nữa mỗi sample
python3 train.py --r 8 --max-seq-length 8192 --epochs 3 --tag lora_r8_s8k
# LoRA rank thấp hơn
python3 train.py --r 4 --max-seq-length 4096 --epochs 2 --tag lora_r4
```

Nếu vẫn tụt so với base → dataset quá chuyên biệt cho model 0.5B, hoặc cần trộn
thêm dữ liệu tổng quát (xem §8).

---

## 8. Mẹo dành cho dataset lớn / tránh quên

- **Mix data:** gộp nhiều nguồn vào 1 lần train (tránh finetune chồng làm
  catastrophic forgetting). Cần thêm flag `--data-mix` vào `train.py`.
- **Dataset lớn:** giảm `max-seq-length` + `--qlora`, tăng `--batch-size`,
  giảm `--epochs` xuống 1.
- **Model 0.5B giới hạn:** với > vài chục triệu token, 0.5B không học nổi tốt
  → cân nhắc nâng model (7B/32B) hoặc dùng LoRA rank nhỏ + 1 epoch.

---

## 9. Chia sẻ model lên Hugging Face

Adapter LoRA nhỏ → push lên HF (chuẩn):

```bash
# trên GPU
pip install huggingface_hub
huggingface-cli login        # token: https://huggingface.co/settings/tokens

# đổi USERNAME/REPO theo tài khoản của bạn
huggingface-cli upload giangkh1908/qwen-0.5b-lora-r8 outputs/lora_r8 --repo-type model
```

Dùng lại từ bất kỳ đâu:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "giangkh1908/qwen-0.5b-lora-r8")
```

---

## 10. Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `python: command not found` | dùng `python3` |
| `No module named vllm` | `source ~/miniforge/etc/profile.d/conda.sh && conda activate llm` |
| vLLM lỗi `nvcc not found` | **bỏ vLLM**, dùng `serving/server.py` |
| VRAM đầy trong train | giảm `--max-seq-length`, thêm `--qlora` |
| Full VRAM bất thường (0.5B mà 16GB) | kiểm tra process cũ `nvidia-smi` / `ps aux | grep python` |
| `git pull` báo file đè | `rm -rf results/base_before && git pull` |
| scp trên Windows hỏi password | mở **Command Prompt/PowerShell** riêng, nhập password thủ công |
| `[transformers] 'temperature' not valid` | vô hại — generation dùng `do_sample=False` (greedy) |

---

## 11. Bước tiếp theo

1. Tải adapter `outputs/lora_r8/` về Windows + commit kết quả lên repo.
2. Chạy `lora_r32` + `qlora_r32` để so sánh 4 cấu hình.
3. Đánh giá qualitative (so sánh `outputs_qualitative.jsonl` giữa base và các
   bản finetune) — xem model có giữ style dài, có lặp, có hallucinate không.
