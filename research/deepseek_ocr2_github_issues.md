# DeepSeek-OCR-2: Performance, Speed & Optimization Research

## Date: 2026-02-06

Sources: GitHub issues/repos, HuggingFace, vLLM docs, community guides, Rust implementation repos.

---

## 1. Official Speed Benchmarks

### DeepSeek-OCR (v1)
- **~2,500 tokens/second** for PDF processing on a single NVIDIA A100-40GB GPU
- **~200,000+ pages/day** on a single A100-40G GPU
- **~33 million pages/day** with a 20-GPU cluster
- **1-2 seconds per page** on A100 with Flash Attention
- **30-60 seconds per image** on RTX 4090 (significantly slower than A100)

### DeepSeek-OCR-2
- Released January 27, 2026 (3B parameters, BF16, Apache-2.0)
- Architecture: DeepEncoder V2 + DeepSeek3B-MoE-A570M decoder
- No official speed benchmarks published yet by DeepSeek for v2
- Community reports: vLLM inference is the fastest local option with batching
- Transformers-based inference is moderate speed, maximum flexibility

### Comparison vs Competitors
- **MistralOCR API**: Fastest (cloud-based, not local)
- **DeepSeek OCR 2 (vLLM)**: Fast local option with batching
- **DeepSeek OCR 2 (Transformers)**: Moderate speed
- **LightOnOCR-2-1B**: Claimed 1.73x faster than DeepSeek-OCR v1 (HuggingFace discussion, no independent verification)

---

## 2. Recommended Inference Settings

### Transformers Inference (Default)

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',   # CRITICAL for speed
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file=image_file,
    output_path=output_path,
    base_size=1024,      # Base resolution
    image_size=768,      # Target image size
    crop_mode=True,      # Enable document cropping
    save_results=True
)
```

### Key Parameters
- **base_size**: 1024 (default)
- **image_size**: 768 (standard), 640 (for Unsloth/smaller VRAM)
- **crop_mode**: True (recommended for documents)
- **Dynamic resolution**: (0-6) x 768x768 + 1 x 1024x1024 = (0-6) x 144 + 256 visual tokens

### Prompts
- **Layout-preserving OCR**: `<image>\n<|grounding|>Convert the document to markdown.`
- **Text-only OCR (faster, no layout)**: `<image>\nFree OCR.`
  - Note: Non-grounding prompts are faster but lose layout/table structure

---

## 3. vLLM Inference (Recommended for Production)

### Why vLLM
- Official recommendation for production speed
- Supports batched inference (multiple images simultaneously)
- Supports PDF concurrent processing
- "On-par speed with DeepSeek-OCR" for PDF processing
- DeepSeek-OCR is officially supported in upstream vLLM

### vLLM Configuration (from official config.py)

```python
# Key engine args
block_size = 256
max_model_len = 8192
gpu_memory_utilization = 0.75
tensor_parallel_size = 1

# Sampling params
temperature = 0.0          # Deterministic output
max_tokens = 8192           # Max generation length
# Custom logits processor to prevent repetition:
logits_processor = NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)
skip_special_tokens = False

# Image processing
BASE_SIZE = 1024
IMAGE_SIZE = 640            # Note: 640 in official vLLM config, 768 in transformers example
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6

# Concurrency
MAX_CONCURRENCY = 100
NUM_WORKERS = 64            # For image preprocessing (resize/pad)
```

### vLLM Installation

```bash
# Official recommended versions
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install flash-attn==2.7.3 --no-build-isolation
```

### vLLM Performance Tips (from vLLM Recipes docs)

1. **Use custom logits processor**: NGramPerReqLogitsProcessor is critical for OCR quality
2. **Disable prefix caching**: OCR tasks don't benefit from it; disabling avoids overhead
3. **Set `mm-processor-cache-gb` to 0**: For online serving
4. **Plain prompts > instruction format**: "DeepSeek-OCR works better with plain prompts"
5. **Tune `max_num_batched_tokens`**: Adjust based on GPU VRAM. Test with 16K, 32K, 64K, 96K
6. **Batch processing**: Process multiple images in a single batch for significantly better throughput
7. **VLLM_USE_V1=0**: The official scripts disable the v1 engine

### vLLM Version Compatibility

| Version | Status |
|---------|--------|
| vLLM 0.8.5 | Official supported version |
| vLLM 0.11.0+ | Requires custom adapter for logits processor (see Issue #231) |
| vLLM 0.13.0, 0.14.0, 0.14.1 | Supported via Kalorda project (github.com/vlmOCR/Kalorda) |
| Latest nightly | `uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly` |

---

## 4. Critical Bug: Text Repetition/Looping

### The Problem
Both DeepSeek-OCR and OCR-2 have a known failure mode where the model enters a repetition loop, outputting the same text block 10-20+ times. This is especially prevalent with:
- Multi-page PDFs
- Dense documents (financial statements, invoices, bank statements)
- Non-English/multilingual text
- The `<|grounding|>` tag intensifies the problem on dense documents

### Root Causes
1. Dense text regions compressing into insufficient visual tokens
2. Grounding mode layout detection conflicting with text extraction
3. Long documents exceeding internal context thresholds
4. **KV cache quantization**: Small models suffer significantly with quantized KV caches

### Workarounds

#### NGramPerReqLogitsProcessor (Primary Fix)
```python
from vllm import SamplingParams

# Standard settings
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=[NGramPerReqLogitsProcessor(
        ngram_size=30,     # Scans last 30 tokens for repeating sequences
        window_size=90,    # Sliding window over last 90 tokens
    )],
    whitelist_token_ids={128821, 128822}  # ChatML boundary tokens, exempt from penalty
)

# For edge cases (blank images, severe repetition):
# ngram_size=8, window_size=256
```

#### KV Cache Fix (Most Effective for Ollama/quantized deployments)
- Use **bf16** precision for KV cache, NOT quantized formats
- For vLLM: works reliably with proper precision
- For Ollama: `OLLAMA_KV_CACHE_TYPE=f16` may not fully resolve; vLLM recommended instead

#### Generation Config Fix (Partial)
```python
model.generation_config.max_new_tokens = 2048
base_size = 640
image_size = 640
crop_mode = True
```

#### Prompt Workaround
- Remove `<|grounding|>` tag for dense documents (loses layout info but prevents looping)

---

## 5. GPU Requirements & Memory

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| Full precision (bfloat16) | 16GB+ | RTX 4080 or better recommended |
| 4-bit quantization | 8GB+ | RTX 3070 or better |
| Fine-tuning | 24GB+ | Recommended minimum |
| vLLM serving (gpu_memory_utilization=0.75) | 16GB+ | Adjust utilization based on GPU |

### Multi-GPU Setup (DeepSeek-VL2 reference)
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="sequential",       # Sequential layer distribution
    offload_state_dict=True,
    max_memory={0: "44GIB", 1: "44GIB", "cpu": "20GIB"},
)
# IMPORTANT: Do NOT call .cuda() on Accelerate-dispatched models
model.eval()
```

### Incremental Prefilling (for limited VRAM)
- Use `--chunk_size 512` for 40GB GPUs (slower but fits in memory)
- Skip chunk_size on >40GB GPUs for faster inference

---

## 6. Alternative Inference Approaches

### Rust Implementation: deepseek-ocr.rs
- **Repo**: github.com/TimmyOVO/deepseek-ocr.rs
- Uses Candle tensor library + Rocket async web framework
- OpenAI API-compatible server + CLI
- **Benchmarks (vs Python, single-request MacOS)**:
  - Overall decode: **1.88x faster**
  - Token loop: **1.46x faster**
  - Prompt prefill: **1.83x faster**
  - Prompt tokenization: **97.42x faster**
  - Vision embedding: 0.62x (slower than Python)
- Supports DSQ quantization: Q4_K, Q6_K, Q8_0
- FP16 model size: ~6.3GB weights, ~13GB runtime
- **Metal** (Apple Silicon), **CUDA** (alpha), **Intel MKL** (preview) backends
- Zero Python runtime dependency
- Release builds mandatory (debug mode 10x slower)

### Rust Implementation: DeepSeek-OCR-2-burn
- **Repo**: github.com/huahuadeliaoliao/DeepSeek-OCR-2-burn
- Uses Burn framework (v0.20.1)
- Vulkan (wgpu) and CPU (NdArray) backends
- **Status**: Experimental, "performance not yet competitive with official Python stacks"

### GGUF / Ollama
- **NexaAI/DeepSeek-OCR-GGUF**: Available on HuggingFace
- **Ollama**: Requires v0.13.0+; `ollama run mike/deepseek-ocr`
- Q4_K_M or Q5_K_M quantization = best speed/quality balance
- **Warning**: GGUF/Ollama deployments more prone to repetition loop bug due to KV cache quantization

### Unsloth (Fine-tuning Optimization)
- **1.4x faster training, 40% less VRAM, 5x longer context**
- Works with both DeepSeek-OCR and OCR-2
- Uses image_size=640 (vs standard 768)
- Fine-tuning params: batch_size=2, gradient_accumulation=4, lr=2e-4, LoRA rank=16

---

## 7. Key GitHub Issues Summary

### DeepSeek-OCR-2 (github.com/deepseek-ai/DeepSeek-OCR-2/issues)

| Issue | Title | Status | Relevance |
|-------|-------|--------|-----------|
| #6 | How to deploy via vLLM | Open | Community request, no official guide yet |
| #13 | vllm-0.8.5 wheel installation failure | Open | Dependency resolution |
| #19 | Unofficial Rust (Burn) inference | Open | Alternative fast inference |
| #26 | Kalorda supports higher vLLM versions (0.13-0.14) | Open | vLLM compatibility |
| #28 | Text repetition/looping with multi-page PDFs | Open | Critical perf bug |
| #40 | Request to upgrade vLLM version | Open | Version lock concern |
| #41 | Flash-Attention install fails on T4 (Colab) | Open | Setup blocker |
| #42 | Word repetition during name generation | Open | Repetition variant |

### DeepSeek-OCR v1 (github.com/deepseek-ai/DeepSeek-OCR/issues) - Performance-related

| Issue | Title | Status | Relevance |
|-------|-------|--------|-----------|
| #77 | vLLM support | Open | vLLM serve didn't work initially; nightly build fixed |
| #170 | Error trying transformer inference | Open | Inference compatibility |
| #231 | Enable DeepSeek-OCR in vLLM 0.11.0 v1 engine | Open | Custom adapter needed for logits processor |
| #240 | [Solved] vLLM on RTX 5090 | Open | Installation order matters, "much faster than other models" |
| #299 | Triton CUDA illegal memory access on vLLM 0.11.2 | Open | vLLM compatibility bug |
| #300 | How to set MODE in upstream vLLM | Open | Configuration for different resolution modes |
| #307 | vLLM version issues | Open | Version compatibility |
| #311 | Can't run fine-tuned model on vLLM | Open | Fine-tuned model deployment |
| #318 | Comparison to LightOnOCR-2-1B | Open | Speed comparison request (unanswered) |

### DeepSeek-VL2 (github.com/deepseek-ai/DeepSeek-VL2/issues) - Performance-related

| Issue | Title | Status | Relevance |
|-------|-------|--------|-----------|
| #73 | Flash Attention 2 support for tiny/small | Closed | "Training too slow without FA2" |
| #132 | Running inference on multiple GPUs | Open | Multi-GPU setup with Accelerate |
| #148 | Inference time for vl2-small | Open | Timing question (unanswered) |

---

## 8. Practical Speed Optimization Checklist

### Must-Do
1. **Enable Flash Attention 2**: `_attn_implementation='flash_attention_2'` + `flash-attn==2.7.3`
2. **Use bfloat16**: `model.to(torch.bfloat16)` -- do NOT use float32
3. **Use vLLM for production**: Significantly faster than raw transformers inference
4. **Use NGramPerReqLogitsProcessor**: Prevents repetition that wastes tokens/time

### Should-Do
5. **Batch process images**: vLLM batch inference >> sequential processing
6. **Tune max_num_batched_tokens**: Start at 32K, test up to 96K based on GPU VRAM
7. **Disable prefix caching**: No benefit for OCR tasks, adds overhead
8. **Set temperature=0.0**: Deterministic, no wasted computation on sampling
9. **Use crop_mode=True**: Better document handling
10. **Set appropriate max_tokens**: 8192 for offline, 2048 for serving (avoid unnecessary generation)

### Consider
11. **Reduce image_size to 640**: Fewer visual tokens, faster processing, slight quality tradeoff
12. **Use text-only prompt** (`Free OCR.`) when layout isn't needed: Faster, avoids grounding bugs
13. **Try Rust implementation** (deepseek-ocr.rs): 1.5-2x faster decode on supported platforms
14. **GGUF quantization** (Q4_K_M): For VRAM-constrained setups, but watch for repetition bugs
15. **Unsloth for fine-tuning**: 1.4x faster training, 40% less VRAM

### Avoid
- Float32 precision (unnecessary, 2x slower)
- KV cache quantization on small models (causes repetition loops)
- `<|grounding|>` tag on very dense multi-page documents (triggers looping)
- Debug builds of Rust implementations (10x slower)
- vLLM prefix caching for OCR workloads (overhead with no benefit)
- Calling `.cuda()` on Accelerate-dispatched multi-GPU models

---

## 9. Environment & Dependency Reference

```bash
# Tested environment
Python 3.12.9
CUDA 11.8
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
transformers==4.46.3 to 4.52.4
tokenizers==0.20.3
vllm==0.8.5 (official) or nightly
flash-attn==2.7.3
einops
addict
easydict
```

### RTX 5090 / CUDA 12.8 Setup (from Issue #240)
1. Install nightly xformers first
2. Use pre-built wheels for flash-attention and vLLM (don't compile from source)
3. Install vLLM wheel with `--no-build-isolation --no-deps`
4. If torchvision breaks torch, reinstall xformers to restore correct torch version
5. Test iteratively: `python -c "import vllm; print(vllm.__version__)"`

---

## 10. Sources

### GitHub Repositories
- [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)
- [deepseek-ocr.rs (Rust)](https://github.com/TimmyOVO/deepseek-ocr.rs)
- [DeepSeek-OCR-2-burn (Rust/Burn)](https://github.com/huahuadeliaoliao/DeepSeek-OCR-2-burn)
- [Kalorda (higher vLLM versions)](https://github.com/vlmOCR/Kalorda)

### GitHub Issues (Key)
- [OCR-2 #28: Text repetition/looping](https://github.com/deepseek-ai/DeepSeek-OCR-2/issues/28)
- [OCR-2 #26: Kalorda higher vLLM support](https://github.com/deepseek-ai/DeepSeek-OCR-2/issues/26)
- [OCR #77: vLLM support](https://github.com/deepseek-ai/DeepSeek-OCR/issues/77)
- [OCR #231: vLLM 0.11.0 custom modifications](https://github.com/deepseek-ai/DeepSeek-OCR/issues/231)
- [OCR #240: RTX 5090 vLLM setup](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
- [VL2 #73: Flash Attention 2 speed](https://github.com/deepseek-ai/DeepSeek-VL2/issues/73)
- [VL2 #132: Multi-GPU inference](https://github.com/deepseek-ai/DeepSeek-VL2/issues/132)

### HuggingFace
- [DeepSeek-OCR-2 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR Repetition Discussion](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/89)
- [NexaAI/DeepSeek-OCR-GGUF](https://huggingface.co/NexaAI/DeepSeek-OCR-GGUF)
- [Unsloth/DeepSeek-OCR-2](https://huggingface.co/unsloth/DeepSeek-OCR-2)

### Documentation & Guides
- [vLLM Recipes: DeepSeek-OCR](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [Unsloth: DeepSeek-OCR-2 Fine-tune Guide](https://unsloth.ai/docs/models/deepseek-ocr-2)
- [DEV.to: Complete Guide 2026](https://dev.to/czmilo/deepseek-ocr-2-complete-guide-to-running-fine-tuning-in-2026-3odb)
