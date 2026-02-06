# DeepSeek-OCR-2 Speed Optimization Research

**Date:** 2026-02-06
**Problem:** 3B parameter VLM on RTX 3090 (24GB), ~7.5 sec/page, bottleneck is autoregressive text generation (~7ms/token, ~1000 tokens/page)
**Target:** Significant reduction in per-page latency

---

## Current Architecture Understanding

Your Rust app is an HTTP client that sends entire PDFs to a Python server running DeepSeek-OCR-2. The server does the heavy lifting: image extraction, vision encoding, and autoregressive text generation. The bottleneck is **decode-phase latency** -- each output token takes ~7ms, and a typical page generates ~1000 tokens, so generation alone is ~7 seconds.

The model uses:
- **DeepEncoder V2** with visual causal flow (reorders visual tokens by semantic structure)
- **Qwen2-0.5B** repurposed as vision encoder
- **MoE backbone with Multi-Head Latent Attention (MLA)** -- compresses KV cache via low-rank projections
- **256-1,120 visual input tokens** per image (already efficient vs. competitors using 6,000-7,000)

---

## Fix 1: Switch from Transformers to vLLM (Highest Impact)

**Expected speedup: 3-5x**

If you are running inference through HuggingFace Transformers (`AutoModel.from_pretrained`), you are leaving massive performance on the table. vLLM provides:

- **PagedAttention** -- near-zero KV cache memory waste
- **Continuous batching** -- GPU stays saturated between requests
- **CUDA graphs** -- eliminates kernel launch overhead on decode steps (the exact bottleneck)
- **Optimized attention kernels**

### vLLM Offline Inference Setup

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR-2",
    enable_prefix_caching=False,       # CRITICAL: disable for OCR workloads
    mm_processor_cache_gb=0,           # CRITICAL: disable image caching
    logits_processors=[NGramPerReqLogitsProcessor],
    trust_remote_code=True,
    gpu_memory_utilization=0.92,       # RTX 3090: use most of 24GB
    max_model_len=16384,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td> and </td> for tables
    ),
)
```

### vLLM HTTP Server Setup

```bash
vllm serve deepseek-ai/DeepSeek-OCR-2 \
    --host 0.0.0.0 --port 8000 \
    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92
```

### Why this matters for your bottleneck

CUDA graphs compile the decode step into a single fused operation. Instead of launching dozens of individual CUDA kernels per token (attention, FFN, sampling, etc.), the entire decode step runs as one graph replay. This directly attacks the 7ms/token overhead by reducing kernel launch latency and memory access patterns.

### Required environment

```bash
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.8.5  # or download specific wheel from DeepSeek repo
pip install flash-attn==2.7.3 --no-build-isolation
```

**Pin these versions exactly.** Mismatches between torch/flash-attn/vLLM cause cryptic runtime errors.

---

## Fix 2: Batch Multiple Pages Concurrently (High Impact)

**Expected speedup: 2-4x throughput (not per-page latency)**

Instead of processing pages sequentially, send multiple page images in a single batch:

```python
# vLLM batch inference
model_input = [
    {"prompt": prompt, "multi_modal_data": {"image": page_image_1}},
    {"prompt": prompt, "multi_modal_data": {"image": page_image_2}},
    {"prompt": prompt, "multi_modal_data": {"image": page_image_3}},
    {"prompt": prompt, "multi_modal_data": {"image": page_image_4}},
]
outputs = llm.generate(model_input, sampling_params)
```

With vLLM's continuous batching, the GPU processes decode steps for multiple pages simultaneously. The model is only 3B params (~6GB in bf16), so on a 24GB RTX 3090 you have ~18GB for KV cache, which can hold several concurrent sequences.

### Pipeline architecture for PDFs

The official DeepSeek-OCR code uses `ThreadPoolExecutor` for CPU-bound preprocessing (PDF-to-image conversion) and batches GPU inference:

1. **CPU threads** -- Convert PDF pages to images in parallel
2. **GPU batch** -- Feed all page images to vLLM in one batch
3. **Post-processing** -- Assemble markdown output

This pipeline keeps the GPU busy while CPU handles I/O.

### Tuning batch size on RTX 3090

```
Model weights:           ~6 GB (bf16)
Flash-attn workspace:    ~0.5 GB
Available for KV cache:  ~15-17 GB (at 0.92 utilization)

Per sequence KV cache (max_model_len=16384):
  MLA compressed cache ≈ much less than standard MHA
  Estimate: can fit 4-8 concurrent sequences comfortably
```

Tune `max_num_batched_tokens` for your hardware:
- Default: 2048
- For throughput on small models: try 8192-16384
- Higher = better throughput, but watch for OOM

---

## Fix 3: FP8 Dynamic Quantization (Medium Impact)

**Expected speedup: 1.3-1.8x decode speed + 42% smaller model**

A community-contributed FP8 quantized version exists: [richarddavison/DeepSeek-OCR-2-FP8](https://huggingface.co/richarddavison/DeepSeek-OCR-2-FP8)

- Model size: ~3.5GB (down from ~6GB in bf16)
- Quantized with `llmcompressor` using `FP8_DYNAMIC` scheme
- Frees ~2.5GB VRAM for larger batch sizes / longer sequences

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"]
)
oneshot(model=model, recipe=recipe)
```

**Caveat:** FP8 (E4M3) compute is natively supported on Ada Lovelace (RTX 4090) and Hopper (H100) GPUs. The RTX 3090 (Ampere) does NOT have native FP8 tensor cores. On RTX 3090, the benefit is purely from reduced memory bandwidth (smaller weights = faster loads), not faster compute. Still helps because decode is memory-bandwidth-bound.

### INT4/INT8 Quantization Alternative (Better for RTX 3090)

Since RTX 3090 has native INT8 tensor cores:

```python
# Unsloth 4-bit loading
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/DeepSeek-OCR-2",
    load_in_4bit=True,
)
```

- 4-bit: ~8GB VRAM minimum, freeing massive space for batching
- INT8: better accuracy preservation, ~3GB model
- Expected: 97%+ accuracy retention for 3B models at INT4

---

## Fix 4: Reduce Output Token Count (Medium Impact)

**Expected speedup: proportional to token reduction**

Since generation is 7ms/token and you are generating ~1000 tokens/page, anything that reduces output token count directly reduces latency.

### Lower max_new_tokens

If your pages don't need 8192 tokens of output, cap it:

```python
# For typical pages, 2048-4096 is usually sufficient
sampling_params = SamplingParams(
    max_tokens=4096,  # down from 8192
    # ... other params
)
```

### Use a simpler prompt

Prompts that ask for "detailed markdown with all formatting" produce more tokens than "extract text content." Tailor the prompt to your actual needs.

### The NGramPerReqLogitsProcessor

This logits processor is **mandatory** for DeepSeek-OCR-2. It prevents text repetition (a known issue) using:
- `ngram_size=30` -- detects 30-token repeated sequences
- `window_size=90` -- looks back 90 tokens for repetition
- `whitelist_token_ids={128821, 128822}` -- allows `<td>`/`</td>` repetition (tables need them)

Without this processor, the model can enter repetition loops that waste hundreds of tokens and time.

---

## Fix 5: Lower Image Resolution (Medium Impact)

**Expected speedup: 1.3-2x (fewer visual tokens = faster prefill + shorter generation)**

The default configuration uses `base_size=1024, image_size=768, crop_mode=True`, which generates up to 1,120 visual tokens per image. You can reduce this:

### Resolution modes available

| Mode | Resolution | Visual Tokens | Use Case |
|------|-----------|--------------|----------|
| **Tiny** | 512x512 | ~64 | Quick scans, simple pages |
| **Small** | 640x640 | ~100 | Balanced default |
| **Base** (default) | 1024+768 crops | 256-1,120 | Fine details, complex layouts |
| **Large** | 1280x1280 | Maximum | Complex layouts with small text |

### Recommended tuning

```python
# For typical book pages with standard-size text:
result = model.infer(
    image=img,
    base_size=768,      # down from 1024
    image_size=640,     # down from 768
    crop_mode=False,    # disable multi-crop tiling
)
```

Disabling `crop_mode` alone drops from up to 7 crops (7 x 144 = 1,008 tokens) down to just the global view (256 tokens). This is a **4x reduction in visual tokens**, which speeds up both the prefill (vision encoding) and can influence generation length.

**Trade-off:** At 512x640, small text and fine table borders may blur. Test on your actual documents. For typical book pages with 10pt+ text, `crop_mode=False` with `base_size=768` usually works fine.

---

## Fix 6: Flash Attention 2 (Low-Medium Impact)

**Expected speedup: 1.2-1.5x on attention operations**

If you are not already using Flash Attention 2, enable it:

```python
model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR-2",
    torch_dtype=torch.bfloat16,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
)
```

Requires: `pip install flash-attn==2.7.3 --no-build-isolation`

Flash Attention 2 reduces attention from O(N^2) memory to O(N) and fuses the attention computation into a single kernel. This helps both prefill (processing visual tokens) and decode (generating text tokens).

**Note:** If using vLLM, Flash Attention is enabled by default. This fix only matters if you are using raw Transformers inference.

---

## Fix 7: Speculative Decoding (Experimental, High Potential)

**Expected speedup: 1.5-2.3x end-to-end**

Recent research (SpecVLM, 2025) shows speculative decoding works well for vision-language models. The idea: a small "draft" model proposes multiple tokens at once, and the full model verifies them in a single forward pass.

### vLLM speculative decoding setup

```bash
vllm serve deepseek-ai/DeepSeek-OCR-2 \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 5}' \
    # ... other flags
```

Available methods in vLLM:
- **N-Gram Matching** -- uses patterns from the input to predict next tokens (no draft model needed)
- **Draft model** -- pairs a smaller model (e.g., 1B) with the 3B target
- **EAGLE-3** -- learned speculative heads

**Caveat:** vLLM docs note speculative decoding "is not yet optimized and does not usually yield inter-token latency reductions for all prompt datasets." For OCR output (which is highly structured markdown), n-gram matching may work well since table rows and markdown patterns repeat predictably.

### The 3B model challenge

Speculative decoding typically pairs a large target with a small draft. With a 3B target model, finding a compatible draft model small enough to provide speedup is tricky. N-gram matching avoids this problem entirely since it does not need a separate model.

---

## Fix 8: deepseek-ocr.rs (Rust Native Inference)

**Expected speedup: 1.5-1.9x over Python Transformers**

Since your project is already in Rust, [deepseek-ocr.rs](https://github.com/TimmyOVO/deepseek-ocr.rs) is worth investigating:

- **1.88x faster** overall vs Python reference pipeline (benchmarked on macOS)
- **1.46x faster** token loop generation
- **1.83x faster** prompt prefill
- Supports DSQ quantization (Q4_K, Q6_K, Q8_0 tiers)
- OpenAI-compatible HTTP server and CLI
- No Python dependency

### DSQ Quantization tiers

| Tier | Size | Speed | Accuracy |
|------|------|-------|----------|
| Q8_0 | Largest | Fastest decode | Best |
| Q6_K | Medium | Good | Good |
| Q4_K | Smallest | Most aggressive | Lower precision |

### Current limitations

- CUDA support is **alpha stage**
- Best performance currently on Apple Metal (FP16)
- Prefill: ~40-50 tok/s on M-series, ~12 tok/s on CPU
- No continuous batching (single-request at a time)

**Verdict:** Interesting but not production-ready for your CUDA use case yet. Monitor the repo. If CUDA support matures, this eliminates all Python overhead.

---

## Fix 9: Pipeline-Level Optimizations (Architectural)

These are changes to how your Rust client and Python server interact:

### A. Parallel page preprocessing

Convert PDF pages to images on CPU threads while GPU processes previous pages:

```
Thread pool (CPU):  [page1->img] [page2->img] [page3->img] ...
                         |            |            |
GPU inference:      [generate(img1)] [generate(img2)] [generate(img3)]
```

Your Rust client already sends the entire PDF. If the server extracts pages serially before starting inference, there is wasted time. The server should pipeline extraction and inference.

### B. Streaming results per page

Instead of waiting for the entire PDF to finish, stream page results back as they complete. Your Rust client already has a `WorkerEvent::Completed` event per file -- extend this to per-page granularity if the server supports it.

### C. Reduce DPI for image extraction

200 DPI is sufficient for OCR. 300+ DPI wastes memory and processing time for no accuracy gain. Verify what your server uses.

### D. Multiple vLLM workers on one GPU

The 3B model uses ~6GB in bf16 (or ~3.5GB in FP8). On a 24GB card, you could potentially run 2-3 independent vLLM instances, each handling different PDFs. This is a poor man's batching but may help if vLLM's built-in batching is not available in your setup.

---

## Compound Strategy: Recommended Optimization Order

Apply these in sequence, measuring after each step:

| Step | Fix | Expected Cumulative | Effort |
|------|-----|:---:|--------|
| 1 | Switch to vLLM (Fix 1) | **3-5x** | Medium (re-deploy server) |
| 2 | Batch pages (Fix 2) | **5-10x throughput** | Medium (modify server) |
| 3 | Lower resolution / disable crop_mode (Fix 5) | **6-15x** | Low (config change) |
| 4 | INT4/INT8 quantization (Fix 3) | **8-20x** | Low (swap model) |
| 5 | Tune max_tokens down (Fix 4) | **10-25x** | Trivial |
| 6 | Speculative decoding (Fix 7) | **15-35x** | Medium-High (experimental) |

### Realistic target on RTX 3090

Starting from 7.5 sec/page (0.13 pages/sec):

- **With vLLM + batching + lower resolution:** ~1.0-2.5 sec/page (0.4-1.0 pages/sec)
- **Adding quantization:** ~0.7-1.5 sec/page (0.7-1.5 pages/sec)
- **735-page book:** ~8-18 minutes (down from ~92 minutes)

This would be a **5-10x improvement**, bringing you within striking distance of the sub-10-minute target for 735 pages.

---

## Performance Reference Numbers

| Setup | Tokens/sec | Pages/sec | Source |
|-------|:---:|:---:|--------|
| DeepSeek-OCR-2, A100-40G, vLLM | ~2,500 | ~4.65 | Official |
| DeepSeek-OCR-2, RTX 3090, Transformers (current) | ~143 | ~0.13 | Your measurement |
| DeepSeek-OCR-2, RTX 3090, vLLM (estimated) | ~500-800 | ~0.5-0.8 | Estimated (A100/3090 ratio) |
| DeepSeek-OCR-2, RTX 3090, vLLM + quant + tuned (target) | ~800-1,500 | ~0.7-1.5 | Estimated |
| deepseek-ocr.rs, macOS Metal | ~40-50 (prefill) | N/A | Benchmark |
| RTX 3090 general 3B decode | ~140-180 tok/s (single) | N/A | Community reports |

The gap between your 143 tok/s and the A100's 2,500 tok/s is ~17x. The expected hardware gap is only 3-5x, meaning **you have 4-6x of software optimization headroom** from switching to vLLM, batching, and tuning.

---

## Alternative: Switch Models Entirely

If the above optimizations are insufficient, your speed_comparison.md already covers this. The fastest options for your RTX 3090:

| Model | Est. Pages/sec (RTX 3090) | Quality |
|-------|:---:|---------|
| **Marker** (pipeline) | 3-10 | Good for books, weaker tables |
| **LightOnOCR-2-1B** | ~1.8 | Good general, 1B params |
| **MinerU 2.5** | ~0.9 | Best accuracy (90.67 OmniDoc) |

---

## Sources

- [DeepSeek-OCR-2 GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 README](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/main/README.md)
- [DeepSeek-OCR-2 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [vLLM DeepSeek-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [vLLM CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- [richarddavison/DeepSeek-OCR-2-FP8](https://huggingface.co/richarddavison/DeepSeek-OCR-2-FP8)
- [Unsloth DeepSeek-OCR-2](https://unsloth.ai/docs/models/deepseek-ocr-2)
- [deepseek-ocr.rs (Rust implementation)](https://github.com/TimmyOVO/deepseek-ocr.rs)
- [DeepSeek-OCR-2 Complete Guide (DEV Community)](https://dev.to/czmilo/deepseek-ocr-2-complete-guide-to-running-fine-tuning-in-2026-3odb)
- [DeepSeek VL2 Memory Optimization (DeepWiki)](https://deepwiki.com/deepseek-ai/DeepSeek-VL2/4.1-memory-optimization-techniques)
- [SpecVLM: Fast Speculative Decoding in VLMs (arXiv)](https://arxiv.org/abs/2509.11815)
- [Multi-Head Latent Attention Explained (Vizuara)](https://vizuara.substack.com/p/decoding-multi-head-latent-attention)
- [AMD ROCm: Multimodal Inference in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html)
- [vLLM Speculators Library](https://github.com/vllm-project/speculators)
- [Maximizing Inference with Dual RTX 3090 (Medium)](https://thamizhelango.medium.com/maximizing-inference-speed-with-dual-rtx-3090-gpus-deepseek-model-optimization-2139d15f7b55)
- [NVIDIA Speculative Decoding Blog](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [DeepSeek-OCR-2 GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR-2/issues)
- [vLLM Forum: Running DeepSeek-OCR-2](https://discuss.vllm.ai/t/how-to-run-deep-seek-ocr-2-in-vllm/2280)
