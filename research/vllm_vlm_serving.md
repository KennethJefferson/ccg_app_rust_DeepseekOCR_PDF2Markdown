# vLLM & Alternatives for Serving Vision-Language Models

Research date: 2026-02-06

## Executive Summary

vLLM is the leading open-source inference engine for serving VLMs (vision-language models) in
production. It officially supports **DeepSeek-VL2**, **DeepSeek-OCR**, and (with caveats)
**DeepSeek-OCR-2**. For our use case (PDF-to-Markdown via DeepSeek-OCR-2/VL2), vLLM is the
strongest option: it provides 14-24x throughput over raw HuggingFace Transformers, has an
OpenAI-compatible API, and has an official DeepSeek-OCR recipe. SGLang is a viable alternative
with slightly higher throughput in some benchmarks. llama.cpp multimodal support exists but does
not cover DeepSeek architectures. TensorRT-LLM is powerful but complex and also lacks DeepSeek-VL2
support.

---

## 1. vLLM Multimodal Support

### 1.1 Architecture

vLLM V1 (current stable) has purpose-built multimodal features:

- **Encoder cache**: Multimodal embeddings (from ViT) are computed once and stored on GPU.
  A single 1024x1024 image can generate 4096 embeddings in Pixtral -- caching avoids
  recomputing on every prefill.
- **Encoder-aware scheduler**: Tracks positions of multimodal embeddings within each request,
  retrieves cached data, eliminates redundant encoder execution.
- **Hybrid parallelism**: ViT Data Parallel + LLM Tensor Parallel on multi-GPU setups.
  Vision encoder runs with data parallelism across GPUs while the language model uses
  tensor parallelism. Dramatically reduces TTFT (time to first token).
- **Continuous batching**: Iteration-level scheduling fills GPU gaps, absorbs latency
  variance between requests. Multi-step scheduling runs the model for N consecutive steps,
  spreading CPU overhead and reducing GPU idle time.

### 1.2 Supported VLMs (partial list)

| Model | Architecture | Notes |
|-------|-------------|-------|
| DeepSeek-VL2 | `DeepseekVLV2ForCausalLM` | Requires `--hf-overrides` flag |
| DeepSeek-OCR | `DeepseekOCRForCausalLM` | Official vLLM recipe, upstream since v0.8.5 |
| DeepSeek-OCR-2 | `DeepseekOCR2ForCausalLM` | **Not yet in upstream vLLM** -- see Section 2 |
| Qwen2-VL | `Qwen2VLForConditionalGeneration` | Fully supported |
| InternVL2.5 / InternVL3 | Various | Fully supported |
| LLaVA variants | Various | Fully supported |
| GLM-OCR (0.9B) | - | Supported since vLLM 0.11.0 |
| Pixtral 12B | - | Fully supported |

### 1.3 Performance Numbers

**vLLM vs HuggingFace Transformers (text LLMs)**:
- **14-24x higher throughput** depending on batch size and workload.
- Batch size 32: vLLM ~3.38s vs HF ~12.9s.
- **19-27% less GPU memory** via PagedAttention.
- Same infrastructure can handle **5x more traffic** without extra GPUs.

**vLLM v0.6.0 -> v0.8.1**: 24% throughput improvement on generation-heavy workloads with V1 engine.

**DeepSeek-OCR on vLLM**:
- **~2,500 tokens/second** on a single A100-40G (vision tokens, streaming).
- **~200,000 pages/day** processing capacity on A100-40G.

**LMCache integration (repeated images)**:
- ~100% cache hit rate for repeated images = zero GPU vision passes.
- End-to-end latency drops from tens of seconds to ~1 second for cached images.

**Single request latency caveat**: The 14-24x gains are primarily for batched/high-concurrency
scenarios. For single sequential requests, vLLM still helps (PagedAttention, optimized kernels)
but the multiplier is lower. The biggest wins come from continuous batching across concurrent
requests.

**Speculative decoding**: Up to 2.5-2.8x additional speedup with Eagle 3 speculative decoding
(available in vLLM). Most impactful for synchronous/low-concurrency workloads.

---

## 2. DeepSeek-OCR-2 on vLLM (Key Finding)

### 2.1 Current Status

DeepSeek-OCR-2 (released Jan 27, 2026) uses the `DeepseekOCR2ForCausalLM` architecture, which
is **NOT yet supported in upstream/latest vLLM**. Running with latest vLLM yields:

```
ValueError: Model architectures ['DeepseekOCR2ForCausalLM'] are not supported.
```

### 2.2 Workarounds

**Option A: Use the DeepSeek-provided vLLM fork** (recommended for now)

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm

# Install the pinned vLLM wheel
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install flash-attn==2.7.3
pip install -r requirements.txt

# Run image OCR (streaming)
python run_dpsk_ocr2_image.py

# Run PDF processing (concurrent)
python run_dpsk_ocr2_pdf.py
```

Configuration is in `config.py` for input/output paths and processing parameters.

**Option B: Wait for upstream vLLM support** -- likely coming given that DeepSeek-OCR (v1) was
upstreamed quickly. Track vLLM GitHub issues.

### 2.3 DeepSeek-OCR (v1) on vLLM (fully supported)

The original DeepSeek-OCR works out of the box with upstream vLLM:

```bash
# Install
uv pip install -U vllm --torch-backend auto

# Serve (online API)
vllm serve deepseek-ai/DeepSeek-OCR \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0

# Key flags:
# --no-enable-prefix-caching   (OCR tasks don't benefit from prefix caching)
# --mm-processor-cache-gb 0    (no image reuse expected)
```

Offline batch processing:

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

model = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

sampling_params = SamplingParams(
    max_tokens=8192,
    # Custom OCR params:
    # ngram_size=30, window_size=90
)
```

### 2.4 DeepSeek-VL2 on vLLM

```bash
vllm serve deepseek-ai/deepseek-vl2-small \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name deepseek_vl2_small \
  --limit-mm-per-prompt "image=2" \
  --hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
  --chat_template ./template_deepseek_vl2.jinja
```

The `--hf-overrides` flag is **required** because the checkpoint's `model_type` is
`deepseek_vl_v2` which Transformers doesn't recognize by default.

---

## 3. SGLang (Alternative)

### 3.1 Overview

SGLang is a high-performance serving framework for LLMs and VLMs from LMSYS.
Key differentiator: **RadixAttention** for KV cache reuse across requests.

### 3.2 DeepSeek-VL2 Support

As of Dec 2024, there was a feature request (sgl-project/sglang#2653) for DeepSeek-VL2 support.
Current status unclear -- check SGLang's supported models page at
`docs.sglang.ai/supported_models/multimodal_language_models`.

### 3.3 Performance vs vLLM

| Metric | SGLang | vLLM | Notes |
|--------|--------|------|-------|
| Batch inference (H100, 8B) | 16,215 tok/s | 12,553 tok/s | SGLang ~29% faster |
| Peak throughput (B300, 64 users) | 1,876 tok/s | 1,605 tok/s | SGLang ~17% faster |
| High concurrency (100 requests) | Lower | 4,741 tok/s | vLLM excels at very high concurrency |
| Multi-turn conversation | Better (RadixAttention) | Good | ~10% boost from cache reuse |

**Summary**: SGLang tends to be faster at moderate concurrency and for multi-turn workloads.
vLLM excels at very high concurrency. Both are strong choices.

### 3.4 Verdict for Our Use Case

SGLang is a viable alternative but **DeepSeek-VL2/OCR-2 support is uncertain**. vLLM has
confirmed support for all three DeepSeek models. Stick with vLLM unless SGLang adds explicit
DeepSeek-OCR-2 support.

---

## 4. TensorRT-LLM

### 4.1 Overview

NVIDIA's TensorRT-LLM provides maximum performance on NVIDIA GPUs through graph optimization,
kernel fusion, and quantization. Supports VLMs including LLaVA-NeXT, Qwen2-VL, VILA, Llama 3.2
Vision, and Llama 4 Vision.

### 4.2 DeepSeek Support

**No DeepSeek-VL2 or DeepSeek-OCR support** in TensorRT-LLM as of Feb 2026.
DeepSeek text models (V3, R1) are supported, but not the vision variants.

### 4.3 Trade-offs

| Pro | Con |
|-----|-----|
| Best raw performance on NVIDIA GPUs | Complex build/setup process |
| NVFP4 quantization, Eagle-3 speculative decoding | NVIDIA-only (no AMD, no CPU) |
| Triton Inference Server integration | No DeepSeek-VL2/OCR support |
| TensorRT Edge-LLM for edge devices | Model conversion pipeline required |

### 4.4 Verdict

Not viable for our use case due to missing DeepSeek model support. Revisit if NVIDIA adds
DeepSeek-VL2 to their support matrix.

---

## 5. llama.cpp Multimodal

### 5.1 Overview

llama.cpp added multimodal support in April 2025 via `libmtmd`. Supports image and audio inputs
through external projection models (mmproj) that encode non-text modalities into the text model's
embedding space.

### 5.2 Supported VLMs

- Gemma 3 (4B, 12B, 27B)
- SmolVLM
- Pixtral 12B
- Qwen2 VL, Qwen2.5 VL
- Mistral Small 3.1

### 5.3 DeepSeek Support

**No DeepSeek-VL2 or DeepSeek-OCR support** in llama.cpp. The DeepSeek text models have some
GGUF support, but the vision-language variants (VL2, OCR, OCR-2) are not supported.

### 5.4 Trade-offs

| Pro | Con |
|-----|-----|
| CPU + GPU inference | No DeepSeek VLM support |
| GGUF quantization (2-8 bit) | Limited VLM model selection |
| Runs on consumer hardware | Lower throughput than vLLM/SGLang |
| Mobile/edge deployment (Android) | No continuous batching for serving |

### 5.5 Verdict

Not viable for DeepSeek models. Useful for other VLMs on consumer/edge hardware.

---

## 6. Quantization for VLMs on vLLM

vLLM supports GPTQ, AWQ, INT4, INT8, FP8, and AutoRound quantization. For VLMs specifically:

- **GPTQ**: Works via GPTQModel library for models like Qwen2-VL. Some compatibility issues
  between GPTQModel's multi-modal support and vLLM's layer mapping.
- **AWQ**: Recommended for best accuracy. Protects salient weights by scaling channels based
  on activation distributions.
- **FP8**: Supported on AMD and NVIDIA. Typically only the language component is quantized,
  not the vision encoder.

**Important caveat**: VLM quantization in vLLM is still maturing. The vision encoder is
typically left at full precision while only the language model backbone gets quantized.

---

## 7. Windows / Local Deployment

vLLM does **not** support Windows natively. Options:

1. **WSL2 + NVIDIA GPU**: Install vLLM inside WSL2 with CUDA drivers.
   Requirements: WSL2, NVIDIA GPU (GTX 1080+), compatible drivers.
2. **Docker Desktop for Windows**: Docker Model Runner now supports vLLM on Docker Desktop
   with WSL2 and NVIDIA GPUs. Provides unified workflow and production parity.
3. **Community fork**: `github.com/SystemPanic/vllm-windows` provides a native Windows build.

For our Rust-based tool, the most practical approach would be:
- Run vLLM as a local server (via WSL2 or Docker) exposing the OpenAI-compatible API.
- Call the API from Rust using an HTTP client.
- This decouples the inference engine from the Rust application.

---

## 8. Recommendations for This Project

### Short-term (now)

1. **Use DeepSeek-OCR (v1) with upstream vLLM** for immediate speedup. It's fully supported,
   has an official recipe, and delivers ~2,500 tok/s on A100-40G.
2. Serve via `vllm serve` with OpenAI-compatible API.
3. Call from Rust via HTTP (reqwest + serde).

### Medium-term (when upstream support lands)

1. **Switch to DeepSeek-OCR-2 on vLLM** once `DeepseekOCR2ForCausalLM` is upstreamed
   (or use the DeepSeek fork now if you're willing to pin vLLM 0.8.5).
2. Consider AWQ/GPTQ quantization of the language backbone for lower VRAM usage.

### Alternative path

1. If running on consumer GPU (RTX 3090/4090 with 24GB VRAM), DeepSeek-OCR-2 at 3B params
   should fit comfortably. vLLM's memory efficiency via PagedAttention helps here.
2. For maximum local performance on Windows: Docker + vLLM + NVIDIA Container Toolkit.

### What NOT to pursue

- TensorRT-LLM: No DeepSeek-VL2/OCR support.
- llama.cpp: No DeepSeek VLM support.
- SGLang: Uncertain DeepSeek-VL2/OCR-2 support; vLLM is the safer bet.

---

## 9. Expected Speedup Summary

| Setup | Throughput | vs HF Transformers |
|-------|------------|-------------------|
| HF Transformers (single request, no batching) | Baseline | 1x |
| vLLM (single request) | ~2-3x faster | PagedAttention + optimized kernels |
| vLLM (batched, moderate concurrency) | ~5-14x faster | Continuous batching |
| vLLM (high concurrency, 32+ requests) | ~14-24x faster | Full pipeline utilization |
| vLLM + speculative decoding | Additional 2-2.8x | Eagle 3, best for low concurrency |
| vLLM + LMCache (repeated images) | Near-instant for cached | 100% encoder cache hit |

For our PDF-to-Markdown pipeline processing many pages sequentially, the **batched** scenario
is most relevant. We can submit multiple pages concurrently to vLLM and get 5-14x throughput
improvement over raw HuggingFace.

---

## Sources

- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM DeepSeek-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [vLLM DeepSeek-VL2 Module](https://docs.vllm.ai/en/v0.9.0/api/vllm/vllm.model_executor.models.deepseek_vl2.html)
- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 HuggingFace Discussion (vLLM compat)](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/discussions/3)
- [DeepSeek-VL2 GitHub](https://github.com/deepseek-ai/DeepSeek-VL2)
- [vLLM V1 Multimodal Acceleration (Red Hat)](https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models)
- [vLLM vs HuggingFace Benchmark](https://medium.com/@alishafique3/vllm-vs-hugging-face-for-high-performance-offline-llm-inference-2d953b4fb3b4)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang vs vLLM Benchmark](https://medium.com/@saidines12/sglang-vs-vllm-part-1-benchmark-performance-3231a41033ca)
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
- [llama.cpp Multimodal Docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [vLLM on Windows via Docker](https://www.docker.com/blog/docker-model-runner-vllm-windows/)
- [LMCache Multimodal in vLLM V1](https://blog.lmcache.ai/2025-07-03-multimodal-models/)
- [vLLM Speculative Decoding Blog](https://blog.vllm.ai/2024/10/17/spec-decode.html)
- [AMD ROCm vLLM ViT DP Optimization](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html)
- [vLLM DeepSeek-OCR Tweet (~2500 tok/s on A100-40G)](https://x.com/vllm_project/status/1980235518706401405)
