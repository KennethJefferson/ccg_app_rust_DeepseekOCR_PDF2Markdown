# VLM Quantization & Inference Speedup Research

**Goal**: Reduce per-token latency from ~7ms/token to 2-3ms/token for DeepSeek-VL2 (3B activated params, MoE) on RTX 3090.

**RTX 3090 specs**: 936 GB/s memory bandwidth, 35.6 TFLOPS FP16, 24GB GDDR6X, Ampere (SM 8.6).

---

## TL;DR: Can We Hit 2-3ms/token?

**Probably yes, with the right stack.** Here's the math and evidence:

| Technique | Expected Speedup | New Latency (from 7ms) | Confidence |
|-----------|-----------------|----------------------|------------|
| INT4 AWQ + Marlin kernel | 2-3x | ~2.3-3.5ms | High |
| + CUDA Graphs | +15-25% on top | ~2.0-3.0ms | High |
| + torch.compile | +10-20% on top | ~1.8-2.5ms | Medium |
| + Speculative decoding | 2.5-2.9x standalone | ~2.4-2.8ms | Medium |
| Stacked (INT4 + CUDA graphs + compile) | ~3-4x total | ~1.8-2.3ms | Medium |

**Best realistic target: 2-3ms/token with INT4 AWQ (Marlin) + CUDA graphs.**

---

## 1. Quantization Methods

### 1.1 AWQ (Activation-Aware Weight Quantization)

- Protects salient weights by observing activations, not weights themselves
- 4-bit weight-only quantization (INT W4A16: 4-bit weights, 16-bit activations)
- Particularly good for instruction-tuned and multi-modal LMs
- **Memory**: ~4x reduction (3B model: ~6GB FP16 -> ~1.5GB INT4)
- **Speed**: 3-4x speedup on token generation claimed by AutoAWQ authors
- **Quality**: Indistinguishable from full-precision on most benchmarks; best accuracy among INT4 methods

**Key insight**: For small-batch inference (batch size <= 4), weight-only quantization (INT4 AWQ) excels because decode is memory-bandwidth-bound, and reading 4x fewer weight bytes directly translates to speed.

### 1.2 GPTQ

- Post-training quantization using approximate second-order information
- Also 4-bit weight-only
- **Speed**: Similar throughput to AWQ when both use Marlin kernels
- **Quality**: Slightly worse than AWQ; may overfit calibration data
- **Verdict**: AWQ preferred for quality; GPTQ is fine if model is already available in GPTQ format

### 1.3 bitsandbytes (bnb) 4-bit

- Easy to use via HuggingFace Transformers BitsAndBytesConfig
- NF4 (NormalFloat4) data type, optional double quantization
- **Speed**: SLOW for inference. Benchmarks show 168 tok/s vs 741 tok/s for Marlin-AWQ on same model (H200). That's 4.4x slower than Marlin.
- **Quality**: Best perplexity among quantization methods (6.67 vs 6.56 baseline)
- **Verdict**: Great for fine-tuning (QLoRA), terrible for production inference. The dequantization overhead kills latency.

### 1.4 Quantization Benchmarks (Qwen2.5-32B on H200)

Source: JarvisLabs vLLM quantization guide

| Method | tok/s | ITL (ms) | Perplexity |
|--------|-------|----------|------------|
| FP16 baseline | 461 | 20.4 | 6.56 |
| Marlin-AWQ | 741 | 12.6 | 6.84 |
| Marlin-GPTQ | 712 | 13.1 | 6.84 |
| GPTQ (no Marlin) | 276 | 35.0 | - |
| bitsandbytes | 168 | 56.5 | 6.67 |
| AWQ (no Marlin) | 68 | 138.7 | - |
| GGUF Q4_K_M | 93 | 101.6 | 6.74 |

**Critical finding**: The kernel matters more than the quantization method. Marlin-AWQ is 10.9x faster than plain AWQ. Always use Marlin kernels.

### 1.5 VLM-Specific Quantization Results (Qwen2.5-VL-3B)

Source: Red Hat VLM quantization article

For the 3B VLM specifically:
- FP8 W8A8 and INT W8A8: >99% accuracy recovery
- INT W4A16 (4-bit weights): ~92-98% accuracy recovery depending on task
- **Speedup for 3B model: 1.1-1.5x** (modest -- smaller models are already fast and less memory-bound)
- Larger models (72B) see up to 3.5x speedup

**Important caveat for 3B models**: Smaller models get less speedup from quantization because they're closer to the compute-bound regime. The 7ms/token baseline suggests we ARE memory-bound though, so there's room.

---

## 2. Kernel-Level Optimizations

### 2.1 Marlin Kernel (FP16xINT4)

- Achieves near-ideal ~4x speedup for batch sizes up to 16-32
- Theoretical max: 3.87x (accounting for group scale overhead)
- **Better speedups on consumer GPUs** (RTX 3090) vs datacenter (A100), because RTX has lower baseline bandwidth making the relative improvement larger
- Compatible with Ampere and newer (RTX 30xx/40xx, A10, A100)
- Available in vLLM via Marlin-GPTQ and Marlin-AWQ backends

### 2.2 CUDA Graphs

- Captures entire decode step as a single graph, launched with one API call
- Eliminates CPU-GPU synchronization overhead and kernel launch latency
- TensorRT-LLM uses CUDA graph padding for batch flexibility
- **Typical improvement**: 15-30% latency reduction on decode steps
- Most impactful for small models where kernel launch overhead is a larger fraction of total time

### 2.3 torch.compile

- Fuses operations, eliminates memory round-trips
- **Typical speedup**: 1.5-2x over eager mode
- vLLM V1 integrates torch.compile by default
- Modes: `default` (fast compile), `reduce-overhead`, `max-autotune` (slow compile, best inference)
- For ViT encoder: ~13% speedup on A100 (1000ms -> 870ms for batch of 256 images)
- **Caveat**: May conflict with some quantization backends; test compatibility

### 2.4 Flash Attention / Flash Decoding

- Flash Attention v2: Reduces memory IO for attention computation
- Flash Decoding: Parallelizes across KV cache sequence dimension during decode
- Already used in most modern inference stacks (vLLM, SGLang, TensorRT-LLM)
- **Impact**: Primarily helps with long sequences. For short OCR outputs, effect is moderate.

---

## 3. Decoding Acceleration Techniques

### 3.1 Speculative Decoding for VLMs

**SpecVLM** (2025):
- 2.5-2.9x end-to-end speedup on VLM tasks
- Uses elastic visual token compression + lightweight draft model
- Tested on LLaVA models (7B/13B), batch size 1
- Lossless -- preserves target model output distribution
- Requires training a draft model (~5 epochs)

**FLASH** (2025):
- Up to 2.68x speedup on video captioning, 2.55x on visual instruction tuning
- Semi-autoregressive decoding: generates multiple tokens per forward pass
- Leverages visual token redundancy

**EAGLE/EAGLE-2**:
- Feature-level speculative decoding
- 1.5-2.3x speedup as baseline for VLMs

**Verdict**: Speculative decoding is promising but requires a draft model. For a custom model like DeepSeek-VL2, you'd need to train one. Not plug-and-play.

### 3.2 Continuous Batching

- vLLM/SGLang process requests token-by-token, insert new requests as old ones finish
- **23x throughput improvement** over naive batching (vLLM benchmark)
- Primary benefit is throughput (multiple concurrent requests), not single-request latency
- For OCR pipeline processing many pages: huge win for overall throughput
- Does NOT reduce per-token latency for a single request

### 3.3 KV Cache Optimization

- DeepSeek-VL2 uses Multi-Head Latent Attention (MLA) which compresses KV cache into latent vectors
- Already more memory-efficient than standard MHA
- Paged attention (vLLM) prevents memory fragmentation
- For 3B model on 24GB GPU: KV cache is not the bottleneck

---

## 4. Theoretical Analysis: RTX 3090 + 3B Model

### 4.1 Roofline Analysis

The decode phase is **memory-bandwidth-bound** for typical LLM inference:
- Each token generation requires reading all model weights once
- RTX 3090 bandwidth: 936 GB/s

**FP16 (no quantization):**
- 3B params * 2 bytes = 6 GB of weights
- Theoretical max: 936 / 6 = 156 tokens/sec = 6.4ms/token
- Real-world with overhead: ~7-9ms/token (matches our 7ms observation)

**INT4 (4-bit quantization):**
- 3B params * 0.5 bytes = 1.5 GB of weights
- Theoretical max: 936 / 1.5 = 624 tokens/sec = 1.6ms/token
- Real-world with overhead (dequant, activations, KV cache reads): ~2.5-4ms/token

**INT4 with Marlin kernel (near-ideal):**
- Marlin achieves ~3.87x of FP16 throughput at batch=1
- 6.4ms / 3.87 = 1.65ms/token theoretical
- Real-world: ~2-3ms/token

### 4.2 MoE Consideration

DeepSeek-VL2 is MoE with ~3B activated parameters but more total parameters:
- Only activated expert weights need to be read per token
- But routing overhead + all-expert storage is still in memory
- MoE actually benefits MORE from quantization: total model fits more comfortably in VRAM, and activated-parameter reads are smaller
- INT4 MoE: activated weights ~0.75-1.5 GB per token step

### 4.3 The Vision Encoder Factor

- Vision encoder (SigLip/SAM-style) runs once per image during prefill
- Does NOT affect per-token decode latency
- Can be optimized separately with TensorRT or torch.compile
- For OCR: prefill (image encoding + prompt) is a one-time cost per page

---

## 5. Recommended Implementation Stack

### Option A: Maximum Speed (Recommended)

```
vLLM + AWQ INT4 (Marlin kernel) + CUDA Graphs
```

- **Expected latency**: 2-3ms/token
- **Setup complexity**: Medium
- **Steps**:
  1. Quantize model with AutoAWQ to INT4
  2. Serve with vLLM (Marlin backend auto-selected for AWQ)
  3. CUDA graphs enabled by default in vLLM
  4. Enable chunked prefill for throughput: `--enable-chunked-prefill`

### Option B: Maximum Speed + Quality

```
vLLM + GPTQ INT4 (Marlin kernel) + CUDA Graphs + KV cache INT8
```

- Similar speed to Option A
- GPTQ may work better if calibration data matches OCR domain

### Option C: Easy Integration (HuggingFace native)

```
AutoAWQ + torch.compile(mode="max-autotune")
```

- **Expected latency**: 3-4ms/token
- **Setup complexity**: Low
- **Steps**:
  1. Load model with AutoAWQ
  2. Apply torch.compile to the model
  3. Warm up with dummy inputs

### Option D: Maximum Throughput (batch processing)

```
vLLM/SGLang + AWQ INT4 + Continuous Batching
```

- For processing many PDF pages concurrently
- Single-request latency similar to Option A
- But 10-20x higher throughput when processing multiple pages
- SGLang may edge out vLLM on Ampere GPUs at high batch sizes

### Option E: Extreme (requires engineering effort)

```
TensorRT-LLM + INT4 AWQ + CUDA Graphs + Speculative Decoding
```

- **Expected latency**: 1.5-2.5ms/token
- **Setup complexity**: High
- Requires building TensorRT engine for DeepSeek-VL2 architecture
- Speculative decoding needs a draft model (could use DeepSeek-VL2-Tiny as draft)

---

## 6. What NOT to Use

| Approach | Why Not |
|----------|---------|
| bitsandbytes for inference | 4.4x slower than Marlin. It's for training, not serving. |
| GGUF/llama.cpp | Great for CPU, suboptimal on GPU vs Marlin kernels. 101ms ITL vs 12.6ms. |
| FP8 quantization | RTX 3090 lacks native FP8 support (that's Hopper/Ada). Software FP8 won't help. |
| INT8 SmoothQuant | Only ~1.2x speedup vs FP16. Not worth the complexity for 3B model. |
| AWQ without Marlin | 10.9x slower than Marlin-AWQ. Always use Marlin backend. |

---

## 7. Answer: 7ms -> 2-3ms Feasibility

**Yes, achievable.** The path:

1. **Current**: FP16, ~7ms/token (matches roofline: 936GB/s / 6GB = 156 tok/s)
2. **INT4 AWQ + Marlin**: ~2.5-3.5ms/token (3.87x theoretical, ~2-3x practical)
3. **+ CUDA Graphs**: ~2.0-3.0ms/token (saves kernel launch overhead)
4. **+ torch.compile**: ~1.8-2.5ms/token (fused operations)

The roofline theoretical minimum for INT4 on RTX 3090 is ~1.6ms/token. With real-world overhead (activation computation, KV cache, dequantization), **2-3ms/token is realistic and achievable**.

For the OCR pipeline specifically:
- Prefill (image encode + prompt): one-time cost, ~50-200ms depending on image size
- Decode: 2-3ms/token * ~200 tokens avg = 400-600ms per page
- vs current: 7ms * 200 = 1400ms per page
- **Net improvement: ~2-3x faster per page**

---

## Sources

- [Red Hat: 3.5x Faster VLMs with Quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization)
- [JarvisLabs: vLLM Quantization Complete Guide & Benchmarks](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
- [Marlin: FP16xINT4 Kernel (IST-DASLab)](https://github.com/IST-DASLab/marlin)
- [PyTorch: INT4 Decoding GQA CUDA Optimizations](https://pytorch.org/blog/int4-decoding/)
- [NVIDIA: Mastering LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [NVIDIA: Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)
- [Faster LLMs with Quantization (bitbasti.com)](https://bitbasti.com/blog/faster-llms-with-quantization)
- [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/abs/2509.11815)
- [FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding](https://arxiv.org/abs/2505.12728)
- [vLLM Blog: torch.compile Integration](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [LLM Inference Unveiled: Roofline Model Insights](https://arxiv.org/html/2402.16363v4)
- [DeepSeek-VL2 GitHub](https://github.com/deepseek-ai/DeepSeek-VL2)
- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)
- [VLMQ: Post-Training Quantization for VLMs](https://arxiv.org/html/2508.03351v1)
- [VEQ: Modality-Adaptive Quantization for MoE VLMs](https://arxiv.org/html/2602.01037)
- [Anyscale: 23x Throughput with Continuous Batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
