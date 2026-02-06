# Why Pipeline OCR (Marker/MinerU) Is Fundamentally Faster Than Autoregressive VLM OCR (DeepSeek-OCR-2)

## Deep Architectural Analysis

---

## Executive Summary

The speed difference between pipeline-based OCR tools (Marker, MinerU) and autoregressive VLM-based OCR (DeepSeek-OCR-2) is not a matter of implementation quality -- it stems from **fundamentally different computational paradigms**. Pipeline tools decompose the problem into parallelizable sub-tasks with specialized small models, while VLM-based OCR forces all output through a sequential, token-by-token autoregressive bottleneck. The key insight: **the autoregressive decoder is the speed ceiling**, and no amount of encoder optimization can fully overcome it.

---

## 1. The Core Bottleneck: Autoregressive Token Generation

### Why Autoregressive Decoding Is Inherently Slow

Large language models (including VLM decoders) generate text **one token at a time**. Each token depends on ALL previous tokens, creating a sequential dependency chain:

```
Token[N+1] = f(Token[1], Token[2], ..., Token[N], image_features)
```

This means:
- **O(n) forward passes** for n output tokens -- you CANNOT parallelize this
- A 1000-word markdown page (~1500 tokens) requires ~1500 sequential model forward passes
- Each forward pass involves attention over the full KV cache
- **GPU utilization is low** during decoding because each step produces only 1 token, leaving most CUDA cores idle

### Concrete Math

For DeepSeek-OCR-2 with ~570M active parameters (MoE decoder):
- Prefill (encoding the image): **1 forward pass** -- parallelizable, fast
- Decoding 1500 tokens of markdown: **1500 sequential forward passes**
- Even at ~2,500 tokens/sec on A100, a dense page takes ~0.6 seconds minimum
- Complex pages with tables/equations can produce 3000-5000 tokens, taking 1-2+ seconds

For a 100-page PDF: **60-200 seconds** just for decoding, ignoring encoding entirely.

### Why GPUs Cannot Help

Modern GPUs excel at parallel computation (matrix multiplications across thousands of cores). But autoregressive decoding is fundamentally **memory-bandwidth bound**, not compute-bound:
- Each decoding step reads the entire KV cache from GPU memory
- Only produces 1 output token per step
- Arithmetic intensity (FLOPs / bytes) is extremely low
- Result: GPU utilization during decoding is typically 5-30%

---

## 2. How Pipeline OCR Avoids This Bottleneck

### Marker's Pipeline Architecture (Powered by Surya)

Marker decomposes document conversion into **5 specialized stages**, each using a purpose-built model:

```
PDF Page Image
    |
    v
[1] Text Detection (EfficientViT-based segmentation model)
    -> Detects bounding boxes of text regions
    -> ViT processes full page at LOW resolution (just finding locations)
    -> Output: list of bounding box coordinates
    |
    v
[2] Layout Detection (ViT-based segmentation model)
    -> Classifies regions: text, table, header, footer, equation, figure
    -> Single forward pass per page
    -> Output: labeled layout regions
    |
    v
[3] Reading Order Detection
    -> Determines logical reading sequence of detected blocks
    -> Resolves column layouts, sidebars, footnotes
    |
    v
[4] Text Recognition (Modified Donut: Swin encoder + BART decoder)
    -> Processes EACH text line/region individually (small crops)
    -> Key: operates on SMALL image crops, not full pages
    -> Uses GQA, MoE layers, UTF-16 decoding
    -> BATCH PROCESSES hundreds of line crops simultaneously on GPU
    |
    v
[5] Post-processing & Markdown Assembly
    -> Merges recognized text with layout information
    -> Applies formatting rules
    -> CPU-bound string operations (microseconds)
```

### Why This Is Fast: The Key Architectural Advantages

**A. Parallelizable Batch Processing of Small Crops**

The recognition model processes TEXT LINES, not full pages. A typical page has 30-60 text lines. Surya batches these into a single GPU inference call:
- Default `RECOGNITION_BATCH_SIZE = 256` (~12.8 GB VRAM at 50MB per item)
- All 30-60 lines from a page processed in ONE batch forward pass
- Even though the decoder is autoregressive (Donut/BART), the text per line is SHORT (~10-20 tokens)
- 20 tokens x 30 lines = 600 total decode steps, but **batched across 30 lines simultaneously**
- Effective throughput: 30x higher than processing the page as a single sequence

**B. Small Specialized Models**

Each stage uses a tiny, task-specific model:
- Detection model: ~25M parameters (EfficientViT)
- Layout model: ~25M parameters (SegFormer variant)
- Recognition model: ~200M parameters (modified Donut)
- Compare to DeepSeek-OCR-2: 500M encoder + 3B decoder (570M active via MoE)

**C. Short Output Sequences**

The recognition model only needs to produce ~10-20 tokens per text line crop. Autoregressive overhead is minimal for short sequences. A full page's text spread across 40 lines means 40 parallel short sequences rather than 1 long sequential sequence of 1500+ tokens.

**D. Detection/Layout Are NOT Autoregressive**

Steps 1-3 are **encoder-only** operations (segmentation, classification):
- Single forward pass per page
- Fully parallelizable on GPU
- No sequential token generation whatsoever
- Each takes ~10-50ms on GPU

### Marker Speed Benchmarks

| Configuration | Speed | Notes |
|---|---|---|
| Single process, H100 | ~5-6 pages/sec | Conservative baseline |
| Batch mode, single H100 | ~25 pages/sec | Batched line recognition |
| 22 processes, H100 (80GB) | ~122 pages/sec | Multi-process parallelism |
| Single page latency | ~0.18 sec/page | Batched mode |
| CPU only (x86) | ~16+ sec/page | No GPU acceleration |
| CPU only (M3 Max) | ~4.2 sec/page | Apple Silicon MPS |
| L4 GPU | ~0.86 sec/page | Mid-range GPU |

---

## 3. MinerU's Decoupled Two-Stage Architecture

MinerU (especially v2.5) takes a different but equally clever approach to avoiding the autoregressive bottleneck.

### Architecture: Coarse-to-Fine Decoupling

```
PDF Page Image (high-res, e.g., 2048x2048)
    |
    v
[Stage 1: Layout Analysis - FAST]
    -> Downsampled thumbnail (1036x1036 fixed)
    -> NaViT vision encoder + Qwen2-0.5B LLM
    -> Identifies: text blocks, tables, figures, equations
    -> Processes at LOW resolution = few visual tokens
    -> Output: bounding boxes + element types
    |
    v
[Stage 2: Content Recognition - TARGETED]
    -> Crops ONLY identified content regions at NATIVE resolution
    -> Max 2048x2048 per crop
    -> Fine-grained recognition within local windows
    -> Autoregressive decoding, but on SMALL targeted crops
```

### Why This Is Fast

**A. Avoids O(N^2) Token Explosion**

End-to-end VLMs that process full high-res pages face quadratic scaling:
- A 2048x2048 image at standard tokenization = thousands of visual tokens
- Self-attention cost = O(T^2) where T = number of tokens
- MinerU Stage 1 processes a downsized thumbnail with far fewer tokens
- Stage 2 only processes small crops, not the full page

MinerU's paper states this provides **"computational savings by an order of magnitude"** compared to native-resolution end-to-end approaches.

**B. Blank Space Elimination**

A typical document page is 40-70% whitespace. End-to-end VLMs process ALL of it. MinerU's two-stage approach:
- Stage 1 identifies where content is (fast, low-res)
- Stage 2 ONLY processes content regions (skipping blank areas)
- This alone can reduce computation by 50%+

**C. Pixel-Unshuffle Token Reduction**

MinerU2.5 applies pixel-unshuffle on 2x2 vision token blocks, reducing Stage 1 FLOPs by ~25% with <0.1% accuracy loss.

### MinerU Speed Benchmarks

| Configuration | Speed | Notes |
|---|---|---|
| v1.x pipeline (L4 GPU) | 0.21 sec/page (4.8 pages/sec) | YOLO + PP-OCR pipeline |
| v2.5 baseline (A100) | 0.95 pages/sec | Single inference stream |
| v2.5 optimized (A100 80G) | 4.47 pages/sec | vllm-async-engine |
| v2.5 optimized (RTX 4090) | 2.12 pages/sec | vllm-async-engine |
| CPU only (x86) | 3.3 sec/page | No GPU |
| MonkeyOCR-Pro-3B | 0.47 pages/sec | MinerU2.5 is 4x faster |
| dots.ocr | 0.28 pages/sec | MinerU2.5 is 7x faster |

---

## 4. DeepSeek-OCR-2: The Autoregressive Architecture

### How It Works

```
PDF Page Image
    |
    v
[DeepEncoder V2] (Qwen2-0.5B based, ~500M params)
    -> Multi-resolution cropping: global view (1024x1024) + up to 6 local crops (768x768)
    -> Visual tokens: bidirectional attention (ViT-style)
    -> Causal flow tokens: learnable queries with causal attention
    -> "Visual Causal Flow" reorders visual information semantically
    -> Output: 256-1,120 visual tokens total
    |
    v
[MoE Decoder] (DeepSeek-3B-MoE, ~570M active params per token)
    -> Receives causal flow tokens (not raw visual tokens)
    -> Autoregressive generation of markdown output
    -> Activates 6 of 64 experts + 2 shared per token
    -> Token-by-token sequential output
    |
    v
Markdown text output (500-5000+ tokens per page)
```

### Why It Is Slower

**A. Single Long Autoregressive Sequence Per Page**

Unlike Marker (which batches 30-60 short line recognitions), DeepSeek-OCR-2 generates the ENTIRE page's markdown in one autoregressive sequence. A typical page:
- 500-1500 tokens for simple text
- 2000-5000 tokens for tables, equations, complex layouts
- Each token requires a full decoder forward pass

**B. Vision Token Processing Overhead**

Even with the efficient DeepEncoder V2 (256-1,120 tokens), the prefill step processes all visual tokens through cross-attention at every decoding step. With 1,120 visual tokens and a 3000-token output, that is:
- 3000 decoding steps, each attending to 1,120 visual tokens
- Plus growing KV cache of previously generated tokens

**C. MoE Routing Overhead**

While MoE reduces active parameters (570M vs 3B total), the routing decision (selecting 6/64 experts) adds latency per token. At 2,500 tokens/sec on A100, this is respectable but fundamentally limited by the sequential nature.

**D. Cannot Batch Multiple Pages Through the Decoder**

The autoregressive decoder processes ONE page's sequence at a time (or with limited batching via vLLM). You cannot interleave tokens from different pages within a single forward pass without significant complexity (continuous batching helps but doesn't eliminate the fundamental bottleneck).

### DeepSeek-OCR Speed

| Metric | Value | Notes |
|---|---|---|
| Throughput (A100 40G) | ~2-3 pages/sec | ~200K pages/day |
| Token generation rate | ~2,500 tokens/sec | On A100 |
| Tokens per page (typical) | 500-2000 | Simple to complex |

---

## 5. The Fundamental Architectural Comparison

### The Three Paradigms

| Aspect | Traditional Pipeline (Marker) | Decoupled VLM (MinerU 2.5) | Monolithic VLM (DeepSeek-OCR-2) |
|---|---|---|---|
| **Text generation** | Batched short autoregressive per line | Autoregressive per crop region | Single long autoregressive per page |
| **Detection** | Encoder-only (parallel, no AR) | Stage 1 VLM (low-res, fast) | Implicit in encoder attention |
| **Layout understanding** | Separate specialized model | Stage 1 global analysis | Learned in causal flow |
| **Tokens decoded per page** | ~20 tokens x 40 lines (batched) | Varies per crop | 500-5000 sequential |
| **GPU utilization** | High (batch inference) | Moderate | Low during decode |
| **Blank space processing** | Skipped (only text regions cropped) | Skipped (targeted crops) | Processed (full page encoded) |
| **Model size** | ~250M total | 1.2B | 3.5B (570M active) |
| **Parallelism** | High (multi-model pipeline) | Medium (two-stage) | Low (sequential decode) |

### Why Token Count Is Everything

The single most important factor determining OCR speed is: **how many autoregressive decoding steps are required per page?**

- **Marker/Surya**: ~20 tokens per line x 40 lines = 800 total decode steps, BUT batched 40-at-a-time = effectively ~20 sequential steps
- **MinerU 2.5**: Variable per crop, but crops are small and can be processed in parallel
- **DeepSeek-OCR-2**: 500-5000 sequential decode steps, no batching across regions

This creates an order-of-magnitude difference in effective sequential computation.

---

## 6. CTC vs. Autoregressive Decoders for Text Recognition

### CTC (Connectionist Temporal Classification)

Used by: Tesseract, PaddleOCR (PP-OCR), traditional CRNN pipelines

```
Image -> CNN features -> BiLSTM -> CTC alignment -> Text
         (parallel)      (parallel)  (parallel)
```

- ALL character positions predicted in a SINGLE forward pass
- Post-processing removes blanks and duplicates
- No sequential token dependency
- Speed: **up to 12x faster** than attention-based decoders (documented by SVTRv2 research)
- Trade-off: no built-in language model context

### SVTRv2: CTC Beating Encoder-Decoder Models (ICCV 2025)

SVTRv2 demonstrated that CTC-based recognition can **surpass** encoder-decoder models in both accuracy AND speed:
- Uses multi-size resizing strategy to handle text irregularity
- Semantic guidance module integrates linguistic context (can be dropped at inference for speed)
- Feature rearrangement module handles CTC alignment challenges
- Result: CTC + smart encoder > encoder-decoder for scene text

### Autoregressive Attention Decoder

Used by: Surya/Donut, TrOCR, Nougat, DeepSeek-OCR-2

```
Image -> Encoder -> Decoder (token by token, sequential)
```

- Each output token attends to encoder features AND all previous tokens
- Produces higher quality output with implicit language modeling
- But: O(n) sequential decoding steps

### Speed Comparison

| Decoder Type | Relative Speed | Accuracy | Language Context |
|---|---|---|---|
| CTC (CRNN) | **Fastest** (1x baseline) | Good | None (external LM needed) |
| CTC + smart encoder (SVTRv2) | Fast (~1-2x CRNN) | Excellent | Integrated via semantic guidance |
| Attention decoder (Donut/TrOCR) | Slow (~5-12x slower than CTC) | Excellent | Built-in |
| Full VLM decoder (DeepSeek-OCR-2) | Slowest (for full page) | Best (semantic understanding) | Full LLM capabilities |

---

## 7. Can DeepSeek-OCR-2 Be Made Faster?

### Option A: Speculative Decoding (1.5-2x speedup)

SpecVLM (2025) demonstrated that speculative decoding can accelerate VLM inference:
- A small "draft" model proposes multiple tokens
- The full model verifies them in parallel
- Achieves **1.5-2x speedup** while preserving output quality
- Limitation: speedup diminishes with large batch sizes (less spare compute for draft)

### Option B: Non-Autoregressive Decoding (Possible but unproven for OCR)

Research exists on non-autoregressive VLMs (CVPR 2024):
- Replace causal attention mask with bidirectional attention
- Use learnable query tokens that generate all outputs in parallel
- Results: "consistently outperform autoregressive models on visual grounding tasks"
- BUT: not yet proven for high-fidelity document OCR where output quality is critical

### Option C: Hybrid Pipeline (Most Practical)

Use DeepSeek-OCR-2's encoder for layout understanding, but delegate text recognition to a faster CTC/pipeline-based recognizer:
- Encoder detects and classifies regions
- CTC-based recognizer handles plain text (fast)
- VLM decoder only invoked for complex elements (tables, equations)
- Could achieve 3-5x speedup on typical documents

### Option D: Token Compression (Already in Use)

DeepSeek-OCR's original innovation was vision token compression:
- Compress rich 2D pages into compact vision token sets
- At 10x compression: 97% OCR precision
- At 20x compression: ~60% accuracy
- This is already deployed; further compression trades accuracy for speed

### Option E: Parallel Page Processing via vLLM Continuous Batching

Multiple pages CAN be processed simultaneously through the decoder using vLLM's continuous batching:
- Different pages at different decoding stages share GPU resources
- While page A is decoding token 500, page B is decoding token 200
- Improves **throughput** (pages/sec) but NOT **latency** (time per page)
- DeepSeek-OCR with vLLM: ~100 concurrent sequences (default)

---

## 8. Head-to-Head Speed Benchmarks Summary

### GPU (L4 - Mid-Range)

| Tool | sec/page | pages/sec | Paradigm |
|---|---|---|---|
| MinerU v1 (pipeline) | 0.21 | 4.8 | Pipeline (YOLO + PP-OCR) |
| Docling | 0.49 | 2.0 | Pipeline |
| Marker | 0.86 | 1.2 | Pipeline (Surya) |
| DeepSeek-OCR-2 | ~0.5-1.0 | ~1-2 | Monolithic VLM |

### GPU (A100 80G - High-End)

| Tool | pages/sec | Notes |
|---|---|---|
| Marker (22 processes) | ~122 | Multi-process parallelism |
| Marker (single, batched) | ~25 | Batched line recognition |
| MinerU 2.5 (optimized) | ~4.5 | vllm-async-engine |
| LightOnOCR-2-1B | ~5.7 | End-to-end VLM, but optimized |
| DeepSeek-OCR | ~2-3 | Single A100 40G |
| OlmOCR | ~0.4-4 | Depends on optimization |

### CPU Only

| Tool | sec/page (x86) | sec/page (M3 Max) |
|---|---|---|
| Docling | 3.1 | 1.27 |
| MinerU v1 | 3.3 | Did not finish |
| Marker | 16+ | 4.2 |

---

## 9. The Speed-Quality Spectrum (2025-2026 State of the Art)

```
SPEED <<<--------------------------------------------------->>> QUALITY

Tesseract/     PaddleOCR    Marker    MinerU     LightOn   DeepSeek  DeepSeek
CRNN+CTC      (PP-OCRv5)   (Surya)   2.5        OCR-2     OCR       OCR-2
~50+ pg/s     ~10+ pg/s    ~25 pg/s  ~4.5 pg/s  ~5.7pg/s  ~2-3pg/s  ~2-3pg/s
Low qual      Good qual    Good qual V.Good     V.Good    Excellent  SOTA
No layout     Basic layout Full MD   Full MD    Full MD   Full MD    Full MD
```

Key observations:
- **Marker achieves the best speed/quality balance** for most use cases
- **MinerU 2.5** bridges pipeline and VLM approaches with its decoupled design
- **LightOnOCR-2** shows that end-to-end VLMs CAN be fast (5.7 pg/s) with the right architecture (1B params, no pipeline overhead)
- **DeepSeek-OCR-2** trades speed for SOTA quality and semantic understanding

---

## 10. Key Takeaways for Our Project

### The Fundamental Truth
> The autoregressive decoder is the bottleneck. Any approach that produces ALL output through sequential token generation will be slower than approaches that decompose the problem into parallel sub-tasks.

### Why Marker Is 10-50x Faster Than Naive VLM OCR
1. **Batched short-sequence recognition** vs single long-sequence generation
2. **Detection/layout models are encoder-only** (no AR decoding at all)
3. **Processes only text regions**, skipping whitespace
4. **Small specialized models** with high GPU utilization
5. **Multi-process parallelism** (22 processes on H100)

### Why MinerU 2.5 Is a Smart Middle Ground
1. **Decoupled stages** avoid processing full-res pages end-to-end
2. **Coarse-to-fine** reduces token count by order of magnitude
3. **VLM quality** with pipeline-like efficiency
4. **Targeted high-res crops** only where content exists

### Implications for Our DeepSeek-OCR-2 Integration
1. **Speed ceiling is ~2-3 pages/sec on A100** for pure autoregressive VLM approach
2. **Hybrid approach is most promising**: use DeepSeek-OCR-2 for complex elements only, faster tools for bulk text
3. **vLLM continuous batching** improves throughput but not per-page latency
4. **Speculative decoding** could add 1.5-2x speedup with minimal quality loss
5. **For speed-critical use cases**: Marker (25+ pg/s) or MinerU 2.5 (4.5 pg/s) are objectively better choices

---

## Sources

### Architecture and Technical Details
- [Surya OCR - GitHub](https://github.com/datalab-to/surya) - Detection (EfficientViT) + Recognition (modified Donut)
- [Marker - GitHub](https://github.com/datalab-to/marker) - Pipeline architecture
- [How Marker Converts PDFs to Markdown](https://kevinhu.io/notes/how-marker-works/) - Pipeline breakdown
- [MinerU2.5 Paper (arXiv)](https://arxiv.org/abs/2509.22186) - Decoupled VLM architecture
- [MinerU2.5 Full Paper](https://arxiv.org/html/2509.22186v2) - Technical details
- [DeepSeek-OCR-2 Paper (arXiv)](https://arxiv.org/abs/2601.20552) - Visual Causal Flow
- [DeepSeek-OCR Paper (arXiv)](https://arxiv.org/abs/2510.18234) - Contexts Optical Compression
- [Donut - OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- [Nougat Paper (arXiv)](https://arxiv.org/abs/2308.13418) - Neural Optical Understanding

### Speed and Decoder Analysis
- [Technical Analysis of Modern Non-LLM OCR Engines (IntuitionLabs)](https://intuitionlabs.ai/articles/non-llm-ocr-technologies) - CTC vs autoregressive deep dive
- [SVTRv2: CTC Beats Encoder-Decoder Models (arXiv)](https://arxiv.org/abs/2411.15858) - CTC superiority
- [Non-autoregressive Seq2Seq VLMs (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_Non-autoregressive_Sequence-to-Sequence_Vision-Language_Models_CVPR_2024_paper.pdf)
- [SpecVLM: Fast Speculative Decoding in VLMs](https://arxiv.org/html/2509.11815v1) - 1.5-2x speedup
- [Lookahead Decoding (LMSYS)](https://lmsys.org/blog/2023-11-21-lookahead-decoding/) - Breaking sequential dependency
- [LLM Inference Basics (BentoML)](https://bentoml.com/llm/llm-inference-basics/how-does-llm-inference-work)

### Benchmarks and Comparisons
- [VLM-OCR Recipes on GPU Infrastructure (HuggingFace)](https://huggingface.co/blog/florentgbelidji/vlm-ocr-recipes-gpu-infra) - Cost/speed analysis
- [Supercharge OCR Pipelines with Open Models (HuggingFace)](https://huggingface.co/blog/ocr-open-models) - Pipeline vs end-to-end
- [LightOnOCR-2-1B (HuggingFace)](https://huggingface.co/blog/lightonai/lightonocr-2) - 5.7 pg/s, SOTA accuracy
- [8 Top Open-Source OCR Models Compared (Modal)](https://modal.com/blog/8-top-open-source-ocr-models-compared)
- [7 Best Open-Source OCR Models 2025 (E2E Networks)](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- [12 Open-Source PDF Parsing Comparison](https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html)
- [PDF to Markdown Tools Deep Dive (Jimmy Song)](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
- [Docling Technical Report (arXiv)](https://arxiv.org/html/2501.17887v1) - Speed benchmarks
- [DeepSeek-OCR 2 Guide (AiCybr)](https://aicybr.com/blog/deepseek-ocr-2-guide)
- [Batch Inference with DeepSeek-OCR (SkyPilot)](https://blog.skypilot.co/skypilot-pools-deepseek-ocr/)
- [Replicate Datalab Marker Blog](https://replicate.com/blog/datalab-marker-and-ocr-fast-parsing)
- [OlmOCR Hacker News Discussion](https://news.ycombinator.com/item?id=43174298)
- [fast360 OCR Arena](https://github.com/shijincai/fast360) - Benchmark 7 models
