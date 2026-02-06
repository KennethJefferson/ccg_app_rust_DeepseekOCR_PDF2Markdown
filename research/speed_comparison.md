# PDF-to-Markdown OCR Speed Comparison

**Date:** 2026-02-06
**Baseline:** DeepSeek-OCR-2 on RTX 3090 at ~8 pages/min (~0.13 pages/sec), 735-page book = ~92 minutes

---

## Executive Summary

Our current setup is **60-400x slower** than what's achievable. The fastest open-source tools on datacenter GPUs (H100/H200) process 5-25 pages/sec. Even on consumer GPUs (RTX 3090/4090), several tools hit 1-5 pages/sec. Cloud APIs like Mistral OCR 3 hit ~33 pages/sec. A 735-page book in under 10 minutes requires **>1.2 pages/sec sustained**, which multiple tools can achieve.

---

## Master Comparison Table

All speed figures are pages/second unless noted. Costs are USD per 1M pages.

### Open-Source / Self-Hosted Models

| Tool | Pages/sec (H100) | Pages/sec (RTX 4090) | Pages/sec (RTX 3090) | Accuracy (olmOCR-Bench) | VRAM Needed | Cost/1M pages (H100) | 735pg Book Time (H100) | 735pg Book Time (RTX 4090) |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Marker (batch, multi-worker)** | 25-122 | 14.5-18.0 | ~0.03-5* | N/A (pipeline) | 3.5-5GB/worker | ~$80-320 | **6-30 sec** | **41-51 sec** |
| **LightOnOCR-2-1B** | 5.71 | ~3.0 (est.) | ~1.8 (est.) | 76.1 | ~4GB | $141 | **2.1 min** | ~4.1 min |
| **DeepSeek-OCR** | 4.65 | ~2.5 (est.) | **~0.13 (measured)** | 75.7 | ~8-12GB | $168 | **2.6 min** | ~4.9 min |
| **MinerU 2.5** | ~3.5 (est.) | 1.70 | ~0.9 (est.) | 90.67 (OmniDoc) | ~6-8GB | ~$220 (est.) | **3.5 min** | **7.2 min** |
| **PaddleOCR-VL** | 2.20 | ~1.2 (est.) | ~0.7 (est.) | 80.0 | ~2-4GB | $355 | **5.6 min** | ~10.2 min |
| **dots.ocr** | 1.94 | ~1.0 (est.) | ~0.6 (est.) | 79.1 | ~4-6GB | $402 | **6.3 min** | ~12.3 min |
| **OlmOCR-2** | 1.78 | ~0.9 (est.) | ~0.5 (est.) | 82.4 | ~8-12GB | $439 | **6.9 min** | ~13.6 min |
| **Chandra** | 1.29 | ~0.7 (est.) | ~0.4 (est.) | **83.1** (highest) | ~12-16GB (9B params) | $605 | **9.5 min** | ~17.5 min |
| **Uni-Parser (8x4090D cluster)** | N/A | 20.0 (8 GPUs) | N/A | N/A | 8x 24GB | N/A | N/A | **37 sec (8 GPUs)** |
| **MinerU 2.5 (H200)** | N/A | N/A | N/A | 90.67 (OmniDoc) | H200 | N/A | **2.7 min** | N/A |
| **Docling** | ~2.0 | ~1.0 (est.) | ~0.5 (est.) | Good (qualitative) | ~4-8GB | ~$390 (est.) | **6.1 min** | ~12.3 min |
| **Nougat (Meta)** | ~0.5 (est.) | ~0.3 (est.) | ~0.15 (est.) | Good on arXiv only | ~6-10GB | ~$1,500+ | ~24.5 min | ~40.8 min |
| **Nanonets OCR 2** | 1.12 | ~0.6 (est.) | ~0.35 (est.) | 69.5 | ~8-12GB | $697 | **10.9 min** | ~20.4 min |

*Marker on RTX 3090: GitHub issue #919 reports 0.014-0.03 pages/sec due to a known performance bug; fixed configurations should yield ~5 pages/sec.*

### Cloud API Services

| Service | Speed (pages/min) | Speed (pages/sec) | Cost/1K pages | 735pg Book Time | Batch Discount | Quality |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Mistral OCR 3** | ~2,000 | ~33 | $2.00 | **~22 sec** | 50% ($1/1K) | SOTA tables (96.6%), handwriting (88.9%) |
| **Gemini 3 Flash** | ~500+ (est.) | ~8+ (est.) | ~$0.17* | **~1.5 min** | Batch API available | >95% printed, 85% handwriting |
| **Gemini 2.5 Pro** | ~200 (est.) | ~3 (est.) | ~$1.25* | **~4.1 min** | Batch API available | Best overall formatting preservation |
| **Mathpix** | High (enterprise) | N/A | $3.50 | Minutes | N/A | Excellent for STEM/math |
| **Google Document AI** | 120 | 2.0 | $1.50** | **~6.1 min** | Volume discounts | 98% mixed datasets |
| **Amazon Textract** | ~60-120 (est.) | ~1-2 (est.) | $1.50** | **~6-12 min** | Volume pricing | Good general OCR |
| **LlamaParse** | ~120 (est.) | ~2 (est.) | ~$3.00 (est.) | **~6 min** | N/A | Good structure preservation |
| **Reducto** | N/A | N/A | $15.00 | N/A | Volume tiers | High accuracy, multi-pass |

*Gemini costs estimated based on token pricing (~750 tokens/page). **Per 1K pages at standard pricing.

---

## Answering Your 5 Questions

### 1. Which tools can process a 735-page PDF in under 10 minutes?

**On datacenter GPU (H100/H200):**
- Marker batch mode: **6-30 seconds** (the clear winner for raw speed)
- LightOnOCR-2: ~2.1 min
- DeepSeek-OCR: ~2.6 min
- MinerU 2.5 (H200): ~2.7 min
- PaddleOCR-VL: ~5.6 min
- dots.ocr: ~6.3 min
- OlmOCR-2: ~6.9 min
- Chandra: ~9.5 min

**On consumer GPU (RTX 4090):**
- Marker batch mode: **41-51 seconds**
- LightOnOCR-2: ~4.1 min
- MinerU 2.5: ~7.2 min
- Uni-Parser (8x 4090): ~37 seconds

**On RTX 3090 (your hardware):**
- Marker (properly configured): likely ~2.5-5 min (single-worker, need to verify bug fix)
- LightOnOCR-2: ~6.8 min (estimated)

**Cloud APIs:**
- Mistral OCR 3: **~22 seconds**
- Gemini 3 Flash: ~1.5 min
- Gemini 2.5 Pro: ~4 min

### 2. What's the fastest open-source PDF-to-markdown with decent quality?

**Winner: Marker** -- 25-122 pages/sec on H100 in batch mode, 14.5-18 pages/sec on RTX 4090. Purpose-built for PDF-to-markdown conversion. Outputs markdown + JSON natively. Supports multi-GPU, multi-worker parallelism.

**Runner-up for quality: MinerU 2.5** -- Scores 90.67 on OmniDocBench (highest among all tested tools), 1.70 pages/sec on RTX 4090. Only 1.2B parameters but outperforms 72B models on document parsing. Best balance of speed and accuracy.

**Runner-up for speed: LightOnOCR-2-1B** -- 5.71 pages/sec on H100, single forward pass per page (no retries). Cheapest self-hosted option at $141/1M pages. 1.73x faster than DeepSeek-OCR.

### 3. Are there cloud API solutions that are faster?

Yes, significantly:

| API | 735-page Time | Cost for 735 pages |
|-----|:-:|:-:|
| Mistral OCR 3 | ~22 sec | $1.47 (standard) / $0.74 (batch) |
| Gemini 3 Flash | ~1.5 min | ~$0.12 |
| Gemini 2.5 Pro | ~4 min | ~$0.92 |
| Google Document AI | ~6 min | ~$1.10 |

**Gemini 3 Flash is the best value**: ~$0.12 per book, sub-2-minute processing, >95% accuracy. For 20 books that's ~$2.40 total.

**Mistral OCR 3 is the fastest**: 22 seconds per book, best table/handwriting accuracy, $0.74/book with batch discount.

### 4. Can any of these handle batch processing of 20+ PDFs efficiently?

**Best for local batch processing:**

| Tool | 20 x 735pg Books | Hardware | Estimated Time |
|------|:-:|:-:|:-:|
| Marker (multi-GPU) | 14,700 pages | 1x H100 | ~2-10 min |
| Marker (multi-GPU) | 14,700 pages | 1x RTX 4090 | ~14-17 min |
| Marker (multi-GPU) | 14,700 pages | 4x RTX 4090 | ~4-5 min |
| MinerU 2.5 | 14,700 pages | 1x RTX 4090 | ~2.4 hours |
| LightOnOCR-2 | 14,700 pages | 1x H100 | ~44 min |
| Uni-Parser | 14,700 pages | 8x RTX 4090D | ~12 min |

**Best for cloud batch processing:**

| API | 20 x 735pg Books | Estimated Time | Total Cost |
|-----|:-:|:-:|:-:|
| Mistral OCR 3 (batch) | 14,700 pages | ~7-8 min | $14.70 |
| Gemini 3 Flash (batch) | 14,700 pages | ~20-30 min | ~$2.50 |
| Gemini 2.5 Pro | 14,700 pages | ~1-2 hours | ~$18.40 |

Marker has explicit batch CLI: `marker_chunk_convert` with `NUM_DEVICES` and `NUM_WORKERS` environment variables. MinerU and Docling also support batch directories.

### 5. What's the state of the art for speed vs quality tradeoff?

The Pareto frontier as of early 2026:

```
Accuracy
  ^
  |  * MinerU 2.5 (90.67 OmniDoc, 1.70 p/s on 4090)
  |
  |         * Chandra (83.1, 1.29 p/s on H100)
  |       * OlmOCR-2 (82.4, 1.78 p/s)
  |     * PaddleOCR-VL (80.0, 2.20 p/s)
  |   * dots.ocr (79.1, 1.94 p/s)
  |                       * LightOnOCR (76.1, 5.71 p/s)
  |                     * DeepSeek-OCR (75.7, 4.65 p/s)
  |
  |                                           * Marker (pipeline, 25+ p/s)
  +---------------------------------------------------------> Speed (p/s)
```

**Key insight:** MinerU 2.5 breaks the speed-accuracy tradeoff by achieving the highest document parsing accuracy with a tiny 1.2B model, but Marker remains the raw speed champion because it's a pipeline (layout + OCR + LLM cleanup) rather than a single model.

For technical books specifically, MinerU 2.5 and Marker are the two best options, with MinerU handling complex tables and formulas better while Marker handles book-structured documents faster.

---

## Recommendations for Your Use Case

**Use case: Batch converting technical/programming books, currently ~8 pages/min on RTX 3090**

### Option A: Quick Win -- Fix Current Setup (Free)
Your DeepSeek-OCR-2 at 0.13 pages/sec on RTX 3090 is suspiciously slow compared to the H100 benchmark of 4.65 pages/sec. The ~36x gap is larger than the typical H100/3090 performance difference (~3-5x for inference). Investigate:
- Are you using batch inference or processing page-by-page?
- Is the model quantized (FP16/INT8)?
- Check GPU utilization -- is the GPU actually being used?
- Review image preprocessing DPI (200 DPI is sufficient, 300+ is wasteful)

Expected improvement: **2-5x** (getting to ~0.3-0.6 pages/sec = ~20-36 pages/min)

### Option B: Switch to Marker (Free, Best Speed)
- Install marker-pdf, run in batch mode with multiple workers
- On RTX 3090 with 24GB VRAM: can run 4-5 workers at 3.5-5GB each
- **CAUTION:** GitHub issue #919 reports severe performance regression on RTX 3090 (0.014 pages/sec). Verify this is resolved in latest version before committing.
- Expected: **3-10 pages/sec** if working correctly = 735 pages in **1.2-4.1 min**
- Batch 20 books: possibly under 1 hour on a single RTX 3090
- **Risk:** Quality for tables/math may be lower than DeepSeek-OCR-2

### Option C: Switch to MinerU 2.5 (Free, Best Quality)
- 1.70 pages/sec on RTX 4090, ~0.9 pages/sec on RTX 3090 (estimated)
- **7x faster than your current setup** with significantly better accuracy
- Excels at scientific documents, tables, formulas
- 735 pages in ~13.6 min on RTX 3090 (vs 92 min now)
- 20 books: ~4.5 hours on RTX 3090

### Option D: Switch to LightOnOCR-2-1B (Free, Best Value)
- Only 1B parameters, minimal VRAM
- Estimated ~1.5-2.0 pages/sec on RTX 3090
- 735 pages in ~6-8 min
- Cheapest to run, single forward pass per page
- **Trade-off:** Lower accuracy (76.1 vs 83+ for Chandra/OlmOCR)

### Option E: Cloud API -- Gemini 3 Flash (Cheapest)
- ~$0.12 per 735-page book
- 20 books = ~$2.50
- Sub-2-minute per book
- >95% accuracy
- No GPU needed at all
- **Best option if you have internet and budget is not zero**

### Option F: Cloud API -- Mistral OCR 3 (Fastest + Highest Quality)
- 22 seconds per 735-page book
- $0.74/book with batch discount
- 20 books = $14.70, done in ~8 minutes
- 96.6% table accuracy, 88.9% handwriting
- **Best option if time is the constraint**

### Recommended Path

1. **Immediate:** Try Gemini 3 Flash on a test book ($0.12). Compare quality to your current output.
2. **Short-term:** Install Marker, test on your RTX 3090. If the performance bug is fixed, this is your local speed champion.
3. **For quality-critical books:** Use MinerU 2.5 -- best document parsing accuracy in any benchmark.
4. **For batch production runs:** Use Mistral OCR 3 batch API. $14.70 for all 20 books, done in under 10 minutes, highest quality.

---

## Hardware Scaling Reference

Approximate throughput multipliers relative to RTX 3090 for inference workloads:

| GPU | vs RTX 3090 | Typical OCR pages/sec |
|-----|:-:|:-:|
| RTX 3090 (24GB) | 1.0x | baseline |
| RTX 4090 (24GB) | 1.8-2.2x | 1.5-2.5x faster |
| A100 80GB | 2.5-3.5x | 2-3x faster |
| H100 80GB | 4-6x | 3-5x faster |
| H200 141GB | 5-8x | 4-6x faster |

---

## Sources

- [E2E Networks: 7 Best Open-Source OCR Models 2025 (H100 benchmarks)](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- [Skywork: Chandra OCR Benchmarks](https://skywork.ai/blog/sheets/chandra-ocr-benchmark/)
- [GitHub: Marker README (batch mode benchmarks)](https://github.com/datalab-to/marker)
- [GitHub: Marker Issue #919 (RTX 3090 performance regression)](https://github.com/datalab-to/marker/issues/919)
- [NeurOhive: MinerU 2.5 Benchmarks](https://neurohive.io/en/state-of-the-art/mineru2-5-open-source-1-2b-model-for-pdf-parsing-outperforms-gemini-2-5-pro-on-benchmarks/)
- [GitHub: MinerU Discussion #1226 (speed)](https://github.com/opendatalab/MinerU/discussions/1226)
- [Allen AI: OlmOCR-2 Paper](https://allenai.org/blog/olmocr-2)
- [Medium: LightOnOCR-2-1B vs DeepSeek OCR](https://medium.com/data-science-in-your-pocket/lightonocr-2-1b-best-ocr-model-beats-deepseek-ocr-55871623e0a6)
- [Mistral AI: Mistral OCR 3 Announcement](https://mistral.ai/news/mistral-ocr-3)
- [PyImageSearch: Mistral OCR 3 Technical Review](https://pyimagesearch.com/2025/12/23/mistral-ocr-3-technical-review-sota-document-parsing-at-commodity-pricing/)
- [IntuitionLabs: AI OCR Models Comparison](https://intuitionlabs.ai/articles/ai-ocr-models-pdf-structured-text-comparison)
- [Sergey.fyi: Ingesting Millions of PDFs with Gemini 2.0 Flash](https://www.sergey.fyi/articles/gemini-flash-2)
- [Mathpix Pricing](https://mathpix.com/pricing/api)
- [Google Cloud: Document AI Pricing](https://cloud.google.com/document-ai/pricing)
- [AWS Textract Pricing](https://aws.amazon.com/textract/pricing/)
- [Quantum Zeitgeist: Uni-Parser 20 Pages/Sec](https://quantumzeitgeist.com/analysis-uni-parser-achieves-pdf-pages-per-second-enabling/)
- [Skywork: DeepSeek-OCR Benchmarks 2025](https://skywork.ai/blog/llm/deepseek-ocr-benchmarks-and-performance-test-2025-update/)
- [Procycons: PDF Data Extraction Benchmark 2025](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [HuggingFace: LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- [Docling Technical Report (Arxiv)](https://arxiv.org/html/2408.09869v5)
- [OmniDocBench (CVPR 2025)](https://github.com/opendatalab/OmniDocBench)
