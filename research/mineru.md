# MinerU (opendatalab/MinerU) - Research Notes

**Date:** 2026-02-06
**Repository:** https://github.com/opendatalab/MinerU
**Current Version:** 2.7.2 (Jan 2026) / Model: MinerU2.5
**License:** AGPL-3.0
**Pricing:** Free / Open Source (AGPL restrictions apply to commercial use)

---

## 1. Architecture & Pipeline

MinerU has evolved through two major generations:

### MinerU Classic Pipeline (v1.x - v2.4)
A modular pipeline composed of separate specialized models:
- **Layout Detection:** DocLayout-YOLO (AGPL-licensed) for region detection
- **OCR:** PaddleOCR for text recognition
- **Table Recognition:** Dedicated table structure model
- **Formula Recognition:** Separate LaTeX conversion model
- Reading order analysis, header/footer removal, page number stripping

### MinerU 2.5 (Current State-of-the-Art)
A **decoupled vision-language model** with a unified 1.2B parameter architecture:

**Stage I - Coarse Layout Analysis:**
- Downsamples page images to 1036x1036 px
- Rapid structural parsing identifies text blocks, figures, tables, formulas
- Classifies formula regions as atomic or compound

**Stage II - Fine-Grained Recognition:**
- Crops high-resolution patches from detected layout regions
- Performs text OCR, formula-to-LaTeX, table-to-HTML at native resolution
- Uses the ADR (Atomic Decomposition & Recombination) framework for complex formulas

**Model Components:**
- **Vision Encoder:** NaViT-based native-resolution encoder, 675M parameters, dynamic 2D positional embeddings
- **Language Model:** Qwen2-Instruct, ~500M parameters
- **Patch Merger:** Pixel-unshuffle for connecting vision to language
- **Total:** ~1.2B parameters

Available on HuggingFace: `opendatalab/MinerU2.5-2509-1.2B` (also available in GGUF and MLX formats)

---

## 2. Conversion Speed Benchmarks

### MinerU 2.5 (VLM Backend) - GPU Throughput

| GPU | Pages/Second | Seconds/Page |
|-----|-------------|--------------|
| NVIDIA H200 (141GB) | 4.47 | 0.22 |
| NVIDIA A100 (80GB) | 2.12 | 0.47 |
| NVIDIA RTX 4090 (24GB) | 1.70 | 0.59 |

Token generation speed: 2,337 tokens/second (A100), up from 1,045 tokens/sec unoptimized.

### MinerU Classic Pipeline - Cross-Tool Comparison (from Docling paper, arxiv 2501.17887)

**x86 CPU (8 threads):**

| Tool | Seconds/Page | Pages/Minute |
|------|-------------|-------------|
| Docling | 3.1 | ~19 |
| **MinerU (pipeline)** | **3.3** | **~18** |
| Unstructured | 4.2 | ~14 |
| Marker | 16+ | ~4 |

**NVIDIA L4 GPU:**

| Tool | Seconds/Page | Pages/Minute |
|------|-------------|-------------|
| **MinerU (pipeline)** | **0.21** | **~286** |
| Docling | 0.49 | ~122 |
| Marker | 0.86 | ~70 |
| Unstructured | N/A (no GPU accel) | N/A |

**Key takeaway:** MinerU is the fastest tool with GPU acceleration by a wide margin - 2.3x faster than Docling, 4x faster than Marker on NVIDIA L4. On CPU it is competitive with Docling.

### Speed Summary
- **GPU (pipeline, L4):** ~0.21 sec/page (~286 pages/min)
- **GPU (VLM, RTX 4090):** ~0.59 sec/page (~102 pages/min)
- **GPU (VLM, A100):** ~0.47 sec/page (~128 pages/min)
- **CPU (pipeline):** ~3.3 sec/page (~18 pages/min)

Note: The pipeline backend is faster but less accurate (82+ score). The VLM backend is slower but significantly more accurate (90+ score). Choose based on your quality requirements.

---

## 3. GPU Requirements

### Minimum VRAM by Backend

| Backend | Min VRAM | Min RAM | Accuracy Score |
|---------|----------|---------|---------------|
| Pipeline (CPU-capable) | 6 GB (or 0 for CPU-only) | 8 GB | 82+ |
| VLM auto-engine | 8 GB | 16 GB | 90+ |
| Hybrid auto-engine | 10 GB | 16 GB | 90+ |
| HTTP client modes | 3 GB | 8 GB | Depends on server |

### Peak VRAM Usage
- **Typical workloads:** 8-16 GB
- **Large/complex multi-page PDFs:** peaks at 20-25 GB (observed on 48GB GPU)
- **Minimum GPU architecture:** NVIDIA Volta or later (RTX 20xx+), or Apple Silicon

### Recommended Hardware
- **Minimum practical:** 8 GB VRAM GPU for VLM backend
- **Comfortable:** 16-24 GB VRAM
- **Optimal:** A100 80GB or H200 for maximum throughput
- **RAM:** 32 GB+ recommended for local inference
- **Disk:** 20 GB+ SSD for model weights and dependencies

---

## 4. Accuracy & Quality

### OmniDocBench v1.5 Scores (MinerU 2.5)

| Metric | MinerU 2.5 Score |
|--------|-----------------|
| **Overall Score** | **90.67** |
| Text Edit Distance | 0.047 (lower is better) |
| Formula CDM | 88.46 |
| Table TEDS | 88.22 |
| Table TEDS-S | 92.38 |
| Reading Order Edit Distance | 0.044 (lower is better) |

### OCR Accuracy (Ocean-OCR Benchmark)

| Language | Edit Distance | F1-Score |
|----------|--------------|----------|
| English | 0.033 | 0.945 |
| Chinese | 0.082 | 0.965 |

### Comparative Standing
- **Outperforms** MonkeyOCR-pro-3B (3.7B params) by +1.82 points at 4x throughput
- **Outperforms** dots.ocr (3.0B params) by +2.26 points at 7x throughput
- **Outperforms** general-purpose VLMs including GPT-4o, Gemini-2.5 Pro, Qwen2.5-VL-72B on most document parsing tasks
- **Best-in-class** reading order prediction consistency
- Pipeline tools (MinerU, Mathpix) outperform VLMs like GPT-4o for English text (Edit distance ~0.058-0.101) and formulas (CDM ~71-76%)

### Quality Strengths
- Excellent table recognition rendered as HTML
- Strong formula-to-LaTeX conversion via ADR framework
- Reliable header/footer/page number removal
- Good handling of multi-column layouts and complex structures
- Strong reading order prediction

### Quality Weaknesses
- Mixed Chinese/English content accuracy drops (PaddleOCR limitation in classic pipeline)
- Handwritten content detected as figures rather than recognized
- Lesser-known languages (diacritical marks, Arabic script) may have OCR inaccuracies
- Image cropping can be incomplete in some cases

---

## 5. RTX 3090 Compatibility (24GB VRAM)

**Yes, it can run on an RTX 3090.** Here is the detailed analysis:

- RTX 3090 uses Ampere architecture (post-Volta), so it meets the architecture requirement
- 24 GB VRAM is above the 8-10 GB minimum for VLM/hybrid backends
- The RTX 4090 (also 24 GB VRAM) benchmarks at **1.70 pages/second** with MinerU 2.5
- The RTX 3090 will be somewhat slower than the 4090 due to fewer CUDA cores and lower memory bandwidth, but should still achieve roughly 1.0-1.5 pages/second (estimated)

### Potential Issues on RTX 3090
- **Large complex PDFs** may peak at 20-25 GB VRAM, which could cause OOM on 24GB
- **Mitigation:** Process very large documents in smaller batches, or use the pipeline backend (6 GB VRAM minimum)
- **Recommendation:** Use the VLM backend for quality, fall back to pipeline for extremely large/complex documents that cause VRAM pressure

### Expected Performance on RTX 3090

| Backend | Estimated Speed | VRAM Usage |
|---------|----------------|-----------|
| Pipeline | ~0.3-0.5 sec/page | 6-10 GB |
| VLM | ~0.7-1.0 sec/page | 8-20 GB |
| Hybrid | ~0.5-0.8 sec/page | 10-20 GB |

(Estimates based on RTX 4090 benchmarks with ~15-30% reduction for 3090's lower compute)

---

## 6. Multi-Page PDF Handling

- Processes multi-page PDFs natively - no manual splitting required
- Pages are processed sequentially through the pipeline
- Cross-page table merging is supported (added in v2.7.2)
- Reading order is maintained across pages
- Output is a single concatenated markdown file with all pages
- Headers, footers, and page numbers are automatically stripped for semantic coherence
- VRAM usage scales with page complexity, not page count (processes one page at a time)

---

## 7. Batch Processing Support

### CLI Batch Mode
```bash
# Process entire directory of PDFs
mineru -p /path/to/pdf_directory -o /path/to/output -m auto

# Specify backend
mineru -p /path/to/pdf_directory -o /path/to/output -b pipeline

# CPU-only mode
mineru -p /path/to/pdf_directory -o /path/to/output -b pipeline
```

### Python SDK
Full Python API available for programmatic batch processing and integration.

### Concurrency Considerations
- **Single process:** Safe and straightforward
- **Multi-threading:** NOT safe due to PyTorch CUDA limitations
- **Multi-process:** Supported, but each worker must initialize its own model instance and device
- **Table parsing:** Still CPU-bound and single-threaded - can be a bottleneck
- **vLLM async engine:** Supports concurrent inference at 2.12 fps on A100

### Output Per Document
Each processed PDF generates:
- Main markdown file
- Extracted images directory
- JSON structure data
- Auxiliary debugging files

---

## 8. Known Limitations

### Content Type Limitations
- **Vertical text:** Limited support
- **Code blocks:** Not supported in layout model
- **Comic books, art albums, primary school textbooks:** Cannot parse well
- **Handwritten notes:** Detected as figures, not recognized as text
- **Complex tables:** May have row/column recognition errors

### Language Limitations
- Mixed Chinese/English content accuracy decreases
- Lesser-known languages (diacritical marks in Latin script, Arabic script) may have OCR inaccuracies
- PaddleOCR-dependent quality for classic pipeline

### Technical Limitations
- Table of contents and lists recognized via rules - uncommon formats may fail
- Reading order can be wrong under extremely complex layouts
- Image cropping may be incomplete in some cases
- Multi-threading not safe (CUDA limitation) - must use multi-process
- Table parsing is CPU-bound and single-threaded
- Windows limited to Python 3.10-3.12 (Ray dependency)
- VRAM peaks of 20-25 GB possible on complex documents

### AGPL License Implications
- AGPL-3.0 requires derivative works to be open-sourced
- DocLayout-YOLO also AGPL-licensed
- Commercial use requires careful license compliance or potential commercial licensing arrangement
- Future versions may explore more permissive model licenses

---

## 9. License & Pricing

| Aspect | Details |
|--------|---------|
| **License** | AGPL-3.0 |
| **Cost** | Free / Open Source |
| **Commercial Use** | Allowed under AGPL terms (must open-source derivative works) |
| **Model Weights** | Freely available on HuggingFace |
| **Key Dependencies** | DocLayout-YOLO (AGPL), PaddleOCR (Apache 2.0), pypdfium2 (used instead of pymupdf to avoid additional AGPL issues) |

The AGPL license is the main consideration for commercial deployment. If you build a service using MinerU, you must make your source code available. This is a significant constraint for proprietary SaaS applications.

---

## 10. Installation Complexity

### Simplified Installation (v2.7+)
```bash
# Create virtual environment
python -m venv mineru_env
source mineru_env/bin/activate  # Linux/Mac
# or: mineru_env\Scripts\activate  # Windows

# Install everything
uv pip install -U "mineru[all]"
```

This installs all backends and dependencies in one command. Major improvement over earlier versions.

### System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.10-3.13 (Windows: 3.10-3.12) | 3.11 |
| RAM | 16 GB | 32 GB+ |
| Disk | 20 GB (SSD preferred) | 40 GB+ |
| GPU | Optional (Volta+) | 16-24 GB VRAM |
| OS | Linux 2019+, Windows, macOS 14+ | Linux |

### Installation Methods
1. **pip/uv** (recommended): `uv pip install -U "mineru[all]"`
2. **From source:** Clone repo + `uv pip install -e .[all]`
3. **Docker:** Pre-built images available

### Installation Pain Points
- Model weights download (~5-10 GB) on first run
- CUDA toolkit must match PyTorch version
- Windows has Python 3.13 limitation (Ray dependency)
- Some older `magic-pdf[full]` installs needed `detectron2` compiled from source (no longer an issue in v2.7+)
- Overall complexity: **Medium** - significantly improved from earlier versions but still requires GPU driver/CUDA setup

---

## Summary Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Speed (GPU)** | Excellent | 0.21 sec/page (pipeline/L4), fastest in class |
| **Speed (CPU)** | Good | 3.3 sec/page, competitive |
| **Accuracy** | Excellent | 90.67 overall, beats GPT-4o on document parsing |
| **RTX 3090** | Yes | Works well, may hit VRAM limits on very complex docs |
| **Batch Processing** | Good | CLI directory support, Python SDK |
| **Installation** | Medium | Much improved, but GPU setup still needed |
| **License** | Restrictive | AGPL-3.0 limits commercial proprietary use |
| **Active Development** | Very Active | Monthly releases, strong community |

### Bottom Line
MinerU is the leading open-source PDF-to-markdown tool in terms of both speed and accuracy when GPU-accelerated. The 1.2B parameter VLM model achieves state-of-the-art results while being deployable on consumer hardware (RTX 3090/4090). The main concerns are the AGPL license for commercial use and potential VRAM pressure on very complex documents with 24GB cards.

---

## Sources

- [MinerU GitHub Repository](https://github.com/opendatalab/MinerU)
- [MinerU Official Documentation](https://opendatalab.github.io/MinerU/)
- [MinerU Quick Start Guide](https://opendatalab.github.io/MinerU/quick_start/)
- [MinerU2.5 Paper (arxiv 2509.22186)](https://arxiv.org/html/2509.22186v2)
- [MinerU2.5 HuggingFace Model](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)
- [Docling Paper with Speed Benchmarks (arxiv 2501.17887)](https://arxiv.org/html/2501.17887v1)
- [OmniDocBench (CVPR 2025)](https://github.com/opendatalab/OmniDocBench)
- [MinerU 2.5 Tutorial - Sonu Sahani](https://sonusahani.com/blogs/mineru)
- [12-Tool Comparative Evaluation - Liduos](https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html)
- [PDF to Markdown Deep Dive - Jimmy Song](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
- [MinerU Batch Processing Discussion](https://github.com/opendatalab/MinerU/discussions/3738)
- [MinerU NeuralHive Overview](https://neurohive.io/en/state-of-the-art/mineru2-5-open-source-1-2b-model-for-pdf-parsing-outperforms-gemini-2-5-pro-on-benchmarks/)
