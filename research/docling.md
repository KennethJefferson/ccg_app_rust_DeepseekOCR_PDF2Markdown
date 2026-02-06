# Docling (IBM) - PDF-to-Markdown Conversion Research

**Last updated:** 2026-02-06
**Project:** [github.com/docling-project/docling](https://github.com/docling-project/docling)
**Version reviewed:** v2.72.0 (released 2026-02-03)
**Stars:** 52.3k | **Contributors:** 176 | **Used by:** 2.7k projects
**License:** MIT (code); individual model licenses apply separately

---

## 1. Architecture and Pipeline

Docling uses a modular, multi-stage pipeline architecture with specialized AI models at each step. There are two main pipeline options: the **Standard PDF Pipeline** and the newer **VLM Pipeline** (using Granite-Docling).

### Standard PDF Pipeline

The StandardPdfPipeline is a multi-threaded pipeline with these stages:

1. **PDF Parsing** -- Backend parses the PDF, extracts text tokens, and renders bitmap images of each page. Two backends available: Docling Parse v4 (default, best accuracy) and PyPdfium.
2. **Layout Analysis** -- An RT-DETR object detector (trained on DocLayNet dataset) predicts bounding boxes and classifies page elements (headings, paragraphs, tables, figures, code blocks, etc.).
3. **OCR** (optional) -- For scanned content. Supports EasyOCR, Tesseract, RapidOCR (GPU-accelerated), and macOS native OCR.
4. **Table Structure Recognition** -- TableFormer, a vision-transformer model, recovers table structure including merged cells, headers, and hierarchy. Two modes: `FAST` and `ACCURATE`.
5. **Optional Enrichments** -- Formula detection (CodeFormulaModel), picture classification, image description via VLMs.
6. **Assembly & Post-processing** -- ReadingOrderModel applies heuristics for document structure (sections, lists, nesting). Aggregates all pages, detects language, concatenates fragmented paragraphs across pages.

### Key AI Models

| Model | Architecture | Purpose | Speed (L4 GPU) | Speed (x86 CPU) |
|-------|-------------|---------|-----------------|------------------|
| DocLayNet | RT-DETR object detector | Layout analysis | 44ms/page | 633ms/page |
| TableFormer | Vision-transformer | Table structure | 400ms/table | 1.74s/table |
| CodeFormulaModel | Transformer | Formula detection | -- | -- |

### VLM Pipeline (Granite-Docling)

The newer pipeline uses **Granite-Docling-258M**, a compact vision-language model:

- **Architecture:** Built on Idefics3, uses Granite 3 LLM backbone + SigLIP2 visual encoder
- **Size:** 258M parameters -- delivers accuracy on par with models several times its size
- **DocTags:** A purpose-built markup language that separates content from layout while preserving tables, code, math, and hierarchy
- **Training:** Used the nanoVLM framework for lightweight/efficient training
- **Serving:** Optimized for vLLM acceleration; also supports transformers, ONNX, mlx-vlm, llama.cpp
- **llama.cpp speed:** 403 tokens/second (100x faster than transformers implementation)

---

## 2. Conversion Speed and Benchmarks

### Official Benchmarks (from Docling Technical Report, arXiv:2501.17887)

Test dataset: 89 PDFs, 4,008 pages, 56,246 text items, 1,842 tables, 4,676 pictures.

#### GPU Performance (NVIDIA L4 -- 24GB VRAM)

| Tool | Sec/Page | Pages/Sec |
|------|----------|-----------|
| **MinerU** | 0.21 | 4.76 |
| **Docling** | 0.49 | 2.04 |
| **Marker** | 0.86 | 1.16 |

#### CPU Performance (x86, 8 threads)

| Tool | Sec/Page |
|------|----------|
| Unstructured | 0.24 |
| MinerU | 0.30 |
| **Docling** | 0.32 |
| Marker | 0.06* |

*Marker's CPU number appears to be an outlier -- possibly measured differently.

#### macOS M3 Max

| Tool | Sec/Page |
|------|----------|
| Unstructured | 0.37 |
| **Docling** | 0.79 |

### RTX Consumer GPU Benchmarks (from official docs)

**Standard Pipeline (No OCR):**

| GPU | Pages/Sec | CPU-only Pages/Sec | Speedup |
|-----|-----------|---------------------|---------|
| RTX 5090 (32GB) | 7.9 | 1.5 | ~5.3x |
| RTX 5070 (12GB) | 4.2 | 1.2 | ~3.5x |
| AWS L40S (48GB) | 3.1 | -- | -- |

**Standard Pipeline (With OCR):**

| GPU | Pages/Sec |
|-----|-----------|
| RTX 5090 | 1.6 |
| RTX 5070 | 1.1 |

**VLM Pipeline (Granite-Docling via inference server):**

| GPU | Pages/Sec |
|-----|-----------|
| RTX 5090 | 3.8-4.5 |
| RTX 5070 | 2.8-3.2 |
| AWS L40S | 2.4 |

### Scaling Characteristics

From the Procycons benchmark:
- 1-page PDF: ~6.28 seconds (includes model loading/warm-up overhead)
- 50-page PDF: ~65.12 seconds (~1.3 sec/page amortized)
- Scaling is roughly linear with page count after initial overhead

### Summary: Practical Throughput Expectations

For a typical setup with an RTX-class GPU (no OCR, standard pipeline):
- **~2-8 pages/second** depending on GPU tier
- **~120-480 pages/minute**
- A 100-page PDF: **~12-50 seconds**

With OCR enabled, expect roughly 3-5x slower throughput.

---

## 3. GPU Requirements (VRAM)

### Standard Pipeline (Layout + Table -- no formula enrichment)

The standard pipeline with layout analysis and table structure is relatively lightweight:

- **Layout model (DocLayNet):** ~50-100MB per page in batch
- **Table model (TableFormer):** Moderate additional overhead
- **Without formula enrichment:** Comfortably runs on 8GB+ VRAM

Recommended batch sizes by VRAM:

| VRAM | layout_batch_size | ocr_batch_size | table_batch_size |
|------|-------------------|----------------|------------------|
| 8GB | 16-32 | 16-32 | 4 |
| 12GB (RTX 5070) | 16-32 | 32 | 4 |
| 24GB (RTX 4090/3090) | 32-64 | 64 | 4 |
| 32GB+ (RTX 5090) | 64-128 | 64 | 4 |

### Formula Enrichment (CodeFormulaModel) -- HEAVY

This is the VRAM killer:
- **Default settings:** 18-20GB VRAM
- **With beam search + batch size 2:** ~40GB VRAM
- **RTX 3090 (24GB):** CUDA OOM errors reported at default settings
- **RTX 2080 Ti (11GB):** OOM even at batch_size=1

Workaround for limited VRAM:
```python
from docling.datamodel.settings import settings
settings.perf.elements_batch_size = 2
```

### VLM Pipeline (Granite-Docling-258M)

The 258M model is compact and should run comfortably on most GPUs. Exact VRAM not documented, but at 258M parameters (~0.5GB weights), overhead is minimal. The inference server (vLLM) adds some memory for KV cache.

---

## 4. Can It Run on a Single RTX 3090 (24GB VRAM)?

**Yes, with caveats.**

| Configuration | RTX 3090 Feasibility | Notes |
|---------------|---------------------|-------|
| Standard pipeline (no OCR, no formula) | Works well | batch_size 32-64 comfortable |
| Standard pipeline + OCR | Works well | Slightly slower, still fine |
| Standard pipeline + formula enrichment | Likely OOM | Users report CUDA OOM at defaults. Reduce batch_size to 1-2, disable beam search |
| VLM pipeline (Granite-Docling-258M) | Works well | Model is tiny (258M params) |
| All enrichments enabled | Problematic | Formula enrichment alone can exceed 24GB |

**Recommendation for RTX 3090:** Use the standard pipeline with layout + table + OCR but **disable formula enrichment** (`do_formula_enrichment=False`). This gives the best accuracy-to-VRAM ratio. If you need formula support, reduce `elements_batch_size` to 1-2 and test carefully.

---

## 5. Accuracy and Quality of Output

### Strengths

From the Procycons benchmark (2025):
- **Table extraction:** 97.9% accuracy on complex tables (cell-level)
- **Text extraction:** 100% accuracy for core content in dense paragraphs
- **Table of contents:** 100% text fidelity with accurate reconstruction
- **Digital PDFs:** Excellent quality -- Docling reads PDF text tokens directly, avoiding OCR errors

From the Systenics comparison:
- Transaction tables converted "flawlessly" into clean Markdown tables
- Document structure and page numbers preserved accurately
- Called "the most robust framework" with superior structural preservation

### Weaknesses

From GitHub issues and community reports:
- **Complex tables:** Merged cells, multi-level headers, or rows with varying column counts can produce garbled output. TableFormer enforces a rectangular grid.
- **Scanned/image PDFs:** OCR capabilities are limited compared to specialized OCR tools. Handwritten content and camera-captured images often produce poor results.
- **Formatting accuracy:** 94%+ accuracy on numerical/textual tables, but formatting fidelity can suffer.
- **Character confusion:** Numbers like '0' sometimes misread as 'o'; spurious '=' and '~' characters inserted.
- **Formula extraction:** Inconsistent accuracy even when it works.

### Versus Competitors

| Metric | Docling | LlamaParse | Unstructured |
|--------|---------|------------|--------------|
| Complex table accuracy | 97.9% | Fails with complex structures | 75% |
| Simple table accuracy | High | 100% | 100% |
| Text fidelity | 100% | Struggles with multi-column | High but inconsistent formatting |
| TOC reconstruction | 100% | Failed | Severely deficient |
| Speed (1 page) | 6.28s | ~6s | 51s |
| Speed (50 pages) | 65s | ~6s (cloud API) | 141s |

---

## 6. Multi-Page PDF Handling

Docling handles multi-page PDFs natively:

1. **Page-level processing:** Each page is parsed, rendered, and processed through layout/table models independently.
2. **Threaded pipeline:** Pages are processed in batches across threads for parallelism.
3. **Cross-page assembly:** After all pages complete, `_assemble_document()` aggregates results. The ReadingOrderModel constructs the final document hierarchy.
4. **Paragraph concatenation:** Docling detects and concatenates fragmented paragraphs that span across pages into single text blocks.
5. **Bounding boxes:** Components can have bounding boxes spanning multiple pages.
6. **Page range selection:** Process specific pages with `--page-range` CLI flag or `page_range` Python parameter.

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
# Process specific pages
result = converter.convert("document.pdf", page_range=(1, 10))
```

Scaling is roughly linear -- a 50-page PDF takes about 10x longer than a 5-page PDF, which is expected since each page goes through the full model pipeline.

---

## 7. Batch Processing Support

Docling has first-class batch processing support.

### CLI Batch Processing

```bash
# Convert all PDFs in a directory to Markdown
docling --from pdf --to md --output ./output/ ./input_dir/

# Convert PDF and DOCX files to Markdown and JSON
docling --from pdf --from docx --to md --to json --output ./output/ ./input_dir/

# Abort on first error
docling --abort-on-error --from pdf --to md --output ./output/ ./input_dir/
```

### Python API Batch Processing

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
input_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
# Or pass a directory
results = converter.convert_all(input_paths)

for result in results:
    md = result.document.export_to_markdown()
    # process output...
```

### Performance Tuning for Batch

- GPU batch sizes are configurable per pipeline stage
- The pipeline uses multi-threading between stages
- For GPU: set `layout_batch_size=64`, `ocr_batch_size=64`
- For VLM pipeline: set `page_batch_size >= vlm_options.concurrency` (recommended concurrency: 64)

---

## 8. Known Limitations

### Accuracy Limitations
- **Complex tables:** Merged cells, nested structures, multi-level headers produce garbled output
- **Scanned documents:** OCR quality is limited; poor results on handwritten or camera-captured content
- **Formula detection:** Memory-heavy, inconsistent accuracy, can cause OOM on consumer GPUs
- **Character misrecognition:** Occasional confusion between similar characters (0/o, etc.)

### Reliability Limitations
- **Silent failures:** When Docling fails, it can fail silently or hang -- production systems need graceful error handling and retries
- **Memory spikes:** Content complexity (not just page count) can trigger VRAM exhaustion
- **3x memory regression:** Issue #2786 reports 3x memory consumption increase between certain versions

### Scope Limitations
- **No handwriting recognition:** Focused on typed/printed content
- **Limited image-based PDFs:** Struggles with PDFs that are essentially scanned images
- **Table batch GPU:** Table batch processing on GPU not fully implemented (table_batch_size capped at 4)
- **Windows support:** Tested but some GPU features (like vLLM for VLM pipeline) work best on Linux
- **Python 3.9 dropped:** Requires Python 3.10+ as of v2.70.0

### Performance Limitations
- **GPU vs CPU gap:** Without GPU, the pipeline is 3-5x slower
- **OCR penalty:** Enabling OCR reduces throughput by 3-5x
- **Formula penalty:** Enabling formula enrichment can add minutes per document
- **Initial overhead:** First document is slow due to model loading (~6s for a single page)

---

## 9. License and Pricing

| Component | License | Cost |
|-----------|---------|------|
| Docling codebase | MIT | Free |
| Docling-core | MIT | Free |
| Docling-serve (API server) | MIT | Free |
| DocLayNet (layout model) | CDLA-Permissive 2.0 | Free |
| TableFormer (table model) | MIT | Free |
| Granite-Docling-258M | Apache 2.0 | Free |

**Everything is free and open-source.** No SaaS pricing, no API costs, no usage limits. Runs entirely locally. The MIT license is maximally permissive -- commercial use is allowed.

Individual model licenses should be checked in their respective repositories, but all current models use permissive licenses (MIT, Apache 2.0, CDLA-Permissive).

---

## 10. Installation Complexity

### Basic Installation

```bash
pip install docling
```

That's it for the basic setup. Models are downloaded automatically on first use.

### Requirements
- **Python:** 3.10+ (3.9 dropped in v2.70.0, 3.13 may have issues on Intel Mac)
- **Platforms:** macOS, Linux, Windows (x86_64 and arm64)
- **Disk space:** ~1-2GB for models (downloaded on first run)

### Optional Extras

```bash
pip install "docling[vlm]"          # Granite-Docling VLM pipeline
pip install "docling[easyocr]"      # EasyOCR engine
pip install "docling[tesserocr]"    # Tesseract OCR (requires system Tesseract)
pip install "docling[rapidocr]"     # RapidOCR with ONNX (GPU-capable)
pip install "docling[asr]"          # Audio/speech processing
pip install "docling[ocrmac]"       # macOS native OCR
```

### GPU Setup

For CUDA GPU acceleration, you need:
- CUDA 13.0+ with compatible drivers
- PyTorch with CUDA support
- For the VLM pipeline on Linux: vLLM for best performance

```bash
# Linux CPU-only (skip CUDA)
pip install docling --extra-index-url https://download.pytorch.org/whl/cpu
```

### OCR Engine Dependencies (if needed)

Tesseract requires system-level installation:
```bash
# Debian/Ubuntu
apt-get install tesseract-ocr tesseract-ocr-eng libtesseract-dev

# macOS
brew install tesseract leptonica pkg-config

# Then set TESSDATA_PREFIX=/path/to/tessdata/
```

### Complexity Rating: Low-Medium

- **Basic usage (digital PDFs):** Very easy -- single pip install, works out of the box
- **With GPU acceleration:** Medium -- needs CUDA setup, batch size tuning
- **With OCR:** Medium -- OCR engines have system dependencies
- **With VLM pipeline:** Medium-High on Linux (vLLM setup), simpler on other platforms with transformers
- **Production deployment:** docling-serve provides a REST API server for easier integration

### Development Setup

```bash
git clone https://github.com/docling-project/docling.git
cd docling
uv sync --all-extras
```

---

## Quick Comparison: Docling vs Alternatives

| Feature | Docling | Marker | MinerU | LlamaParse |
|---------|---------|--------|--------|------------|
| License | MIT | GPL-3.0 | AGPL-3.0 | Commercial |
| GPU speed (L4) | 0.49 s/pg | 0.86 s/pg | 0.21 s/pg | ~6s (cloud) |
| CPU speed | 0.32 s/pg | >16 s/pg* | 0.30 s/pg | N/A (cloud) |
| Table accuracy | 97.9% | Good | Good | Fails complex |
| Runs locally | Yes | Yes | Yes | No (API) |
| VRAM (basic) | ~4-8 GB | ~4-8 GB | ~4-8 GB | N/A |
| Batch support | Yes | Yes | Yes | Yes |
| Multi-format | PDF,DOCX,PPTX,HTML,images,audio | PDF only | PDF only | PDF,DOCX |
| VLM option | Granite-Docling-258M | No | No | No |

*Marker CPU numbers vary significantly across benchmarks.

---

## Bottom Line

Docling is a strong choice for local PDF-to-Markdown conversion:

- **Speed:** ~0.49 sec/page on an L4 GPU, ~2-8 pages/sec on RTX consumer cards. Not the fastest (MinerU wins on GPU), but solid.
- **Accuracy:** Best-in-class for table extraction (97.9%) and structural preservation among open-source tools.
- **RTX 3090 compatibility:** Yes, works well for the standard pipeline. Disable formula enrichment to stay within 24GB VRAM.
- **Cost:** Completely free and open-source (MIT).
- **Trade-off:** Slower than MinerU on GPU, but better accuracy and broader format support. Much faster than Marker on CPU.
- **Watch out for:** Complex merged tables, scanned document OCR quality, formula enrichment memory usage, silent failure modes.

---

## Sources

- [Docling GitHub Repository](https://github.com/docling-project/docling)
- [Docling Technical Report (arXiv:2501.17887)](https://arxiv.org/html/2501.17887v1)
- [Docling Architecture Docs](https://docling-project.github.io/docling/concepts/architecture/)
- [Docling GPU Support Docs](https://docling-project.github.io/docling/usage/gpu/)
- [Docling RTX GPU Docs](https://docling-project.github.io/docling/getting_started/rtx/)
- [Docling Installation Docs](https://docling-project.github.io/docling/getting_started/installation/)
- [Granite-Docling-258M on HuggingFace](https://huggingface.co/ibm-granite/granite-docling-258M)
- [IBM Granite-Docling Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
- [Procycons PDF Extraction Benchmark 2025](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [Systenics PDF-to-Markdown Deep Dive](https://systenics.ai/blog/2025-07-28-pdf-to-markdown-conversion-tools/)
- [InfoQ: Granite-Docling-258M Release](https://www.infoq.com/news/2025/10/granite-docling-ibm/)
- [GPU VRAM for Formula Detection (Issue #871)](https://github.com/docling-project/docling/issues/871)
- [Docling Batch Conversion Examples](https://docling-project.github.io/docling/examples/batch_convert/)
- [Docling RTX Acceleration Blog (DEV Community)](https://dev.to/aairom/supercharge-your-document-workflows-docling-now-unleashes-the-power-of-nvidia-rtx-14n4)
