# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-06

### Changed
- **Replaced DeepSeek-OCR-2 with Marker** (marker-pdf) as the OCR engine on the server
- ~18x speedup: from ~7.5s/page to ~0.4s/page on RTX 3090 (224-page book in 90s)
- Server now processes entire PDFs in a single call instead of per-page image inference
- Removed `pdf_pipeline.py` (PyMuPDF) -- Marker handles PDF rendering internally
- Simplified `requirements.txt` from 12 deps to 4 (marker-pdf pulls transitive deps)
- `start.sh` detects stale venv and recreates; symlinks datalab model cache to persistent storage
- API contract unchanged: Rust client works without modification

### Removed
- DeepSeek-OCR-2 model dependency (~6GB), replaced by Marker models (~3.3GB)
- PyMuPDF PDF-to-PNG pipeline
- Per-page OOM handling (Marker processes whole PDFs atomically)
- GPU semaphore (no longer needed with single whole-PDF calls)

## [0.1.1] - 2026-02-06

### Fixed
- Corrected model API: use `AutoModel` with built-in `model.infer()` instead of `AutoModelForCausalLM` with manual `generate()`
- Fixed model name from `DeepSeek-OCR-2-3B` to `DeepSeek-OCR-2`
- Fixed dtype from `float16` to `bfloat16` and attention param from `attn_implementation` to `_attn_implementation`
- Fixed `model.infer()` crash on `output_path=""` by using `tempfile.TemporaryDirectory`
- Added `torch_dtype=torch.bfloat16` to `from_pretrained()` for proper Flash Attention 2 initialization

### Changed
- Server deployment now uses persistent `/workspace/` paths (venv, model, server code) instead of ephemeral container disk
- `start.sh` is now idempotent: skips venv creation and model download if already present
- Added per-page timing instrumentation to `/convert` endpoint
- Reduced default PDF render DPI from 144 to 68 to avoid unnecessary image cropping

### Added
- Performance research and analysis in `research/` directory
- Detailed comparisons of OCR alternatives: Marker, MinerU, Docling, Surya, Nougat, GOT-OCR, Mistral OCR
- Speed benchmarks, architecture analysis, vLLM serving research, quantization optimization paths
- Root cause analysis: autoregressive decoding is the primary bottleneck (~7ms/token on RTX 3090)

## [0.1.0] - 2026-02-05

### Added
- Rust CLI client with clap-based argument parsing (`-i`, `-r`, `-o`, `-w`, `--api-url`)
- Directory scanner with recursive walk, pre-existence skip, and auto-rename collision handling
- Async worker pool using tokio channels for concurrent PDF processing
- HTTP multipart upload client with exponential backoff retry (3 attempts)
- Live TUI dashboard with ratatui: worker status, progress bar, file list, elapsed timer
- Green color theme with braille spinner animation for active workers
- Two-stage Ctrl+C shutdown (graceful then force) with terminal state restoration
- File-only logging via tracing-appender when TUI is active
- Python FastAPI server for Runpod GPU deployment
- DeepSeek-OCR-2-3B model integration with GPU semaphore for safe single-inference access
- PDF-to-PNG pipeline via PyMuPDF at 144 DPI
- `/convert` endpoint (multipart PDF upload, returns combined Markdown)
- `/health` endpoint (model and GPU status)
- OOM handling with partial result return on GPU memory exhaustion
- 7 unit tests for scanner module (recursive scan, skip existing, collision rename, edge cases)
