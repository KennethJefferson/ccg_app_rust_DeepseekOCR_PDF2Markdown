# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
