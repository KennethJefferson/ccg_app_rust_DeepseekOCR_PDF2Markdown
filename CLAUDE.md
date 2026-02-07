# CLAUDE.md - Project Instructions for Claude Code

## Project Overview
PDF2Markdown: A Rust CLI client with TUI dashboard that sends PDFs to a Python FastAPI server running Marker for OCR, returning converted Markdown.

## Architecture
- **Client** (`src/`): Rust + tokio async runtime, ratatui TUI, reqwest HTTP client
- **Server** (`server/`): Python FastAPI + gunicorn/uvicorn multi-worker, Marker (marker-pdf) for PDF-to-Markdown conversion
- **Deployment**: Runpod GPU pod (RTX 3090), persistent storage at `/workspace/` (venv, HF cache, server code)

## Build & Test
```bash
cargo build              # Debug build
cargo build --release    # Release build
cargo test               # Run all unit tests
cargo run -- --help      # Show CLI usage
```

## Toolchain
- Rust 1.93+ (required by transitive deps: time, darling, instability)
- Python 3.10+ for server
- CUDA 11.8+ with RTX 3090 or equivalent for server GPU inference

## Key Conventions
- No shared mutable state in async code; use mpsc channels only
- All worker communication flows through `WorkerEvent` variants
- `AppState` is sole-owner in the main event loop
- Server returns 200 for all responses; check `success` field in body
- Scanner pre-checks for existing `.md` files and skips them
- Client workers capped at 4; default is 2
- Server uses `gunicorn --workers N` with `uvicorn.workers.UvicornWorker` (separate OS processes, each with own CUDA context)
- Workers auto-recycle after ~100 requests (`--max-requests 100 --max-requests-jitter 20`) for VRAM leak prevention
- PDFs pre-validated with pypdfium2 before GPU processing (catches bad PDFs immediately)
- `/health` runs CUDA probe (trivial GPU op) to detect silently corrupted contexts
- `MARKER_WORKERS` env var controls worker count (default 4, recommend 3 for RTX 3090)
- Client sends MD5 hash of PDF bytes via `X-File-MD5` header; server rejects corrupted uploads
- Client retries transient errors: CUDA crashes, upload corruption, timeouts (3 attempts, exponential backoff)

## File Layout
```
src/
  main.rs         - Entry point, CLI parse, scanner, app launch
  cli.rs          - clap derive argument definitions
  scanner.rs      - Directory walking, PDF discovery, collision rename
  types.rs        - Core data types (QueueItem, AppState, Stats, etc.)
  error.rs        - ScanError, ApiError enums
  api_client.rs   - HTTP multipart upload with MD5 integrity + retry logic
  worker.rs       - Async worker task pulling from channel
  app.rs          - Orchestration: channels, workers, TUI event loop
  shutdown.rs     - Two-stage Ctrl+C handler
  logging.rs      - tracing-subscriber + file appender setup
  tui/
    mod.rs        - Terminal init/restore with panic hook
    event.rs      - Crossterm key reader + event enum
    ui.rs         - Dashboard layout rendering
    widgets.rs    - Worker rows, file lines, spinner, formatting
server/
  app/main.py     - FastAPI endpoints (/convert, /health), CUDA error recovery, MD5 verification, PDF pre-validation
  app/model.py    - Single PdfConverter per worker process + CUDA health probe
  app/schemas.py  - Pydantic response models
```

## Testing
- Scanner has 7 unit tests using `tempfile` crate
- Tests cover: recursive scan, skip existing, collision rename, empty dir, nonexistent dir
- API client designed for `wiremock` integration tests (not yet implemented)

## Server Deployment (Runpod)
- All deps installed to `/workspace/venv/` (persistent across pod restarts)
- Marker/surya models cached at `/workspace/datalab_cache/` (~3.3GB, symlinked from `/root/.cache/datalab`)
- Server code at `/workspace/pdf2md-server/`
- `start.sh` is idempotent: skips install if marker-pdf already in venv
- Always expose TCP port 8000 for direct API access (faster than proxy)
- When killing gunicorn, use `kill -9 <PID>` by exact PID; `pkill -f gunicorn` may kill SSH sessions
- After killing, verify GPU memory freed with `nvidia-smi`; zombie VRAM requires pod restart

## Model API (Marker)
- Uses `PdfConverter` from `marker.converters.pdf` with `create_model_dict()` from `marker.models`
- Converter takes a file path, returns rendered output; extract text with `text_from_rendered()`
- Page count from `rendered.metadata["page_stats"]`
- Marker handles PDF rendering internally (no separate PyMuPDF step)
- Each gunicorn worker loads one PdfConverter instance with its own CUDA context
- `MARKER_WORKERS` env var controls worker count (default 4, tune based on VRAM)
- 3 workers: ~10.5 GB idle, safest for mixed workloads on RTX 3090
- 4 workers: ~14 GB idle, ~18-24 GB active (risky for large PDFs)
- Fatal CUDA errors (device-side assert, OOM) cause worker to call `os._exit(1)`; gunicorn auto-restarts with fresh context
- `torch.cuda.empty_cache()` called after each conversion and on failure
- GPL-3.0 license (server-only, private deployment, no impact on MIT Rust client)

## Common Pitfalls
- `time` crate v0.3.47+ requires Rust 1.88+; do not downgrade toolchain
- `ratatui 0.29` pulls `darling 0.23` which also needs Rust 1.88+
- Don't add `chrono`/`uuid`/`futures` unless actually used; heavy dep trees
- `ScanResult` needs `#[derive(Debug)]` for test assertions with `unwrap_err()`
- Use `#[allow(dead_code)]` on enums with Debug derive when fields are consumed via pattern matching
- Server `__pycache__` can serve stale code after SCP updates; always `rm -rf __pycache__` before restart
- Use gunicorn `--workers N` with UvicornWorker for parallel processing (each worker = separate OS process)
- If 4 workers OOM, reduce with `MARKER_WORKERS=3` or `MARKER_WORKERS=2`
- CUDA device-side assert corrupts a worker's GPU context permanently; server auto-kills worker so gunicorn restarts it
- Client retries transient CUDA errors (OOM, device-side assert) since gunicorn restarts the failed worker
- Client retries on upload corruption detected by MD5 mismatch
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces VRAM fragmentation
- Client timeout is 600s (10 min) to accommodate large PDFs over WAN
- `start.sh` cleans zombie processes, stale temp files, and __pycache__ on every start
- Killing gunicorn master does NOT kill Marker's multiprocessing.spawn children; must `kill -9` them manually
- `pkill -f gunicorn` may kill SSH sessions; always kill by exact PID
- Some PDFs cause `malloc_consolidate` glibc heap corruption in Marker/PDFium (uncatchable, crashes worker)
