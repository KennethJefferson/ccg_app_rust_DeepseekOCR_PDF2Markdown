# CLAUDE.md - Project Instructions for Claude Code

## Project Overview
PDF2Markdown: A Rust CLI client with TUI dashboard that sends PDFs to a Python FastAPI server running Marker for OCR, returning converted Markdown.

## Architecture
- **Client** (`src/`): Rust + tokio async runtime, ratatui TUI, reqwest HTTP client
- **Server** (`server/`): Python FastAPI, Marker (marker-pdf) for PDF-to-Markdown conversion
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

## File Layout
```
src/
  main.rs         - Entry point, CLI parse, scanner, app launch
  cli.rs          - clap derive argument definitions
  scanner.rs      - Directory walking, PDF discovery, collision rename
  types.rs        - Core data types (QueueItem, AppState, Stats, etc.)
  error.rs        - ScanError, ApiError enums
  api_client.rs   - HTTP multipart upload with retry logic
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
  app/main.py     - FastAPI endpoints (/convert, /health)
  app/model.py    - Marker PdfConverter loading + async conversion wrapper
  app/schemas.py  - Pydantic response models
```

## Testing
- Scanner has 7 unit tests using `tempfile` crate
- Tests cover: recursive scan, skip existing, collision rename, empty dir, nonexistent dir
- API client designed for `wiremock` integration tests (not yet implemented)

## Server Deployment (Runpod)
- All deps installed to `/workspace/venv/` (persistent across pod restarts)
- Marker/surya models cached at `/workspace/datalab_cache/` (~3.3GB, symlinked from `/root/.cache/datalab`)
- Server code at `/workspace/deepseek-ocr-server/`
- `start.sh` is idempotent: skips install if marker-pdf already in venv
- Always expose TCP port 8000 for direct API access (faster than proxy)
- When killing uvicorn, use `kill -9 <PID>` by exact PID; `pkill -f uvicorn` kills SSH sessions too
- After killing, verify GPU memory freed with `nvidia-smi`; zombie VRAM requires pod restart

## Model API (Marker)
- Uses `PdfConverter` from `marker.converters.pdf` with `create_model_dict()` from `marker.models`
- Converter takes a file path, returns rendered output; extract text with `text_from_rendered()`
- Page count from `rendered.metadata["page_stats"]`
- Marker handles PDF rendering internally (no separate PyMuPDF step)
- GPL-3.0 license (server-only, private deployment, no impact on MIT Rust client)

## Common Pitfalls
- `time` crate v0.3.47+ requires Rust 1.88+; do not downgrade toolchain
- `ratatui 0.29` pulls `darling 0.23` which also needs Rust 1.88+
- Don't add `chrono`/`uuid`/`futures` unless actually used; heavy dep trees
- `ScanResult` needs `#[derive(Debug)]` for test assertions with `unwrap_err()`
- Use `#[allow(dead_code)]` on enums with Debug derive when fields are consumed via pattern matching
- Server `__pycache__` can serve stale code after SCP updates; always `rm -rf __pycache__` before restart
