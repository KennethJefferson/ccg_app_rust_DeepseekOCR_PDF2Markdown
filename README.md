# PDF2Markdown

A Rust-based CLI tool with a live TUI dashboard that batch-converts PDF documents to Markdown using [Marker](https://github.com/datalab-to/marker) running on a GPU server.

## How It Works

```
PDF files on disk                          Runpod GPU Server
     |                                          |
     v                                          v
 [Scanner] --> [Work Queue] --> [Workers] --> [FastAPI]
                                    |         PDF -> Marker -> Markdown
                                    v              |
                               [TUI Dashboard]     v
                                    |          Markdown response
                                    v              |
                              [.md files]  <-------+
```

1. **Scanner** walks input directories for `.pdf` files, skipping any with existing `.md` output
2. **Workers** (1-4, default 2) pull PDFs from a channel and upload them to the API server
3. **Server** runs a pool of Marker model instances (configurable, default 4) for parallel conversion
4. **TUI** displays real-time worker status, progress bar, file results, and statistics

## Requirements

### Client (Windows/Linux/macOS)
- Rust 1.93+

### Server (Runpod / any Linux with GPU)
- Python 3.10+
- CUDA 11.8+
- GPU with 24GB+ VRAM (tested on RTX 3090)

## Quick Start

### 1. Deploy the Server

```bash
# On your Runpod instance
cd server/
chmod +x start.sh
./start.sh
```

This installs dependencies, downloads Marker models (~3.3GB on first run), and starts the API on port 8000.

### 2. Build the Client

```bash
cargo build --release
```

### 3. Convert PDFs

```bash
# Convert all PDFs in a directory
deepseek-ocr-pdf2md -i ./documents --api-url https://<pod-id>-8000.proxy.runpod.net

# Recursive scan with 4 parallel workers, output to a flat directory
deepseek-ocr-pdf2md -i ./docs -r -w 4 -o ./markdown_output --api-url https://<pod-id>-8000.proxy.runpod.net
```

## TUI Dashboard

```
+-- DeepSeek OCR - PDF to Markdown ----------- [Ctrl+C to stop] --+
|                                                                   |
| Workers                                                           |
|  # Worker 0: report.pdf                            (12s)          |
|  v Worker 1: invoice.pdf                           (8s)           |
|  - Worker 2: idle                                                 |
|                                                                   |
| Progress [===============-------] 23/67               34%         |
|                                                                   |
| Files                                                             |
|  v report.md                                        3.2s          |
|  v invoice.md                                       8.1s          |
|  x broken.md                             API timeout 2.0s         |
|  - already_done.md                                  skipped        |
|                                                                   |
| Completed: 22  Failed: 1  Skipped: 3  Elapsed: 2m 14s            |
+-------------------------------------------------------------------+
```

- Green color theme with braille spinner animation
- Workers section shows active/idle/done status
- File list sorted by: processing > failed > completed > pending > skipped
- Auto-scrolling with max 25 visible entries

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/convert` | POST | Multipart upload of PDF, returns `{ success, markdown, pages_processed, error? }` |
| `/health` | GET | Returns `{ status, model_loaded, gpu_available, pool_size }` |

## Project Structure

```
.
├── Cargo.toml
├── src/
│   ├── main.rs           # Entry point
│   ├── cli.rs            # CLI argument definitions
│   ├── scanner.rs        # PDF discovery + collision handling
│   ├── types.rs          # Core data structures
│   ├── error.rs          # Error types
│   ├── api_client.rs     # HTTP client with retry logic
│   ├── worker.rs         # Async worker tasks
│   ├── app.rs            # Orchestration + event loop
│   ├── shutdown.rs       # Ctrl+C handler
│   ├── logging.rs        # File + stderr logging setup
│   └── tui/
│       ├── mod.rs        # Terminal management
│       ├── event.rs      # Input event handling
│       ├── ui.rs         # Dashboard layout
│       └── widgets.rs    # Custom TUI components
└── server/
    ├── app/
    │   ├── main.py       # FastAPI application
    │   ├── model.py      # Marker model pool + parallel conversion
    │   └── schemas.py    # Response models
    ├── requirements.txt
    └── start.sh          # Server startup script
```

## Design Decisions

- **No shared mutable state**: Workers send events through channels. The main loop is the sole owner of `AppState`.
- **Whole-PDF conversion**: Marker processes entire PDFs at once (no per-page loop).
- **Model pool**: Server loads multiple Marker instances into an async queue for parallel GPU inference. Configurable via `MARKER_POOL_SIZE` env var (default 4).
- **Pipeline parallelism**: Client workers overlap network I/O with server-side GPU processing — while one PDF is converting, the next is uploading.
- **Pre-existence check**: Already-converted PDFs are skipped before queuing.
- **Auto-rename on collision**: When using `-o`, duplicate filenames get `_1`, `_2` suffixes.
- **Two-stage shutdown**: First Ctrl+C finishes current work gracefully; second forces immediate exit.
- **Retry with backoff**: Failed API calls retry up to 3 times with exponential backoff (1s, 2s, 4s).

## Performance

Using Marker on RTX 3090 with 4 model instances: **~0.4-0.5s per page per instance**. With 4 workers saturating the pool, throughput scales near-linearly up to the GPU's compute limit.

| Setup | Throughput (est.) |
|-------|------------------|
| 1 worker, 1 instance | ~150 pages/min |
| 2 workers, 2 instances | ~280 pages/min |
| 4 workers, 4 instances | ~500 pages/min (GPU-bound) |

## License

MIT
