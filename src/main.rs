mod api_client;
mod app;
mod cli;
mod error;
mod logging;
mod scanner;
mod shutdown;
mod tui;
mod types;
mod worker;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use tracing::info;

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app_start = Instant::now();
    let cli = Cli::parse();

    // Determine log directory
    let log_dir = cli
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("."));

    // Initialize logging (file-only since TUI will be active)
    logging::init_logging(&log_dir, true);

    info!(
        inputs = ?cli.input,
        recursive = cli.recursive,
        output = ?cli.output,
        workers = cli.workers,
        api_url = %cli.api_url,
        "Starting PDF2Markdown"
    );

    let num_workers = cli.workers as usize;

    // Create output directory if specified
    if let Some(ref out_dir) = cli.output {
        if !out_dir.exists() {
            std::fs::create_dir_all(out_dir)?;
            info!(path = %out_dir.display(), "Created output directory");
        }
    }

    // Scan for PDFs
    let scan_start = Instant::now();
    let scan_result = scanner::scan_directories(
        &cli.input,
        cli.recursive,
        cli.output.as_deref(),
    )?;
    let scan_elapsed = scan_start.elapsed();

    info!(
        found = scan_result.total_found,
        queued = scan_result.queue.len(),
        skipped = scan_result.skipped,
        scan_ms = scan_elapsed.as_millis() as u64,
        "Scan complete"
    );

    // Run the application
    app::run(
        scan_result.queue,
        scan_result.files,
        scan_result.total_found,
        scan_result.skipped,
        num_workers,
        &cli.api_url,
        app_start,
    )
    .await?;

    Ok(())
}
