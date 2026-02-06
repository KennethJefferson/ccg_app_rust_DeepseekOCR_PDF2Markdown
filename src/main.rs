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

use clap::Parser;
use tracing::info;

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
        "Starting DeepSeek OCR"
    );

    // Validate workers count
    if cli.workers == 0 {
        anyhow::bail!("Workers must be at least 1");
    }

    // Create output directory if specified
    if let Some(ref out_dir) = cli.output {
        if !out_dir.exists() {
            std::fs::create_dir_all(out_dir)?;
            info!(path = %out_dir.display(), "Created output directory");
        }
    }

    // Scan for PDFs
    let scan_result = scanner::scan_directories(
        &cli.input,
        cli.recursive,
        cli.output.as_deref(),
    )?;

    info!(
        found = scan_result.total_found,
        queued = scan_result.queue.len(),
        skipped = scan_result.skipped,
        "Scan complete"
    );

    // Run the application
    app::run(
        scan_result.queue,
        scan_result.files,
        scan_result.total_found,
        scan_result.skipped,
        cli.workers,
        &cli.api_url,
    )
    .await?;

    Ok(())
}
