use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "pdf2md")]
#[command(about = "Convert PDFs to Markdown via Marker OCR")]
pub struct Cli {
    /// Input directories containing PDF files
    #[arg(short, long, required = true, num_args = 1..)]
    pub input: Vec<PathBuf>,

    /// Scan subdirectories recursively
    #[arg(short, long, default_value_t = false)]
    pub recursive: bool,

    /// Flat output directory (default: next to source PDF)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Number of parallel workers (1-4)
    #[arg(short, long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub workers: u8,

    /// Marker API URL (e.g., https://{pod-id}-8000.proxy.runpod.net)
    #[arg(long)]
    pub api_url: String,
}
