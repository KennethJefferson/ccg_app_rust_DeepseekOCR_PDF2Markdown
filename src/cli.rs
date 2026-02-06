use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "deepseek-ocr")]
#[command(about = "Convert PDFs to Markdown using DeepSeek-OCR")]
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

    /// Number of parallel workers
    #[arg(short, long, default_value_t = 1)]
    pub workers: usize,

    /// DeepSeek-OCR API URL (e.g., https://{pod-id}-8000.proxy.runpod.net)
    #[arg(long)]
    pub api_url: String,
}
