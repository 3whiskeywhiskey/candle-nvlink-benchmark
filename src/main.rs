use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber;

mod backend;
mod benchmark;
mod config;
mod models;
mod parallelization;
mod metrics;

use crate::benchmark::BenchmarkRunner;
use crate::config::{BenchmarkConfig, HardwareConfig};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a specific benchmark
    Run {
        /// Configuration file path
        #[arg(short, long, default_value = "config/default.json")]
        config: String,
        
        /// Override model type
        #[arg(short, long)]
        model: Option<String>,
        
        /// Override parallelization strategy
        #[arg(short, long)]
        strategy: Option<String>,
        
        /// Enable verbose logging
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run all benchmarks in suite
    Suite {
        /// Configuration file path
        #[arg(short, long, default_value = "config/suite.json")]
        config: String,
        
        /// Output results to file
        #[arg(short, long)]
        output: Option<String>,
    },
    /// List available models and strategies
    List,
    /// Generate sample configuration files
    GenConfig {
        /// Target hardware (m4-metal, v100-nvlink)
        #[arg(long, default_value = "m4-metal")]
        hardware: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let level = match &cli.command {
        Commands::Run { verbose, .. } if *verbose => Level::DEBUG,
        _ => Level::INFO,
    };
    
    tracing_subscriber::fmt()
        .with_max_level(level)
        .init();

    info!("🚀 Distributed Training Benchmark Suite Starting");
    
    match &cli.command {
        Commands::Run { config, model, strategy, .. } => {
            let mut benchmark_config = BenchmarkConfig::load(config)?;
            
            // Apply CLI overrides
            if let Some(model_override) = model {
                benchmark_config.model.model_type = model_override.clone();
            }
            if let Some(strategy_override) = strategy {
                benchmark_config.parallelization.strategy = strategy_override.clone();
            }
            
            let mut runner = BenchmarkRunner::new(benchmark_config)?;
            
            // Wrap benchmark execution with proper error handling
            let results = match runner.run_single().await {
                Ok(results) => results,
                Err(e) => {
                    eprintln!("❌ Benchmark failed: {}", e);
                    
                    // If on macOS with Metal, give time for cleanup
                    if cfg!(target_os = "macos") {
                        info!("🧹 Allowing Metal resources to cleanup...");
                        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                    }
                    
                    return Err(e);
                }
            };
            
            println!("\n📊 Benchmark Results:");
            println!("{}", results.display_table());
            
            // Explicit cleanup for Metal resources
            if cfg!(target_os = "macos") {
                info!("🧹 Final Metal resource cleanup...");
                drop(runner);
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        },
        Commands::Suite { config, output } => {
            let suite_config = BenchmarkConfig::load_suite(config)?;
            let mut runner = BenchmarkRunner::new_suite(suite_config)?;
            let results = runner.run_suite().await?;
            
            println!("\n📊 Suite Results:");
            println!("{}", results.display_summary());
            
            if let Some(output_path) = output {
                results.save_to_file(output_path)?;
                info!("Results saved to: {}", output_path);
            }
        },
        Commands::List => {
            println!("📋 Available Models:");
            println!("  - transformer-small (125M params)");
            println!("  - transformer-medium (350M params)"); 
            println!("  - transformer-large (1.3B params)");
            println!("  - cnn-resnet18");
            println!("  - cnn-resnet50");
            println!("  - mlp-large");
            
            println!("\n🔀 Available Parallelization Strategies:");
            println!("  - data-parallel: Standard data parallelism");
            println!("  - model-parallel: Model split across devices");
            println!("  - pipeline-parallel: Pipeline parallelism");
            println!("  - hybrid: Combination of strategies");
        },
        Commands::GenConfig { hardware } => {
            let hardware_config = match hardware.as_str() {
                "m4-metal" => HardwareConfig::m4_metal(),
                "v100-nvlink" => HardwareConfig::v100_nvlink(),
                _ => {
                    eprintln!("❌ Unknown hardware type: {}", hardware);
                    std::process::exit(1);
                }
            };
            
            let config = BenchmarkConfig::default_for_hardware(hardware_config);
            std::fs::create_dir_all("config")?;
            
            let config_path = format!("config/{}.json", hardware);
            config.save(&config_path)?;
            
            println!("✅ Generated configuration: {}", config_path);
        },
    }

    Ok(())
} 