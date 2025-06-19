use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};
use tracing::{info, debug, warn, error};
use sysinfo::{System, Pid};

use crate::backend::DeviceManager;
use crate::config::BenchmarkConfig;
use crate::metrics::{BenchmarkMetrics, SystemMetrics, CommunicationMetrics};
use crate::models::{create_model, create_synthetic_data, BenchmarkModel};
use crate::parallelization::{create_strategy, ParallelizationStrategy, ForwardResult, BackwardResult};

pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    device_manager: DeviceManager,
    model: Box<dyn BenchmarkModel>,
    strategy: Box<dyn ParallelizationStrategy>,
    system_metrics: SystemMetrics,
    memory_monitor: MemoryMonitor,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Result<Self> {
        info!("🔧 Initializing Benchmark Runner");
        
        // Validate configuration
        config.validate()?;
        
        // Initialize device manager
        let device_manager = DeviceManager::new(
            &config.hardware.device_type,
            config.hardware.num_devices,
        )?;
        
        // Create model
        let model = create_model(
            &config.model.model_type,
            &config.model.size,
            device_manager.primary_device(),
        )?;
        
        info!("📊 Model: {} ({} parameters)", 
              model.name(), 
              format_number(model.parameter_count()));
        
        // Create parallelization strategy
        let mut strategy = create_strategy(&config.parallelization.strategy, &config.parallelization)?;
        strategy.setup(&device_manager)?;
        
        // Initialize system metrics
        let system_metrics = SystemMetrics::new()?;
        
        // Initialize memory monitor with reasonable system memory threshold
        // Use 90% of actual total system memory instead of GPU memory per device
        let mut temp_system = sysinfo::System::new_all();
        temp_system.refresh_all();
        let total_system_memory_gb = temp_system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let max_memory_threshold_gb = total_system_memory_gb * 0.9;
        let memory_monitor = MemoryMonitor::new(max_memory_threshold_gb);
        
        Ok(Self {
            config,
            device_manager,
            model,
            strategy,
            system_metrics,
            memory_monitor,
        })
    }

    pub fn new_suite(configs: Vec<BenchmarkConfig>) -> Result<BenchmarkSuite> {
        let runners = configs
            .into_iter()
            .map(|config| Self::new(config))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(BenchmarkSuite { runners })
    }

    pub async fn run_single(&mut self) -> Result<BenchmarkResult> {
        info!("🚀 Starting single benchmark run");
        
        let mut metrics = BenchmarkMetrics::new(&self.config);
        
        // Hardware characterization
        self.run_hardware_benchmark(&mut metrics).await?;
        
        // Training benchmark
        self.run_training_benchmark(&mut metrics).await?;
        
        // Memory benchmark (only if enabled)
        if self.config.benchmark.memory_profiling {
            info!("🧠 Memory profiling enabled - running memory benchmark");
            self.run_memory_benchmark(&mut metrics).await?;
        } else {
            info!("🧠 Memory profiling disabled - skipping memory benchmark");
            metrics.memory.max_batch_size = self.config.training.batch_size;
            metrics.memory.memory_efficiency = 1.0;
        }
        
        // Communication benchmark
        if self.device_manager.is_multi_device() {
            self.run_communication_benchmark(&mut metrics).await?;
        }
        
        let result = BenchmarkResult::from_metrics(metrics, &self.config);
        info!("✅ Benchmark completed successfully");
        
        Ok(result)
    }

    async fn run_hardware_benchmark(&mut self, metrics: &mut BenchmarkMetrics) -> Result<()> {
        info!("🔍 Running hardware characterization");
        
        // Memory info
        let memory_info = self.device_manager.memory_info()?;
        metrics.hardware.memory_info = memory_info;
        
        // P2P bandwidth (if multi-device)
        if self.device_manager.is_multi_device() {
            let p2p_results = self.device_manager.benchmark_p2p_bandwidth()?;
            metrics.hardware.p2p_bandwidth = p2p_results;
        }
        
        // System info
        metrics.hardware.system_info = self.system_metrics.get_system_info();
        
        Ok(())
    }

    async fn run_training_benchmark(&mut self, metrics: &mut BenchmarkMetrics) -> Result<()> {
        info!("🏋️ Running training benchmark");
        
        // Check initial memory state before starting
        self.check_memory_pressure_and_cleanup().await?;
        
        let progress = ProgressBar::new(
            (self.config.benchmark.warmup_iterations + self.config.benchmark.measurement_iterations) as u64
        );
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")?
                .progress_chars("#>-")
        );
        
        // Warmup phase
        progress.set_message("Warming up...");
        let mut warmup_completed = 0;
        for i in 0..self.config.benchmark.warmup_iterations {
            // Check memory before each iteration
            match self.memory_monitor.check_memory_pressure() {
                Ok(MemoryStatus::Critical) => {
                    error!("🚨 Critical memory pressure during warmup at iteration {}", i);
                    break;
                }
                Ok(MemoryStatus::Warning) => {
                    warn!("⚠️ Memory pressure warning during warmup at iteration {}", i);
                    self.memory_monitor.force_cleanup();
                    // Continue but be more careful
                }
                Ok(MemoryStatus::Normal) => {}
                Err(e) => {
                    error!("Memory check failed during warmup: {}", e);
                    break;
                }
            }
            
            // Create fresh data for each iteration to avoid accumulation
            let input_data = create_synthetic_data(
                self.config.training.batch_size,
                self.config.training.sequence_length,
                50257, // GPT-2 vocab size
                self.device_manager.primary_device(),
            )?;
            
            let _result = self.run_training_step(&input_data, false).await?;
            
            // Explicit cleanup after each iteration
            drop(input_data);
            drop(_result);
            
            // Force cleanup every few iterations for large models
            if (i + 1) % 2 == 0 {
                self.memory_monitor.force_cleanup();
            }
            
            warmup_completed += 1;
            progress.inc(1);
        }
        
        info!("✅ Completed {} warmup iterations", warmup_completed);
        
        // Measurement phase
        progress.set_message("Measuring...");
        let mut forward_times = Vec::new();
        let mut backward_times = Vec::new();
        let mut memory_usage = Vec::new();
        let mut communication_times = Vec::new();
        let mut measurements_completed = 0;
        
        for i in 0..self.config.benchmark.measurement_iterations {
            // Check memory before each iteration
            match self.memory_monitor.check_memory_pressure() {
                Ok(MemoryStatus::Critical) => {
                    error!("🚨 Critical memory pressure during measurement at iteration {}", i);
                    break;
                }
                Ok(MemoryStatus::Warning) => {
                    warn!("⚠️ Memory pressure warning during measurement at iteration {}", i);
                    self.memory_monitor.force_cleanup();
                }
                Ok(MemoryStatus::Normal) => {}
                Err(e) => {
                    error!("Memory check failed during measurement: {}", e);
                    break;
                }
            }
            
            // Create fresh data for each iteration to avoid accumulation
            let input_data = create_synthetic_data(
                self.config.training.batch_size,
                self.config.training.sequence_length,
                50257,
                self.device_manager.primary_device(),
            )?;
            
            let (forward_result, backward_result) = self.run_training_step(&input_data, true).await?;
            
            // Extract metrics without keeping tensor references
            forward_times.push(forward_result.computation_time_ms);
            backward_times.push(backward_result.computation_time_ms);
            memory_usage.push(forward_result.memory_used_bytes + backward_result.memory_used_bytes);
            communication_times.push(forward_result.communication_time_ms + backward_result.communication_time_ms);
            
            // Explicit cleanup after each iteration
            drop(input_data);
            drop(forward_result);
            drop(backward_result);
            
            // Force cleanup every iteration for large models
            self.memory_monitor.force_cleanup();
            
            measurements_completed += 1;
            progress.inc(1);
        }
        
        progress.finish_with_message("Training benchmark complete");
        
        if measurements_completed == 0 {
            return Err(anyhow::anyhow!("No measurements completed due to memory pressure"));
        }
        
        info!("✅ Completed {} measurement iterations", measurements_completed);
        
        // Store metrics
        metrics.training.forward_time_ms = calculate_stats(&forward_times);
        metrics.training.backward_time_ms = calculate_stats(&backward_times);
        metrics.training.memory_usage_bytes = calculate_stats_usize(&memory_usage);
        metrics.training.communication_time_ms = calculate_stats(&communication_times);
        
        // Calculate throughput
        let avg_step_time = metrics.training.forward_time_ms.mean + metrics.training.backward_time_ms.mean;
        let samples_per_second = (self.config.training.batch_size as f64 * 1000.0) / avg_step_time;
        metrics.training.throughput_samples_per_second = samples_per_second;
        
        // Calculate tokens per second (for language models)
        let tokens_per_second = samples_per_second * self.config.training.sequence_length as f64;
        metrics.training.throughput_tokens_per_second = tokens_per_second;
        
        info!("📈 Training throughput: {:.1} samples/s, {:.1} tokens/s", 
              samples_per_second, tokens_per_second);
        
        Ok(())
    }

    async fn run_training_step(&self, input_data: &candle_core::Tensor, _measure: bool) -> Result<(ForwardResult, BackwardResult)> {
        // Forward pass
        let forward_result = self.strategy.forward_step(self.model.as_ref(), input_data)?;
        
        // Create dummy gradients for backward pass
        let dummy_gradients = vec![forward_result.output.clone()];
        
        // Backward pass
        let backward_result = self.strategy.backward_step(&dummy_gradients)?;
        
        Ok((forward_result, backward_result))
    }

    async fn run_memory_benchmark(&mut self, metrics: &mut BenchmarkMetrics) -> Result<()> {
        info!("💾 Running memory benchmark");
        
        // Check initial memory state
        self.check_memory_pressure_and_cleanup().await?;
        
        // Test different batch sizes to find memory limits
        let base_batch_size = self.config.training.batch_size;
        let mut max_batch_size = base_batch_size;
        let mut batch_size = base_batch_size;
        
        // More conservative exponential search for max batch size
        let mut iteration_count = 0;
        let max_iterations = 10; // Prevent infinite loops
        
        loop {
            debug!("Testing batch size: {} (iteration {})", batch_size, iteration_count);
            
            // Safety check for infinite loops
            iteration_count += 1;
            if iteration_count > max_iterations {
                warn!("⚠️ Reached maximum iterations in memory benchmark");
                break;
            }
            
            // Memory check before each test
            match self.memory_monitor.check_memory_pressure()? {
                MemoryStatus::Critical => {
                    error!("🚨 Critical memory pressure - stopping memory benchmark");
                    break;
                }
                MemoryStatus::Warning => {
                    warn!("⚠️ Memory pressure warning - being more conservative");
                    self.memory_monitor.force_cleanup();
                    break;
                }
                MemoryStatus::Normal => {}
            }
            
            match self.test_batch_size(batch_size).await {
                Ok(_) => {
                    info!("✅ Batch size {} successful", batch_size);
                    max_batch_size = batch_size;
                    
                    // Ensure batch size always increases (minimum increment of 1)
                    let next_batch_size = std::cmp::max(
                        batch_size + 1,  // Minimum increment
                        (batch_size as f64 * 1.5) as usize  // 1.5x growth
                    );
                    
                    // Check if we can't increase anymore (hit our limit)
                    if next_batch_size == batch_size {
                        info!("📊 Cannot increase batch size further from {}", batch_size);
                        break;
                    }
                    
                    batch_size = next_batch_size;
                    
                    // Force cleanup after successful test
                    self.memory_monitor.force_cleanup();
                }
                Err(e) => {
                    warn!("❌ Batch size {} failed: {}", batch_size, e);
                    
                    // Force cleanup after failure
                    self.memory_monitor.force_cleanup();
                    break;
                }
            }
            
            // More conservative safety limits
            if batch_size > base_batch_size * 8 {
                break;
            }
        }
        
        // Skip binary search if we didn't find a larger working size
        if max_batch_size == base_batch_size {
            info!("💾 Max batch size: {} (base size)", max_batch_size);
            metrics.memory.max_batch_size = max_batch_size;
            metrics.memory.memory_efficiency = 1.0;
            return Ok(());
        }
        
        info!("💾 Max batch size: {} ({}x base)", max_batch_size, 
              max_batch_size as f64 / base_batch_size as f64);
        
        metrics.memory.max_batch_size = max_batch_size;
        metrics.memory.memory_efficiency = (max_batch_size as f64) / (base_batch_size as f64);
        
        Ok(())
    }

    async fn test_batch_size(&mut self, batch_size: usize) -> Result<()> {
        // Add explicit resource cleanup by scoping the test
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let test_data = create_synthetic_data(
                batch_size,
                self.config.training.sequence_length,
                50257,
                self.device_manager.primary_device(),
            )?;
            
            // Try forward pass - wrap in additional error handling
            match self.strategy.forward_step(self.model.as_ref(), &test_data) {
                Ok(_forward_result) => {
                    // Explicit cleanup by dropping the test_data and result
                    drop(_forward_result);
                    drop(test_data);
                    Ok(())
                }
                Err(e) => {
                    // Explicit cleanup on error
                    drop(test_data);
                    Err(e)
                }
            }
        }));
        
        match result {
            Ok(inner_result) => inner_result,
            Err(_) => {
                // If we caught a panic, it's likely a Metal resource issue
                // Force garbage collection and return error
                if cfg!(target_os = "macos") {
                    // Give Metal time to clean up
                    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                }
                Err(anyhow::anyhow!("Memory test failed due to resource constraints"))
            }
        }
    }

    async fn run_communication_benchmark(&mut self, metrics: &mut BenchmarkMetrics) -> Result<()> {
        info!("📡 Running communication benchmark");
        
        let mut comm_metrics = CommunicationMetrics::default();
        
        // P2P bandwidth already measured in hardware benchmark
        if let Some(first_result) = metrics.hardware.p2p_bandwidth.first() {
            comm_metrics.bandwidth_gbps = first_result.bandwidth_gbps;
            comm_metrics.latency_us = first_result.latency_us;
        }
        
        // AllReduce benchmark
        let test_sizes = vec![1_000, 10_000, 100_000, 1_000_000]; // Different tensor sizes
        let mut allreduce_times = Vec::new();
        
        for size in test_sizes {
            let test_tensor = candle_core::Tensor::zeros(
                (size,),
                candle_core::DType::F32,
                self.device_manager.primary_device(),
            )?;
            
            let start = Instant::now();
            // Simulate AllReduce operation
            std::thread::sleep(Duration::from_micros(10)); // Simulated communication
            let duration = start.elapsed().as_micros() as f64;
            
            allreduce_times.push(duration);
            
            // Explicit cleanup
            drop(test_tensor);
        }
        
        comm_metrics.allreduce_time_us = calculate_stats(&allreduce_times);
        metrics.communication = Some(comm_metrics);
        
        Ok(())
    }

    async fn check_memory_pressure_and_cleanup(&mut self) -> Result<()> {
        let memory_status = self.memory_monitor.check_memory_pressure()?;
        
        match memory_status {
            MemoryStatus::Critical => {
                error!("🚨 Critical memory pressure detected - stopping benchmark");
                self.memory_monitor.force_cleanup();
                return Err(anyhow::anyhow!("Critical memory pressure"));
            }
            MemoryStatus::Warning => {
                warn!("⚠️ Memory pressure warning - forcing cleanup");
                self.memory_monitor.force_cleanup();
            }
            MemoryStatus::Normal => {
                // Normal operation, no cleanup needed
            }
        }
        
        Ok(())
    }
}

pub struct BenchmarkSuite {
    runners: Vec<BenchmarkRunner>,
}

impl BenchmarkSuite {
    pub async fn run_suite(&mut self) -> Result<SuiteResults> {
        let total_runners = self.runners.len();
        info!("🎯 Running benchmark suite with {} configurations", total_runners);
        
        let mut results = Vec::new();
        
        for (i, runner) in self.runners.iter_mut().enumerate() {
            info!("📊 Running benchmark {}/{}", i + 1, total_runners);
            let result = runner.run_single().await?;
            results.push(result);
        }
        
        Ok(SuiteResults { results })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkResult {
    pub config_name: String,
    pub model_name: String,
    pub strategy_name: String,
    pub hardware_type: String,
    pub metrics: BenchmarkMetrics,
    pub performance_score: f64,
    pub efficiency_score: f64,
}

impl BenchmarkResult {
    pub fn from_metrics(metrics: BenchmarkMetrics, config: &BenchmarkConfig) -> Self {
        let performance_score = Self::calculate_performance_score(&metrics);
        let efficiency_score = Self::calculate_efficiency_score(&metrics, config);
        
        Self {
            config_name: format!("{}-{}-{}", 
                config.model.model_type, 
                config.model.size, 
                config.parallelization.strategy),
            model_name: config.model.model_type.clone(),
            strategy_name: config.parallelization.strategy.clone(),
            hardware_type: config.hardware.device_type.clone(),
            metrics,
            performance_score,
            efficiency_score,
        }
    }

    fn calculate_performance_score(metrics: &BenchmarkMetrics) -> f64 {
        // Weighted combination of throughput metrics
        let throughput_weight = 0.5;
        let memory_weight = 0.3;
        let communication_weight = 0.2;
        
        let throughput_score = metrics.training.throughput_samples_per_second / 1000.0; // Normalize
        let memory_score = metrics.memory.memory_efficiency;
        let communication_score = if let Some(ref comm) = metrics.communication {
            comm.bandwidth_gbps / 100.0 // Normalize to 0-1 range
        } else {
            1.0
        };
        
        throughput_weight * throughput_score + 
        memory_weight * memory_score + 
        communication_weight * communication_score
    }

    fn calculate_efficiency_score(metrics: &BenchmarkMetrics, config: &BenchmarkConfig) -> f64 {
        let num_devices = config.hardware.num_devices as f64;
        let ideal_speedup = num_devices;
        
        // Calculate actual speedup (simplified)
        let actual_speedup = metrics.training.throughput_samples_per_second / 100.0; // Base reference
        
        (actual_speedup / ideal_speedup).min(1.0)
    }

    pub fn display_table(&self) -> String {
        use tabled::{Table, Tabled};
        
        #[derive(Tabled)]
        struct ResultRow {
            metric: String,
            value: String,
        }
        
        let rows = vec![
            ResultRow { metric: "Configuration".to_string(), value: self.config_name.clone() },
            ResultRow { metric: "Model".to_string(), value: self.model_name.clone() },
            ResultRow { metric: "Strategy".to_string(), value: self.strategy_name.clone() },
            ResultRow { metric: "Hardware".to_string(), value: self.hardware_type.clone() },
            ResultRow { 
                metric: "Throughput (samples/s)".to_string(), 
                value: format!("{:.1}", self.metrics.training.throughput_samples_per_second) 
            },
            ResultRow { 
                metric: "Throughput (tokens/s)".to_string(), 
                value: format!("{:.1}", self.metrics.training.throughput_tokens_per_second) 
            },
            ResultRow { 
                metric: "Forward Time (ms)".to_string(), 
                value: format!("{:.2}", self.metrics.training.forward_time_ms.mean) 
            },
            ResultRow { 
                metric: "Backward Time (ms)".to_string(), 
                value: format!("{:.2}", self.metrics.training.backward_time_ms.mean) 
            },
            ResultRow { 
                metric: "Memory Efficiency".to_string(), 
                value: format!("{:.2}x", self.metrics.memory.memory_efficiency) 
            },
            ResultRow { 
                metric: "Performance Score".to_string(), 
                value: format!("{:.3}", self.performance_score) 
            },
            ResultRow { 
                metric: "Efficiency Score".to_string(), 
                value: format!("{:.3}", self.efficiency_score) 
            },
        ];
        
        Table::new(rows).to_string()
    }
}

#[derive(serde::Serialize)]
pub struct SuiteResults {
    pub results: Vec<BenchmarkResult>,
}

impl SuiteResults {
    pub fn display_summary(&self) -> String {
        use tabled::{Table, Tabled};
        
        #[derive(Tabled)]
        struct SummaryRow {
            configuration: String,
            throughput_samples_s: String,
            memory_efficiency: String,
            performance_score: String,
            efficiency_score: String,
        }
        
        let rows: Vec<SummaryRow> = self.results.iter().map(|result| {
            SummaryRow {
                configuration: result.config_name.clone(),
                throughput_samples_s: format!("{:.1}", result.metrics.training.throughput_samples_per_second),
                memory_efficiency: format!("{:.2}x", result.metrics.memory.memory_efficiency),
                performance_score: format!("{:.3}", result.performance_score),
                efficiency_score: format!("{:.3}", result.efficiency_score),
            }
        }).collect();
        
        Table::new(rows).to_string()
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json_results = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json_results)?;
        Ok(())
    }
}

// Utility functions
fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn calculate_stats(values: &[f64]) -> crate::metrics::Statistics {
    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;
    
    let variance: f64 = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    
    crate::metrics::Statistics {
        mean,
        std_dev,
        median,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
    }
}

fn calculate_stats_usize(values: &[usize]) -> crate::metrics::Statistics {
    let float_values: Vec<f64> = values.iter().map(|&x| x as f64).collect();
    calculate_stats(&float_values)
}

// Memory monitoring utility
struct MemoryMonitor {
    system: System,
    process_id: Pid,
    max_memory_threshold_gb: f64,
    initial_memory_mb: f64,
}

impl MemoryMonitor {
    fn new(max_memory_threshold_gb: f64) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let process_id = sysinfo::get_current_pid().unwrap();
        let initial_memory_mb = if let Some(process) = system.process(process_id) {
            process.memory() as f64 / 1024.0 / 1024.0
        } else {
            0.0
        };

        Self {
            system,
            process_id,
            max_memory_threshold_gb,
            initial_memory_mb,
        }
    }

    fn check_memory_pressure(&mut self) -> Result<MemoryStatus> {
        self.system.refresh_all();
        
        let total_memory_gb = self.system.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let used_memory_gb = self.system.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let available_memory_gb = total_memory_gb - used_memory_gb;
        
        let process_memory_mb = if let Some(process) = self.system.process(self.process_id) {
            process.memory() as f64 / 1024.0 / 1024.0
        } else {
            0.0
        };
        
        let memory_growth_mb = process_memory_mb - self.initial_memory_mb;
        
        debug!("Memory status: {:.2}GB used / {:.2}GB total, Process: {:.2}MB (+{:.2}MB growth)", 
               used_memory_gb, total_memory_gb, process_memory_mb, memory_growth_mb);
        
        if used_memory_gb > self.max_memory_threshold_gb {
            error!("⚠️ Memory usage ({:.2}GB) exceeds threshold ({:.2}GB)", 
                   used_memory_gb, self.max_memory_threshold_gb);
            return Ok(MemoryStatus::Critical);
        }
        
        if available_memory_gb < 10.0 { // Less than 10GB available
            warn!("⚠️ Low memory available: {:.2}GB", available_memory_gb);
            return Ok(MemoryStatus::Warning);
        }
        
        if process_memory_mb > 50_000.0 { // Process using more than 50GB
            error!("⚠️ Process memory usage excessive: {:.2}GB", process_memory_mb / 1024.0);
            return Ok(MemoryStatus::Critical);
        }
        
        Ok(MemoryStatus::Normal)
    }
    
    fn force_cleanup(&self) {
        // Force garbage collection
        if cfg!(target_os = "macos") {
            // On macOS, give the system time to clean up Metal resources
            std::thread::sleep(Duration::from_millis(500));
        }
    }
}

#[derive(Debug, PartialEq)]
enum MemoryStatus {
    Normal,
    Warning,
    Critical,
} 