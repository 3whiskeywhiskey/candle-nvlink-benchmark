use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sysinfo::System;
use chrono::{DateTime, Utc};

use crate::backend::{MemoryInfo, P2PBenchmarkResult};
use crate::config::BenchmarkConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub timestamp: DateTime<Utc>,
    pub config_hash: String,
    pub hardware: HardwareMetrics,
    pub training: TrainingMetrics,
    pub memory: MemoryMetrics,
    pub communication: Option<CommunicationMetrics>,
    pub system: SystemPerformanceMetrics,
}

impl BenchmarkMetrics {
    pub fn new(config: &BenchmarkConfig) -> Self {
        Self {
            timestamp: Utc::now(),
            config_hash: format!("{:x}", md5::compute(serde_json::to_string(config).unwrap())),
            hardware: HardwareMetrics::default(),
            training: TrainingMetrics::default(),
            memory: MemoryMetrics::default(),
            communication: None,
            system: SystemPerformanceMetrics::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareMetrics {
    pub memory_info: Vec<MemoryInfo>,
    pub p2p_bandwidth: Vec<P2PBenchmarkResult>,
    pub system_info: SystemInfo,
    pub device_utilization: Vec<DeviceUtilization>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingMetrics {
    pub forward_time_ms: Statistics,
    pub backward_time_ms: Statistics,
    pub communication_time_ms: Statistics,
    pub memory_usage_bytes: Statistics,
    pub throughput_samples_per_second: f64,
    pub throughput_tokens_per_second: f64,
    pub loss_progression: Vec<f64>,
    pub gradient_norms: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryMetrics {
    pub max_batch_size: usize,
    pub memory_efficiency: f64,
    pub peak_memory_usage_bytes: usize,
    pub memory_fragmentation: f64,
    pub oom_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommunicationMetrics {
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
    pub allreduce_time_us: Statistics,
    pub broadcast_time_us: Statistics,
    pub p2p_transfer_efficiency: f64,
    pub communication_overhead_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemPerformanceMetrics {
    pub cpu_utilization: Statistics,
    pub memory_utilization: Statistics,
    pub network_bandwidth_mbps: Statistics,
    pub power_consumption_watts: Option<Statistics>,
    pub temperature_celsius: Option<Statistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemInfo {
    pub os: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub hostname: String,
    pub rust_version: String,
    pub candle_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceUtilization {
    pub device_id: usize,
    pub gpu_utilization_percent: f64,
    pub memory_utilization_percent: f64,
    pub temperature_celsius: f64,
    pub power_usage_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PowerMetrics {
    pub timestamp: DateTime<Utc>,
    pub total_power_watts: f64,
    pub cpu_power_watts: f64,
    pub gpu_power_watts: f64,
    pub memory_power_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessMetrics {
    pub cpu_usage: f64,
    pub memory_usage_bytes: u64,
    pub virtual_memory_bytes: u64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Statistics {
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

// System metrics collection
pub struct SystemMetrics {
    system: System,
}

impl SystemMetrics {
    pub fn new() -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();
        Ok(Self { system })
    }

    pub fn get_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: format!("{} {}", 
                       System::name().unwrap_or_else(|| "Unknown".to_string()),
                       System::os_version().unwrap_or_else(|| "Unknown".to_string())),
            cpu_brand: self.system.cpus().first()
                .map(|cpu| cpu.brand().to_string())
                .unwrap_or_else(|| "Unknown".to_string()),
            cpu_cores: self.system.cpus().len(),
            total_memory_gb: self.system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            hostname: System::host_name().unwrap_or_else(|| "Unknown".to_string()),
            rust_version: rustc_version_runtime::version().to_string(),
            candle_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn get_cpu_utilization(&mut self) -> f64 {
        self.system.refresh_cpu_all();
        self.system.global_cpu_usage() as f64
    }

    #[allow(dead_code)]
    pub fn get_memory_utilization(&mut self) -> f64 {
        self.system.refresh_memory();
        let used = self.system.used_memory() as f64;
        let total = self.system.total_memory() as f64;
        (used / total) * 100.0
    }

    #[allow(dead_code)]
    pub fn get_process_metrics(&mut self, pid: Option<u32>) -> ProcessMetrics {
        self.system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        
        let pid = pid.unwrap_or_else(|| std::process::id());
        
        if let Some(process) = self.system.process(sysinfo::Pid::from(pid as usize)) {
            ProcessMetrics {
                cpu_usage: process.cpu_usage() as f64,
                memory_usage_bytes: process.memory() * 1024, // Convert KB to bytes
                virtual_memory_bytes: process.virtual_memory() * 1024,
                disk_read_bytes: process.disk_usage().read_bytes,
                disk_write_bytes: process.disk_usage().written_bytes,
            }
        } else {
            ProcessMetrics::default()
        }
    }
}

// Performance monitoring utilities
#[allow(dead_code)]
pub struct PerformanceMonitor {
    start_time: std::time::Instant,
    measurements: Vec<PerformanceMeasurement>,
    system_metrics: SystemMetrics,
}

#[allow(dead_code)]
impl PerformanceMonitor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            start_time: std::time::Instant::now(),
            measurements: Vec::new(),
            system_metrics: SystemMetrics::new()?,
        })
    }

    pub fn start_measurement(&mut self, name: String) {
        let measurement = PerformanceMeasurement {
            name,
            start_time: std::time::Instant::now(),
            end_time: None,
            cpu_usage_start: self.system_metrics.get_cpu_utilization(),
            memory_usage_start: self.system_metrics.get_memory_utilization(),
            cpu_usage_end: 0.0,
            memory_usage_end: 0.0,
        };
        self.measurements.push(measurement);
    }

    pub fn end_measurement(&mut self, name: &str) {
        if let Some(measurement) = self.measurements.iter_mut()
            .find(|m| m.name == name && m.end_time.is_none()) {
            measurement.end_time = Some(std::time::Instant::now());
            measurement.cpu_usage_end = self.system_metrics.get_cpu_utilization();
            measurement.memory_usage_end = self.system_metrics.get_memory_utilization();
        }
    }

    pub fn get_measurements(&self) -> &[PerformanceMeasurement] {
        &self.measurements
    }

    pub fn get_summary(&self) -> PerformanceSummary {
        let mut summary = PerformanceSummary {
            total_runtime_ms: self.start_time.elapsed().as_millis() as f64,
            measurements: HashMap::new(),
        };

        for measurement in &self.measurements {
            if let Some(end_time) = measurement.end_time {
                let duration_ms = (end_time - measurement.start_time).as_millis() as f64;
                let avg_cpu = (measurement.cpu_usage_start + measurement.cpu_usage_end) / 2.0;
                let avg_memory = (measurement.memory_usage_start + measurement.memory_usage_end) / 2.0;

                summary.measurements.insert(measurement.name.clone(), MeasurementSummary {
                    duration_ms,
                    avg_cpu_usage: avg_cpu,
                    avg_memory_usage: avg_memory,
                    peak_memory_usage: measurement.memory_usage_end.max(measurement.memory_usage_start),
                });
            }
        }

        summary
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    pub name: String,
    pub start_time: std::time::Instant,
    pub end_time: Option<std::time::Instant>,
    pub cpu_usage_start: f64,
    pub memory_usage_start: f64,
    pub cpu_usage_end: f64,
    pub memory_usage_end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_runtime_ms: f64,
    pub measurements: HashMap<String, MeasurementSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementSummary {
    pub duration_ms: f64,
    pub avg_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub peak_memory_usage: f64,
}

// GPU monitoring (placeholder - would need actual GPU monitoring libraries)
#[allow(dead_code)]
pub struct GpuMonitor {
    device_count: usize,
}

#[allow(dead_code)]
impl GpuMonitor {
    pub fn new(device_count: usize) -> Self {
        Self { device_count }
    }

    pub fn get_device_utilizations(&self) -> Vec<DeviceUtilization> {
        // Placeholder implementation
        // In a real implementation, this would use NVIDIA ML or similar
        (0..self.device_count).map(|i| DeviceUtilization {
            device_id: i,
            gpu_utilization_percent: 85.0 + (i as f64 * 2.0), // Simulated
            memory_utilization_percent: 70.0 + (i as f64 * 3.0), // Simulated
            temperature_celsius: 65.0 + (i as f64 * 1.5), // Simulated
            power_usage_watts: 200.0 + (i as f64 * 10.0), // Simulated
        }).collect()
    }

    pub fn get_memory_usage(&self, device_id: usize) -> Option<(u64, u64)> {
        // Returns (used_bytes, total_bytes)
        // Placeholder implementation
        if device_id < self.device_count {
            Some((12 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)) // 12GB used, 16GB total
        } else {
            None
        }
    }
}

// Utility functions for metrics
impl Statistics {
    #[allow(dead_code)]
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

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
        
        Self {
            mean,
            std_dev,
            median,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }

    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        !self.mean.is_nan() && !self.std_dev.is_nan()
    }
}

// Export utilities
#[allow(dead_code)]
pub fn export_metrics_to_csv(metrics: &BenchmarkMetrics, path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;
    
    // Write CSV header
    writeln!(file, "timestamp,config_hash,model,strategy,hardware,throughput_samples_s,throughput_tokens_s,forward_time_ms,backward_time_ms,memory_efficiency")?;
    
    // Write data row
    writeln!(file, "{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2}",
        metrics.timestamp.format("%Y-%m-%d %H:%M:%S"),
        metrics.config_hash,
        "transformer", // Would get from config
        "data-parallel", // Would get from config  
        metrics.hardware.system_info.os,
        metrics.training.throughput_samples_per_second,
        metrics.training.throughput_tokens_per_second,
        metrics.training.forward_time_ms.mean,
        metrics.training.backward_time_ms.mean,
        metrics.memory.memory_efficiency
    )?;
    
    Ok(())
}

#[allow(dead_code)]
pub fn export_metrics_to_json(metrics: &BenchmarkMetrics, path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(metrics)?;
    std::fs::write(path, json)?;
    Ok(())
}

// Comparative analysis utilities
#[allow(dead_code)]
pub fn compare_metrics(baseline: &BenchmarkMetrics, comparison: &BenchmarkMetrics) -> ComparisonResult {
    ComparisonResult {
        throughput_improvement: (comparison.training.throughput_samples_per_second / baseline.training.throughput_samples_per_second) - 1.0,
        memory_efficiency_improvement: comparison.memory.memory_efficiency - baseline.memory.memory_efficiency,
        forward_time_improvement: (baseline.training.forward_time_ms.mean / comparison.training.forward_time_ms.mean) - 1.0,
        backward_time_improvement: (baseline.training.backward_time_ms.mean / comparison.training.backward_time_ms.mean) - 1.0,
        communication_overhead_change: 0.0, // Simplified
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub throughput_improvement: f64,
    pub memory_efficiency_improvement: f64,
    pub forward_time_improvement: f64,
    pub backward_time_improvement: f64,
    pub communication_overhead_change: f64,
}

impl ComparisonResult {
    #[allow(dead_code)]
    pub fn display(&self) -> String {
        format!(
            "Performance Comparison:\n\
             Throughput: {:+.1}%\n\
             Memory Efficiency: {:+.2}x\n\
             Forward Time: {:+.1}%\n\
             Backward Time: {:+.1}%\n\
             Communication Overhead: {:+.1}%",
            self.throughput_improvement * 100.0,
            self.memory_efficiency_improvement,
            self.forward_time_improvement * 100.0,
            self.backward_time_improvement * 100.0,
            self.communication_overhead_change
        )
    }
} 