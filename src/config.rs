use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub hardware: HardwareConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub parallelization: ParallelizationConfig,
    pub benchmark: BenchmarkSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub device_type: String,          // "metal", "cuda", "cpu"
    pub num_devices: usize,
    pub memory_per_device_gb: f32,
    pub interconnect: String,         // "nvlink", "system", "none"
    pub network_bandwidth_gbps: f32,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,           // "transformer", "cnn", "mlp"
    pub size: String,                 // "small", "medium", "large"
    pub parameters: u64,              // Total parameter count
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub layers: usize,
    pub hidden_size: usize,
    pub attention_heads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub warmup_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub mixed_precision: bool,
    pub checkpoint_interval: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConfig {
    pub strategy: String,             // "data", "model", "pipeline", "hybrid"
    pub data_parallel_size: usize,
    pub model_parallel_size: usize,
    pub pipeline_parallel_size: usize,
    pub gradient_sync_freq: usize,
    pub communication_backend: String, // "nccl", "gloo", "mpi"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSettings {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub memory_profiling: bool,
    pub communication_profiling: bool,
    pub power_profiling: bool,
    pub output_detailed_logs: bool,
}

impl HardwareConfig {
    pub fn m4_metal() -> Self {
        Self {
            device_type: "metal".to_string(),
            num_devices: 1,
            memory_per_device_gb: 128.0,
            interconnect: "unified".to_string(),
            network_bandwidth_gbps: 10.0,
            compute_capability: Some("M4".to_string()),
        }
    }

    pub fn v100_nvlink() -> Self {
        Self {
            device_type: "cuda".to_string(),
            num_devices: 4,
            memory_per_device_gb: 16.0,
            interconnect: "nvlink".to_string(),
            network_bandwidth_gbps: 25.0,
            compute_capability: Some("7.0".to_string()),
        }
    }

    pub fn total_memory_gb(&self) -> f32 {
        self.memory_per_device_gb * self.num_devices as f32
    }

    pub fn is_multi_gpu(&self) -> bool {
        self.num_devices > 1
    }

    #[allow(dead_code)]
    pub fn supports_fast_interconnect(&self) -> bool {
        matches!(self.interconnect.as_str(), "nvlink" | "unified")
    }
}

impl BenchmarkConfig {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: BenchmarkConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn load_suite(path: &str) -> Result<Vec<Self>> {
        let content = fs::read_to_string(path)?;
        let configs: Vec<BenchmarkConfig> = serde_json::from_str(&content)?;
        Ok(configs)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn default_for_hardware(hardware: HardwareConfig) -> Self {
        let is_multi_gpu = hardware.is_multi_gpu();
        let total_memory = hardware.total_memory_gb();

        // Adjust model size based on available memory
        let (model_size, parameters, hidden_size, layers) = if total_memory >= 100.0 {
            ("large", 1_300_000_000, 2048, 24)
        } else if total_memory >= 40.0 {
            ("medium", 350_000_000, 1024, 12)
        } else {
            ("small", 125_000_000, 768, 6)
        };

        // Adjust batch size based on memory and multi-GPU setup
        let batch_size = if is_multi_gpu {
            if total_memory >= 100.0 { 64 } else { 32 }
        } else {
            if total_memory >= 100.0 { 16 } else { 8 }
        };

        // Choose parallelization strategy
        let (strategy, data_parallel_size, model_parallel_size) = if is_multi_gpu {
            ("hybrid".to_string(), hardware.num_devices / 2, hardware.num_devices / 2)
        } else {
            ("data".to_string(), 1, 1)
        };

        Self {
            hardware: hardware.clone(),
            model: ModelConfig {
                model_type: "transformer".to_string(),
                size: model_size.to_string(),
                parameters,
                input_shape: vec![batch_size, 512],
                output_shape: vec![batch_size, 512, 50257],
                layers,
                hidden_size,
                attention_heads: Some(hidden_size / 64),
            },
            training: TrainingConfig {
                batch_size,
                sequence_length: 512,
                learning_rate: 1e-4,
                num_epochs: 1,
                warmup_steps: 100,
                gradient_accumulation_steps: if is_multi_gpu { 1 } else { 4 },
                mixed_precision: true,
                checkpoint_interval: 1000,
            },
            parallelization: ParallelizationConfig {
                strategy,
                data_parallel_size,
                model_parallel_size,
                pipeline_parallel_size: 1,
                gradient_sync_freq: 1,
                communication_backend: if hardware.device_type == "cuda" {
                    "nccl".to_string()
                } else {
                    "gloo".to_string()
                },
            },
            benchmark: BenchmarkSettings {
                warmup_iterations: 10,
                measurement_iterations: 50,
                memory_profiling: true,
                communication_profiling: is_multi_gpu,
                power_profiling: false,
                output_detailed_logs: false,
            },
        }
    }

    pub fn estimate_memory_usage(&self) -> f32 {
        // Rough estimate: parameters * 4 bytes (fp32) + activations + gradients + optimizer states
        let param_memory = (self.model.parameters as f32 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        let activation_memory = (self.training.batch_size as f32 * 
                               self.training.sequence_length as f32 * 
                               self.model.hidden_size as f32 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        
        // Rough multiplier for gradients + optimizer states
        let total_memory = param_memory * 3.0 + activation_memory * 2.0;
        total_memory
    }

    pub fn validate(&self) -> Result<()> {
        let estimated_memory = self.estimate_memory_usage();
        let available_memory = self.hardware.total_memory_gb();

        if estimated_memory > available_memory * 0.9 {
            return Err(anyhow::anyhow!(
                "Estimated memory usage ({:.2}GB) exceeds available memory ({:.2}GB)",
                estimated_memory, available_memory
            ));
        }

        if self.parallelization.data_parallel_size * self.parallelization.model_parallel_size 
           > self.hardware.num_devices {
            return Err(anyhow::anyhow!(
                "Total parallel size exceeds available devices"
            ));
        }

        Ok(())
    }
} 