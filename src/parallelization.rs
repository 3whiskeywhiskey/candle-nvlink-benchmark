use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tracing::{info, debug};

use crate::backend::{DeviceManager, CommunicationBackend, NaiveCommunicationBackend};
use crate::models::BenchmarkModel;

pub trait ParallelizationStrategy: Send + Sync {
    #[allow(dead_code)]
    fn name(&self) -> &str;
    fn setup(&mut self, device_manager: &DeviceManager) -> Result<()>;
    fn forward_step(&self, model: &dyn BenchmarkModel, input: &Tensor) -> Result<ForwardResult>;
    fn backward_step(&self, gradients: &[Tensor]) -> Result<BackwardResult>;
    fn communication_overhead(&self) -> f64;
    #[allow(dead_code)]
    fn memory_efficiency(&self) -> f64;
    #[allow(dead_code)]
    fn expected_speedup(&self, num_devices: usize) -> f64;
}

#[derive(Debug, Clone)]
pub struct ForwardResult {
    pub output: Tensor,
    pub activations: Vec<Tensor>,
    pub computation_time_ms: f64,
    pub communication_time_ms: f64,
    pub memory_used_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct BackwardResult {
    pub gradients: Vec<Tensor>,
    pub computation_time_ms: f64,
    pub communication_time_ms: f64,
    pub memory_used_bytes: usize,
}

// Data Parallel Strategy
pub struct DataParallelStrategy {
    name: String,
    devices: Vec<Device>,
    #[allow(dead_code)]
    replica_models: Vec<Box<dyn BenchmarkModel>>,
    communication_backend: Box<dyn CommunicationBackend>,
    #[allow(dead_code)]
    sync_frequency: usize,
    #[allow(dead_code)]
    step_count: usize,
}

impl DataParallelStrategy {
    pub fn new(sync_frequency: usize) -> Self {
        Self {
            name: "data-parallel".to_string(),
            devices: Vec::new(),
            replica_models: Vec::new(),
            communication_backend: Box::new(NaiveCommunicationBackend),
            sync_frequency,
            step_count: 0,
        }
    }
}

impl ParallelizationStrategy for DataParallelStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn setup(&mut self, device_manager: &DeviceManager) -> Result<()> {
        self.devices = device_manager.devices().to_vec();
        info!("🔄 Setting up data parallel across {} devices", self.devices.len());
        
        // In a real implementation, we would create model replicas here
        // For now, we'll simulate the setup
        debug!("Data parallel setup complete");
        Ok(())
    }

    fn forward_step(&self, model: &dyn BenchmarkModel, input: &Tensor) -> Result<ForwardResult> {
        let start_time = std::time::Instant::now();
        
        // Split batch across devices
        let batch_size = input.dim(0)?;
        let per_device_batch = batch_size / self.devices.len();
        
        let mut device_outputs = Vec::new();
        let mut total_memory = 0;
        
        // Simulate forward pass on each device
        for (i, device) in self.devices.iter().enumerate() {
            let start_idx = i * per_device_batch;
            let end_idx = if i == self.devices.len() - 1 {
                batch_size
            } else {
                (i + 1) * per_device_batch
            };
            
            // Create sub-batch for this device
            let device_input = input.narrow(0, start_idx, end_idx - start_idx)?;
            let device_input = device_input.to_device(device)?;
            
            // Forward pass
            let output = model.forward(&device_input)?;
            device_outputs.push(output);
            
            total_memory += device_input.elem_count() * 4; // 4 bytes per f32
        }
        
        // Gather outputs and clear the vector to avoid keeping references
        let combined_output = Tensor::cat(&device_outputs, 0)?;
        
        let computation_time = start_time.elapsed().as_millis() as f64;
        
        Ok(ForwardResult {
            output: combined_output,
            activations: Vec::new(), // Don't keep activation references
            computation_time_ms: computation_time,
            communication_time_ms: 0.0, // No communication in forward pass for data parallel
            memory_used_bytes: total_memory,
        })
    }

    fn backward_step(&self, gradients: &[Tensor]) -> Result<BackwardResult> {
        let _start_time = std::time::Instant::now();
        
        // Simulate gradient computation
        let computation_time = 50.0; // Simulated backward pass time
        
        // AllReduce communication for gradient synchronization
        let comm_start = std::time::Instant::now();
        let _synchronized_gradients = self.communication_backend
            .all_reduce(&gradients[0], &self.devices)?;
        let communication_time = comm_start.elapsed().as_millis() as f64;
        
        // Calculate memory usage without keeping gradient references
        let memory_used = gradients.iter().map(|g| g.elem_count() * 4).sum();
        
        Ok(BackwardResult {
            gradients: Vec::new(), // Don't keep gradient references to avoid memory accumulation
            computation_time_ms: computation_time,
            communication_time_ms: communication_time,
            memory_used_bytes: memory_used,
        })
    }

    fn communication_overhead(&self) -> f64 {
        // Data parallel has communication overhead during gradient sync
        0.15 // 15% overhead typical for data parallel
    }

    fn memory_efficiency(&self) -> f64 {
        // Each device stores full model, less memory efficient
        1.0 / self.devices.len() as f64
    }

    fn expected_speedup(&self, num_devices: usize) -> f64 {
        // Near-linear speedup for data parallel with good interconnect
        num_devices as f64 * 0.85 // 85% efficiency
    }
}

// Model Parallel Strategy
pub struct ModelParallelStrategy {
    name: String,
    devices: Vec<Device>,
    layer_assignments: HashMap<usize, usize>, // layer_id -> device_id
    #[allow(dead_code)]
    communication_backend: Box<dyn CommunicationBackend>,
}

impl ModelParallelStrategy {
    pub fn new() -> Self {
        Self {
            name: "model-parallel".to_string(),
            devices: Vec::new(),
            layer_assignments: HashMap::new(),
            communication_backend: Box::new(NaiveCommunicationBackend),
        }
    }
}

impl ParallelizationStrategy for ModelParallelStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn setup(&mut self, device_manager: &DeviceManager) -> Result<()> {
        self.devices = device_manager.devices().to_vec();
        info!("🔀 Setting up model parallel across {} devices", self.devices.len());
        
        // Assign layers to devices (simplified)
        let num_layers = 24; // Assume transformer with 24 layers
        let layers_per_device = num_layers / self.devices.len();
        
        for layer_id in 0..num_layers {
            let device_id = layer_id / layers_per_device;
            let device_id = device_id.min(self.devices.len() - 1);
            self.layer_assignments.insert(layer_id, device_id);
        }
        
        debug!("Model parallel layer assignments: {:?}", self.layer_assignments);
        Ok(())
    }

    fn forward_step(&self, _model: &dyn BenchmarkModel, input: &Tensor) -> Result<ForwardResult> {
        let start_time = std::time::Instant::now();
        
        // Simulate pipeline forward pass
        let mut current_activations = input.clone();
        let mut total_comm_time = 0.0;
        let mut total_memory = input.elem_count() * 4;
        
        // Simulate passing through layers on different devices
        for (_layer_id, &device_id) in &self.layer_assignments {
            let device = &self.devices[device_id];
            
            // Transfer to appropriate device
            let comm_start = std::time::Instant::now();
            current_activations = current_activations.to_device(device)?;
            total_comm_time += comm_start.elapsed().as_millis() as f64;
            
            // Simulate layer computation
            // In real implementation, this would be actual layer forward pass
            let scale = Tensor::from_slice(&[1.01f32], (), current_activations.device())?;
            current_activations = current_activations.broadcast_mul(&scale)?; // Dummy operation
            
            total_memory += current_activations.elem_count() * 4;
        }
        
        let computation_time = start_time.elapsed().as_millis() as f64 - total_comm_time;
        
        Ok(ForwardResult {
            output: current_activations,
            activations: Vec::new(), // Don't keep activation references
            computation_time_ms: computation_time,
            communication_time_ms: total_comm_time,
            memory_used_bytes: total_memory,
        })
    }

    fn backward_step(&self, gradients: &[Tensor]) -> Result<BackwardResult> {
        let start_time = std::time::Instant::now();
        
        // Simulate backward pass through model parallel layers
        let mut current_gradients = gradients[0].clone();
        let mut total_comm_time = 0.0;
        
        // Reverse pass through layers (convert to sorted vec first)
        let mut assignments: Vec<_> = self.layer_assignments.iter().collect();
        assignments.sort_by_key(|&(layer_id, _)| layer_id);
        for (_layer_id, &device_id) in assignments.iter().rev() {
            let device = &self.devices[device_id];
            
            // Transfer gradients
            let comm_start = std::time::Instant::now();
            current_gradients = current_gradients.to_device(device)?;
            total_comm_time += comm_start.elapsed().as_millis() as f64;
            
            // Simulate gradient computation
            let scale = Tensor::from_slice(&[0.99f32], (), current_gradients.device())?;
            current_gradients = current_gradients.broadcast_mul(&scale)?; // Dummy operation
        }
        
        let computation_time = start_time.elapsed().as_millis() as f64 - total_comm_time;
        let memory_used = current_gradients.elem_count() * 4;
        
        Ok(BackwardResult {
            gradients: Vec::new(), // Don't keep gradient references
            computation_time_ms: computation_time,
            communication_time_ms: total_comm_time,
            memory_used_bytes: memory_used,
        })
    }

    fn communication_overhead(&self) -> f64 {
        // Model parallel has high communication overhead between layers
        0.35 // 35% overhead due to activation transfers
    }

    fn memory_efficiency(&self) -> f64 {
        // Better memory efficiency as model is split across devices
        self.devices.len() as f64 * 0.8 // 80% efficiency
    }

    fn expected_speedup(&self, num_devices: usize) -> f64 {
        // Limited speedup due to sequential nature
        1.0 + (num_devices as f64 - 1.0) * 0.3
    }
}

// Pipeline Parallel Strategy
pub struct PipelineParallelStrategy {
    name: String,
    devices: Vec<Device>,
    pipeline_stages: usize,
    micro_batch_size: usize,
    #[allow(dead_code)]
    communication_backend: Box<dyn CommunicationBackend>,
}

impl PipelineParallelStrategy {
    pub fn new(micro_batch_size: usize) -> Self {
        Self {
            name: "pipeline-parallel".to_string(),
            devices: Vec::new(),
            pipeline_stages: 4,
            micro_batch_size,
            communication_backend: Box::new(NaiveCommunicationBackend),
        }
    }
}

impl ParallelizationStrategy for PipelineParallelStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn setup(&mut self, device_manager: &DeviceManager) -> Result<()> {
        self.devices = device_manager.devices().to_vec();
        self.pipeline_stages = self.devices.len();
        info!("⚡ Setting up pipeline parallel with {} stages", self.pipeline_stages);
        Ok(())
    }

    fn forward_step(&self, _model: &dyn BenchmarkModel, input: &Tensor) -> Result<ForwardResult> {
        let start_time = std::time::Instant::now();
        
        let batch_size = input.dim(0)?;
        let num_micro_batches = (batch_size + self.micro_batch_size - 1) / self.micro_batch_size;
        
        let mut final_outputs = Vec::new();
        let mut total_comm_time = 0.0;
        let mut total_memory = 0;
        
        // Simulate pipelined execution
        for micro_batch_id in 0..num_micro_batches {
            let start_idx = micro_batch_id * self.micro_batch_size;
            let end_idx = (start_idx + self.micro_batch_size).min(batch_size);
            
            let micro_batch = input.narrow(0, start_idx, end_idx - start_idx)?;
            
            // Process through pipeline stages
            let mut stage_output = micro_batch;
            for stage_id in 0..self.pipeline_stages {
                let device = &self.devices[stage_id];
                
                let comm_start = std::time::Instant::now();
                stage_output = stage_output.to_device(device)?;
                total_comm_time += comm_start.elapsed().as_millis() as f64;
                
                // Simulate stage computation
                let scale = Tensor::from_slice(&[1.01f32], (), stage_output.device())?;
                stage_output = stage_output.broadcast_mul(&scale)?;
                total_memory += stage_output.elem_count() * 4;
            }
            
            final_outputs.push(stage_output);
        }
        
        let final_output = Tensor::cat(&final_outputs, 0)?;
        let computation_time = start_time.elapsed().as_millis() as f64 - total_comm_time;
        
        Ok(ForwardResult {
            output: final_output,
            activations: Vec::new(), // Don't keep activation references
            computation_time_ms: computation_time,
            communication_time_ms: total_comm_time,
            memory_used_bytes: total_memory,
        })
    }

    fn backward_step(&self, gradients: &[Tensor]) -> Result<BackwardResult> {
        let start_time = std::time::Instant::now();
        
        // Simulate pipelined backward pass
        let mut total_comm_time = 0.0;
        let mut total_memory = 0;
        
        for gradient in gradients {
            let mut stage_gradient = gradient.clone();
            
            // Backward through pipeline stages (reverse order)
            for stage_id in (0..self.pipeline_stages).rev() {
                let device = &self.devices[stage_id];
                
                let comm_start = std::time::Instant::now();
                stage_gradient = stage_gradient.to_device(device)?;
                total_comm_time += comm_start.elapsed().as_millis() as f64;
                
                // Simulate gradient computation
                let scale = Tensor::from_slice(&[0.99f32], (), stage_gradient.device())?;
                stage_gradient = stage_gradient.broadcast_mul(&scale)?;
            }
            
            total_memory += stage_gradient.elem_count() * 4;
        }
        
        let computation_time = start_time.elapsed().as_millis() as f64 - total_comm_time;
        
        Ok(BackwardResult {
            gradients: Vec::new(), // Don't keep gradient references
            computation_time_ms: computation_time,
            communication_time_ms: total_comm_time,
            memory_used_bytes: total_memory,
        })
    }

    fn communication_overhead(&self) -> f64 {
        // Pipeline parallel has moderate communication overhead
        0.20 // 20% overhead
    }

    fn memory_efficiency(&self) -> f64 {
        // Good memory efficiency with pipeline stages
        self.pipeline_stages as f64 * 0.9
    }

    fn expected_speedup(&self, num_devices: usize) -> f64 {
        // Good speedup with pipeline parallelism
        num_devices as f64 * 0.75 // 75% efficiency
    }
}

// Hybrid Strategy (combines data and model parallelism)
pub struct HybridParallelStrategy {
    name: String,
    data_parallel_size: usize,
    model_parallel_size: usize,
    data_strategy: DataParallelStrategy,
    model_strategy: ModelParallelStrategy,
}

impl HybridParallelStrategy {
    pub fn new(data_parallel_size: usize, model_parallel_size: usize) -> Self {
        Self {
            name: "hybrid-parallel".to_string(),
            data_parallel_size,
            model_parallel_size,
            data_strategy: DataParallelStrategy::new(1),
            model_strategy: ModelParallelStrategy::new(),
        }
    }
}

impl ParallelizationStrategy for HybridParallelStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn setup(&mut self, device_manager: &DeviceManager) -> Result<()> {
        info!("🔄🔀 Setting up hybrid parallel: {}x data, {}x model", 
              self.data_parallel_size, self.model_parallel_size);
        
        // Setup both strategies
        self.data_strategy.setup(device_manager)?;
        self.model_strategy.setup(device_manager)?;
        
        Ok(())
    }

    fn forward_step(&self, model: &dyn BenchmarkModel, input: &Tensor) -> Result<ForwardResult> {
        // Combine data and model parallel forward passes
        // This is a simplified implementation
        let data_result = self.data_strategy.forward_step(model, input)?;
        let model_result = self.model_strategy.forward_step(model, &data_result.output)?;
        
        Ok(ForwardResult {
            output: model_result.output,
            activations: [data_result.activations, model_result.activations].concat(),
            computation_time_ms: data_result.computation_time_ms + model_result.computation_time_ms,
            communication_time_ms: data_result.communication_time_ms + model_result.communication_time_ms,
            memory_used_bytes: data_result.memory_used_bytes + model_result.memory_used_bytes,
        })
    }

    fn backward_step(&self, gradients: &[Tensor]) -> Result<BackwardResult> {
        // Combine backward passes
        let model_result = self.model_strategy.backward_step(gradients)?;
        let data_result = self.data_strategy.backward_step(&model_result.gradients)?;
        
        Ok(BackwardResult {
            gradients: data_result.gradients,
            computation_time_ms: data_result.computation_time_ms + model_result.computation_time_ms,
            communication_time_ms: data_result.communication_time_ms + model_result.communication_time_ms,
            memory_used_bytes: data_result.memory_used_bytes + model_result.memory_used_bytes,
        })
    }

    fn communication_overhead(&self) -> f64 {
        // Hybrid has combined overhead
        (self.data_strategy.communication_overhead() + self.model_strategy.communication_overhead()) / 2.0
    }

    fn memory_efficiency(&self) -> f64 {
        // Better memory efficiency than pure data parallel
        (self.data_parallel_size * self.model_parallel_size) as f64 * 0.7
    }

    fn expected_speedup(&self, num_devices: usize) -> f64 {
        // Good speedup combining both strategies
        num_devices as f64 * 0.8 // 80% efficiency
    }
}

// Factory function to create parallelization strategies
pub fn create_strategy(strategy_name: &str, config: &crate::config::ParallelizationConfig) -> Result<Box<dyn ParallelizationStrategy>> {
    match strategy_name {
        "data" => Ok(Box::new(DataParallelStrategy::new(config.gradient_sync_freq))),
        "model" => Ok(Box::new(ModelParallelStrategy::new())),
        "pipeline" => Ok(Box::new(PipelineParallelStrategy::new(8))), // 8 micro-batch size
        "hybrid" => Ok(Box::new(HybridParallelStrategy::new(
            config.data_parallel_size,
            config.model_parallel_size,
        ))),
        _ => Err(anyhow::anyhow!("Unknown parallelization strategy: {}", strategy_name)),
    }
} 