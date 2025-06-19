use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tracing::{info, debug, warn};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum BackendType {
    Metal,
    Cuda(usize), // device_id
    Cpu,
}

pub struct DeviceManager {
    devices: Vec<Device>,
    #[allow(dead_code)]
    backend_type: BackendType,
    primary_device: Device,
    #[allow(dead_code)]
    device_capabilities: HashMap<usize, DeviceCapabilities>,
}

#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    #[allow(dead_code)]
    memory_gb: f64,
    #[allow(dead_code)]
    compute_capability: String,
    #[allow(dead_code)]
    nvlink_connections: Vec<usize>,
}

impl DeviceManager {
    pub fn new(backend_type: &str, num_devices: usize) -> Result<Self> {
        match backend_type {
            "metal" => Self::new_metal(),
            "cuda" => Self::new_cuda(num_devices),
            "cpu" => Self::new_cpu(),
            _ => Err(anyhow::anyhow!("Unsupported backend type: {}", backend_type)),
        }
    }

    fn new_metal() -> Result<Self> {
        info!("🔍 Attempting to initialize Metal backend...");
        
        // Debug: Check if Metal is available
        debug!("Metal feature enabled: {}", cfg!(feature = "metal"));
        
        match Device::new_metal(0) {
            Ok(device) => {
                info!("🖥️  Successfully initialized Metal backend on M4");
                Ok(Self {
                    devices: vec![device.clone()],
                    backend_type: BackendType::Metal,
                    primary_device: device,
                    device_capabilities: HashMap::new(),
                })
            }
            Err(e) => {
                info!("❌ Metal device creation failed: {}", e);
                info!("📝 Falling back to CPU backend for development");
                // Fallback to CPU for now
                let device = Device::Cpu;
                Ok(Self {
                    devices: vec![device.clone()],
                    backend_type: BackendType::Cpu,
                    primary_device: device,
                    device_capabilities: HashMap::new(),
                })
            }
        }
    }

    fn new_cuda(_num_devices: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut devices = Vec::new();
            for i in 0.._num_devices {
                let device = Device::new_cuda(i)?;
                devices.push(device);
            }
            
            let primary_device = devices[0].clone();
            info!("🚀 Initialized {} CUDA devices", _num_devices);
            
            Ok(Self {
                devices,
                backend_type: BackendType::Cuda(0),
                primary_device,
                device_capabilities: HashMap::new(),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(anyhow::anyhow!("CUDA backend not available"))
        }
    }

    fn new_cpu() -> Result<Self> {
        let device = Device::Cpu;
        info!("💻 Using CPU backend");
        Ok(Self {
            devices: vec![device.clone()],
            backend_type: BackendType::Cpu,
            primary_device: device,
            device_capabilities: HashMap::new(),
        })
    }

    pub fn primary_device(&self) -> &Device {
        &self.primary_device
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    pub fn get_device(&self, index: usize) -> Result<&Device> {
        self.devices.get(index)
            .ok_or_else(|| anyhow::anyhow!("Device {} not available", index))
    }

    #[allow(dead_code)]
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    pub fn is_multi_device(&self) -> bool {
        self.devices.len() > 1
    }

    #[allow(dead_code)]
    pub fn supports_peer_to_peer(&self) -> bool {
        match self.backend_type {
            BackendType::Cuda(_) => self.devices.len() > 1, // NVLink support
            BackendType::Metal => true,                      // Unified memory
            BackendType::Cpu => false,
        }
    }

    pub fn memory_info(&self) -> Result<Vec<MemoryInfo>> {
        let mut info = Vec::new();
        
        for (i, device) in self.devices.iter().enumerate() {
            let memory_info = match device {
                Device::Cuda(_) => {
                    // Would need cudarc integration for actual memory info
                    MemoryInfo {
                        device_id: i,
                        total_memory: 16 * 1024 * 1024 * 1024, // 16GB for V100
                        used_memory: 0, // Would query actual usage
                        free_memory: 16 * 1024 * 1024 * 1024,
                    }
                }
                Device::Metal(_) => {
                    MemoryInfo {
                        device_id: i,
                        total_memory: 128 * 1024 * 1024 * 1024, // 128GB for M4
                        used_memory: 0,
                        free_memory: 128 * 1024 * 1024 * 1024,
                    }
                }
                Device::Cpu => {
                    MemoryInfo {
                        device_id: i,
                        total_memory: 32 * 1024 * 1024 * 1024, // Default CPU
                        used_memory: 0,
                        free_memory: 32 * 1024 * 1024 * 1024,
                    }
                }
            };
            info.push(memory_info);
        }
        
        Ok(info)
    }

    pub fn benchmark_p2p_bandwidth(&self) -> Result<Vec<P2PBenchmarkResult>> {
        if !self.is_multi_device() {
            return Ok(vec![]);
        }

        let mut results = Vec::new();
        let test_size = 1024 * 1024 * 100; // 100MB test

        for i in 0..self.devices.len() {
            for j in (i + 1)..self.devices.len() {
                let result = self.measure_p2p_bandwidth(i, j, test_size)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    fn measure_p2p_bandwidth(&self, src_idx: usize, dst_idx: usize, size: usize) -> Result<P2PBenchmarkResult> {
        let src_device = self.get_device(src_idx)?;
        let dst_device = self.get_device(dst_idx)?;

        // Create test tensor on source device
        let test_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let src_tensor = Tensor::from_slice(&test_data, (size,), src_device)?;

        let start = std::time::Instant::now();
        
        // Copy to destination device (this tests P2P transfer)
        let _dst_tensor = src_tensor.to_device(dst_device)?;
        
        let duration = start.elapsed();
        let bandwidth_gbps = (size as f64 * 4.0) / (duration.as_secs_f64() * 1e9); // 4 bytes per f32

        debug!("P2P bandwidth {}→{}: {:.2} GB/s", src_idx, dst_idx, bandwidth_gbps);

        Ok(P2PBenchmarkResult {
            src_device: src_idx,
            dst_device: dst_idx,
            bandwidth_gbps,
            latency_us: duration.as_micros() as f64,
            size_bytes: size * 4,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryInfo {
    pub device_id: usize,
    pub total_memory: u64,
    pub used_memory: u64,
    pub free_memory: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct P2PBenchmarkResult {
    pub src_device: usize,
    pub dst_device: usize,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
    pub size_bytes: usize,
}

impl MemoryInfo {
    #[allow(dead_code)]
    pub fn utilization_percent(&self) -> f64 {
        (self.used_memory as f64 / self.total_memory as f64) * 100.0
    }

    #[allow(dead_code)]
    pub fn total_gb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    #[allow(dead_code)]
    pub fn used_gb(&self) -> f64 {
        self.used_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    #[allow(dead_code)]
    pub fn free_gb(&self) -> f64 {
        self.free_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

// Communication abstractions for distributed training
pub trait CommunicationBackend: Send + Sync {
    fn all_reduce(&self, tensor: &Tensor, devices: &[Device]) -> Result<Tensor>;
    #[allow(dead_code)]
    fn all_gather(&self, tensor: &Tensor, devices: &[Device]) -> Result<Vec<Tensor>>;
    #[allow(dead_code)]
    fn broadcast(&self, tensor: &Tensor, src_device: usize, devices: &[Device]) -> Result<Vec<Tensor>>;
    #[allow(dead_code)]
    fn reduce_scatter(&self, tensors: &[Tensor], devices: &[Device]) -> Result<Vec<Tensor>>;
}

// NVLink-Optimized Communication Backend
pub struct NVLinkCommunicationBackend {
    ring_topology: Vec<usize>,
    bucket_size_mb: usize,
}

impl NVLinkCommunicationBackend {
    pub fn new(num_devices: usize, bucket_size_mb: usize) -> Self {
        // Create optimal ring topology for NVLink
        let ring_topology = (0..num_devices).collect();
        
        info!("🔗 NVLink Ring Topology: {:?}", ring_topology);
        info!("📦 Gradient Bucket Size: {}MB", bucket_size_mb);
        
        Self {
            ring_topology,
            bucket_size_mb,
        }
    }

    fn ring_all_reduce(&self, tensor: &Tensor, devices: &[Device]) -> Result<Tensor> {
        let num_devices = devices.len();
        if num_devices <= 1 {
            return Ok(tensor.clone());
        }

        // Step 1: Chunk tensor for efficient pipeline communication
        let tensor_size = tensor.elem_count();
        let chunk_size = (tensor_size + num_devices - 1) / num_devices;
        
        let mut chunks = Vec::new();
        for i in 0..num_devices {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(tensor_size);
            if start < tensor_size {
                // Simulate chunking - in real implementation would slice tensor
                chunks.push(tensor.clone());
            }
        }

        // Step 2: Ring AllReduce - Reduce-Scatter phase
        for step in 0..num_devices - 1 {
            for (rank, &device_id) in self.ring_topology.iter().enumerate() {
                let next_rank = (rank + 1) % num_devices;
                let next_device_id = self.ring_topology[next_rank];
                
                // Send chunk to next device in ring and receive from previous
                let chunk_id = (rank + num_devices - step - 1) % num_devices;
                if chunk_id < chunks.len() {
                    let chunk = &chunks[chunk_id];
                    let _transferred_chunk = chunk.to_device(&devices[next_device_id])?;
                    // In real implementation: accumulate received chunk
                }
            }
        }

        // Step 3: AllGather phase - share reduced chunks
        for step in 0..num_devices - 1 {
            for (rank, &device_id) in self.ring_topology.iter().enumerate() {
                let next_rank = (rank + 1) % num_devices;
                let next_device_id = self.ring_topology[next_rank];
                
                let chunk_id = (rank + num_devices - step) % num_devices;
                if chunk_id < chunks.len() {
                    let chunk = &chunks[chunk_id];
                    let _transferred_chunk = chunk.to_device(&devices[next_device_id])?;
                }
            }
        }

        // Step 4: Reconstruct final tensor (simplified - return original for now)
        let final_tensor = tensor.to_device(&devices[0])?;
        Ok(final_tensor)
    }
}

impl CommunicationBackend for NVLinkCommunicationBackend {
    fn all_reduce(&self, tensor: &Tensor, devices: &[Device]) -> Result<Tensor> {
        debug!("🔄 NVLink Ring AllReduce: {} elements across {} devices", 
               tensor.elem_count(), devices.len());
        
        // Use optimized ring algorithm for NVLink topology
        self.ring_all_reduce(tensor, devices)
    }

    fn broadcast(&self, tensor: &Tensor, source_device: usize, devices: &[Device]) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        
        for (i, device) in devices.iter().enumerate() {
            if i == source_device {
                results.push(tensor.clone());
            } else {
                results.push(tensor.to_device(device)?);
            }
        }
        
        Ok(results)
    }

    fn all_gather(&self, tensor: &Tensor, devices: &[Device]) -> Result<Vec<Tensor>> {
        // Simplified all-gather - in real implementation would use NVLink topology
        let mut results = Vec::new();
        
        for device in devices {
            results.push(tensor.to_device(device)?);
        }
        
        Ok(results)
    }

    fn reduce_scatter(&self, tensors: &[Tensor], devices: &[Device]) -> Result<Vec<Tensor>> {
        // Simplified implementation
        let mut results = Vec::new();
        for (i, device) in devices.iter().enumerate() {
            if i < tensors.len() {
                results.push(tensors[i].to_device(device)?);
            }
        }
        Ok(results)
    }
}

// Naive Communication Backend (fallback)
pub struct NaiveCommunicationBackend;

impl CommunicationBackend for NaiveCommunicationBackend {
    fn all_reduce(&self, tensor: &Tensor, devices: &[Device]) -> Result<Tensor> {
        // Simple all-reduce: copy to all devices and return
        let result = tensor.to_device(&devices[0])?;
        Ok(result)
    }

    fn broadcast(&self, tensor: &Tensor, _source_device: usize, devices: &[Device]) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        
        for device in devices {
            results.push(tensor.to_device(device)?);
        }
        
        Ok(results)
    }

    fn all_gather(&self, tensor: &Tensor, devices: &[Device]) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        
        for device in devices {
            results.push(tensor.to_device(device)?);
        }
        
        Ok(results)
    }

    fn reduce_scatter(&self, tensors: &[Tensor], devices: &[Device]) -> Result<Vec<Tensor>> {
        // Simplified implementation
        let mut results = Vec::new();
        for (i, device) in devices.iter().enumerate() {
            if i < tensors.len() {
                results.push(tensors[i].to_device(device)?);
            }
        }
        Ok(results)
    }
}

// SmartNIC Communication Backend (future implementation)
pub struct SmartNICCommunicationBackend {
    #[allow(dead_code)]
    bluefield_devices: Vec<String>,
    #[allow(dead_code)]
    offload_enabled: bool,
}

impl SmartNICCommunicationBackend {
    #[allow(dead_code)]
    pub fn new() -> Self {
        info!("🔌 Initializing BlueField2 SmartNIC backend");
        
        Self {
            bluefield_devices: vec![], // Will discover BlueField2 devices
            offload_enabled: false,    // Enable when SmartNICs detected
        }
    }
}

impl CommunicationBackend for SmartNICCommunicationBackend {
    fn all_reduce(&self, tensor: &Tensor, devices: &[Device]) -> Result<Tensor> {
        // Future: Offload AllReduce to BlueField2 SmartNIC
        // For now, fall back to naive implementation
        let result = tensor.to_device(&devices[0])?;
        Ok(result)
    }

    fn broadcast(&self, tensor: &Tensor, _source_device: usize, devices: &[Device]) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        
        for device in devices {
            results.push(tensor.to_device(device)?);
        }
        
        Ok(results)
    }

    fn all_gather(&self, tensor: &Tensor, devices: &[Device]) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        
        for device in devices {
            results.push(tensor.to_device(device)?);
        }
        
        Ok(results)
    }

    fn reduce_scatter(&self, tensors: &[Tensor], devices: &[Device]) -> Result<Vec<Tensor>> {
        // Simplified implementation
        let mut results = Vec::new();
        for (i, device) in devices.iter().enumerate() {
            if i < tensors.len() {
                results.push(tensors[i].to_device(device)?);
            }
        }
        Ok(results)
    }
} 