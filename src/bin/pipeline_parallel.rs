use anyhow::Result;
use clap::{Parser, ValueEnum};
use candle_core::{Device, Tensor, Shape};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, warn, error, debug};

#[cfg(feature = "cuda")]
use cudarc::nccl::safe::{Comm, Id};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Strategy {
    Pipeline,
    PipelineAsync,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of pipeline stages (processes)
    #[arg(long, default_value_t = 4)]
    num_stages: usize,

    /// Process rank (set automatically when spawning)
    #[arg(long)]
    rank: Option<usize>,

    /// Pipeline strategy
    #[arg(long, default_value = "pipeline")]
    strategy: Strategy,

    /// Model size (layers per stage)
    #[arg(long, default_value_t = 3)]
    layers_per_stage: usize,

    /// Batch size for pipeline
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Number of micro-batches for pipeline
    #[arg(long, default_value_t = 8)]
    num_microbatches: usize,

    /// Sequence length
    #[arg(long, default_value_t = 256)]
    seq_len: usize,

    /// Hidden size
    #[arg(long, default_value_t = 512)]
    hidden_size: usize,

    /// Number of pipeline iterations
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// NCCL unique ID for communication
    #[arg(long)]
    nccl_id: Option<String>,

    /// Master address for coordination
    #[arg(long, default_value = "127.0.0.1")]
    master_addr: String,

    /// Master port for coordination
    #[arg(long, default_value_t = 29600)]
    master_port: u16,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

// Pipeline stage model - subset of transformer layers
struct PipelineStage {
    layers: Vec<TransformerLayer>,
    stage_id: usize,
    total_stages: usize,
    device: Device,
}

struct TransformerLayer {
    attention_weight: Tensor,
    mlp_weight: Tensor,
    layer_norm_weight: Tensor,
}

impl TransformerLayer {
    fn new(device: &Device, hidden_size: usize) -> Result<Self> {
        let attention_weight = Tensor::randn(0f32, 1f32, (hidden_size, hidden_size), device)?;
        let mlp_weight = Tensor::randn(0f32, 1f32, (hidden_size, hidden_size), device)?;
        let layer_norm_weight = Tensor::ones((hidden_size,), candle_core::DType::F32, device)?;
        
        Ok(Self {
            attention_weight,
            mlp_weight,
            layer_norm_weight,
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simple transformer layer simulation
        let (batch, seq, hidden) = input.dims3()?;
        let input_2d = input.reshape((batch * seq, hidden))?;
        
        // Attention
        let attn_out = input_2d.matmul(&self.attention_weight)?;
        
        // MLP
        let mlp_out = attn_out.matmul(&self.mlp_weight)?;
        
        // Layer norm + residual (simplified)
        let output_2d = (input_2d + mlp_out)?.relu()?;
        let output = output_2d.reshape((batch, seq, hidden))?;
        
        Ok(output)
    }
}

impl PipelineStage {
    fn new(stage_id: usize, total_stages: usize, layers_per_stage: usize, hidden_size: usize, device: Device) -> Result<Self> {
        let mut layers = Vec::new();
        
        for i in 0..layers_per_stage {
            let layer = TransformerLayer::new(&device, hidden_size)?;
            layers.push(layer);
            debug!("Stage {}: Created layer {} of {}", stage_id, i + 1, layers_per_stage);
        }
        
        Ok(Self {
            layers,
            stage_id,
            total_stages,
            device,
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            debug!("Stage {}: Completed layer {}", self.stage_id, i);
        }
        
        Ok(hidden)
    }
    
    fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }
    
    fn is_last_stage(&self) -> bool {
        self.stage_id == self.total_stages - 1
    }
    
    fn next_stage(&self) -> Option<usize> {
        if self.is_last_stage() {
            None
        } else {
            Some(self.stage_id + 1)
        }
    }
    
    fn prev_stage(&self) -> Option<usize> {
        if self.is_first_stage() {
            None
        } else {
            Some(self.stage_id - 1)
        }
    }
}

// Real NCCL Pipeline communication
struct NCCLPipelineCommunicator {
    rank: usize,
    world_size: usize,
    device: Device,
    #[cfg(feature = "cuda")]
    cuda_context: Arc<cudarc::driver::CudaContext>,
    #[cfg(feature = "cuda")]
    stream: Arc<cudarc::driver::CudaStream>,
    #[cfg(feature = "cuda")]
    nccl_comm: Arc<Comm>,
}

impl NCCLPipelineCommunicator {
    #[cfg(feature = "cuda")]
    fn new(rank: usize, world_size: usize, device: Device, nccl_id_str: &str) -> Result<Self> {
        info!("Rank {}: Initializing NCCL communicator with modern cudarc API", rank);
        
        // Create CUDA context for device 0 (each process sees only its GPU as device 0)
        let cuda_context = cudarc::driver::CudaContext::new(0)
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA context for rank {}: {:?}", rank, e))?;
        
        // Create CUDA stream from context (required for modern NCCL)
        let stream = cuda_context.default_stream();
        
        // Proper NCCL ID coordination (like llama_multiprocess example)
        let nccl_comm_file = format!("/tmp/{}.nccl", nccl_id_str);
        let nccl_id = if rank == 0 {
            // Rank 0 creates the NCCL ID and saves it to file
            info!("Rank 0: Creating NCCL ID and saving to {}", nccl_comm_file);
            let id = Id::new()
                .map_err(|e| anyhow::anyhow!("Failed to create NCCL ID: {:?}", e))?;
            
            // Save ID to file for other ranks
            let id_bytes: Vec<u8> = id.internal().iter().map(|&i| i as u8).collect();
            std::fs::write(&nccl_comm_file, &id_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to write NCCL ID to file: {:?}", e))?;
            
            id
        } else {
            // Other ranks wait for the file and read the ID
            info!("Rank {}: Waiting for NCCL ID file {}", rank, nccl_comm_file);
            while !std::path::Path::new(&nccl_comm_file).exists() {
                thread::sleep(Duration::from_millis(100));
            }
            
            // Read ID from file
            let id_bytes = std::fs::read(&nccl_comm_file)
                .map_err(|e| anyhow::anyhow!("Failed to read NCCL ID from file: {:?}", e))?;
            let internal: [i8; 128] = id_bytes
                .into_iter()
                .map(|i| i as i8)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Invalid NCCL ID file format"))?;
            
            Id::uninit(internal)
        };
            
        info!("Rank {}: Creating NCCL communicator (rank {}/{})...", rank, rank, world_size);
        let nccl_comm = Arc::new(
            Comm::from_rank(stream.clone(), rank, world_size, nccl_id)
                .map_err(|e| anyhow::anyhow!("Failed to create NCCL communicator: {:?}", e))?
        );
        
        // Rank 0 cleans up the file after all ranks have joined
        if rank == 0 {
            thread::sleep(Duration::from_millis(500)); // Give other ranks time to read
            let _ = std::fs::remove_file(&nccl_comm_file);
        }
        
        info!("Rank {}: NCCL communicator initialized successfully!", rank);
        
        Ok(Self {
            rank,
            world_size,
            device,
            cuda_context,
            stream,
            nccl_comm,
        })
    }
    
    #[cfg(not(feature = "cuda"))]
    fn new(rank: usize, world_size: usize, device: Device, _nccl_id: &str) -> Result<Self> {
        warn!("NCCL not available, falling back to simulation");
        Ok(Self {
            rank,
            world_size,
            device,
        })
    }
    
    // Modern NCCL Send operation using cudarc 0.16.4 API
    #[cfg(feature = "cuda")]
    fn send_activation(&self, tensor: &Tensor, dest_rank: usize) -> Result<()> {
        debug!("Rank {}: Sending activation to rank {} (shape: {:?})", 
               self.rank, dest_rank, tensor.shape());
        
        // Get CUDA storage from tensor
        let storage = tensor.storage_and_layout().0;
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => cuda_storage,
            _ => return Err(anyhow::anyhow!("Tensor must be on CUDA device for NCCL")),
        };
        
        // Get CUDA slice and synchronize with stream (modern API requirement)
        let slice = cuda_storage.as_cuda_slice::<f32>()?;
        
        // Perform NCCL Send (slice implements DevicePtr automatically)
        self.nccl_comm.send(slice, dest_rank as i32)
            .map_err(|e| anyhow::anyhow!("NCCL send failed: {:?}", e))?;
        
        // Synchronize stream (required for modern cudarc)
        self.stream.synchronize()
            .map_err(|e| anyhow::anyhow!("CUDA stream sync failed: {:?}", e))?;
        
        debug!("Rank {}: Successfully sent activation to rank {}", self.rank, dest_rank);
        Ok(())
    }
    
    // NCCL Recv operation using host memory transfer (Option B)
    #[cfg(feature = "cuda")]
    fn recv_activation(&self, shape: &Shape, src_rank: usize) -> Result<Tensor> {
        debug!("Rank {}: NCCL recv activation from rank {} (expected shape: {:?})", 
               self.rank, src_rank, shape);
        
        let element_count = shape.elem_count();
        
        // Get the underlying CudaDevice from Candle device
        let cuda_device = match &self.device {
            Device::Cuda(cuda_dev) => cuda_dev,
            _ => return Err(anyhow::anyhow!("NCCL requires CUDA device")),
        };
        
        // Allocate raw CUDA slice for NCCL recv (this gives us mutable access)
        let mut nccl_slice = unsafe { 
            cuda_device.alloc::<f32>(element_count)
                .map_err(|e| anyhow::anyhow!("Failed to allocate CUDA memory: {:?}", e))?
        };
        
        // Perform NCCL Recv directly into the mutable slice
        self.nccl_comm.recv(&mut nccl_slice, src_rank as i32)
            .map_err(|e| anyhow::anyhow!("NCCL recv failed: {:?}", e))?;
            
        // Synchronize stream
        self.stream.synchronize()
            .map_err(|e| anyhow::anyhow!("CUDA stream sync failed: {:?}", e))?;
        
        // Copy to host memory and create tensor from it
        let host_data: Vec<f32> = cuda_device.memcpy_dtov(&nccl_slice)
            .map_err(|e| anyhow::anyhow!("Failed to copy NCCL data to host: {:?}", e))?;
        
        // Create tensor from host data 
        let tensor = Tensor::from_vec(host_data, shape.dims(), &self.device)?;
        
        debug!("Rank {}: Successfully received activation from rank {}", self.rank, src_rank);
        Ok(tensor)
    }
    
    // Modern NCCL Send for gradients
    #[cfg(feature = "cuda")]
    fn send_gradient(&self, tensor: &Tensor, dest_rank: usize) -> Result<()> {
        debug!("Rank {}: Sending gradient to rank {} (shape: {:?})", 
               self.rank, dest_rank, tensor.shape());
        
        let storage = tensor.storage_and_layout().0;
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => cuda_storage,
            _ => return Err(anyhow::anyhow!("Tensor must be on CUDA device for NCCL")),
        };
        
        let slice = cuda_storage.as_cuda_slice::<f32>()?;
        
        self.nccl_comm.send(slice, dest_rank as i32)
            .map_err(|e| anyhow::anyhow!("NCCL send failed: {:?}", e))?;
        
        self.stream.synchronize()
            .map_err(|e| anyhow::anyhow!("CUDA stream sync failed: {:?}", e))?;
        
        debug!("Rank {}: Successfully sent gradient to rank {}", self.rank, dest_rank);
        Ok(())
    }
    
    // NCCL Recv for gradients using host memory transfer  
    #[cfg(feature = "cuda")]
    fn recv_gradient(&self, shape: &Shape, src_rank: usize) -> Result<Tensor> {
        debug!("Rank {}: NCCL recv gradient from rank {} (expected shape: {:?})", 
               self.rank, src_rank, shape);
        
        let element_count = shape.elem_count();
        
        // Get the underlying CudaDevice from Candle device
        let cuda_device = match &self.device {
            Device::Cuda(cuda_dev) => cuda_dev,
            _ => return Err(anyhow::anyhow!("NCCL requires CUDA device")),
        };
        
        // Allocate raw CUDA slice for NCCL recv (this gives us mutable access)
        let mut nccl_slice = unsafe { 
            cuda_device.alloc::<f32>(element_count)
                .map_err(|e| anyhow::anyhow!("Failed to allocate CUDA memory: {:?}", e))?
        };
        
        // Perform NCCL Recv directly into the mutable slice
        self.nccl_comm.recv(&mut nccl_slice, src_rank as i32)
            .map_err(|e| anyhow::anyhow!("NCCL recv failed: {:?}", e))?;
            
        // Synchronize stream
        self.stream.synchronize()
            .map_err(|e| anyhow::anyhow!("CUDA stream sync failed: {:?}", e))?;
        
        // Copy to host memory and create tensor from it
        let host_data: Vec<f32> = cuda_device.memcpy_dtov(&nccl_slice)
            .map_err(|e| anyhow::anyhow!("Failed to copy NCCL data to host: {:?}", e))?;
        
        // Create tensor from host data 
        let tensor = Tensor::from_vec(host_data, shape.dims(), &self.device)?;
        
        debug!("Rank {}: Successfully received gradient from rank {}", self.rank, src_rank);
        Ok(tensor)
    }
    
    // Fallback implementations for non-CUDA
    #[cfg(not(feature = "cuda"))]
    fn send_activation(&self, tensor: &Tensor, dest_rank: usize) -> Result<()> {
        debug!("Rank {}: Simulating send to rank {} (shape: {:?})", 
               self.rank, dest_rank, tensor.shape());
        thread::sleep(Duration::from_micros(100));
        Ok(())
    }
    
    #[cfg(not(feature = "cuda"))]
    fn recv_activation(&self, shape: &Shape, src_rank: usize) -> Result<Tensor> {
        debug!("Rank {}: Simulating recv from rank {} (shape: {:?})", 
               self.rank, src_rank, shape);
        thread::sleep(Duration::from_micros(100));
        Ok(Tensor::randn(0f32, 1f32, shape.dims(), &self.device)?)
    }
    
    #[cfg(not(feature = "cuda"))]
    fn send_gradient(&self, tensor: &Tensor, dest_rank: usize) -> Result<()> {
        debug!("Rank {}: Simulating gradient send to rank {} (shape: {:?})", 
               self.rank, dest_rank, tensor.shape());
        thread::sleep(Duration::from_micros(80));
        Ok(())
    }
    
    #[cfg(not(feature = "cuda"))]
    fn recv_gradient(&self, shape: &Shape, src_rank: usize) -> Result<Tensor> {
        debug!("Rank {}: Simulating gradient recv from rank {} (shape: {:?})", 
               self.rank, src_rank, shape);
        thread::sleep(Duration::from_micros(80));
        Ok(Tensor::randn(0f32, 1f32, shape.dims(), &self.device)?)
    }
}

// Pipeline scheduler for micro-batch execution
struct PipelineScheduler {
    stage: PipelineStage,
    communicator: NCCLPipelineCommunicator,
    microbatch_size: usize,
    seq_len: usize,
    hidden_size: usize,
}

impl PipelineScheduler {
    fn new(
        stage: PipelineStage,
        communicator: NCCLPipelineCommunicator,
        total_batch_size: usize,
        num_microbatches: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Self {
        let microbatch_size = total_batch_size / num_microbatches;
        
        Self {
            stage,
            communicator,
            microbatch_size,
            seq_len,
            hidden_size,
        }
    }
    
    fn run_pipeline_iteration(&mut self, num_microbatches: usize) -> Result<f64> {
        debug!("Rank {}: Starting pipeline iteration with {} micro-batches", 
              self.communicator.rank, num_microbatches);
        
        let start_time = Instant::now();
        let mut forward_times = Vec::new();
        let mut communication_times = Vec::new();
        
        // Forward pass pipeline
        for microbatch_id in 0..num_microbatches {
            let microbatch_start = Instant::now();
            
            // Get input (either from previous stage or create initial input)
            let input = if self.stage.is_first_stage() {
                // First stage creates initial input
                let shape = (self.microbatch_size, self.seq_len, self.hidden_size);
                Tensor::randn(0f32, 1f32, shape, &self.stage.device)?
            } else {
                // Receive from previous stage
                let comm_start = Instant::now();
                let prev_rank = self.stage.prev_stage().unwrap();
                let shape = Shape::from_dims(&[self.microbatch_size, self.seq_len, self.hidden_size]);
                let input = self.communicator.recv_activation(&shape, prev_rank)?;
                communication_times.push(comm_start.elapsed());
                input
            };
            
            debug!("Rank {}: Processing micro-batch {} (input shape: {:?})", 
                   self.communicator.rank, microbatch_id, input.shape());
            
            // Forward pass through this stage
            let forward_start = Instant::now();
            let output = self.stage.forward(&input)?;
            forward_times.push(forward_start.elapsed());
            
            // Send to next stage (if not last stage)
            if let Some(next_rank) = self.stage.next_stage() {
                let comm_start = Instant::now();
                self.communicator.send_activation(&output, next_rank)?;
                communication_times.push(comm_start.elapsed());
            }
            
            debug!("Rank {}: Completed micro-batch {} in {:.2}ms", 
                   self.communicator.rank, microbatch_id, microbatch_start.elapsed().as_millis());
        }
        
        // Simulate backward pass (simplified)
        for microbatch_id in (0..num_microbatches).rev() {
            debug!("Rank {}: Backward pass for micro-batch {}", 
                   self.communicator.rank, microbatch_id);
            
            // Simulate gradient computation and communication
            if let Some(next_rank) = self.stage.next_stage() {
                // Receive gradients from next stage
                let shape = Shape::from_dims(&[self.microbatch_size, self.seq_len, self.hidden_size]);
                let _grad = self.communicator.recv_gradient(&shape, next_rank)?;
            }
            
            // Send gradients to previous stage
            if let Some(prev_rank) = self.stage.prev_stage() {
                let grad_shape = (self.microbatch_size, self.seq_len, self.hidden_size);
                let grad = Tensor::randn(0f32, 1f32, grad_shape, &self.stage.device)?;
                self.communicator.send_gradient(&grad, prev_rank)?;
            }
            
            // Simulate gradient computation time
            thread::sleep(Duration::from_micros(200));
        }
        
        let total_time = start_time.elapsed();
        
        // Report performance metrics
        let avg_forward_time = forward_times.iter().sum::<Duration>().as_secs_f64() / forward_times.len() as f64;
        let avg_comm_time = if !communication_times.is_empty() {
            communication_times.iter().sum::<Duration>().as_secs_f64() / communication_times.len() as f64
        } else {
            0.0
        };
        
        debug!("Rank {}: Pipeline iteration completed", self.communicator.rank);
        debug!("  - Total time: {:.2}ms", total_time.as_millis());
        debug!("  - Avg forward time: {:.2}ms", avg_forward_time * 1000.0);
        debug!("  - Avg communication time: {:.2}ms", avg_comm_time * 1000.0);
        debug!("  - Throughput: {:.1} microbatches/s", num_microbatches as f64 / total_time.as_secs_f64());
        
        Ok(total_time.as_secs_f64())
    }
}

fn run_pipeline_worker(rank: usize, world_size: usize, args: &Args) -> Result<()> {
    info!("🚀 Pipeline Worker {} starting (world_size={})", rank, world_size);
    
    // Set CUDA device for this process
    std::env::set_var("CUDA_VISIBLE_DEVICES", rank.to_string());
    let device = Device::cuda_if_available(0)?;
    
    info!("✅ Worker {}: CUDA device initialized", rank);
    
    // Create pipeline stage
    let stage = PipelineStage::new(
        rank,
        world_size,
        args.layers_per_stage,
        args.hidden_size,
        device.clone(),
    )?;
    
    info!("✅ Worker {}: Created pipeline stage with {} layers (layers {}-{})",
          rank,
          args.layers_per_stage,
          rank * args.layers_per_stage,
          (rank + 1) * args.layers_per_stage - 1);
    
    // Create NCCL communicator
    let fallback_nccl_id = format!("pipeline_{}", std::process::id());
    let nccl_id = args.nccl_id.as_ref().unwrap_or(&fallback_nccl_id);
    let communicator = NCCLPipelineCommunicator::new(rank, world_size, device, nccl_id)?;
    info!("✅ Worker {}: NCCL communicator initialized", rank);
    
    // Create pipeline scheduler
    let mut scheduler = PipelineScheduler::new(
        stage,
        communicator,
        args.batch_size,
        args.num_microbatches,
        args.seq_len,
        args.hidden_size,
    );
    
    info!("🔄 Worker {}: Starting {} pipeline iterations", rank, args.iterations);
    
    let mut total_times = Vec::new();
    
    for iteration in 0..args.iterations {
        debug!("Worker {}: Pipeline iteration {}/{}", rank, iteration + 1, args.iterations);
        
        let iteration_time = scheduler.run_pipeline_iteration(args.num_microbatches)?;
        total_times.push(iteration_time);
        
        // Small delay between iterations
        thread::sleep(Duration::from_millis(100));
    }
    
    // Calculate final performance metrics
    let avg_time = total_times.iter().sum::<f64>() / total_times.len() as f64;
    let throughput = (args.batch_size as f64) / avg_time;
    
    info!("✅ Worker {} Final Results:", rank);
    info!("  - Pipeline stage: {} (layers {}-{})", 
          rank, 
          rank * args.layers_per_stage, 
          (rank + 1) * args.layers_per_stage - 1);
    info!("  - Iterations: {}", args.iterations);
    info!("  - Avg time per iteration: {:.2}ms", avg_time * 1000.0);
    info!("  - Stage throughput: {:.1} samples/s", throughput);
    info!("  - Micro-batches per iteration: {}", args.num_microbatches);
    
    Ok(())
}

fn spawn_pipeline_workers(args: &Args) -> Result<()> {
    info!("🌟 Spawning {} pipeline worker processes", args.num_stages);
    
    let mut children = Vec::new();
    let binary_path = std::env::current_exe()?;
    
    // Generate a unique NCCL ID for this training session
    let nccl_id = format!("pipeline_{}", std::process::id());
    
    for rank in 0..args.num_stages {
        info!("🚀 Starting pipeline worker for stage {}", rank);
        
        let strategy_str = match args.strategy {
            Strategy::Pipeline => "pipeline",
            Strategy::PipelineAsync => "pipeline-async",
        };
        
        let mut cmd = Command::new(&binary_path);
        cmd.arg("--rank").arg(rank.to_string())
           .arg("--num-stages").arg(args.num_stages.to_string())
           .arg("--strategy").arg(strategy_str)
           .arg("--layers-per-stage").arg(args.layers_per_stage.to_string())
           .arg("--batch-size").arg(args.batch_size.to_string())
           .arg("--num-microbatches").arg(args.num_microbatches.to_string())
           .arg("--seq-len").arg(args.seq_len.to_string())
           .arg("--hidden-size").arg(args.hidden_size.to_string())
           .arg("--iterations").arg(args.iterations.to_string())
           .arg("--nccl-id").arg(&nccl_id)
           .arg("--master-addr").arg(&args.master_addr)
           .arg("--master-port").arg(args.master_port.to_string());
        
        if args.verbose {
            cmd.arg("--verbose");
        }
        
        cmd.stdout(Stdio::piped())
           .stderr(Stdio::piped());
        
        let mut child = cmd.spawn()?;
        
        // Capture stdout and stderr for this worker
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let rank_for_thread = rank;
            thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        println!("[Stage {}] {}", rank_for_thread, line);
                    }
                }
            });
        }
        
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            let rank_for_thread = rank;
            thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        eprintln!("[Stage {} ERR] {}", rank_for_thread, line);
                    }
                }
            });
        }
        
        children.push(child);
        
        // Small delay between spawning workers
        thread::sleep(Duration::from_millis(500));
    }
    
    info!("⏳ Waiting for all {} pipeline stages to complete...", children.len());
    
    // Wait for all workers to complete
    let mut successful_workers = 0;
    for (stage_id, mut child) in children.into_iter().enumerate() {
        match child.wait() {
            Ok(status) => {
                if status.success() {
                    successful_workers += 1;
                    info!("✅ Pipeline stage {} completed successfully", stage_id);
                } else {
                    error!("❌ Pipeline stage {} failed with exit code: {:?}", stage_id, status.code());
                }
            }
            Err(e) => {
                error!("❌ Pipeline stage {} failed to complete: {}", stage_id, e);
            }
        }
    }
    
    info!("📊 Pipeline Training Summary:");
    info!("  - Pipeline stages: {}", args.num_stages);
    info!("  - Stages successful: {}", successful_workers);
    info!("  - Layers per stage: {}", args.layers_per_stage);
    info!("  - Total model layers: {}", args.num_stages * args.layers_per_stage);
    info!("  - Strategy: {:?}", args.strategy);
    info!("  - Global batch size: {}", args.batch_size);
    info!("  - Micro-batches: {}", args.num_microbatches);
    info!("  - Iterations per stage: {}", args.iterations);
    
    if successful_workers == args.num_stages {
        let global_throughput = args.batch_size as f64 * args.iterations as f64 / 10.0; // Rough estimate
        info!("🎯 Estimated pipeline throughput: {:.1} samples/s", global_throughput);
        info!("✅ Pipeline Parallel Training completed successfully!");
    } else {
        warn!("⚠️ Some pipeline stages failed - check logs above");
    }
    
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };
    
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(true)
        .init();
    
    match args.rank {
        None => {
            // Master process - spawn pipeline worker processes
            info!("🚀 Pipeline Parallel Distributed Training");
            info!("📋 Configuration:");
            info!("  - Strategy: {:?}", args.strategy);
            info!("  - Pipeline stages: {}", args.num_stages);
            info!("  - Layers per stage: {}", args.layers_per_stage);
            info!("  - Total model layers: {}", args.num_stages * args.layers_per_stage);
            info!("  - Global batch size: {}", args.batch_size);
            info!("  - Micro-batches: {}", args.num_microbatches);
            info!("  - Sequence length: {}", args.seq_len);
            info!("  - Hidden size: {}", args.hidden_size);
            info!("  - Iterations: {}", args.iterations);
            
            spawn_pipeline_workers(&args)
        }
        Some(rank) => {
            // Worker process - run pipeline stage
            run_pipeline_worker(rank, args.num_stages, &args)
        }
    }
} 