# Distributed Training Benchmark Suite

A comprehensive benchmark suite for distributed machine learning training using Candle and Rust. Designed to demonstrate HPC techniques and evaluate performance across different hardware configurations, from M4 Metal systems to multi-GPU NVLink setups.

## 🚀 Features

- **Multi-Backend Support**: Metal (Apple Silicon), CUDA, and CPU backends
- **Multiple Parallelization Strategies**: Data, Model, Pipeline, and Hybrid parallelism
- **Comprehensive Benchmarking**: Training throughput, memory efficiency, communication overhead
- **Hardware Characterization**: P2P bandwidth, memory utilization, system performance
- **Flexible Configuration**: JSON-based configuration system for different hardware setups
- **Detailed Metrics**: Performance analysis with statistical reporting and comparison tools

## 🏗️ Architecture

### Core Components

1. **Backend Abstraction** (`src/backend.rs`)
   - Device management for Metal/CUDA/CPU
   - P2P bandwidth measurement
   - Communication primitives

2. **Model Definitions** (`src/models.rs`)
   - Transformer models (small/medium/large)
   - Benchmarkable model trait
   - Synthetic data generation

3. **Parallelization Strategies** (`src/parallelization.rs`)
   - Data Parallel: Split batches across devices
   - Model Parallel: Split model layers across devices
   - Pipeline Parallel: Pipelined execution with micro-batches
   - Hybrid: Combination of strategies

4. **Benchmarking Framework** (`src/benchmark.rs`)
   - Coordinated benchmark execution
   - Performance measurement and profiling
   - Results analysis and reporting

5. **Metrics Collection** (`src/metrics.rs`)
   - Hardware performance tracking
   - System resource monitoring
   - Statistical analysis utilities

## 🖥️ Hardware Support

### Current Development Platform
- **M4 Metal (128GB)**: Development and testing with unified memory architecture

### Target Production Platform
- **4x V100 16GB + NVLink**: Multi-GPU distributed training demonstration
- **ConnectX-5 25Gbit**: Network communication benchmarking
- **Future: BlueField-2 DPU**: Advanced networking offload

## 🛠️ Installation

### Prerequisites
```bash
# Rust (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# For CUDA support (on target system)
# CUDA toolkit and drivers must be installed
```

### Build
```bash
git clone <repository>
cd distributed-training-benchmark

# Build for current platform (M4 Metal by default)
cargo build --release

# Build with CUDA support (for V100 system)
cargo build --release --features cuda --no-default-features
```

## 🎯 Usage

### Basic Commands

```bash
# List available models and strategies
cargo run -- list

# Generate configuration for your hardware
cargo run -- gen-config --hardware m4-metal
cargo run -- gen-config --hardware v100-nvlink

# Run a single benchmark
cargo run -- run --config config/m4-metal.json

# Run a benchmark suite
cargo run -- suite --config config/benchmark-suite.json --output results.json

# Override model or strategy
cargo run -- run --config config/m4-metal.json --model transformer-large --strategy hybrid
```

### Sample Benchmark Run

```bash
# On M4 Metal system
cargo run -- run --config config/m4-metal.json --verbose

# Expected output:
🚀 Distributed Training Benchmark Suite Starting
🔧 Initializing Benchmark Runner
🖥️ Initialized Metal backend on M4
📊 Model: transformer-large (1.3B parameters)
🔄 Setting up data parallel across 1 devices
🚀 Starting single benchmark run
🔍 Running hardware characterization
🏋️ Running training benchmark
💾 Running memory benchmark
📈 Training throughput: 156.2 samples/s, 320,204.8 tokens/s
✅ Benchmark completed successfully
```

## 📊 Benchmark Results

The suite generates comprehensive performance metrics:

### Performance Metrics
- **Throughput**: Samples/second and tokens/second
- **Latency**: Forward/backward pass timing
- **Memory Efficiency**: Maximum batch size, memory utilization
- **Communication Overhead**: P2P bandwidth, AllReduce timing

### Hardware Metrics
- **System Information**: CPU, memory, OS details
- **Device Utilization**: GPU usage, memory usage, temperature
- **Network Performance**: Bandwidth utilization, latency measurements

### Example Results Table
```
┌─────────────────────┬────────────────┐
│ Metric              │ Value          │
├─────────────────────┼────────────────┤
│ Configuration       │ transformer-large-hybrid │
│ Model               │ transformer    │
│ Strategy            │ hybrid         │
│ Hardware            │ metal          │
│ Throughput (samples/s) │ 156.2       │
│ Throughput (tokens/s)  │ 320,204.8   │
│ Forward Time (ms)   │ 45.23          │
│ Backward Time (ms)  │ 78.45          │
│ Memory Efficiency   │ 2.4x           │
│ Performance Score   │ 0.847          │
│ Efficiency Score    │ 0.923          │
└─────────────────────┴────────────────┘
```

## 🔧 Configuration

### Configuration Structure
```json
{
  "hardware": {
    "device_type": "metal|cuda|cpu",
    "num_devices": 1,
    "memory_per_device_gb": 128.0,
    "interconnect": "unified|nvlink|system",
    "network_bandwidth_gbps": 10.0
  },
  "model": {
    "model_type": "transformer",
    "size": "small|medium|large",
    "parameters": 1300000000,
    "layers": 24,
    "hidden_size": 2048
  },
  "parallelization": {
    "strategy": "data|model|pipeline|hybrid",
    "data_parallel_size": 2,
    "model_parallel_size": 2
  }
}
```

### Pre-configured Settings

- **M4 Metal**: Optimized for 128GB unified memory
- **V100 NVLink**: Multi-GPU distributed training
- **Small/Medium/Large**: Different model sizes for various hardware

## 🚀 HPC Techniques Demonstrated

### Memory Optimization
- **Large Batch Sizes**: Utilizing full 128GB on M4 or 64GB total on V100s
- **Memory-Mapped Data Loading**: Efficient data pipeline
- **Gradient Accumulation**: Memory-efficient training

### Communication Optimization
- **NVLink Utilization**: High-bandwidth GPU-to-GPU communication
- **AllReduce Algorithms**: Efficient gradient synchronization
- **Pipeline Parallelism**: Overlapping computation and communication

### Performance Analysis
- **Roofline Modeling**: Theoretical vs. actual performance
- **Scalability Analysis**: Performance scaling with device count
- **Bottleneck Identification**: Memory, compute, or communication bound

## 📈 Development Roadmap

### Phase 1: M4 Metal Development ✅
- [x] Core framework implementation
- [x] Metal backend integration
- [x] Basic transformer models
- [x] Single-device benchmarking

### Phase 2: V100 NVLink Integration 🚧
- [ ] CUDA backend optimization
- [ ] Multi-GPU communication
- [ ] NVLink bandwidth utilization
- [ ] Distributed training strategies

### Phase 3: Advanced Features 📋
- [ ] Custom CUDA kernels
- [ ] Mixed precision optimization
- [ ] Dynamic load balancing
- [ ] BlueField-2 DPU integration

### Phase 4: Production Optimization 📋
- [ ] Performance tuning
- [ ] Memory optimization
- [ ] Communication overlap
- [ ] Real-world workload evaluation

## 🤝 Contributing

This project follows senior engineering practices:

1. **Scope Clarification**: All changes must have clear objectives
2. **Minimal Changes**: Only modify what's necessary for the task
3. **Production Safety**: Changes must not break existing functionality
4. **Performance Focus**: Optimizations must be measurable and justified

## 📝 License

[Your License Here]

## 🔗 References

- [Candle Framework](https://github.com/huggingface/candle)
- [NVIDIA NVLink](https://developer.nvidia.com/nvlink)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/)
- [Distributed Training Best Practices](https://arxiv.org/abs/2006.15704)

---

**Note**: This project is designed as a proof of concept for HPC techniques in distributed machine learning. It demonstrates the capabilities of modern hardware configurations and serves as a foundation for production-scale distributed training systems. 