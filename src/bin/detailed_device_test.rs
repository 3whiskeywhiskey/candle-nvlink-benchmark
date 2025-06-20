use candle_core::{Device, Tensor, DType};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🔬 Detailed Device Transfer Failure Analysis\n");
    
    // Create devices
    let devices: Vec<Device> = (0..4)
        .map(|i| Device::cuda_if_available(i))
        .collect::<Result<Vec<_>, _>>()?;
    
    println!("📱 Testing detailed transfer process:");
    
    // Test problematic transfer: GPU0 -> GPU1
    println!("\n🔬 Analyzing GPU0 -> GPU1 transfer (known to fail):");
    
    let gpu0 = &devices[0];
    let gpu1 = &devices[1];
    
    // Create test tensor on GPU0
    println!("  Step 1: Creating tensor on GPU0...");
    let tensor_gpu0 = match Tensor::ones((2, 2), DType::F32, gpu0) {
        Ok(t) => {
            println!("    ✅ Tensor created on GPU0: shape {:?}", t.shape());
            t
        }
        Err(e) => {
            println!("    ❌ Failed to create tensor on GPU0: {}", e);
            return Ok(());
        }
    };
    
    // Check if same_device (should be false for different GPUs)
    println!("  Step 2: Checking same_device...");
    let same_dev = gpu0.same_device(gpu1);
    println!("    same_device(GPU0, GPU1) = {} (expected: false)", same_dev);
    
    // Manual CPU roundtrip to diagnose where it fails
    println!("  Step 3: Manual CPU roundtrip simulation...");
    
    // Step 3a: GPU0 -> CPU
    println!("    Step 3a: GPU0 -> CPU...");
    let tensor_cpu = match tensor_gpu0.to_device(&Device::Cpu) {
        Ok(t) => {
            println!("      ✅ GPU0 -> CPU: SUCCESS");
            t
        }
        Err(e) => {
            println!("      ❌ GPU0 -> CPU: FAILED ({})", e);
            return Ok(());
        }
    };
    
    // Step 3b: CPU -> GPU1 
    println!("    Step 3b: CPU -> GPU1...");
    let tensor_gpu1 = match tensor_cpu.to_device(gpu1) {
        Ok(t) => {
            println!("      ✅ CPU -> GPU1: SUCCESS");
            t
        }
        Err(e) => {
            println!("      ❌ CPU -> GPU1: FAILED ({})", e);
            println!("      🚨 FAILURE POINT IDENTIFIED: CPU -> GPU1 transfer");
            return Ok(());
        }
    };
    
    // Step 4: Direct transfer (should use the CPU roundtrip internally)
    println!("  Step 4: Direct GPU0 -> GPU1 transfer...");
    match tensor_gpu0.to_device(gpu1) {
        Ok(_) => println!("    ✅ Direct transfer: SUCCESS"),
        Err(e) => println!("    ❌ Direct transfer: FAILED ({})", e),
    }
    
    // Test working transfer: GPU0 -> GPU2
    println!("\n🔬 Analyzing GPU0 -> GPU2 transfer (known to work):");
    
    let gpu2 = &devices[2];
    
    println!("  Step 1: Direct GPU0 -> GPU2 transfer...");
    match tensor_gpu0.to_device(gpu2) {
        Ok(_) => println!("    ✅ Direct transfer: SUCCESS"),
        Err(e) => println!("    ❌ Direct transfer: FAILED ({})", e),
    }
    
    println!("  Step 2: Manual roundtrip GPU0 -> CPU -> GPU2...");
    
    // CPU -> GPU2
    println!("    Step 2a: CPU -> GPU2...");
    match tensor_cpu.to_device(gpu2) {
        Ok(_) => println!("      ✅ CPU -> GPU2: SUCCESS"),
        Err(e) => println!("      ❌ CPU -> GPU2: FAILED ({})", e),
    }
    
    // Test all CPU -> GPU transfers to identify pattern
    println!("\n🔬 Testing all CPU -> GPU transfers:");
    for (i, device) in devices.iter().enumerate() {
        print!("  CPU -> GPU{}: ", i);
        match tensor_cpu.to_device(device) {
            Ok(_) => println!("✅ SUCCESS"),
            Err(e) => println!("❌ FAILED ({})", e),
        }
    }
    
    println!("\n🏁 Detailed Analysis Complete");
    Ok(())
} 