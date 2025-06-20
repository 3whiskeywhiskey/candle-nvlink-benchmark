use candle_core::{Device, Tensor, DType};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🔍 Testing Candle Device Operations\n");
    
    // Test device detection
    let devices: Vec<Device> = (0..4)
        .map(|i| Device::cuda_if_available(i))
        .collect::<Result<Vec<_>, _>>()?;
    
    println!("📱 Available devices:");
    for (i, device) in devices.iter().enumerate() {
        println!("  GPU{}: {:?}", i, device);
    }
    
    // Test basic tensor creation on each device
    println!("\n🧪 Testing tensor creation on each device:");
    let mut tensors = Vec::new();
    
    for (i, device) in devices.iter().enumerate() {
        match Tensor::rand(0f32, 1f32, (4, 4), device) {
            Ok(tensor) => {
                println!("  ✅ GPU{}: Created tensor shape {:?}", i, tensor.shape());
                tensors.push(tensor);
            }
            Err(e) => {
                println!("  ❌ GPU{}: Failed to create tensor: {}", i, e);
            }
        }
    }
    
    // Test tensor transfers between devices
    println!("\n🔄 Testing tensor transfers:");
    if tensors.len() >= 2 {
        let source_tensor = &tensors[0]; // GPU0
        
        for (target_idx, target_device) in devices.iter().enumerate().skip(1) {
            print!("  GPU0 -> GPU{}: ", target_idx);
            match source_tensor.to_device(target_device) {
                Ok(_) => println!("✅ SUCCESS"),
                Err(e) => println!("❌ FAILED ({})", e),
            }
        }
    }
    
    // Test the specific problematic case: GPU2 operations
    println!("\n🎯 Testing GPU2 specific operations:");
    let gpu2_device = &devices[2];
    
    // Test simple scalar tensor creation
    print!("  Creating scalar tensor on GPU2: ");
    match Tensor::from_slice(&[1.0f32], (), gpu2_device) {
        Ok(_) => println!("✅ SUCCESS"),
        Err(e) => println!("❌ FAILED ({})", e),
    }
    
    // Test small tensor creation
    print!("  Creating small tensor on GPU2: ");
    match Tensor::zeros((2, 2), DType::F32, gpu2_device) {
        Ok(_) => println!("✅ SUCCESS"),
        Err(e) => println!("❌ FAILED ({})", e),
    }
    
    // Test tensor shape like our gradients [2, 256, 50257]
    print!("  Creating gradient-sized tensor on GPU2: ");
    match Tensor::zeros((2, 256, 50257), DType::F32, gpu2_device) {
        Ok(large_tensor) => {
            println!("✅ SUCCESS");
            
            // Test moving this tensor to GPU0
            print!("  Moving large tensor GPU2 -> GPU0: ");
            match large_tensor.to_device(&devices[0]) {
                Ok(_) => println!("✅ SUCCESS"),
                Err(e) => println!("❌ FAILED ({})", e),
            }
        }
        Err(e) => println!("❌ FAILED ({})", e),
    }
    
    // Test narrow operation (like our gradient splitting)
    if let Ok(test_tensor) = Tensor::rand(0f32, 1f32, (8, 256, 50257), gpu2_device) {
        print!("  Testing narrow operation on GPU2: ");
        match test_tensor.narrow(0, 4, 2) {
            Ok(_) => println!("✅ SUCCESS"),
            Err(e) => println!("❌ FAILED ({})", e),
        }
    }
    
    println!("\n🏁 Candle Device Test Complete");
    Ok(())
} 