use candle_core::Device;

fn main() {
    println!("🔍 Candle vs CUDA Device Mapping Analysis\n");
    
    // Test Candle device enumeration
    println!("📱 Candle Device Enumeration:");
    for i in 0..8 {
        match Device::cuda_if_available(i) {
            Ok(device) => println!("  Candle index {}: {:?}", i, device),
            Err(e) => println!("  Candle index {}: Error - {}", i, e),
        }
    }
    
    // Test which Candle devices can actually be created
    println!("\n✅ Successfully Created Candle Devices:");
    let mut working_devices = Vec::new();
    for i in 0..8 {
        if let Ok(device) = Device::cuda_if_available(i) {
            working_devices.push((i, device));
        }
    }
    
    for (candle_idx, device) in &working_devices {
        println!("  Candle GPU{}: {:?}", candle_idx, device);
    }
    
    // Try to understand the mapping pattern
    println!("\n🔍 Device ID Pattern Analysis:");
    for (candle_idx, device) in &working_devices {
        if let Device::Cuda(cuda_device) = device {
            println!("  Candle GPU{} -> Internal DeviceId: {:?}", candle_idx, cuda_device);
        }
    }
    
    // Test tensor operations on each working device
    println!("\n🧪 Tensor Creation Test:");
    for (candle_idx, device) in &working_devices {
        match candle_core::Tensor::zeros((2, 2), candle_core::DType::F32, device) {
            Ok(_) => println!("  ✅ Candle GPU{}: Tensor creation SUCCESS", candle_idx),
            Err(e) => println!("  ❌ Candle GPU{}: Tensor creation FAILED ({})", candle_idx, e),
        }
    }
    
    // Test transfers between all working devices
    println!("\n🔄 Transfer Matrix (Candle perspective):");
    if working_devices.len() >= 2 {
        // Create test tensors on each device
        let test_tensors: Vec<_> = working_devices
            .iter()
            .filter_map(|(idx, device)| {
                candle_core::Tensor::ones((2, 2), candle_core::DType::F32, device)
                    .ok()
                    .map(|t| (*idx, t))
            })
            .collect();
        
        print!("       ");
        for (target_idx, _) in &test_tensors {
            print!("GPU{:1} ", target_idx);
        }
        println!();
        
        for (source_idx, source_tensor) in &test_tensors {
            print!("  GPU{}: ", source_idx);
            for (target_idx, _) in &test_tensors {
                if source_idx == target_idx {
                    print!(" -- ");
                } else {
                    let target_device = &working_devices
                        .iter()
                        .find(|(idx, _)| idx == target_idx)
                        .unwrap()
                        .1;
                    
                    match source_tensor.to_device(target_device) {
                        Ok(_) => print!(" ✅ "),
                        Err(_) => print!(" ❌ "),
                    }
                }
            }
            println!();
        }
    }
    
    println!("\n🏁 Device Mapping Analysis Complete");
} 