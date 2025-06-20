use candle_core::Device; fn main() { for i in 0..8 { match Device::cuda_if_available(i) { Ok(d) => println!("Candle GPU{}: {:?}", i, d), Err(e) => println!("Candle GPU{}: Error - {}", i, e), } } }
