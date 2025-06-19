use anyhow::Result;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{Linear, VarBuilder, VarMap, Module, layer_norm, LayerNorm};

// Helper function for Metal-compatible layer normalization
fn metal_compatible_layer_norm(layer_norm: &LayerNorm, input: &Tensor) -> Result<Tensor> {
    match layer_norm.forward(input) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Fallback: Manual layer normalization for Metal compatibility
            let mean = input.mean_keepdim(D::Minus1)?;
            let variance = input.var_keepdim(D::Minus1)?;
            let eps = 1e-5;
            
            // Ensure proper broadcasting
            let mean_expanded = mean.broadcast_as(input.shape())?;
            let variance_expanded = variance.broadcast_as(input.shape())?;
            
            let normalized = ((input - mean_expanded)? / (variance_expanded + eps)?.sqrt()?)?;
            Ok(normalized)
        }
    }
}

// Helper function for Metal-compatible softmax
fn metal_compatible_softmax_last_dim(input: &Tensor) -> Result<Tensor> {
    match candle_nn::ops::softmax_last_dim(input) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Fallback: Manual softmax for Metal compatibility
            let max_vals = input.max_keepdim(D::Minus1)?;
            let shifted = input.broadcast_sub(&max_vals)?;
            let exp_vals = shifted.exp()?;
            let sum_exp = exp_vals.sum_keepdim(D::Minus1)?;
            let softmax = exp_vals.broadcast_div(&sum_exp)?;
            Ok(softmax)
        }
    }
}

pub trait BenchmarkModel: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    #[allow(dead_code)]
    fn parameters(&self) -> Vec<Tensor>;
    fn parameter_count(&self) -> usize;
    #[allow(dead_code)]
    fn memory_footprint(&self) -> usize;
    fn name(&self) -> &str;
}

#[derive(Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[allow(dead_code)]
    pub dropout: f64,
    pub layer_norm_eps: f64,
}

impl TransformerConfig {
    pub fn small() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 1024,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn medium() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 1024,
            num_layers: 12,
            num_attention_heads: 16,
            intermediate_size: 4096,
            max_position_embeddings: 2048,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn large() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 2048,
            num_layers: 24,
            num_attention_heads: 32,
            intermediate_size: 8192,
            max_position_embeddings: 4096,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

pub struct BenchmarkTransformer {
    config: TransformerConfig,
    embeddings: Embeddings,
    layers: Vec<TransformerLayer>,
    final_layer_norm: LayerNorm,
    lm_head: Linear,
    name: String,
}

impl BenchmarkTransformer {
    pub fn new(config: TransformerConfig, device: &Device, name: String) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let embeddings = Embeddings::new(&config, &vb)?;
        
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(&config, &vb.pp(&format!("layer_{}", i)))?;
            layers.push(layer);
        }

        let final_layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("final_layer_norm"))?;
        let lm_head = candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            config,
            embeddings,
            layers,
            final_layer_norm,
            lm_head,
            name,
        })
    }
}

impl BenchmarkModel for BenchmarkTransformer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input)?;
        
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        hidden_states = metal_compatible_layer_norm(&self.final_layer_norm, &hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        
        Ok(logits)
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Simplified for benchmark - in real implementation would collect all actual parameters
        vec![]
    }

    fn parameter_count(&self) -> usize {
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_layers;
        let intermediate_size = self.config.intermediate_size;

        // Embeddings: vocab_size * hidden_size + max_pos * hidden_size
        let embedding_params = vocab_size * hidden_size + self.config.max_position_embeddings * hidden_size;
        
        // Each layer: attention + ffn + layer norms
        let attention_params = 4 * hidden_size * hidden_size; // qkv + output projection
        let ffn_params = hidden_size * intermediate_size + intermediate_size * hidden_size;
        let layer_norm_params = 2 * hidden_size; // 2 layer norms per layer
        let layer_params = attention_params + ffn_params + layer_norm_params;
        
        // Final layer norm + lm_head
        let final_params = hidden_size + hidden_size * vocab_size;
        
        embedding_params + num_layers * layer_params + final_params
    }

    fn memory_footprint(&self) -> usize {
        self.parameter_count() * 4 // 4 bytes per f32
    }

    fn name(&self) -> &str {
        &self.name
    }
}

struct Embeddings {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    layer_norm: LayerNorm,
}

impl Embeddings {
    fn new(config: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("word_embeddings"))?;
        let position_embeddings = candle_nn::embedding(config.max_position_embeddings, config.hidden_size, vb.pp("position_embeddings"))?;
        let layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("layer_norm"))?;
        
        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let position_ids = Tensor::arange(0, seq_len as i64, input_ids.device())?
            .unsqueeze(0)?
            .expand((input_ids.dim(0)?, seq_len))?;

        let word_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;
        
        let embeddings = word_embeds.add(&position_embeds)?;
        Ok(metal_compatible_layer_norm(&self.layer_norm, &embeddings)?)
    }

    #[allow(dead_code)]
    fn parameters(&self) -> Vec<Tensor> {
        // Simplified for benchmark - in real implementation would collect actual parameters
        vec![]
    }
}

struct TransformerLayer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,
}

impl TransformerLayer {
    fn new(config: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(config, &vb.pp("attention"))?;
        let feed_forward = FeedForward::new(config, &vb.pp("feed_forward"))?;
        let attention_layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("attention_layer_norm"))?;
        let output_layer_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("output_layer_norm"))?;

        Ok(Self {
            attention,
            feed_forward,
            attention_layer_norm,
            output_layer_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection
        let attention_output = self.attention.forward(hidden_states)?;
        let attention_output = metal_compatible_layer_norm(&self.attention_layer_norm, &attention_output.add(hidden_states)?)?;
        
        // Feed-forward with residual connection
        let ff_output = self.feed_forward.forward(&attention_output)?;
        let output = metal_compatible_layer_norm(&self.output_layer_norm, &ff_output.add(&attention_output)?)?;
        
        Ok(output)
    }

    #[allow(dead_code)]
    fn parameters(&self) -> Vec<Tensor> {
        // Simplified for benchmark
        vec![]
    }
}

struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn new(config: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        
        let q_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        
        let q = self.q_proj.forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.k_proj.forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self.v_proj.forward(hidden_states)?
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scale_tensor = Tensor::from_slice(&[scale as f32], (), q.device())?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attention_scores = q.matmul(&k_t)?.broadcast_mul(&scale_tensor)?;
        let attention_weights = metal_compatible_softmax_last_dim(&attention_scores)?;
        let attention_output = attention_weights.matmul(&v)?;

        // Reshape and project
        let attention_output = attention_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, hidden_size))?;
        
        Ok(self.out_proj.forward(&attention_output)?)
    }

    #[allow(dead_code)]
    fn parameters(&self) -> Vec<Tensor> {
        // Simplified for benchmark
        vec![]
    }
}

struct FeedForward {
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForward {
    fn new(config: &TransformerConfig, vb: &VarBuilder) -> Result<Self> {
        let up_proj = candle_nn::linear(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?;

        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let intermediate = self.up_proj.forward(hidden_states)?;
        let intermediate = intermediate.gelu()?; // GELU activation
        Ok(self.down_proj.forward(&intermediate)?)
    }

    #[allow(dead_code)]
    fn parameters(&self) -> Vec<Tensor> {
        // Note: In real implementation, we'd need to track parameters differently
        // This is a simplified version for the benchmark
        vec![]
    }
}

// Factory function to create models
pub fn create_model(model_type: &str, size: &str, device: &Device) -> Result<Box<dyn BenchmarkModel>> {
    match model_type {
        "transformer" => {
            let config = match size {
                "small" => TransformerConfig::small(),
                "medium" => TransformerConfig::medium(),
                "large" => TransformerConfig::large(),
                _ => return Err(anyhow::anyhow!("Unknown transformer size: {}", size)),
            };
            let name = format!("transformer-{}", size);
            let model = BenchmarkTransformer::new(config, device, name)?;
            Ok(Box::new(model))
        }
        _ => Err(anyhow::anyhow!("Unknown model type: {}", model_type)),
    }
}

// Utility function to create synthetic data
pub fn create_synthetic_data(batch_size: usize, seq_len: usize, vocab_size: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<u32> = (0..batch_size * seq_len)
        .map(|i| (i % vocab_size) as u32)
        .collect();
    
    Ok(Tensor::from_slice(&data, (batch_size, seq_len), device)?)
} 