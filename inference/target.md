Qwen 0.5B Inference Engine Roadmap
Phase 1: Infrastructure & Weights
1. loadConfig() - Parse config.json (hidden_size=896, num_layers=24, num_heads=14, num_kv_heads=2, intermediate_size=4864, vocab_size=151936, rope_theta=1e6, rms_norm_eps=1e-6)
2. Safetensors::load() + get() - Read .safetensors (bf16 → float conversion, handle 8-byte header + JSON index + raw bytes)
3. loadWeights() - Populate QwenWeights struct (embedding, 24 layers with Q/K/V/O, gate/up/down, norms, final_norm)
Phase 2: Core Math Primitives
4. matMul() - GEMM (row-major, no BLAS dependency preferred)
5. addVectors(), softmax(), silu() - Element-wise ops
Phase 3: Normalization & Positional
6. rmsNorm() - No mean subtraction, no bias (unlike GPT-2 LayerNorm)
7. rope() - In-place rotation of Q/K by position, θ=1e6
Phase 4: Attention & FFN
8. groupedQueryAttention() - 14 Q heads, 2 KV heads → repeat KV 7×, causal mask
9. swiglu() - silu(gate) * up → down projection
Phase 5: Transformer Block
10. transformer() - Pre-norm: rmsNorm → GQA → residual → rmsNorm → SwiGLU → residual
Phase 6: Tokenizer (ChatML)
11. loadVocab() - Load tokenizer.json (BPE merges + vocab)
12. encode() / decode() - Byte-level BPE with ChatML special tokens (<|im_start|>, <|im_end|>)
13. applyChatTemplate() - Wrap user message in ChatML format
Phase 7: Generation Loop
14. qwen() - Embedding → 24× transformer → final_norm → logits
15. generate() - Autoregressive loop with sampling (top-p/temperature), KV cache optional