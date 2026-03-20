# llama.go

A Go implementation of the **"LLM in a Flash"** paper (Apple, 2023).  
Run large language models with weights stored on **Flash (SSD)** — no need to fit all parameters into DRAM or GPU memory.

> Paper: [LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514)

---

## Core Idea

Large models like Falcon 7B exhibit ~**95% activation sparsity** in their FFN layers after ReLU — most neurons output zero for any given token. The paper exploits this with three techniques:

| Technique | Purpose |
|-----------|---------|
| **Low-rank activation predictor** | A small MLP predicts which neurons will fire *before* loading their weights from flash |
| **Sliding-window DRAM cache** | Keep the activated neurons from the last *k* tokens in DRAM; only load the incremental difference |
| **Row-column bundling** | Store up-proj column *i* and down-proj row *i* contiguously — one I/O reads both |

Result: run models **2× larger than DRAM capacity**, with Flash reads **4–20× faster** than naïve loading.

---

## Project Layout

```
llama.go/
├── gguf/          — GGUF file parser (v1–v3 format)
├── tensor/        — Tensor primitives (F32/F16/Q4/Q8 dequant, MatMul, RMSNorm …)
├── flash/         — Flash I/O (32-thread parallel reads, bundled format, direct I/O)
├── model/         — Model loader (attention weights → DRAM, FFN weights → Flash)
├── kvcache/       — KV cache (fixed-length, O(1) append)
├── tokenizer/     — BPE tokenizer built directly from GGUF vocab
├── inference/     — Inference engine
│   ├── forward.go    — Full Transformer forward pass (CPU path)
│   ├── ffn_flash.go  — Flash-aware FFN (3-way dispatch: window / sparse / naïve)
│   ├── predictor.go  — Low-rank activation predictor
│   └── window.go     — Sliding-window DRAM cache
├── metal/         — Apple Metal GPU backend (attention projections + RMSNorm)
└── cmd/           — CLI commands: generate, bench, info
```

---

## Quick Start

### Requirements

- Go 1.21+
- macOS (optional Metal GPU acceleration; other platforms: CPU only)
- A GGUF model file (recommended: Falcon 7B with ReLU fine-tune)

### Build

```bash
git clone https://github.com/<your-user>/llama.go
cd llama.go
go build -o llama.go .
```

### Generate text

```bash
# Default: sliding window k=5, 32 flash threads, Metal GPU on
./llama.go generate \
  -model  falcon-7b.gguf \
  -prompt "The key to efficient LLM inference is" \
  -n      200

# Flags
  -window   5       # sliding window size k (0 = disabled, naïve fallback)
  -threads  32      # parallel flash-read threads
  -gpu      true    # enable Metal GPU (attention projections + norms)
  -temp     0.8     # sampling temperature (0 = greedy)
  -n        128     # tokens to generate
```

### Benchmark

```bash
./llama.go bench -model falcon-7b.gguf -sparsity 0.95 -threads 32
```

Sample output:

```
Sequential read  : 3.8 GB/s
Sparse rows (5%) :  19× speedup vs naïve
```

### Offline bundle pre-processing

```bash
./llama.go bundle -model falcon-7b.gguf -out ./bundles/
```

---

## Metal GPU Backend

Automatically enabled on Apple Silicon Macs. The GPU handles:
- Q / K / V / O attention projections (`MatMulVec`)
- LM Head projection (vocab × hidden — the largest matmul)
- RMSNorm / LayerNorm per layer

FFN weights **always stay on Flash** and are loaded on demand — that is the entire point of the design.

---

## Supported Architectures

| Architecture | Activation | Notes |
|-------------|-----------|-------|
| **Falcon** | GELU / ReLU | Primary paper target; 95% sparsity with ReLU fine-tune |
| **LLaMA** | SiLU (SwiGLU) | GQA supported |
| OPT-style | ReLU | Naturally sparse |

---

## License

MIT © 2024

---

## References

- [LLM in a Flash (arXiv:2312.11514)](https://arxiv.org/abs/2312.11514) — Apple, 2023
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format reference implementation
- [GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
