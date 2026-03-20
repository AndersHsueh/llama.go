# llama.go

**"LLM in a Flash"** 论文 (Apple, 2023) 的 Go 语言实现。  
让大语言模型的权重常驻 **Flash（SSD）**，无需将全部参数加载进 DRAM 或显存，即可完成推理。

> 论文：[LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514)

---

## 核心原理

大模型（如 Falcon 7B）的 FFN 层在 ReLU 激活后约有 **95% 的神经元输出为零**（激活稀疏性）。  
基于这一特性，论文提出三项优化：

| 技术 | 作用 |
|------|------|
| **低秩激活预测器** | 在加载权重前，用小 MLP 预测哪些神经元会激活 |
| **滑动窗口 DRAM 缓存** | 将最近 k 个 token 激活过的神经元保留在 DRAM，增量更新 |
| **行列捆绑 (Bundling)** | Up-proj 第 i 列与 Down-proj 第 i 行连续存储，单次 I/O 同时读取 |

理论效果：可运行 **2× DRAM 容量**的模型，Flash 读取速度比 naive 方案快 **4–20×**。

---

## 项目结构

```
llama.go/
├── gguf/          — GGUF 文件格式解析器 (v1–v3)
├── tensor/        — Tensor 基础层（F32/F16/Q4/Q8 反量化，MatMul, RMSNorm …）
├── flash/         — Flash I/O 抽象（32线程并行读、行列捆绑格式、direct I/O）
├── model/         — 模型加载（注意力权重入 DRAM，FFN 权重留 Flash）
├── kvcache/       — KV Cache（固定长度，O(1) append）
├── tokenizer/     — BPE Tokenizer（从 GGUF vocab 直接构建）
├── inference/     — 推理引擎
│   ├── forward.go    — 完整 Transformer 前向传播（CPU 路径）
│   ├── ffn_flash.go  — Flash-aware FFN（三路调度：窗口缓存 / 稀疏加载 / naive）
│   ├── predictor.go  — 低秩激活预测器
│   └── window.go     — 滑动窗口 DRAM 缓存
├── metal/         — Apple Metal GPU 后端（注意力投影 + RMSNorm 加速）
└── cmd/           — CLI（generate, bench, info）
```

---

## 快速开始

### 环境要求

- Go 1.21+
- macOS（Metal GPU 加速可选；其他平台仅 CPU）
- 支持 GGUF 格式的模型文件（推荐：Falcon 7B ReLU 微调版）

### 编译

```bash
git clone https://github.com/<your-user>/llama.go
cd llama.go
go build -o llama.go .
```

### 推理

```bash
# 默认：滑动窗口 k=5，32 线程 Flash 读取，Metal GPU 加速
./llama.go generate \
  -model  falcon-7b.gguf \
  -prompt "The key to efficient LLM inference is" \
  -n      200

# 参数说明
  -window   5       # 滑动窗口大小 k（0 = 关闭，退化为 naive）
  -threads  32      # Flash 并行读线程数
  -gpu      true    # 启用 Metal GPU（注意力投影 + 归一化）
  -temp     0.8     # 采样温度（0 = greedy）
  -n        128     # 生成 token 数量
```

### 基准测试

```bash
./llama.go bench -model falcon-7b.gguf -sparsity 0.95 -threads 32
```

输出示例：

```
Sequential read  : 3.8 GB/s
Sparse rows (5%) :  19× speedup vs naive
```

### 离线预处理（行列捆绑）

```bash
./llama.go bundle -model falcon-7b.gguf -out ./bundles/
```

---

## Metal GPU 后端

在 Apple Silicon Mac 上自动启用。GPU 负责：
- 注意力的 Q/K/V/O 矩阵投影（`MatMulVec`）
- LM Head（vocab × hidden，最大矩阵）
- RMSNorm / LayerNorm

FFN 权重**始终留在 Flash**，按需加载 —— 这是论文设计的核心。

---

## 支持的模型架构

| 架构 | 激活函数 | 备注 |
|------|----------|------|
| **Falcon** | GELU / ReLU | 论文主要测试对象，95% 稀疏度 |
| **LLaMA** | SiLU (SwiGLU) | 支持 GQA |
| OPT-style | ReLU | 天然稀疏 |

---

## License

MIT © 2024

---

## 参考

- [LLM in a Flash (arXiv:2312.11514)](https://arxiv.org/abs/2312.11514) — Apple, 2023
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF 格式参考实现
- [GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
