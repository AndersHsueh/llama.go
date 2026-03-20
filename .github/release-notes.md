## 下载说明 / Download

| 文件 | 适用平台 |
|------|---------|
| `llama.go_*_darwin_arm64.zip` | **Apple Silicon** Mac（M1 / M2 / M3 / M4） |
| `llama.go_*_darwin_universal.zip` | **Universal** — 同时支持 Apple Silicon 和 Intel Mac |

### 安装步骤

```bash
# 解压（以 Silicon 版为例）
unzip llama.go_*_darwin_arm64.zip

# 添加可执行权限
chmod +x llama.go_darwin_arm64

# 首次运行需要在「系统设置 → 隐私与安全性」中允许（或 xattr 清除隔离属性）
xattr -d com.apple.quarantine llama.go_darwin_arm64

# 生成文字（需要 .gguf 模型文件）
./llama.go_darwin_arm64 generate -model falcon-7b.gguf -prompt "Hello" -n 100
```

### 更新内容

- 完整实现「LLM in a Flash」论文三大核心优化
  - 低秩激活预测器（Predictor）
  - 滑动窗口 DRAM 缓存（Windowing）
  - 行列捆绑 Flash 格式（Bundling）
- Apple Metal GPU 后端：注意力投影 + RMSNorm 加速（在 Apple M4 Pro 验证通过）
- 支持 GGUF v1–v3，Falcon / LLaMA / OPT 架构
- BPE Tokenizer、KV Cache、CLI（generate / bench）
