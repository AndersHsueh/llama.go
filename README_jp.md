# llama.go

**「LLM in a Flash」**論文（Apple, 2023）のGo言語実装です。  
大規模言語モデルの重みを **Flash（SSD）** に保存したまま推論を行い、DRAMやGPUメモリに全パラメータをロードする必要をなくします。

> 論文：[LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/abs/2312.11514)

---

## ダウンロード

| プラットフォーム | ファイル |
|----------------|---------|
| 🍎 Apple Silicon（M1 / M2 / M3 / M4） | [llama.go_v0.0.1_darwin_arm64.zip](https://github.com/AndersHsueh/llama.go/releases/download/v0.0.1/llama.go_v0.0.1_darwin_arm64.zip) |
| 🍎 Universal（Apple Silicon + Intel Mac） | [llama.go_v0.0.1_darwin_universal.zip](https://github.com/AndersHsueh/llama.go/releases/download/v0.0.1/llama.go_v0.0.1_darwin_universal.zip) |

全リリース：[Releases ページ](https://github.com/AndersHsueh/llama.go/releases)

```bash
# 解凍後に実行権限を付与。初回起動時に macOS がブロックする場合は以下を実行：
chmod +x llama.go_darwin_arm64
xattr -d com.apple.quarantine llama.go_darwin_arm64
```

---

## 基本的なアイデア

Falcon 7B のような大規模モデルでは、ReLU 活性化後の FFN 層において約 **95% のニューロンがゼロを出力する**（活性化スパース性）という特性があります。  
本プロジェクトはこの特性を利用した3つの最適化技術を実装しています：

| 技術 | 目的 |
|------|------|
| **低ランク活性化予測器** | 重みをフラッシュからロードする *前に*、どのニューロンが発火するかを小さな MLP で予測する |
| **スライディングウィンドウ DRAM キャッシュ** | 直近 k トークンで活性化したニューロンを DRAM に保持し、差分のみをロードする |
| **行列バンドリング** | Up-proj の列 i と Down-proj の行 i を連続して保存し、1回の I/O で両方を読み込む |

理論的な効果：DRAM 容量の **2倍のサイズ**のモデルを動かせ、Flash 読み取り速度はナイーブ手法比 **4〜20倍**向上します。

---

## プロジェクト構成

```
llama.go/
├── gguf/          — GGUF ファイルパーサー（v1〜v3 フォーマット）
├── tensor/        — テンソル基本演算（F32/F16/Q4/Q8 逆量化、MatMul、RMSNorm …）
├── flash/         — Flash I/O（32スレッド並列読み込み、バンドル形式、ダイレクト I/O）
├── model/         — モデルローダー（注意重み → DRAM、FFN 重み → Flash）
├── kvcache/       — KV キャッシュ（固定長、O(1) 追加）
├── tokenizer/     — GGUF の語彙から構築する BPE トークナイザー
├── inference/     — 推論エンジン
│   ├── forward.go    — 完全な Transformer フォワードパス（CPU パス）
│   ├── ffn_flash.go  — Flash 対応 FFN（3方向ディスパッチ：ウィンドウ / スパース / ナイーブ）
│   ├── predictor.go  — 低ランク活性化予測器
│   └── window.go     — スライディングウィンドウ DRAM キャッシュ
├── metal/         — Apple Metal GPU バックエンド（注意投影 + RMSNorm 高速化）
└── cmd/           — CLI コマンド：generate, bench, info
```

---

## クイックスタート

### 必要環境

- Go 1.21+
- macOS（Metal GPU 高速化はオプション；その他のプラットフォームは CPU のみ）
- GGUF 形式のモデルファイル（推奨：Falcon 7B の ReLU ファインチューン版）

### ビルド

```bash
git clone https://github.com/<your-user>/llama.go
cd llama.go
go build -o llama.go .
```

### テキスト生成

```bash
# デフォルト：スライディングウィンドウ k=5、32 スレッド Flash 読み込み、Metal GPU ON
./llama.go generate \
  -model  falcon-7b.gguf \
  -prompt "The key to efficient LLM inference is" \
  -n      200

# フラグ
  -window   5       # スライディングウィンドウサイズ k（0 = 無効、ナイーブにフォールバック）
  -threads  32      # 並列 Flash 読み込みスレッド数
  -gpu      true    # Metal GPU を有効化（注意投影 + 正規化）
  -temp     0.8     # サンプリング温度（0 = greedy）
  -n        128     # 生成するトークン数
```

### ベンチマーク

```bash
./llama.go bench -model falcon-7b.gguf -sparsity 0.95 -threads 32
```

出力例：

```
Sequential read  : 3.8 GB/s
Sparse rows (5%) :  19× speedup vs naïve
```

### オフラインバンドル前処理

```bash
./llama.go bundle -model falcon-7b.gguf -out ./bundles/
```

---

## Metal GPU バックエンド

Apple Silicon Mac では自動的に有効化されます。GPU が担当するのは：
- Q / K / V / O 注意投影（`MatMulVec`）
- LM Head 投影（vocab × hidden — 最も大きな行列積）
- 各層の RMSNorm / LayerNorm

FFN の重みは**常に Flash に留まり**、必要に応じてロードされます — これがこの設計の核心です。

---

## サポートされるアーキテクチャ

| アーキテクチャ | 活性化関数 | 備考 |
|---------------|-----------|------|
| **Falcon** | GELU / ReLU | 論文の主要テスト対象；ReLU ファインチューンで 95% スパース性 |
| **LLaMA** | SiLU (SwiGLU) | GQA サポート |
| OPT 系 | ReLU | 自然にスパース |

---

## ライセンス

MIT © 2024

---

## 参考文献

- [LLM in a Flash (arXiv:2312.11514)](https://arxiv.org/abs/2312.11514) — Apple, 2023
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF フォーマット参照実装
- [GGUF 仕様](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
