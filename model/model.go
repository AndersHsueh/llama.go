// Package model defines the data structures for a transformer language model
// loaded from a GGUF file.
package model

import (
	"llama.go/gguf"
	"llama.go/tensor"
)

// HParams holds the hyperparameters read from GGUF metadata.
type HParams struct {
	VocabSize    int
	HiddenSize   int // d_model (embedding dimension)
	FFNHiddenSize int // d_ffn (intermediate FFN dimension)
	NumLayers    int
	NumHeads     int
	NumKVHeads   int // for grouped-query attention; == NumHeads for MHA
	MaxSeqLen    int
	NormEps      float32
	RopeFreqBase float32 // for RoPE position embeddings (0 if not used)
	ContextLen   int
}

// HeadDim returns the per-head dimension.
func (h *HParams) HeadDim() int { return h.HiddenSize / h.NumHeads }

// Layer holds the weight tensors for a single transformer layer.
// Attention weights are always resident in DRAM.
// FFN weights are loaded on demand from flash (nil when not loaded).
type Layer struct {
	Index int

	// --- Attention (always in DRAM) ---
	AttnNorm   *tensor.Tensor // pre-attention norm weight [hidden]
	AttnNormB  *tensor.Tensor // pre-attention norm bias (nil if RMSNorm)
	Wq         *tensor.Tensor // query weight [n_heads*head_dim, hidden]
	Wk         *tensor.Tensor // key weight   [n_kv_heads*head_dim, hidden]
	Wv         *tensor.Tensor // value weight [n_kv_heads*head_dim, hidden]
	Wo         *tensor.Tensor // output weight [hidden, n_heads*head_dim]
	Bq, Bk, Bv, Bo *tensor.Tensor // biases (nil if absent)

	// --- FFN norm (small, keep in DRAM) ---
	FFNNorm  *tensor.Tensor // pre-FFN norm weight [hidden]
	FFNNormB *tensor.Tensor // pre-FFN norm bias (nil if RMSNorm)

	// --- FFN weights (loaded from flash on demand) ---
	// FFNUp:   [ffn_hidden, hidden]  (up projection / W1)
	// FFNGate: [ffn_hidden, hidden]  (gate projection, nil if no gating)
	// FFNDown: [hidden, ffn_hidden]  (down projection / W2)
	FFNUp   *tensor.Tensor
	FFNGate *tensor.Tensor
	FFNDown *tensor.Tensor

	// --- GGUF tensor infos (used to load FFN from flash) ---
	FFNUpInfo   *gguf.TensorInfo
	FFNGateInfo *gguf.TensorInfo
	FFNDownInfo *gguf.TensorInfo
}

// FFNLoaded returns true if the FFN weights are currently in DRAM.
func (l *Layer) FFNLoaded() bool { return l.FFNUp != nil }

// UnloadFFN frees the FFN weight tensors from DRAM.
func (l *Layer) UnloadFFN() {
	l.FFNUp = nil
	l.FFNGate = nil
	l.FFNDown = nil
}

// ActivationType describes the FFN activation function.
type ActivationType int

const (
	ActivationReLU  ActivationType = iota
	ActivationGELU
	ActivationSiLU  // SwiGLU (SiLU + gate)
)

// Model holds the full transformer model.
type Model struct {
	HParams HParams
	Arch    string // e.g. "falcon", "llama", "gptneox"
	Name    string

	Activation ActivationType

	// Token embedding table [vocab_size, hidden_size] — always in DRAM.
	TokenEmbedding *tensor.Tensor

	// Final layer norm — always in DRAM.
	OutputNorm  *tensor.Tensor
	OutputNormB *tensor.Tensor

	// LM head (output projection) [vocab_size, hidden_size] — always in DRAM.
	// In many models this is tied to TokenEmbedding (same pointer).
	LMHead *tensor.Tensor

	Layers []*Layer

	// GGUFFile is kept open for on-demand flash reads.
	GGUFFile *gguf.File
}
