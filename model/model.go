// Package model defines the data structures for a transformer language model
// loaded from a GGUF file.
package model

import (
	"llama.go/gguf"
	"llama.go/tensor"
)

// HParams holds the hyperparameters read from GGUF metadata.
type HParams struct {
	VocabSize     int
	HiddenSize    int // d_model (embedding dimension)
	FFNHiddenSize int // d_ffn (intermediate FFN dimension, dense or shared)
	NumLayers     int
	NumHeads      int
	NumKVHeads    int     // for grouped-query attention; == NumHeads for MHA
	HeadDimK      int     // explicit key head dim (0 = HiddenSize/NumHeads)
	HeadDimV      int     // explicit value head dim (0 = same as HeadDimK)
	MaxSeqLen     int
	NormEps       float32
	RopeFreqBase  float32 // for RoPE position embeddings (0 if not used)
	ContextLen    int

	// MoE-specific (zero for dense models)
	NumExperts         int // total experts per layer
	NumExpertsPerToken int // active experts per token (top-k)
	ExpertFFNSize      int // intermediate size per expert
}

// HeadDim returns the per-head dimension for queries/keys.
func (h *HParams) HeadDim() int {
	if h.HeadDimK > 0 {
		return h.HeadDimK
	}
	return h.HiddenSize / h.NumHeads
}

// IsMoE returns true when this model uses Mixture-of-Experts FFN layers.
func (h *HParams) IsMoE() bool { return h.NumExperts > 0 }

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

	// Per-head QK normalization (Qwen3 / Qwen3MoE).
	// Applied after Q/K projection, before RoPE.
	QNorm *tensor.Tensor // [head_dim]
	KNorm *tensor.Tensor // [head_dim]

	// --- FFN norm (small, keep in DRAM) ---
	FFNNorm  *tensor.Tensor // pre-FFN norm weight [hidden]
	FFNNormB *tensor.Tensor // pre-FFN norm bias (nil if RMSNorm)

	// --- Dense FFN weights (loaded from flash on demand) ---
	FFNUp   *tensor.Tensor
	FFNGate *tensor.Tensor
	FFNDown *tensor.Tensor

	// --- GGUF tensor infos (used to load dense FFN from flash) ---
	FFNUpInfo   *gguf.TensorInfo
	FFNGateInfo *gguf.TensorInfo
	FFNDownInfo *gguf.TensorInfo

	// --- MoE fields (nil for dense layers) ---
	// MoERouter is the router weight [n_experts, hidden] (always in DRAM, small).
	MoERouter *tensor.Tensor
	// Expert weight TensorInfos — 3D tensors [hidden, expert_ff, n_experts].
	// Individual expert slices are extracted at inference time.
	ExpertGateInfo *gguf.TensorInfo // ffn_gate_exps: [hidden, expert_ff, n_experts]
	ExpertUpInfo   *gguf.TensorInfo // ffn_up_exps:   [hidden, expert_ff, n_experts]
	ExpertDownInfo *gguf.TensorInfo // ffn_down_exps:  [expert_ff, hidden, n_experts]
}

// FFNLoaded returns true if the dense FFN weights are currently in DRAM.
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
