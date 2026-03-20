package model

import (
	"fmt"
	"strings"

	"llama.go/flash"
	"llama.go/gguf"
	"llama.go/tensor"
)

// Loader loads a model from a GGUF file.
// Strategy (per the paper):
//   - Token embedding, attention weights, norms, LM head → loaded into DRAM immediately.
//   - FFN weights (up/gate/down projections) → stay on flash; TensorInfo recorded for lazy load.
type Loader struct {
	storage *flash.Storage
	file    *gguf.File
}

// Load opens path and loads the model. FFN weights remain on flash.
func Load(path string) (*Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("model.Load: %w", err)
	}
	s, err := flash.Open(path)
	if err != nil {
		return nil, fmt.Errorf("model.Load flash: %w", err)
	}

	l := &Loader{storage: s, file: f}
	return l.build()
}

func (l *Loader) build() (*Model, error) {
	arch, err := l.file.MetaString("general.architecture")
	if err != nil {
		return nil, err
	}
	name, _ := l.file.MetaString("general.name")

	hp, err := l.loadHParams(arch)
	if err != nil {
		return nil, fmt.Errorf("hparams: %w", err)
	}

	m := &Model{
		HParams:  hp,
		Arch:     arch,
		Name:     name,
		GGUFFile: l.file,
	}

	// Set activation type based on architecture.
	switch arch {
	case "falcon":
		m.Activation = ActivationGELU
	case "llama", "mistral", "qwen2", "qwen3", "qwen3moe":
		m.Activation = ActivationSiLU
	default:
		m.Activation = ActivationReLU
	}

	// Load token embedding.
	m.TokenEmbedding, err = l.mustLoadTensor("token_embd.weight")
	if err != nil {
		return nil, err
	}

	// Load output norm.
	m.OutputNorm, err = l.mustLoadTensor("output_norm.weight")
	if err != nil {
		return nil, err
	}
	m.OutputNormB, _ = l.tryLoadTensor("output_norm.bias")

	// LM head — try dedicated tensor, fall back to tied embeddings.
	lmHead, err := l.tryLoadTensor("output.weight")
	if err != nil {
		return nil, err
	}
	if lmHead != nil {
		m.LMHead = lmHead
	} else {
		m.LMHead = m.TokenEmbedding // weight tying
	}

	// Load per-layer weights.
	m.Layers = make([]*Layer, hp.NumLayers)
	for i := 0; i < hp.NumLayers; i++ {
		layer, err := l.loadLayer(arch, i)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		m.Layers[i] = layer
	}

	return m, nil
}

func (l *Loader) loadHParams(arch string) (HParams, error) {
	var hp HParams
	var err error

	pfx := arch + "."

	hp.VocabSize, err = l.metaInt(pfx + "vocab_size")
	if err != nil {
		// Fallback: infer from token_embd tensor.
		if ti, ok := l.file.TensorsByName["token_embd.weight"]; ok {
			hp.VocabSize = int(ti.Dimensions[1])
		} else {
			return hp, fmt.Errorf("cannot determine vocab_size")
		}
	}

	hp.HiddenSize, err = l.metaInt(pfx + "embedding_length")
	if err != nil {
		return hp, err
	}
	hp.FFNHiddenSize, err = l.metaInt(pfx + "feed_forward_length")
	if err != nil {
		return hp, err
	}
	hp.NumLayers, err = l.metaInt(pfx + "block_count")
	if err != nil {
		return hp, err
	}
	hp.NumHeads, err = l.metaInt(pfx + "attention.head_count")
	if err != nil {
		return hp, err
	}
	hp.NumKVHeads, err = l.metaInt(pfx + "attention.head_count_kv")
	if err != nil {
		hp.NumKVHeads = hp.NumHeads // default: MHA
	}
	hp.NormEps, err = l.file.MetaFloat32(pfx + "attention.layer_norm_epsilon")
	if err != nil {
		hp.NormEps, err = l.file.MetaFloat32(pfx + "attention.layer_norm_rms_epsilon")
		if err != nil {
			hp.NormEps = 1e-5 // safe default
		}
	}
	hp.RopeFreqBase, _ = l.file.MetaFloat32(pfx + "rope.freq_base")
	hp.ContextLen, err = l.metaInt(pfx + "context_length")
	if err != nil {
		hp.ContextLen = 2048
	}
	hp.MaxSeqLen = hp.ContextLen

	// Explicit per-head key/value dimensions (Qwen3, Qwen3MoE).
	hp.HeadDimK, _ = l.metaInt(pfx + "attention.key_length")
	hp.HeadDimV, _ = l.metaInt(pfx + "attention.value_length")

	// MoE parameters (zero for dense models).
	hp.NumExperts, _ = l.metaInt(pfx + "expert_count")
	hp.NumExpertsPerToken, _ = l.metaInt(pfx + "expert_used_count")
	hp.ExpertFFNSize, _ = l.metaInt(pfx + "expert_feed_forward_length")

	return hp, nil
}

// loadLayer loads a single transformer layer.
// Attention weights → DRAM. FFN weights → only TensorInfo recorded.
func (l *Loader) loadLayer(arch string, i int) (*Layer, error) {
	layer := &Layer{Index: i}
	p := fmt.Sprintf("blk.%d.", i) // GGUF standard prefix

	var err error

	// --- Attention norm ---
	layer.AttnNorm, err = l.mustLoadTensor(p + "attn_norm.weight")
	if err != nil {
		return nil, err
	}
	layer.AttnNormB, _ = l.tryLoadTensor(p + "attn_norm.bias")

	// --- Attention projections ---
	layer.Wq, err = l.mustLoadTensor(p + "attn_q.weight")
	if err != nil {
		return nil, err
	}
	layer.Wk, err = l.mustLoadTensor(p + "attn_k.weight")
	if err != nil {
		return nil, err
	}
	layer.Wv, err = l.mustLoadTensor(p + "attn_v.weight")
	if err != nil {
		return nil, err
	}
	layer.Wo, err = l.mustLoadTensor(p + "attn_output.weight")
	if err != nil {
		return nil, err
	}
	layer.Bq, _ = l.tryLoadTensor(p + "attn_q.bias")
	layer.Bk, _ = l.tryLoadTensor(p + "attn_k.bias")
	layer.Bv, _ = l.tryLoadTensor(p + "attn_v.bias")
	layer.Bo, _ = l.tryLoadTensor(p + "attn_output.bias")

	// Per-head Q/K norm (Qwen3, Qwen3MoE).
	layer.QNorm, _ = l.tryLoadTensor(p + "attn_q_norm.weight")
	layer.KNorm, _ = l.tryLoadTensor(p + "attn_k_norm.weight")

	// --- FFN norm ---
	layer.FFNNorm, err = l.mustLoadTensor(p + "ffn_norm.weight")
	if err != nil {
		return nil, err
	}
	layer.FFNNormB, _ = l.tryLoadTensor(p + "ffn_norm.bias")

	// --- MoE layers (qwen3moe and similar) ---
	if l.file.TensorsByName[p+"ffn_gate_inp.weight"] != nil {
		// Router weight [n_experts, hidden] — small enough to keep in DRAM.
		layer.MoERouter, err = l.mustLoadTensor(p + "ffn_gate_inp.weight")
		if err != nil {
			return nil, err
		}
		// Expert weights: 3D tensors, stay on flash.
		layer.ExpertGateInfo = l.file.TensorsByName[p+"ffn_gate_exps.weight"]
		layer.ExpertUpInfo = l.file.TensorsByName[p+"ffn_up_exps.weight"]
		layer.ExpertDownInfo = l.file.TensorsByName[p+"ffn_down_exps.weight"]
		if layer.ExpertUpInfo == nil || layer.ExpertDownInfo == nil {
			return nil, fmt.Errorf("missing MoE expert tensors at layer %d", i)
		}
		return layer, nil
	}

	// --- Dense FFN weights: record TensorInfo only, do NOT load into DRAM ---
	layer.FFNUpInfo = l.file.TensorsByName[p+"ffn_up.weight"]
	layer.FFNGateInfo = l.file.TensorsByName[p+"ffn_gate.weight"]
	layer.FFNDownInfo = l.file.TensorsByName[p+"ffn_down.weight"]

	if layer.FFNUpInfo == nil {
		return nil, fmt.Errorf("missing tensor %sffn_up.weight", p)
	}
	if layer.FFNDownInfo == nil {
		return nil, fmt.Errorf("missing tensor %sffn_down.weight", p)
	}

	return layer, nil
}

// mustLoadTensor reads a tensor by name into DRAM. Returns error if not found.
func (l *Loader) mustLoadTensor(name string) (*tensor.Tensor, error) {
	ti, ok := l.file.TensorsByName[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in GGUF", name)
	}
	t, err := l.storage.ReadTensor(ti)
	if err != nil {
		return nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return t, nil
}

// tryLoadTensor loads a tensor if it exists, returns nil (no error) if absent.
func (l *Loader) tryLoadTensor(name string) (*tensor.Tensor, error) {
	ti, ok := l.file.TensorsByName[name]
	if !ok {
		return nil, nil
	}
	t, err := l.storage.ReadTensor(ti)
	if err != nil {
		return nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	return t, nil
}

// metaInt reads a uint32 or uint64 metadata key as int.
func (l *Loader) metaInt(key string) (int, error) {
	v, ok := l.file.Meta[key]
	if !ok {
		return 0, fmt.Errorf("metadata key %q not found", key)
	}
	switch val := v.Value.(type) {
	case uint32:
		return int(val), nil
	case uint64:
		return int(val), nil
	case int32:
		return int(val), nil
	case int64:
		return int(val), nil
	default:
		return 0, fmt.Errorf("metadata key %q is not an integer (got %T)", key, v.Value)
	}
}

// PrintSummary prints model info to stdout.
func (m *Model) PrintSummary() {
	hp := m.HParams
	fmt.Printf("Model: %s  arch=%s\n", m.Name, m.Arch)
	fmt.Printf("  vocab=%d  hidden=%d  ffn=%d  layers=%d  heads=%d  kv_heads=%d\n",
		hp.VocabSize, hp.HiddenSize, hp.FFNHiddenSize, hp.NumLayers, hp.NumHeads, hp.NumKVHeads)
	fmt.Printf("  norm_eps=%.1e  rope_base=%.0f  ctx=%d\n",
		hp.NormEps, hp.RopeFreqBase, hp.ContextLen)
	if hp.IsMoE() {
		fmt.Printf("  MoE: total_experts=%d  active_per_token=%d  expert_ffn=%d\n",
			hp.NumExperts, hp.NumExpertsPerToken, hp.ExpertFFNSize)
	}

	var dramBytes int64
	countTensor := func(t *tensor.Tensor) {
		if t != nil {
			dramBytes += int64(len(t.Data)) * 4
		}
	}
	countTensor(m.TokenEmbedding)
	countTensor(m.OutputNorm)
	countTensor(m.LMHead)
	for _, l := range m.Layers {
		for _, t := range []*tensor.Tensor{
			l.AttnNorm, l.AttnNormB, l.Wq, l.Wk, l.Wv, l.Wo,
			l.Bq, l.Bk, l.Bv, l.Bo, l.FFNNorm, l.FFNNormB,
		} {
			countTensor(t)
		}
	}
	fmt.Printf("  DRAM (attention+norms): %.1f MB  (FFN stays on flash)\n",
		float64(dramBytes)/1e6)
	fmt.Printf("  FFN tensors: %s\n", func() string {
		s := []string{}
		for _, name := range []string{"up", "gate", "down"} {
			_ = name
		}
		if len(m.Layers) > 0 {
			lay := m.Layers[0]
			if lay.FFNUpInfo != nil {
				s = append(s, "up")
			}
			if lay.FFNGateInfo != nil {
				s = append(s, "gate")
			}
			if lay.FFNDownInfo != nil {
				s = append(s, "down")
			}
		}
		return strings.Join(s, "+")
	}()+" (on flash)")
}
