// Package inference implements the transformer forward pass with flash-aware
// FFN weight loading, following the "LLM in a Flash" paper.
package inference

import (
	"fmt"
	"math"

	"llama.go/flash"
	"llama.go/kvcache"
	"llama.go/metal"
	"llama.go/model"
	"llama.go/tensor"
)

// Context holds inference state: the model, KV cache, flash storage,
// and optional Metal GPU device for accelerated matrix operations.
type Context struct {
	Model   *model.Model
	KV      *kvcache.KVCache
	Storage *flash.Storage
	GPU     *metal.Device // nil when running CPU-only

	// DRAMBudgetBytes is the soft limit for FFN DRAM usage (0 = unlimited).
	DRAMBudgetBytes int64
}

// NewContext creates an inference context.
// maxSeq is the maximum sequence length to allocate KV cache for.
// useGPU=true attempts to initialise a Metal device; if unavailable it silently
// falls back to CPU.
func NewContext(m *model.Model, storagePath string, maxSeq int, useGPU bool) (*Context, error) {
	s, err := flash.Open(storagePath)
	if err != nil {
		return nil, fmt.Errorf("inference: open flash: %w", err)
	}

	hp := m.HParams
	kvDim := hp.NumKVHeads * hp.HeadDim()
	kv := kvcache.New(hp.NumLayers, maxSeq, kvDim)

	ctx := &Context{
		Model:   m,
		KV:      kv,
		Storage: s,
	}

	if useGPU {
		dev, err := metal.NewDevice()
		if err != nil {
			return nil, fmt.Errorf("inference: metal init: %w", err)
		}
		ctx.GPU = dev // nil if GPU not available (no error)
	}

	return ctx, nil
}

// Close releases the flash storage and GPU handles.
func (ctx *Context) Close() error {
	if ctx.GPU != nil {
		ctx.GPU.Close()
	}
	return ctx.Storage.Close()
}

// Forward runs one token through the model and returns logits [vocabSize].
// tokenID is the current input token; position is its position in the sequence.
func (ctx *Context) Forward(tokenID int32, position int) ([]float32, error) {
	m := ctx.Model
	hp := m.HParams

	// --- Token embedding lookup ---
	x := embedToken(m.TokenEmbedding, tokenID, hp.HiddenSize)

	// --- Transformer layers ---
	for i, layer := range m.Layers {
		var err error
		x, err = ctx.forwardLayer(layer, x, i, position)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	// --- Final layer norm ---
	var err error
	if m.OutputNormB != nil {
		x, err = tensor.LayerNorm(x, m.OutputNorm, m.OutputNormB, hp.NormEps)
	} else {
		x, err = ctx.gpuRMSNorm(x, m.OutputNorm, hp.NormEps)
	}
	if err != nil {
		return nil, err
	}

	// --- LM head: logits = x @ LMHead^T  (biggest matmul → GPU) ---
	logits, err := ctx.gpuMatMulVec(m.LMHead, x)
	if err != nil {
		return nil, err
	}
	return logits.Data, nil
}

// forwardLayer runs a single transformer layer.
func (ctx *Context) forwardLayer(layer *model.Layer, x *tensor.Tensor, layerIdx, pos int) (*tensor.Tensor, error) {
	m := ctx.Model
	hp := m.HParams

	// Falcon uses a single norm before both attention and FFN (parallel attention).
	// Most other architectures (LLaMA, GPT-NeoX) use separate norms.
	// We detect by checking if FFNNorm == AttnNorm (same pointer for Falcon parallel).
	isFalconStyle := (m.Arch == "falcon")

	// --- Pre-attention norm ---
	normedAttn, err := ctx.applyNormGPU(x, layer.AttnNorm, layer.AttnNormB, hp.NormEps)
	if err != nil {
		return nil, err
	}

	// For Falcon: also use attn_norm output for FFN (parallel architecture).
	var normedFFN *tensor.Tensor
	if isFalconStyle {
		normedFFN = normedAttn
	}

	// --- Self-attention ---
	attnOut, err := ctx.attention(layer, normedAttn, layerIdx, pos)
	if err != nil {
		return nil, fmt.Errorf("attention: %w", err)
	}

	if isFalconStyle {
		// Falcon: x = x + attn_out + ffn_out (all in parallel from same normed input)
		ffnOut, err := ctx.ffn(layer, normedFFN)
		if err != nil {
			return nil, fmt.Errorf("ffn: %w", err)
		}
		if err := tensor.Add(x, attnOut); err != nil {
			return nil, err
		}
		if err := tensor.Add(x, ffnOut); err != nil {
			return nil, err
		}
	} else {
		// Standard: x = x + attn_out
		if err := tensor.Add(x, attnOut); err != nil {
			return nil, err
		}
		// --- Pre-FFN norm ---
		normedFFN, err = ctx.applyNormGPU(x, layer.FFNNorm, layer.FFNNormB, hp.NormEps)
		if err != nil {
			return nil, err
		}
		ffnOut, err := ctx.ffn(layer, normedFFN)
		if err != nil {
			return nil, fmt.Errorf("ffn: %w", err)
		}
		if err := tensor.Add(x, ffnOut); err != nil {
			return nil, err
		}
	}

	return x, nil
}

// attention computes multi-head self-attention for one token.
func (ctx *Context) attention(layer *model.Layer, x *tensor.Tensor, layerIdx, pos int) (*tensor.Tensor, error) {
	m := ctx.Model
	hp := m.HParams
	headDim := hp.HeadDim()
	kvDim := hp.NumKVHeads * headDim

	// --- Q, K, V projections (GPU-accelerated) ---
	q, err := ctx.gpuMatMulVec(layer.Wq, x)
	if err != nil {
		return nil, err
	}
	k, err := ctx.gpuMatMulVec(layer.Wk, x)
	if err != nil {
		return nil, err
	}
	v, err := ctx.gpuMatMulVec(layer.Wv, x)
	if err != nil {
		return nil, err
	}

	if layer.Bq != nil {
		if err := tensor.Add(q, layer.Bq); err != nil {
			return nil, err
		}
	}
	if layer.Bk != nil {
		if err := tensor.Add(k, layer.Bk); err != nil {
			return nil, err
		}
	}
	if layer.Bv != nil {
		if err := tensor.Add(v, layer.Bv); err != nil {
			return nil, err
		}
	}

	// Per-head Q/K RMSNorm (Qwen3 / Qwen3MoE): applied BEFORE RoPE.
	if layer.QNorm != nil {
		applyPerHeadNorm(q.Data, hp.NumHeads, headDim, layer.QNorm.Data, hp.NormEps)
	}
	if layer.KNorm != nil {
		applyPerHeadNorm(k.Data[:kvDim], hp.NumKVHeads, headDim, layer.KNorm.Data, hp.NormEps)
	}

	// --- RoPE (if arch uses it) ---
	if hp.RopeFreqBase > 0 {
		applyRoPE(q.Data, hp.NumHeads, headDim, pos, hp.RopeFreqBase)
		applyRoPE(k.Data[:kvDim], hp.NumKVHeads, headDim, pos, hp.RopeFreqBase)
	}

	// --- Append to KV cache ---
	ctx.KV.AppendK(layerIdx, k.Data[:kvDim])
	ctx.KV.AppendV(layerIdx, v.Data[:kvDim])

	seqLen := ctx.KV.SeqLen + 1 // including current position

	// --- Scaled dot-product attention ---
	// For each query head, compute attention over all past keys.
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	// Attention output has shape [NumHeads × headDim], not HiddenSize.
	// In standard MHA they're equal, but in Qwen3MoE: 32×128=4096 ≠ hidden=2048.
	attnOutDim := hp.NumHeads * headDim
	out := tensor.New(attnOutDim)

	allK := ctx.KV.KeysUpTo(layerIdx) // [seqLen, kvDim]
	allV := ctx.KV.ValsUpTo(layerIdx)
	// Extend to include current position.
	allKData := append(allK.Data, k.Data[:kvDim]...)
	allVData := append(allV.Data, v.Data[:kvDim]...)

	for h := 0; h < hp.NumHeads; h++ {
		qHead := q.Data[h*headDim : (h+1)*headDim]
		// GQA: map query head to kv head.
		kvHeadIdx := h * hp.NumKVHeads / hp.NumHeads

		// Compute attention scores for this head over all positions [0..seqLen].
		scores := make([]float32, seqLen)
		for t := 0; t < seqLen; t++ {
			kHead := allKData[t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kHead[d]
			}
			scores[t] = dot * scale
		}

		// Softmax over scores.
		softmaxInPlace(scores)

		// Weighted sum of values.
		outHead := out.Data[h*headDim : (h+1)*headDim]
		for t := 0; t < seqLen; t++ {
			vHead := allVData[t*kvDim+kvHeadIdx*headDim : t*kvDim+(kvHeadIdx+1)*headDim]
			for d := 0; d < headDim; d++ {
				outHead[d] += scores[t] * vHead[d]
			}
		}
	}

	// --- Output projection (GPU-accelerated) ---
	result, err := ctx.gpuMatMulVec(layer.Wo, out)
	if err != nil {
		return nil, err
	}
	if layer.Bo != nil {
		if err := tensor.Add(result, layer.Bo); err != nil {
			return nil, err
		}
	}
	return result, nil
}

// ffn computes the feed-forward network for one token.
// MoE layers are dispatched to moeFFN; dense layers load from flash on demand.
func (ctx *Context) ffn(layer *model.Layer, x *tensor.Tensor) (*tensor.Tensor, error) {
	if layer.MoERouter != nil {
		return moeFFN(ctx.Storage, 32, layer, x, ctx.Model.HParams)
	}

	// Dense FFN: load weights from flash if not already in DRAM.
	if !layer.FFNLoaded() {
		if err := ctx.loadFFNWeights(layer); err != nil {
			return nil, err
		}
	}

	m := ctx.Model
	switch m.Activation {
	case model.ActivationSiLU:
		return ffnSwiGLU(layer, x)
	case model.ActivationGELU:
		return ffnGELU(layer, x)
	default: // ReLU
		return ffnReLU(layer, x)
	}
}

// ffnReLU: out = W_down @ relu(W_up @ x) — used by OPT/Falcon-ReLU.
func ffnReLU(layer *model.Layer, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := tensor.MatMulVec(layer.FFNUp, x)
	if err != nil {
		return nil, err
	}
	tensor.ReLU(h)
	out, err := tensor.MatMulVec(layer.FFNDown, h)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ffnGELU: out = W_down @ gelu(W_up @ x) — used by Falcon.
func ffnGELU(layer *model.Layer, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := tensor.MatMulVec(layer.FFNUp, x)
	if err != nil {
		return nil, err
	}
	tensor.GELU(h)
	out, err := tensor.MatMulVec(layer.FFNDown, h)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ffnSwiGLU: out = W_down @ (silu(W_gate @ x) * (W_up @ x)) — used by LLaMA.
func ffnSwiGLU(layer *model.Layer, x *tensor.Tensor) (*tensor.Tensor, error) {
	gate, err := tensor.MatMulVec(layer.FFNGate, x)
	if err != nil {
		return nil, err
	}
	tensor.SiLU(gate)

	up, err := tensor.MatMulVec(layer.FFNUp, x)
	if err != nil {
		return nil, err
	}
	if err := tensor.Mul(gate, up); err != nil {
		return nil, err
	}
	out, err := tensor.MatMulVec(layer.FFNDown, gate)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// --- GPU-accelerated helpers ---

// gpuMatMulVec dispatches W @ x to the GPU if a Metal device is present;
// otherwise falls back to the CPU tensor implementation.
func (ctx *Context) gpuMatMulVec(w, x *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.GPU != nil && len(w.Shape) == 2 {
		rows, cols := w.Shape[0], w.Shape[1]
		y, err := ctx.GPU.MatMulVec(w.Data, rows, cols, x.Data)
		if err != nil {
			return nil, err
		}
		return &tensor.Tensor{Shape: []int{rows}, Data: y}, nil
	}
	return tensor.MatMulVec(w, x)
}

// gpuRMSNorm dispatches RMSNorm to the GPU if available; falls back to CPU.
// Returns a 1D tensor of shape [n].
func (ctx *Context) gpuRMSNorm(x, weight *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	if ctx.GPU != nil && len(x.Shape) == 1 {
		out, err := ctx.GPU.RMSNorm(x.Data, weight.Data, eps)
		if err != nil {
			return nil, err
		}
		return &tensor.Tensor{Shape: []int{len(out)}, Data: out}, nil
	}
	return tensor.RMSNorm(x, weight, eps)
}

// applyNormGPU is like applyNorm but uses GPU acceleration when available.
func (ctx *Context) applyNormGPU(x, weight, bias *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	if bias != nil {
		// LayerNorm: GPU path not implemented — use CPU (fast enough).
		return tensor.LayerNorm(x, weight, bias, eps)
	}
	return ctx.gpuRMSNorm(x, weight, eps)
}
func (ctx *Context) loadFFNWeights(layer *model.Layer) error {
	var err error
	layer.FFNUp, err = ctx.Storage.ReadTensor(layer.FFNUpInfo)
	if err != nil {
		return fmt.Errorf("load ffn_up: %w", err)
	}
	if layer.FFNGateInfo != nil {
		layer.FFNGate, err = ctx.Storage.ReadTensor(layer.FFNGateInfo)
		if err != nil {
			return fmt.Errorf("load ffn_gate: %w", err)
		}
	}
	layer.FFNDown, err = ctx.Storage.ReadTensor(layer.FFNDownInfo)
	if err != nil {
		return fmt.Errorf("load ffn_down: %w", err)
	}
	return nil
}

// --- helpers ---

func embedToken(emb *tensor.Tensor, tokenID int32, hiddenSize int) *tensor.Tensor {
	row := int(tokenID)
	data := make([]float32, hiddenSize)
	copy(data, emb.Data[row*hiddenSize:(row+1)*hiddenSize])
	return &tensor.Tensor{Shape: []int{hiddenSize}, Data: data}
}

func applyNorm(x, weight, bias *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	if bias != nil {
		return tensor.LayerNorm(x, weight, bias, eps)
	}
	return tensor.RMSNorm(x, weight, eps)
}

func softmaxInPlace(scores []float32) {
	maxV := scores[0]
	for _, v := range scores {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range scores {
		e := float32(math.Exp(float64(v - maxV)))
		scores[i] = e
		sum += e
	}
	for i := range scores {
		scores[i] /= sum
	}
}

// applyRoPE applies Rotary Position Embeddings in-place.
// vec has shape [nHeads * headDim], rotated per-head.
func applyRoPE(vec []float32, nHeads, headDim, pos int, freqBase float32) {
	halfDim := headDim / 2
	for h := 0; h < nHeads; h++ {
		head := vec[h*headDim : (h+1)*headDim]
		for i := 0; i < halfDim; i++ {
			freq := float64(pos) / math.Pow(float64(freqBase), float64(2*i)/float64(headDim))
			cos := float32(math.Cos(freq))
			sin := float32(math.Sin(freq))
			x0 := head[i]
			x1 := head[i+halfDim]
			head[i] = x0*cos - x1*sin
			head[i+halfDim] = x0*sin + x1*cos
		}
	}
}

// applyPerHeadNorm applies RMSNorm independently to each head slice in vec.
// vec has shape [nHeads * headDim]; weight is shared across all heads [headDim].
func applyPerHeadNorm(vec []float32, nHeads, headDim int, weight []float32, eps float32) {
	for h := 0; h < nHeads; h++ {
		head := vec[h*headDim : (h+1)*headDim]
		var ss float32
		for _, v := range head {
			ss += v * v
		}
		scale := float32(1.0 / math.Sqrt(float64(ss/float32(headDim))+float64(eps)))
		for i, v := range head {
			head[i] = v * scale * weight[i]
		}
	}
}
