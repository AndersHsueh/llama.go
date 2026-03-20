package inference

import (
	"fmt"

	"llama.go/flash"
	"llama.go/gguf"
	"llama.go/model"
	"llama.go/tensor"
)

// FlashFFNConfig holds the configuration for flash-aware FFN execution.
// Zero value = naive (load all FFN weights on demand, no caching).
type FlashFFNConfig struct {
	// WindowSize k: number of past tokens whose activated neurons to keep in DRAM.
	// 0 = disable windowing (naive loading).
	WindowSize int

	// Predictors: if non-nil, use low-rank predictors to pre-select active neurons
	// before loading from flash.
	Predictors *PredictorSet

	// UseBundles: if true, load from pre-built bundle files (row-col bundling).
	// Requires bundle files to have been created with flash.CreateBundle.
	UseBundles bool

	// BundlePaths[i] = path to bundle file for layer i.
	BundlePaths []string

	// NThreads: parallelism for flash reads (paper uses 32).
	NThreads int

	// UseGPU: if true (and on Darwin), initialise a Metal device to accelerate
	// attention projections and RMSNorm. FFN weights remain on flash.
	UseGPU bool
}

// FlashContext extends Context with the Phase-4 optimizations.
type FlashContext struct {
	*Context

	cfg     FlashFFNConfig
	caches  []*NeuronCache  // one per layer, nil if windowing disabled
	bundles []*flash.BundleFile // nil if bundles not used
}

// NewFlashContext creates a FlashContext with the given optimizations enabled.
func NewFlashContext(m *model.Model, storagePath string, maxSeq int, cfg FlashFFNConfig) (*FlashContext, error) {
	base, err := NewContext(m, storagePath, maxSeq, cfg.UseGPU)
	if err != nil {
		return nil, err
	}

	if cfg.NThreads <= 0 {
		cfg.NThreads = 32
	}

	fc := &FlashContext{Context: base, cfg: cfg}

	// Set up per-layer neuron caches (windowing).
	if cfg.WindowSize > 0 {
		hp := m.HParams
		fc.caches = make([]*NeuronCache, hp.NumLayers)
		for i, layer := range m.Layers {
			if layer.FFNUpInfo == nil || layer.FFNDownInfo == nil {
				continue
			}
			nNeurons := int(layer.FFNUpInfo.Dimensions[1])
			// Pre-allocate for ~15% active neurons × window size (conservative).
			maxCached := nNeurons * cfg.WindowSize / 5
			if maxCached < 256 {
				maxCached = 256
			}
			fc.caches[i] = NewNeuronCache(
				maxCached, hp.HiddenSize, cfg.WindowSize,
				base.Storage, layer.FFNUpInfo, layer.FFNDownInfo,
			)
		}
	}

	// Open bundle files if enabled.
	if cfg.UseBundles && len(cfg.BundlePaths) > 0 {
		fc.bundles = make([]*flash.BundleFile, len(cfg.BundlePaths))
		for i, p := range cfg.BundlePaths {
			if p == "" {
				continue
			}
			b, err := flash.OpenBundle(p)
			if err != nil {
				return nil, fmt.Errorf("open bundle layer %d: %w", i, err)
			}
			fc.bundles[i] = b
		}
	}

	return fc, nil
}

// Close releases all resources.
func (fc *FlashContext) Close() error {
	for _, b := range fc.bundles {
		if b != nil {
			b.Close()
		}
	}
	return fc.Context.Close()
}

// ForwardFlash runs one token through the model using flash-aware FFN loading.
// This replaces the naive Forward in inference/forward.go.
func (fc *FlashContext) ForwardFlash(tokenID int32, position int) ([]float32, error) {
	m := fc.Model
	hp := m.HParams

	x := embedToken(m.TokenEmbedding, tokenID, hp.HiddenSize)

	for i, layer := range m.Layers {
		var err error
		x, err = fc.forwardLayerFlash(layer, x, i, position)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	x, err := applyNorm(x, m.OutputNorm, m.OutputNormB, hp.NormEps)
	if err != nil {
		return nil, err
	}
	logits, err := fc.Context.gpuMatMulVec(m.LMHead, x)
	if err != nil {
		return nil, err
	}
	return logits.Data, nil
}

// forwardLayerFlash runs one transformer layer with flash-aware FFN.
func (fc *FlashContext) forwardLayerFlash(layer *model.Layer, x *tensor.Tensor, layerIdx, pos int) (*tensor.Tensor, error) {
	m := fc.Model
	hp := m.HParams
	isFalcon := m.Arch == "falcon"

	normedAttn, err := fc.Context.applyNormGPU(x, layer.AttnNorm, layer.AttnNormB, hp.NormEps)
	if err != nil {
		return nil, err
	}

	attnOut, err := fc.Context.attention(layer, normedAttn, layerIdx, pos)
	if err != nil {
		return nil, err
	}

	// The attention output is used as input to the predictor.
	if isFalcon {
		ffnOut, err := fc.ffnFlash(layer, normedAttn, attnOut, layerIdx)
		if err != nil {
			return nil, err
		}
		if err := tensor.Add(x, attnOut); err != nil {
			return nil, err
		}
		if err := tensor.Add(x, ffnOut); err != nil {
			return nil, err
		}
	} else {
		if err := tensor.Add(x, attnOut); err != nil {
			return nil, err
		}
		normedFFN, err := fc.Context.applyNormGPU(x, layer.FFNNorm, layer.FFNNormB, hp.NormEps)
		if err != nil {
			return nil, err
		}
		ffnOut, err := fc.ffnFlash(layer, normedFFN, attnOut, layerIdx)
		if err != nil {
			return nil, err
		}
		if err := tensor.Add(x, ffnOut); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// ffnFlash computes FFN with the following priority:
//  1. If windowing cache exists for this layer → use NeuronCache (windowing + bundling).
//  2. Else if predictor exists → predict active neurons, load only those rows.
//  3. Else → fall back to loading all FFN weights (naive).
//
// attnOut is passed as input to the activation predictor (paper improvement:
// use current layer's attn output, not previous layer's FFN output).
func (fc *FlashContext) ffnFlash(layer *model.Layer, x, attnOut *tensor.Tensor, layerIdx int) (*tensor.Tensor, error) {
	m := fc.Model

	actFn := activationFn(m.Activation)

	// --- Path 1: windowing with DRAM cache ---
	if fc.cfg.WindowSize > 0 && layerIdx < len(fc.caches) && fc.caches[layerIdx] != nil {
		cache := fc.caches[layerIdx]

		// Predict active neurons for this token.
		activeNeurons, err := fc.predictActive(attnOut, layerIdx, layer.FFNUpInfo)
		if err != nil {
			return nil, err
		}

		// Update cache: evict expired, load new neurons.
		if err := cache.Update(activeNeurons); err != nil {
			return nil, err
		}

		// Compute FFN using cached rows (no flash I/O for cached neurons).
		return cache.FFNForward(x, activeNeurons, actFn)
	}

	// --- Path 2: predictor only (no windowing, sparse load per token) ---
	if fc.cfg.Predictors != nil && fc.cfg.Predictors.Get(layerIdx) != nil {
		activeNeurons, err := fc.predictActive(attnOut, layerIdx, layer.FFNUpInfo)
		if err != nil {
			return nil, err
		}
		return fc.ffnSparseLoad(layer, x, activeNeurons, actFn)
	}

	// --- Path 3: naive (load all FFN weights) ---
	return fc.Context.ffn(layer, x)
}

// predictActive runs the predictor for layerIdx to get active neuron indices.
// Falls back to "all neurons active" if predictor is nil.
func (fc *FlashContext) predictActive(attnOut *tensor.Tensor, layerIdx int, upInfo *gguf.TensorInfo) ([]int, error) {
	pred := fc.cfg.Predictors.Get(layerIdx)
	if pred == nil {
		// No predictor: return all neurons.
		nNeurons := int(upInfo.Dimensions[1])
		all := make([]int, nNeurons)
		for i := range all {
			all[i] = i
		}
		return all, nil
	}
	return pred.Predict(attnOut)
}

// ffnSparseLoad loads only activeNeurons rows from flash and computes FFN.
func (fc *FlashContext) ffnSparseLoad(layer *model.Layer, x *tensor.Tensor, activeNeurons []int, actFn func(*tensor.Tensor)) (*tensor.Tensor, error) {
	if len(activeNeurons) == 0 {
		return tensor.New(x.Shape[0]), nil
	}

	h := x.Shape[0]

	// Load up-projection rows for active neurons.
	upRows, err := fc.Storage.ReadRowsParallel(layer.FFNUpInfo, activeNeurons, fc.cfg.NThreads)
	if err != nil {
		return nil, fmt.Errorf("sparse load up: %w", err)
	}
	// Load down-projection rows for active neurons.
	downRows, err := fc.Storage.ReadRowsParallel(layer.FFNDownInfo, activeNeurons, fc.cfg.NThreads)
	if err != nil {
		return nil, fmt.Errorf("sparse load down: %w", err)
	}

	// Compute intermediate = up_rows @ x.
	intermediate := make([]float32, len(activeNeurons))
	for j := range activeNeurons {
		row := upRows.Data[j*h : (j+1)*h]
		var dot float32
		for d := 0; d < h; d++ {
			dot += row[d] * x.Data[d]
		}
		intermediate[j] = dot
	}

	// Apply activation.
	tmp := &tensor.Tensor{Shape: []int{len(intermediate)}, Data: intermediate}
	actFn(tmp)

	// Compute output = sum_j intermediate[j] * down_row[j].
	out := tensor.New(h)
	for j := range activeNeurons {
		if intermediate[j] == 0 {
			continue
		}
		row := downRows.Data[j*h : (j+1)*h]
		scale := intermediate[j]
		for d := 0; d < h; d++ {
			out.Data[d] += scale * row[d]
		}
	}
	return out, nil
}

// activationFn returns an in-place activation function for the given type.
func activationFn(act model.ActivationType) func(*tensor.Tensor) {
	switch act {
	case model.ActivationSiLU:
		return tensor.SiLU
	case model.ActivationGELU:
		return tensor.GELU
	default:
		return tensor.ReLU
	}
}
