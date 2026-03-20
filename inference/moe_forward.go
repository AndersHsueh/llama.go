package inference

// moe_forward.go — Mixture-of-Experts FFN forward pass (Qwen3MoE style).
//
// Architecture: 128 total experts per layer, top-8 active per token.
//   router_logits = MoERouter @ x              → [nExperts]   (F32, tiny matmul)
//   weights       = softmax(router_logits)
//   top-k         = k largest → indices + weights (re-normalised)
//   for each expert e in top-k:
//     gate  = silu( gate_exps[e] @ x )         → [expertFF]
//     up    =       up_exps[e]   @ x            → [expertFF]
//     h     = gate * up                         → [expertFF]
//     out  += w_e * (down_exps[e] @ h)          → [hidden]
//
// 3-D tensor layouts (GGUF stores dims innermost-first):
//   gate_exps / up_exps Dims=[hidden, expertFF, nExperts]
//       → treated as flat 2D with cols=hidden, total_rows=expertFF*nExperts
//       → expert e occupies global rows [e*expertFF .. (e+1)*expertFF)
//   down_exps Dims=[expertFF, hidden, nExperts]
//       → cols=expertFF, total_rows=hidden*nExperts
//       → expert e occupies global rows [e*hidden .. (e+1)*hidden)

import (
	"fmt"
	"math"
	"sort"

	"llama.go/flash"
	"llama.go/model"
	"llama.go/tensor"
)

// moeFFN dispatches the MoE FFN for a single token.
func moeFFN(storage *flash.Storage, nThreads int, layer *model.Layer, x *tensor.Tensor, hp model.HParams) (*tensor.Tensor, error) {
	nExperts := hp.NumExperts
	k := hp.NumExpertsPerToken
	expertFF := hp.ExpertFFNSize
	hidden := hp.HiddenSize

	if k <= 0 || nExperts <= 0 || expertFF <= 0 {
		return nil, fmt.Errorf("moeFFN: invalid hparams nExperts=%d k=%d expertFF=%d", nExperts, k, expertFF)
	}

	// --- Router: logits = MoERouter @ x → [nExperts] ---
	logits, err := tensor.MatMulVec(layer.MoERouter, x)
	if err != nil {
		return nil, fmt.Errorf("moe router matmul: %w", err)
	}
	softmaxInPlace(logits.Data)

	// --- Top-k selection ---
	type es struct {
		idx   int
		score float32
	}
	all := make([]es, nExperts)
	for i, s := range logits.Data {
		all[i] = es{i, s}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].score > all[b].score })
	topK := all[:k]

	// Re-normalise so weights sum to 1.
	var wsum float32
	for _, e := range topK {
		wsum += e.score
	}
	if wsum < 1e-9 {
		wsum = 1
	}

	// --- Accumulate expert outputs ---
	out := tensor.New(hidden)
	for _, e := range topK {
		eOut, err := expertSwiGLU(storage, layer, x, e.idx, expertFF, hidden, nThreads)
		if err != nil {
			return nil, fmt.Errorf("expert %d: %w", e.idx, err)
		}
		w := e.score / wsum
		for d := 0; d < hidden; d++ {
			out.Data[d] += w * eOut[d]
		}
	}
	return out, nil
}

// expertSwiGLU loads expert e's rows from flash and computes SwiGLU FFN.
func expertSwiGLU(s *flash.Storage, layer *model.Layer, x *tensor.Tensor, e, expertFF, hidden, nThreads int) ([]float32, error) {
	// gate/up expert rows: [e*expertFF .. (e+1)*expertFF)
	gateIdxs := makeRange(e*expertFF, expertFF)
	// down expert rows:    [e*hidden .. (e+1)*hidden)
	downIdxs := makeRange(e*hidden, hidden)

	gateRows, err := s.ReadFlatRows(layer.ExpertGateInfo, gateIdxs, nThreads)
	if err != nil {
		return nil, fmt.Errorf("gate_exps: %w", err)
	}
	upRows, err := s.ReadFlatRows(layer.ExpertUpInfo, gateIdxs, nThreads)
	if err != nil {
		return nil, fmt.Errorf("up_exps: %w", err)
	}
	downRows, err := s.ReadFlatRows(layer.ExpertDownInfo, downIdxs, nThreads)
	if err != nil {
		return nil, fmt.Errorf("down_exps: %w", err)
	}

	// gate = gateRows @ x → [expertFF]
	gate := make([]float32, expertFF)
	for j := 0; j < expertFF; j++ {
		row := gateRows.Data[j*hidden : (j+1)*hidden]
		var dot float32
		for d := 0; d < hidden; d++ {
			dot += row[d] * x.Data[d]
		}
		gate[j] = dot
	}

	// up = upRows @ x → [expertFF]
	up := make([]float32, expertFF)
	for j := 0; j < expertFF; j++ {
		row := upRows.Data[j*hidden : (j+1)*hidden]
		var dot float32
		for d := 0; d < hidden; d++ {
			dot += row[d] * x.Data[d]
		}
		up[j] = dot
	}

	// h = silu(gate) * up
	h := make([]float32, expertFF)
	for j := range h {
		g := gate[j]
		silu := g / (1.0 + float32(math.Exp(-float64(g))))
		h[j] = silu * up[j]
	}

	// out = downRows @ h → [hidden]
	out := make([]float32, hidden)
	for d := 0; d < hidden; d++ {
		row := downRows.Data[d*expertFF : (d+1)*expertFF]
		var dot float32
		for j := 0; j < expertFF; j++ {
			dot += row[j] * h[j]
		}
		out[d] = dot
	}
	return out, nil
}

// makeRange returns a slice [base, base+1, ..., base+n-1].
func makeRange(base, n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = base + i
	}
	return s
}
