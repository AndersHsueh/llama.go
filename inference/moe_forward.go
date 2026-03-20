package inference

// moe_forward.go — Mixture-of-Experts FFN forward pass (Qwen3MoE style).
//
// Memory design: all expert weight buffers are pre-allocated in ExpertScratch
// and reused across layers/experts/tokens. This avoids the 18MB/expert
// temporary allocations that would otherwise exhaust memory during prefill.
//
// Per-expert computation (SwiGLU):
//   gate  = silu( gate_exps[e] @ x )   →  [expertFF]
//   up    =       up_exps[e]   @ x      →  [expertFF]
//   h     = gate * up                   →  [expertFF]
//   out  += w_e * (down_exps[e] @ h)    →  [hidden]
//
// 3-D tensor layouts (GGUF stores dims innermost-first):
//   gate_exps / up_exps  Dims=[hidden, expertFF, nExperts]
//       expert e rows:  [e*expertFF .. (e+1)*expertFF),  each row = hidden floats
//   down_exps            Dims=[expertFF, hidden, nExperts]
//       expert e rows:  [e*hidden .. (e+1)*hidden),      each row = expertFF floats

import (
"fmt"
"math"
"sort"

"llama.go/flash"
"llama.go/model"
"llama.go/tensor"
)

// moeFFN dispatches the MoE FFN for a single token using pre-allocated scratch.
func moeFFN(storage *flash.Storage, scratch *ExpertScratch, layer *model.Layer, x *tensor.Tensor, hp model.HParams) (*tensor.Tensor, error) {
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
err := expertSwiGLUInto(storage, scratch, layer, x, e.idx, expertFF, hidden, out.Data, e.score/wsum)
if err != nil {
return nil, fmt.Errorf("expert %d: %w", e.idx, err)
}
}
return out, nil
}

// expertSwiGLUInto computes expert e's SwiGLU and accumulates w*out into accum.
// All intermediate data is written into scratch buffers — zero heap allocation.
func expertSwiGLUInto(s *flash.Storage, scratch *ExpertScratch,
layer *model.Layer, x *tensor.Tensor,
e, expertFF, hidden int,
accum []float32, w float32) error {

gateUpBuf := scratch.GateUpBuf[:expertFF*hidden]
downBuf := scratch.DownBuf[:hidden*expertFF]
gate := scratch.Gate[:expertFF]
up := scratch.Up[:expertFF]
h := scratch.H[:expertFF]

// --- Read gate_exps rows for expert e (single sequential I/O) ---
var err error
scratch.RawBuf, err = s.ReadContiguousRows(layer.ExpertGateInfo, e*expertFF, expertFF, gateUpBuf, scratch.RawBuf)
if err != nil {
return fmt.Errorf("gate_exps: %w", err)
}

// gate[j] = dot(row_j, x)
for j := 0; j < expertFF; j++ {
row := gateUpBuf[j*hidden : (j+1)*hidden]
var dot float32
for d := 0; d < hidden; d++ {
dot += row[d] * x.Data[d]
}
gate[j] = dot
}

// --- Read up_exps rows for expert e (reuse gateUpBuf) ---
scratch.RawBuf, err = s.ReadContiguousRows(layer.ExpertUpInfo, e*expertFF, expertFF, gateUpBuf, scratch.RawBuf)
if err != nil {
return fmt.Errorf("up_exps: %w", err)
}

// up[j] = dot(row_j, x)
for j := 0; j < expertFF; j++ {
row := gateUpBuf[j*hidden : (j+1)*hidden]
var dot float32
for d := 0; d < hidden; d++ {
dot += row[d] * x.Data[d]
}
up[j] = dot
}

// h = silu(gate) * up
for j := range h {
g := gate[j]
silu := g / (1.0 + float32(math.Exp(-float64(g))))
h[j] = silu * up[j]
}

// --- Read down_exps rows for expert e ---
scratch.RawBuf, err = s.ReadContiguousRows(layer.ExpertDownInfo, e*hidden, hidden, downBuf, scratch.RawBuf)
if err != nil {
return fmt.Errorf("down_exps: %w", err)
}

// accum += w * (down @ h)
for d := 0; d < hidden; d++ {
row := downBuf[d*expertFF : (d+1)*expertFF]
var dot float32
for j := 0; j < expertFF; j++ {
dot += row[j] * h[j]
}
accum[d] += w * dot
}
return nil
}

// makeRange returns a slice [base, base+1, ..., base+n-1].
func makeRange(base, n int) []int {
s := make([]int, n)
for i := range s {
s[i] = base + i
}
return s
}
