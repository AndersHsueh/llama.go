// Package kvcache implements the Key-Value cache for autoregressive LLM decoding.
// During inference, past key/value tensors are cached so they don't need to be
// recomputed at each step.
package kvcache

import "llama.go/tensor"

// LayerCache holds the K and V tensors for a single transformer layer.
// Dimensions: [seqLen, nKVHeads, headDim]
type LayerCache struct {
	K *tensor.Tensor // [maxSeq, nKVHeads * headDim]
	V *tensor.Tensor // [maxSeq, nKVHeads * headDim]
}

// KVCache manages the key-value cache across all layers.
type KVCache struct {
	Layers  []*LayerCache
	SeqLen  int // current number of tokens in the cache
	MaxSeq  int
	KVDim   int // nKVHeads * headDim
}

// New creates a KVCache pre-allocated for maxSeq tokens across nLayers.
// kvDim = nKVHeads * headDim.
func New(nLayers, maxSeq, kvDim int) *KVCache {
	layers := make([]*LayerCache, nLayers)
	for i := range layers {
		layers[i] = &LayerCache{
			K: tensor.New(maxSeq, kvDim),
			V: tensor.New(maxSeq, kvDim),
		}
	}
	return &KVCache{
		Layers: layers,
		SeqLen: 0,
		MaxSeq: maxSeq,
		KVDim:  kvDim,
	}
}

// AppendK stores a key vector for the current position in layer layerIdx.
// key must be of length KVDim.
func (c *KVCache) AppendK(layerIdx int, key []float32) {
	pos := c.SeqLen
	copy(c.Layers[layerIdx].K.Data[pos*c.KVDim:], key)
}

// AppendV stores a value vector for the current position in layer layerIdx.
func (c *KVCache) AppendV(layerIdx int, val []float32) {
	pos := c.SeqLen
	copy(c.Layers[layerIdx].V.Data[pos*c.KVDim:], val)
}

// Commit advances the sequence length by 1 after a token has been appended.
func (c *KVCache) Commit() {
	c.SeqLen++
}

// KeysUpTo returns all key vectors [0..seqLen) for layer layerIdx as a
// [seqLen, kvDim] tensor view (no copy).
func (c *KVCache) KeysUpTo(layerIdx int) *tensor.Tensor {
	n := c.SeqLen
	return &tensor.Tensor{
		Shape: []int{n, c.KVDim},
		Data:  c.Layers[layerIdx].K.Data[:n*c.KVDim],
	}
}

// ValsUpTo returns all value vectors [0..seqLen) for layer layerIdx as a
// [seqLen, kvDim] tensor view (no copy).
func (c *KVCache) ValsUpTo(layerIdx int) *tensor.Tensor {
	n := c.SeqLen
	return &tensor.Tensor{
		Shape: []int{n, c.KVDim},
		Data:  c.Layers[layerIdx].V.Data[:n*c.KVDim],
	}
}

// Reset clears the cache (sets SeqLen to 0). Does not free memory.
func (c *KVCache) Reset() {
	c.SeqLen = 0
}
