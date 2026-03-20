package inference

import (
	"llama.go/flash"
	"llama.go/gguf"
	"llama.go/tensor"
)

// NeuronCache manages a DRAM cache of FFN neuron weight rows for one layer.
//
// Implementation of the "Windowing" technique from the paper:
//   - Maintains a sliding window of the last k tokens' activated neurons.
//   - On each new token, only loads the incremental new neurons not already cached.
//   - Evicts neurons that fall outside the window by swapping them to the tail
//     (O(1) swap, no realloc — matches the paper's Figure 6 memory management).
//
// Layout of the internal matrix (pre-allocated):
//   rows 0..numUsed-1 : currently cached neurons
//   Each row stores: [up_proj_row | down_proj_row]  (row-column bundling, Section 3.2)
//   pointers[i] = original neuron index for row i
type NeuronCache struct {
	// matrix: [maxNeurons, 2*hiddenSize] — bundled up+down weights.
	matrix  []float32
	// pointers[i] = original neuron index for matrix row i.
	pointers []int
	// inCache maps neuron index → matrix row (for O(1) lookup).
	inCache map[int]int
	// lastKActive tracks which neurons were active in the last k tokens.
	// lastKActive[t % windowSize] = set of neuron indices active at step t.
	lastKActive [][]int
	windowHead  int // circular buffer head

	numUsed    int
	maxNeurons int
	hiddenSize int
	windowSize int // k in the paper

	// flash refs for loading missing rows.
	storage  *flash.Storage
	upInfo   *gguf.TensorInfo
	downInfo *gguf.TensorInfo
}

// NewNeuronCache allocates a NeuronCache for one FFN layer.
//   maxNeurons: pre-allocated capacity (set to max expected active neurons across window).
//   hiddenSize: d_model.
//   windowSize: k (how many past tokens to keep in cache).
func NewNeuronCache(maxNeurons, hiddenSize, windowSize int,
	storage *flash.Storage, upInfo, downInfo *gguf.TensorInfo) *NeuronCache {

	return &NeuronCache{
		matrix:      make([]float32, maxNeurons*2*hiddenSize),
		pointers:    make([]int, maxNeurons),
		inCache:     make(map[int]int, maxNeurons),
		lastKActive: make([][]int, windowSize),
		maxNeurons:  maxNeurons,
		hiddenSize:  hiddenSize,
		windowSize:  windowSize,
		storage:     storage,
		upInfo:      upInfo,
		downInfo:    downInfo,
	}
}

// Update integrates a new token's activated neurons into the cache.
//
//  1. Determine which neurons to evict (those activated ONLY in the expiring window slot).
//  2. Evict by swap-to-tail.
//  3. Load newly required neurons from flash (incremental delta).
//  4. Advance the window.
func (c *NeuronCache) Update(activeNeurons []int) error {
	// Step 1: evict neurons from the expiring window slot.
	expiredSlot := c.lastKActive[c.windowHead]
	if expiredSlot != nil {
		// Build set of neurons still needed in remaining window slots.
		stillNeeded := make(map[int]bool)
		for i, slot := range c.lastKActive {
			if i == c.windowHead {
				continue
			}
			for _, n := range slot {
				stillNeeded[n] = true
			}
		}
		// Also keep currently active neurons.
		for _, n := range activeNeurons {
			stillNeeded[n] = true
		}
		// Evict any neuron in expired slot that's not still needed.
		for _, n := range expiredSlot {
			if !stillNeeded[n] {
				c.evict(n)
			}
		}
	}

	// Step 2: collect neurons not yet in cache.
	var toLoad []int
	for _, n := range activeNeurons {
		if _, cached := c.inCache[n]; !cached {
			toLoad = append(toLoad, n)
		}
	}

	// Step 3: load missing neurons from flash (parallel).
	if len(toLoad) > 0 {
		if err := c.loadRows(toLoad); err != nil {
			return err
		}
	}

	// Step 4: record active set and advance window.
	c.lastKActive[c.windowHead] = activeNeurons
	c.windowHead = (c.windowHead + 1) % c.windowSize

	return nil
}

// GetUpRow returns the up-projection weight row for neuron n from the cache.
// Returns nil if n is not cached (shouldn't happen after Update).
func (c *NeuronCache) GetUpRow(n int) []float32 {
	row, ok := c.inCache[n]
	if !ok {
		return nil
	}
	start := row * 2 * c.hiddenSize
	return c.matrix[start : start+c.hiddenSize]
}

// GetDownRow returns the down-projection weight row for neuron n.
func (c *NeuronCache) GetDownRow(n int) []float32 {
	row, ok := c.inCache[n]
	if !ok {
		return nil
	}
	start := row*2*c.hiddenSize + c.hiddenSize
	return c.matrix[start : start+c.hiddenSize]
}

// FFNForward computes the FFN for a single token using only the cached active neurons.
// x is the input [hiddenSize]. activeNeurons are the predicted-active indices.
// Activation function applied to intermediate is passed as actFn.
func (c *NeuronCache) FFNForward(x *tensor.Tensor, activeNeurons []int, actFn func(*tensor.Tensor)) (*tensor.Tensor, error) {
	h := c.hiddenSize

	// Compute intermediate: h_i = up_row_i · x  for each active neuron i.
	intermediate := make([]float32, len(activeNeurons))
	for j, n := range activeNeurons {
		row := c.GetUpRow(n)
		if row == nil {
			continue
		}
		var dot float32
		for d := 0; d < h; d++ {
			dot += row[d] * x.Data[d]
		}
		intermediate[j] = dot
	}

	// Apply activation.
	tmp := &tensor.Tensor{Shape: []int{len(intermediate)}, Data: intermediate}
	actFn(tmp)

	// Compute output: out += h_i * down_row_i  for each active neuron i.
	out := tensor.New(h)
	for j, n := range activeNeurons {
		if intermediate[j] == 0 {
			continue // skip zero activations (true sparsity)
		}
		row := c.GetDownRow(n)
		if row == nil {
			continue
		}
		scale := intermediate[j]
		for d := 0; d < h; d++ {
			out.Data[d] += scale * row[d]
		}
	}
	return out, nil
}

// NumCached returns the number of neurons currently in DRAM.
func (c *NeuronCache) NumCached() int { return c.numUsed }

// --- internal helpers ---

// evict removes neuron n from the cache by swapping it with the last row.
func (c *NeuronCache) evict(n int) {
	row, ok := c.inCache[n]
	if !ok {
		return
	}
	last := c.numUsed - 1
	if row != last {
		// Copy last row into row's position.
		rStart := row * 2 * c.hiddenSize
		lStart := last * 2 * c.hiddenSize
		copy(c.matrix[rStart:rStart+2*c.hiddenSize], c.matrix[lStart:lStart+2*c.hiddenSize])
		c.pointers[row] = c.pointers[last]
		c.inCache[c.pointers[row]] = row
	}
	delete(c.inCache, n)
	c.numUsed--
}

// loadRows fetches the given neuron rows from flash (up + down bundled) into the matrix.
func (c *NeuronCache) loadRows(neurons []int) error {
	h := c.hiddenSize

	// Read up-projection rows from flash.
	upRows, err := c.storage.ReadRowsParallel(c.upInfo, neurons, 32)
	if err != nil {
		return err
	}
	// Read down-projection rows from flash (down is transposed: row i = col i of W_down).
	downRows, err := c.storage.ReadRowsParallel(c.downInfo, neurons, 32)
	if err != nil {
		return err
	}

	for j, n := range neurons {
		if c.numUsed >= c.maxNeurons {
			// Cache full: grow by 10% (shouldn't happen if maxNeurons is well-chosen).
			newCap := c.maxNeurons + c.maxNeurons/10 + 1
			newMatrix := make([]float32, newCap*2*h)
			copy(newMatrix, c.matrix)
			c.matrix = newMatrix
			newPtrs := make([]int, newCap)
			copy(newPtrs, c.pointers)
			c.pointers = newPtrs
			c.maxNeurons = newCap
		}
		row := c.numUsed
		c.pointers[row] = n
		c.inCache[n] = row

		// Copy bundled: [up_row | down_row] into matrix.
		dest := c.matrix[row*2*h : (row+1)*2*h]
		copy(dest[:h], upRows.Data[j*h:(j+1)*h])
		copy(dest[h:], downRows.Data[j*h:(j+1)*h])

		c.numUsed++
	}
	return nil
}
