// Package tensor provides the core tensor abstraction and basic arithmetic operations
// needed for LLM inference.
package tensor

import "fmt"

// DType represents the element data type of a tensor.
type DType int

const (
	DTypeF32 DType = iota
	DTypeF16
	DTypeI32
)

func (d DType) String() string {
	switch d {
	case DTypeF32:
		return "F32"
	case DTypeF16:
		return "F16"
	case DTypeI32:
		return "I32"
	}
	return fmt.Sprintf("DType(%d)", int(d))
}

// Tensor is a dense n-dimensional array of float32 values.
// Internally all data is stored as []float32 after dequantization.
type Tensor struct {
	Shape []int   // dimensions, e.g. [rows, cols] for a 2D matrix
	Data  []float32
}

// New allocates a zero-filled Tensor with the given shape.
func New(shape ...int) *Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return &Tensor{Shape: shape, Data: make([]float32, n)}
}

// FromSlice creates a 1D Tensor from an existing float32 slice (no copy).
func FromSlice(data []float32) *Tensor {
	return &Tensor{Shape: []int{len(data)}, Data: data}
}

// NElements returns the total number of elements.
func (t *Tensor) NElements() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Rows returns the first dimension (for 2D tensors: number of rows).
func (t *Tensor) Rows() int { return t.Shape[0] }

// Cols returns the second dimension (for 2D tensors: number of columns).
func (t *Tensor) Cols() int {
	if len(t.Shape) < 2 {
		return 1
	}
	return t.Shape[1]
}

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return &Tensor{Shape: shape, Data: data}
}

// View returns a Tensor sharing the same Data but with a different shape.
// Total elements must match.
func (t *Tensor) View(shape ...int) (*Tensor, error) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if n != t.NElements() {
		return nil, fmt.Errorf("tensor.View: shape %v has %d elements, want %d", shape, n, t.NElements())
	}
	return &Tensor{Shape: shape, Data: t.Data}, nil
}

// Row returns a 1D Tensor view of row i in a 2D tensor.
func (t *Tensor) Row(i int) *Tensor {
	cols := t.Cols()
	return &Tensor{
		Shape: []int{cols},
		Data:  t.Data[i*cols : (i+1)*cols],
	}
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=F32)", t.Shape)
}
