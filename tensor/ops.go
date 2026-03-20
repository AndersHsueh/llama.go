package tensor

import (
	"fmt"
	"math"
)

// MatMul computes C = A @ B^T where:
//   A is [m, k], B is [n, k] (B is stored transposed, so B[i] is row i).
//   Result C is [m, n].
//
// This is the standard "output = input @ weight^T" pattern used in linear layers.
func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("MatMul: need 2D tensors, got %v and %v", a.Shape, b.Shape)
	}
	m, k := a.Shape[0], a.Shape[1]
	n, bk := b.Shape[0], b.Shape[1]
	if k != bk {
		return nil, fmt.Errorf("MatMul: inner dimension mismatch: A[%d,%d] B[%d,%d]", m, k, n, bk)
	}
	out := New(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			aRow := a.Data[i*k : (i+1)*k]
			bRow := b.Data[j*k : (j+1)*k]
			for l := 0; l < k; l++ {
				sum += aRow[l] * bRow[l]
			}
			out.Data[i*n+j] = sum
		}
	}
	return out, nil
}

// MatMulVec computes y = W^T @ x where W is [outDim, inDim] and x is [inDim].
// Returns y of shape [outDim]. This is the hot path for single-token inference.
func MatMulVec(w, x *Tensor) (*Tensor, error) {
	if len(w.Shape) != 2 {
		return nil, fmt.Errorf("MatMulVec: weight must be 2D, got %v", w.Shape)
	}
	if len(x.Shape) != 1 {
		return nil, fmt.Errorf("MatMulVec: input must be 1D, got %v", x.Shape)
	}
	outDim, inDim := w.Shape[0], w.Shape[1]
	if inDim != x.Shape[0] {
		return nil, fmt.Errorf("MatMulVec: dim mismatch W[%d,%d] x[%d]", outDim, inDim, x.Shape[0])
	}
	out := New(outDim)
	for i := 0; i < outDim; i++ {
		var sum float32
		row := w.Data[i*inDim : (i+1)*inDim]
		for j, v := range x.Data {
			sum += row[j] * v
		}
		out.Data[i] = sum
	}
	return out, nil
}

// MatMulVecSparse is like MatMulVec but only computes outputs for rows in activeRows.
// Used in flash-aware FFN: only load and compute the activated neurons.
// Returns a dense output of shape [outDim] where non-active positions are zero.
func MatMulVecSparse(w *Tensor, x *Tensor, activeRows []int, outDim int) (*Tensor, error) {
	if len(x.Shape) != 1 {
		return nil, fmt.Errorf("MatMulVecSparse: input must be 1D")
	}
	inDim := x.Shape[0]
	out := New(outDim)
	for _, i := range activeRows {
		if i >= outDim {
			return nil, fmt.Errorf("MatMulVecSparse: row %d out of range %d", i, outDim)
		}
		var sum float32
		rowStart := i * inDim
		row := w.Data[rowStart : rowStart+inDim]
		for j, v := range x.Data {
			sum += row[j] * v
		}
		out.Data[i] = sum
	}
	return out, nil
}

// Add computes element-wise a + b (in-place into a).
func Add(a, b *Tensor) error {
	if len(a.Data) != len(b.Data) {
		return fmt.Errorf("Add: shape mismatch %v vs %v", a.Shape, b.Shape)
	}
	for i := range a.Data {
		a.Data[i] += b.Data[i]
	}
	return nil
}

// AddBias adds a bias vector b (1D, length == cols) to each row of a (2D).
func AddBias(a, b *Tensor) error {
	if len(a.Shape) < 1 || len(b.Shape) != 1 {
		return fmt.Errorf("AddBias: need 2D input and 1D bias")
	}
	cols := b.Shape[0]
	if a.Shape[len(a.Shape)-1] != cols {
		return fmt.Errorf("AddBias: bias length %d != last dim %d", cols, a.Shape[len(a.Shape)-1])
	}
	for i := 0; i < len(a.Data); i++ {
		a.Data[i] += b.Data[i%cols]
	}
	return nil
}

// RMSNorm applies Root Mean Square normalization along the last axis.
// eps is the epsilon for numerical stability (typically 1e-5 or 1e-6).
func RMSNorm(x, weight *Tensor, eps float32) (*Tensor, error) {
	n := x.Shape[len(x.Shape)-1]
	if weight.Shape[0] != n {
		return nil, fmt.Errorf("RMSNorm: weight dim %d != input last dim %d", weight.Shape[0], n)
	}
	nRows := len(x.Data) / n
	out := New(x.Shape...)
	for row := 0; row < nRows; row++ {
		xRow := x.Data[row*n : (row+1)*n]
		oRow := out.Data[row*n : (row+1)*n]
		var sumSq float32
		for _, v := range xRow {
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(n)) + float64(eps)))
		scale := 1.0 / rms
		for i, v := range xRow {
			oRow[i] = v * scale * weight.Data[i]
		}
	}
	return out, nil
}

// LayerNorm applies Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
func LayerNorm(x, weight, bias *Tensor, eps float32) (*Tensor, error) {
	n := x.Shape[len(x.Shape)-1]
	if weight.Shape[0] != n || bias.Shape[0] != n {
		return nil, fmt.Errorf("LayerNorm: weight/bias dim mismatch")
	}
	nRows := len(x.Data) / n
	out := New(x.Shape...)
	for row := 0; row < nRows; row++ {
		xRow := x.Data[row*n : (row+1)*n]
		oRow := out.Data[row*n : (row+1)*n]
		var mean float32
		for _, v := range xRow {
			mean += v
		}
		mean /= float32(n)
		var variance float32
		for _, v := range xRow {
			d := v - mean
			variance += d * d
		}
		variance /= float32(n)
		std := float32(math.Sqrt(float64(variance) + float64(eps)))
		for i, v := range xRow {
			oRow[i] = (v-mean)/std*weight.Data[i] + bias.Data[i]
		}
	}
	return out, nil
}

// ReLU applies the ReLU activation in-place: max(0, x).
func ReLU(x *Tensor) {
	for i, v := range x.Data {
		if v < 0 {
			x.Data[i] = 0
		}
	}
}

// SiLU applies the SiLU (Swish) activation in-place: x * sigmoid(x).
func SiLU(x *Tensor) {
	for i, v := range x.Data {
		x.Data[i] = v / (1 + float32(math.Exp(float64(-v))))
	}
}

// GELU applies the GELU activation in-place (approximation used by OPT/Falcon).
func GELU(x *Tensor) {
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i, v := range x.Data {
		x.Data[i] = 0.5 * v * (1 + float32(math.Tanh(float64(c*(v+0.044715*v*v*v)))))
	}
}

// Softmax applies softmax in-place along the last dimension.
func Softmax(x *Tensor) {
	n := x.Shape[len(x.Shape)-1]
	nRows := len(x.Data) / n
	for row := 0; row < nRows; row++ {
		d := x.Data[row*n : (row+1)*n]
		maxV := d[0]
		for _, v := range d {
			if v > maxV {
				maxV = v
			}
		}
		var sum float32
		for i, v := range d {
			e := float32(math.Exp(float64(v - maxV)))
			d[i] = e
			sum += e
		}
		for i := range d {
			d[i] /= sum
		}
	}
}

// Scale multiplies all elements in-place by a scalar.
func Scale(x *Tensor, s float32) {
	for i := range x.Data {
		x.Data[i] *= s
	}
}

// Mul computes element-wise multiplication in-place: a *= b.
func Mul(a, b *Tensor) error {
	if len(a.Data) != len(b.Data) {
		return fmt.Errorf("Mul: shape mismatch %v vs %v", a.Shape, b.Shape)
	}
	for i := range a.Data {
		a.Data[i] *= b.Data[i]
	}
	return nil
}

// Argmax returns the index of the maximum value in a 1D tensor.
func Argmax(x *Tensor) int {
	maxIdx := 0
	maxVal := x.Data[0]
	for i, v := range x.Data[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}
