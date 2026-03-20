//go:build darwin

// Package metal provides a GPU-accelerated backend for matrix operations via
// Apple Metal. It is used to offload attention projection matmuls and RMSNorm
// to the GPU while FFN weights remain on flash (the LLM-in-Flash design).
package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation
#include "metal_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Device wraps an Apple Metal GPU context.
// Use NewDevice() to create; call Close() when done.
type Device struct {
	ptr C.LlamaMetalDevice
}

// Available returns true if a Metal GPU is present on this system.
func Available() bool {
	return C.llama_metal_available() != 0
}

// NewDevice creates and initialises a Metal device context.
// Returns (nil, nil) if no GPU is available (not an error — caller falls back to CPU).
func NewDevice() (*Device, error) {
	if !Available() {
		return nil, nil
	}
	ptr := C.llama_metal_init()
	if ptr == nil {
		return nil, fmt.Errorf("metal: device init failed")
	}
	return &Device{ptr: ptr}, nil
}

// Close releases the Metal context.
func (d *Device) Close() {
	if d != nil && d.ptr != nil {
		C.llama_metal_free(d.ptr)
		d.ptr = nil
	}
}

// MatMulVec computes y = W @ x where W is stored row-major as [rows, cols].
// Returns a new slice y of length rows.
func (d *Device) MatMulVec(w []float32, rows, cols int, x []float32) ([]float32, error) {
	if len(w) != rows*cols {
		return nil, fmt.Errorf("metal MatMulVec: w length %d != rows*cols %d", len(w), rows*cols)
	}
	if len(x) != cols {
		return nil, fmt.Errorf("metal MatMulVec: x length %d != cols %d", len(x), cols)
	}
	y := make([]float32, rows)
	rc := C.llama_metal_matvec(d.ptr,
		(*C.float)(unsafe.Pointer(&w[0])),
		C.int(rows), C.int(cols),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal MatMulVec: kernel error %d", int(rc))
	}
	return y, nil
}

// RMSNorm computes out[i] = (x[i] / rms(x)) * weight[i].
// Returns a new slice of length len(x).
func (d *Device) RMSNorm(x, weight []float32, eps float32) ([]float32, error) {
	n := len(x)
	if len(weight) != n {
		return nil, fmt.Errorf("metal RMSNorm: weight length %d != x length %d", len(weight), n)
	}
	out := make([]float32, n)
	rc := C.llama_metal_rmsnorm(d.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&weight[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
		C.float(eps),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal RMSNorm: kernel error %d", int(rc))
	}
	return out, nil
}
