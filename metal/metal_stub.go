//go:build !darwin

// Package metal provides stubs for non-Darwin platforms.
// On macOS/iOS, the real implementation in metal.go is used via CGo.
package metal

import "fmt"

// Device is a no-op stub on non-Darwin platforms.
type Device struct{}

// Available returns false — Metal is only available on Apple platforms.
func Available() bool { return false }

// NewDevice returns nil, nil — no GPU available on this platform.
func NewDevice() (*Device, error) { return nil, nil }

// Close is a no-op.
func (d *Device) Close() {}

// MatMulVec is not available on non-Darwin platforms.
func (d *Device) MatMulVec(w []float32, rows, cols int, x []float32) ([]float32, error) {
	return nil, fmt.Errorf("metal: not available on this platform")
}

// RMSNorm is not available on non-Darwin platforms.
func (d *Device) RMSNorm(x, weight []float32, eps float32) ([]float32, error) {
	return nil, fmt.Errorf("metal: not available on this platform")
}
