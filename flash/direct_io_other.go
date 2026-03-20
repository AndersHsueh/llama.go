//go:build !darwin

package flash

import (
	"os"
	"unsafe"
)

func openDirect(path string) (*os.File, error) {
	return os.Open(path)
}

func dropCache(_ *os.File) {}

func unsafe_ptr(b []byte) unsafe.Pointer {
	return unsafe.Pointer(&b[0])
}
