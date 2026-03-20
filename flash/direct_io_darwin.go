//go:build darwin

package flash

import (
	"os"
	"syscall"
	"unsafe"
)

// openDirect opens a file with F_NOCACHE on macOS to bypass the OS page cache.
func openDirect(path string) (*os.File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	// F_NOCACHE = 48 on macOS: disables caching of data for the file.
	if _, _, errno := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), 48, 1); errno != 0 {
		// Non-fatal: proceed without direct I/O.
		_ = errno
	}
	return f, nil
}

// dropCache sets F_NOCACHE on the file to evict cached pages.
func dropCache(f *os.File) {
	syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), 48, 1)
}

func unsafe_ptr(b []byte) unsafe.Pointer {
	return unsafe.Pointer(&b[0])
}
