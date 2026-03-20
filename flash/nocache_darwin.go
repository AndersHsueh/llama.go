//go:build darwin

package flash

import (
	"os"
	"syscall"
	"unsafe"
)

// F_NOCACHE disables the OS unified buffer cache for this fd on macOS.
// This prevents the 19 GB model file from filling RAM during prefill.
const fNocache = 48 // F_NOCACHE from <sys/fcntl.h>

func setNocache(f *os.File) error {
	_, _, errno := syscall.Syscall(syscall.SYS_FCNTL,
		f.Fd(),
		uintptr(fNocache),
		uintptr(unsafe.Pointer(nil))+1, // arg=1 → enable
	)
	if errno != 0 {
		return errno
	}
	return nil
}
