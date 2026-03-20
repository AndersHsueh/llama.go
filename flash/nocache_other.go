//go:build !darwin

package flash

import "os"

func setNocache(_ *os.File) error { return nil }
