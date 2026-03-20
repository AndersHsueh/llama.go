package main

// Standalone GGUF metadata dumper — run with:
//   go run tools/gguf_dump/main.go <path.gguf>

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"llama.go/gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: gguf_dump <path.gguf>")
		os.Exit(1)
	}
	path := os.Args[1]

	f, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}

	fmt.Printf("=== GGUF File: %s ===\n", path)
	fmt.Printf("Version: %d  |  Tensors: %d  |  Metadata keys: %d\n\n",
		f.Header.Version, f.Header.TensorCount, f.Header.MetaKVCount)

	// Sort metadata keys for readability.
	keys := make([]string, 0, len(f.Meta))
	for k := range f.Meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	fmt.Println("--- Metadata ---")
	for _, k := range keys {
		v := f.Meta[k]
		val := fmt.Sprintf("%v", v.Value)
		if len(val) > 120 {
			val = val[:117] + "..."
		}
		fmt.Printf("  %-55s = %s\n", k, val)
	}

	// Show tensor name structure (grouped by prefix).
	fmt.Printf("\n--- Tensor structure (unique prefixes, up to 50) ---\n")
	var names []string
	nameToTensor := map[string]*gguf.TensorInfo{}
	for _, ti := range f.Tensors {
		names = append(names, ti.Name)
		nameToTensor[ti.Name] = ti
	}
	sort.Strings(names)

	count := 0
	seen := map[string]bool{}
	for _, name := range names {
		parts := strings.Split(name, ".")
		var prefix string
		if len(parts) >= 3 {
			prefix = strings.Join(parts[:3], ".")
		} else {
			prefix = name
		}
		if seen[prefix] {
			continue
		}
		seen[prefix] = true
		ti := nameToTensor[name]
		fmt.Printf("  %-60s  %-8s  %v\n", name, ti.Type, ti.Dimensions)
		count++
		if count >= 50 {
			fmt.Printf("  ... (%d tensors total)\n", len(f.Tensors))
			break
		}
	}
}
