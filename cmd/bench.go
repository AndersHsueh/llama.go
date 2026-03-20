package cmd

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"llama.go/flash"
	"llama.go/gguf"
	"llama.go/model"
)

// Bench runs I/O and inference latency benchmarks, matching the paper's Table 2/3.
func Bench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	modelPath := fs.String("model", "", "path to GGUF model file (required)")
	nThreads := fs.Int("threads", 32, "parallel read threads")
	sparsity := fs.Float64("sparsity", 0.95, "simulated FFN sparsity (0.0–1.0)")
	nTokens := fs.Int("tokens", 50, "tokens to benchmark")
	seed := fs.Int64("seed", 42, "random seed")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}

	fmt.Printf("=== llama.go flash benchmark ===\n")
	fmt.Printf("Model: %s\n", *modelPath)
	fmt.Printf("Threads: %d  Sparsity: %.0f%%  Tokens: %d\n\n",
		*nThreads, *sparsity*100, *nTokens)

	// --- 1. Model info ---
	m, err := model.Load(*modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	m.PrintSummary()
	fmt.Println()

	s, err := flash.Open(*modelPath)
	if err != nil {
		return err
	}
	defer s.Close()

	f := m.GGUFFile

	// --- 2. Sequential throughput (best case) ---
	fmt.Printf("--- Sequential read throughput ---\n")
	if len(m.Layers) > 0 {
		layer := m.Layers[0]
		if layer.FFNUpInfo != nil {
			br, err := s.BenchSequential(layer.FFNUpInfo)
			if err != nil {
				return err
			}
			fmt.Printf("  ffn_up layer 0 (sequential): %s\n", br)
		}
	}

	// --- 3. Sparse row reads (simulated activation pattern) ---
	fmt.Printf("\n--- Sparse row reads (%.0f%% sparsity, %d threads) ---\n",
		*sparsity*100, *nThreads)

	rng := rand.New(rand.NewSource(*seed))
	var totalBytes int64
	var totalElapsed time.Duration
	var totalReads int

	for layerIdx, layer := range m.Layers {
		if layer.FFNUpInfo == nil {
			continue
		}
		ti := layer.FFNUpInfo
		nRows := int(ti.Dimensions[1])
		// Simulate: pick (1-sparsity) fraction of rows (activated neurons).
		nActive := int(float64(nRows) * (1.0 - *sparsity))
		if nActive < 1 {
			nActive = 1
		}
		// Random subset of rows.
		perm := rng.Perm(nRows)
		activeRows := perm[:nActive]

		_, br, err := s.ReadRowsBench(ti, activeRows, *nThreads)
		if err != nil {
			return fmt.Errorf("layer %d: %w", layerIdx, err)
		}
		totalBytes += br.BytesRead
		totalElapsed += br.Elapsed
		totalReads += br.NReads

		if layerIdx < 3 || layerIdx == len(m.Layers)-1 {
			fmt.Printf("  layer %2d: %d/%d rows → %s\n", layerIdx, nActive, nRows, br)
		} else if layerIdx == 3 {
			fmt.Printf("  ... (layers 3–%d omitted) ...\n", len(m.Layers)-2)
		}
	}
	fmt.Printf("\nTotal sparse I/O: %.2f MB in %.1fms → %.2f MB/s (%d reads)\n",
		float64(totalBytes)/1e6,
		float64(totalElapsed.Microseconds())/1000.0,
		float64(totalBytes)/1e6/totalElapsed.Seconds(),
		totalReads)

	// --- 4. Naive baseline (load all FFN weights per token) ---
	fmt.Printf("\n--- Naive baseline: load all FFN weights per token ---\n")
	var naiveBytes int64
	for _, layer := range m.Layers {
		if layer.FFNUpInfo != nil {
			nb, _ := layer.FFNUpInfo.Type.ByteSize(layer.FFNUpInfo.NElements())
			naiveBytes += nb
		}
		if layer.FFNDownInfo != nil {
			nb, _ := layer.FFNDownInfo.Type.ByteSize(layer.FFNDownInfo.NElements())
			naiveBytes += nb
		}
		if layer.FFNGateInfo != nil {
			nb, _ := layer.FFNGateInfo.Type.ByteSize(layer.FFNGateInfo.NElements())
			naiveBytes += nb
		}
	}
	seqBR, err := benchAllFFN(s, f, m)
	if err != nil {
		return err
	}
	fmt.Printf("  All FFN weights (one token): %.2f MB in %.1fms → %.2f MB/s\n",
		float64(naiveBytes)/1e6,
		float64(seqBR.Microseconds())/1000.0,
		float64(naiveBytes)/1e6/seqBR.Seconds())

	// --- 5. Summary table (matching paper Table 2) ---
	fmt.Printf("\n=== Summary (per-token I/O latency estimate) ===\n")
	fmt.Printf("%-30s  %10s  %10s\n", "Configuration", "Flash→DRAM", "I/O Latency")
	fmt.Printf("%-30s  %10.2f  %10.1fms\n", "Naive (all FFN)",
		float64(naiveBytes)/1e6, float64(seqBR.Microseconds())/1000.0)
	fmt.Printf("%-30s  %10.2f  %10.1fms\n",
		fmt.Sprintf("Sparse (%.0f%% sparsity)", *sparsity*100),
		float64(totalBytes)/1e6,
		float64(totalElapsed.Microseconds())/1000.0)
	fmt.Printf("Speedup: %.1fx\n", seqBR.Seconds()/totalElapsed.Seconds())

	_ = *nTokens
	return nil
}

// benchAllFFN measures the time to sequentially read all FFN weights (naive baseline).
func benchAllFFN(s *flash.Storage, f *gguf.File, m *model.Model) (time.Duration, error) {
	t0 := time.Now()
	for _, layer := range m.Layers {
		for _, ti := range []*gguf.TensorInfo{layer.FFNUpInfo, layer.FFNDownInfo, layer.FFNGateInfo} {
			if ti == nil {
				continue
			}
			nElems := ti.NElements()
			byteSize, err := ti.Type.ByteSize(nElems)
			if err != nil {
				continue
			}
			if _, err := s.ReadRaw(int64(ti.Offset), byteSize); err != nil {
				return 0, err
			}
		}
	}
	return time.Since(t0), nil
}
