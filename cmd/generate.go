// Package cmd implements the command-line interface for llama.go.
package cmd

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"llama.go/inference"
	"llama.go/model"
	"llama.go/tokenizer"
)

// Generate runs text generation from a prompt.
func Generate(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	modelPath := fs.String("model", "", "path to GGUF model file (required)")
	prompt := fs.String("prompt", "", "input prompt (required)")
	n := fs.Int("n", 128, "number of tokens to generate")
	temp := fs.Float64("temp", 0.8, "sampling temperature (0 = greedy)")
	seed := fs.Int64("seed", 42, "random seed")
	maxSeq := fs.Int("maxseq", 2048, "maximum sequence length")
	window := fs.Int("window", 5, "sliding window size k (0 = naive, no windowing)")
	nThreads := fs.Int("threads", 32, "flash read threads")
	useGPU := fs.Bool("gpu", true, "use Metal GPU for attention ops (macOS only)")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if *prompt == "" {
		return fmt.Errorf("--prompt is required")
	}

	fmt.Fprintf(os.Stderr, "Loading model: %s\n", *modelPath)
	t0 := time.Now()

	m, err := model.Load(*modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	m.PrintSummary()
	fmt.Fprintf(os.Stderr, "Model loaded in %.2fs\n\n", time.Since(t0).Seconds())

	// Build tokenizer.
	tok, err := tokenizer.FromGGUF(m.GGUFFile)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	// Build inference context with flash optimizations.
	flashCfg := inference.FlashFFNConfig{
		WindowSize: *window,
		NThreads:   *nThreads,
		UseGPU:     *useGPU,
	}
	ctx, err := inference.NewFlashContext(m, *modelPath, *maxSeq, flashCfg)
	if err != nil {
		return fmt.Errorf("inference context: %w", err)
	}
	defer ctx.Close()

	if *window > 0 {
		fmt.Fprintf(os.Stderr, "Flash mode: windowing k=%d, threads=%d\n", *window, *nThreads)
	} else {
		fmt.Fprintf(os.Stderr, "Flash mode: naive (load all FFN per token)\n")
	}
	if *useGPU && ctx.GPU != nil {
		fmt.Fprintf(os.Stderr, "GPU: Metal enabled (attention + norms on GPU)\n")
	} else {
		fmt.Fprintf(os.Stderr, "GPU: CPU-only\n")
	}

	// Tokenize prompt.
	ids := tok.Encode(*prompt, true)
	fmt.Fprintf(os.Stderr, "Prompt: %q (%d tokens)\n\n", *prompt, len(ids))

	rng := rand.New(rand.NewSource(*seed))

	// Prefill: run the prompt tokens through the model.
	var lastLogits []float32
	t1 := time.Now()
	for pos, id := range ids {
		fmt.Fprintf(os.Stderr, "  prefill tok %d/%d...\r", pos+1, len(ids))
		logits, err := ctx.ForwardFlash(id, pos)
		if err != nil {
			return fmt.Errorf("prefill pos %d: %w", pos, err)
		}
		ctx.KV.Commit()
		lastLogits = logits
	}
	fmt.Fprintf(os.Stderr, "\n")
	prefillMs := time.Since(t1).Milliseconds()
	fmt.Fprintf(os.Stderr, "Prefill: %d tokens in %dms (%.1f ms/tok)\n",
		len(ids), prefillMs, float64(prefillMs)/float64(len(ids)))

	// Print prompt echo.
	fmt.Print(*prompt)

	// Generate n new tokens.
	pos := len(ids)
	t2 := time.Now()
	for i := 0; i < *n; i++ {
		nextID := sample(lastLogits, float32(*temp), rng)
		if int32(nextID) == tok.EOSID {
			break
		}
		fmt.Print(tok.TokenString(int32(nextID)))

		logits, err := ctx.ForwardFlash(int32(nextID), pos)
		if err != nil {
			return fmt.Errorf("generate step %d: %w", i, err)
		}
		ctx.KV.Commit()
		lastLogits = logits
		pos++

		if pos >= *maxSeq {
			break
		}
	}
	fmt.Println()

	genTokens := pos - len(ids)
	genMs := time.Since(t2).Milliseconds()
	if genTokens > 0 {
		fmt.Fprintf(os.Stderr, "\nGenerated: %d tokens in %dms (%.1f ms/tok)\n",
			genTokens, genMs, float64(genMs)/float64(genTokens))
	}
	return nil
}

// Info prints model information and DRAM usage.
func Info(args []string) error {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	modelPath := fs.String("model", "", "path to GGUF model file (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}

	m, err := model.Load(*modelPath)
	if err != nil {
		return err
	}
	m.PrintSummary()

	fmt.Printf("\nTensors in GGUF:\n")
	for _, ti := range m.GGUFFile.Tensors {
		nElems := ti.NElements()
		byteSize, _ := ti.Type.ByteSize(nElems)
		fmt.Printf("  %-50s  %s  %v  %.2fMB\n",
			ti.Name, ti.Type, ti.Dimensions, float64(byteSize)/1e6)
	}
	return nil
}

// sample selects the next token ID from logits.
// temp=0 means greedy (argmax); temp>0 uses temperature sampling.
func sample(logits []float32, temp float32, rng *rand.Rand) int {
	if temp == 0 {
		return argmax(logits)
	}
	// Apply temperature and softmax.
	probs := make([]float32, len(logits))
	var maxV float32
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range logits {
		e := float32(expF64(float64((v - maxV) / temp)))
		probs[i] = e
		sum += e
	}
	// Sample.
	r := rng.Float32() * sum
	var cumsum float32
	for i, p := range probs {
		cumsum += p
		if r < cumsum {
			return i
		}
	}
	return len(probs) - 1
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func expF64(x float64) float64 {
	if x < -88 {
		return 0
	}
	return math.Exp(x)
}
