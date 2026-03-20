package main

import (
	"fmt"
	"os"

	"llama.go/cmd"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "generate", "gen":
		err = cmd.Generate(os.Args[2:])
	case "bench":
		err = cmd.Bench(os.Args[2:])
	case "info":
		err = cmd.Info(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`llama.go — LLM inference from flash memory

Usage:
  llama.go generate -model <path.gguf> -prompt <text> [flags]
  llama.go info     -model <path.gguf>

Commands:
  generate   Generate text from a prompt
  info       Print model info and DRAM usage

Generate flags:
  -model    path to GGUF model file (required)
  -prompt   input prompt text (required)
  -n        number of tokens to generate (default 128)
  -temp     sampling temperature (default 0.8, 0 = greedy)
  -seed     random seed (default 42)`)
}
