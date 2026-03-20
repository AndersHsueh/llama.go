// Package tokenizer implements BPE (Byte-Pair Encoding) tokenization
// by reading the vocabulary stored in GGUF metadata.
package tokenizer

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"llama.go/gguf"
)

const (
	tokenBOS = "<s>"
	tokenEOS = "</s>"
	tokenUNK = "<unk>"
)

// Tokenizer holds the BPE vocabulary and merge rules.
type Tokenizer struct {
	vocab  []string         // index → token string
	scores []float32        // index → BPE merge score
	toID   map[string]int32 // token string → id

	BOSID int32
	EOSID int32
	UNKID int32
}

// FromGGUF builds a Tokenizer from an open GGUF file's metadata.
func FromGGUF(f *gguf.File) (*Tokenizer, error) {
	// GGUF stores vocab under the architecture key (e.g., "tokenizer.ggml.tokens")
	tokens, err := f.MetaStringArray("tokenizer.ggml.tokens")
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	// Scores are stored as an array of float32.
	scores, err := readScores(f, len(tokens))
	if err != nil {
		// Non-fatal: use zero scores.
		scores = make([]float32, len(tokens))
	}

	toID := make(map[string]int32, len(tokens))
	for i, tok := range tokens {
		toID[tok] = int32(i)
	}

	t := &Tokenizer{
		vocab:  tokens,
		scores: scores,
		toID:   toID,
		BOSID:  -1,
		EOSID:  -1,
		UNKID:  -1,
	}

	if id, ok := toID[tokenBOS]; ok {
		t.BOSID = id
	}
	if id, ok := toID[tokenEOS]; ok {
		t.EOSID = id
	}
	if id, ok := toID[tokenUNK]; ok {
		t.UNKID = id
	}

	// Some models use numeric IDs from metadata.
	if bosID, err := f.MetaUint32("tokenizer.ggml.bos_token_id"); err == nil {
		t.BOSID = int32(bosID)
	}
	if eosID, err := f.MetaUint32("tokenizer.ggml.eos_token_id"); err == nil {
		t.EOSID = int32(eosID)
	}

	return t, nil
}

// Encode tokenizes text using BPE and returns a slice of token IDs.
// addBOS prepends the BOS token if true.
func (t *Tokenizer) Encode(text string, addBOS bool) []int32 {
	var ids []int32
	if addBOS && t.BOSID >= 0 {
		ids = append(ids, t.BOSID)
	}

	// Byte-fallback: encode each unicode character, then merge using BPE scores.
	// This matches the sentencepiece BPE encoding used by most LLaMA-family models.
	symbols := t.textToSymbols(text)
	ids = append(ids, t.bpeMerge(symbols)...)
	return ids
}

// Decode converts token IDs back to a string.
func (t *Tokenizer) Decode(ids []int32) string {
	var sb strings.Builder
	for _, id := range ids {
		if int(id) < len(t.vocab) {
			tok := t.vocab[id]
			// Undo sentencepiece space encoding (▁ → space).
			tok = strings.ReplaceAll(tok, "▁", " ")
			sb.WriteString(tok)
		}
	}
	return sb.String()
}

// TokenString returns the string form of a token ID.
func (t *Tokenizer) TokenString(id int32) string {
	if int(id) >= len(t.vocab) {
		return fmt.Sprintf("<id:%d>", id)
	}
	s := t.vocab[id]
	s = strings.ReplaceAll(s, "▁", " ")
	return s
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// --- BPE implementation ---

type symbol struct {
	text string
	id   int32 // pre-computed id if it exists as a whole token
}

func (t *Tokenizer) textToSymbols(text string) []symbol {
	// Prepend space (sentencepiece style: first word gets a space prefix).
	text = " " + text
	var syms []symbol
	for _, r := range text {
		ch := string(r)
		// Map space to ▁
		if ch == " " {
			ch = "▁"
		}
		id, ok := t.toID[ch]
		if !ok {
			// Byte fallback: encode the UTF-8 bytes individually.
			buf := make([]byte, utf8.RuneLen(r))
			utf8.EncodeRune(buf, r)
			for _, b := range buf {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				bid, ok2 := t.toID[byteToken]
				if !ok2 {
					bid = t.UNKID
				}
				syms = append(syms, symbol{text: byteToken, id: bid})
			}
			continue
		}
		syms = append(syms, symbol{text: ch, id: id})
	}
	return syms
}

type mergePair struct {
	left, right int // indices into syms
	score       float32
	merged      string
}

func (t *Tokenizer) bpeMerge(syms []symbol) []int32 {
	if len(syms) == 0 {
		return nil
	}

	// Iteratively merge the pair with the highest score until no merges are possible.
	for {
		best := mergePair{score: -math.MaxFloat32, left: -1}
		for i := 0; i < len(syms)-1; i++ {
			merged := syms[i].text + syms[i+1].text
			if id, ok := t.toID[merged]; ok {
				score := t.scores[id]
				if score > best.score {
					best = mergePair{i, i + 1, score, merged}
					best.right = int(id) // reuse right as merged id
				}
			}
		}
		if best.left < 0 {
			break // no more merges
		}
		// Merge best.left and best.left+1
		mergedID := best.right
		syms[best.left] = symbol{text: best.merged, id: int32(mergedID)}
		syms = append(syms[:best.left+1], syms[best.left+2:]...)
	}

	ids := make([]int32, len(syms))
	for i, s := range syms {
		ids[i] = s.id
	}
	return ids
}

// readScores extracts float32 scores from GGUF metadata.
func readScores(f *gguf.File, n int) ([]float32, error) {
	v, ok := f.Meta["tokenizer.ggml.scores"]
	if !ok {
		return nil, fmt.Errorf("no scores")
	}
	arr, ok := v.Value.(gguf.ArrayValue)
	if !ok {
		return nil, fmt.Errorf("scores not array")
	}
	scores := make([]float32, n)
	for i, val := range arr.Values {
		if i >= n {
			break
		}
		switch fv := val.(type) {
		case float32:
			scores[i] = fv
		case float64:
			scores[i] = float32(fv)
		}
	}
	return scores, nil
}

// --- simple greedy tokenizer as a fallback for models without merge scores ---

// EncodeGreedy does a simple greedy (longest-match) tokenization.
// Useful for testing and for models that use unigram / wordpiece.
func (t *Tokenizer) EncodeGreedy(text string, addBOS bool) []int32 {
	var ids []int32
	if addBOS && t.BOSID >= 0 {
		ids = append(ids, t.BOSID)
	}

	// Build sorted vocab for longest-match.
	type entry struct {
		tok string
		id  int32
	}
	entries := make([]entry, 0, len(t.vocab))
	for tok, id := range t.toID {
		entries = append(entries, entry{tok, id})
	}
	sort.Slice(entries, func(i, j int) bool {
		return len(entries[i].tok) > len(entries[j].tok)
	})

	for pos := 0; pos < len(text); {
		found := false
		for _, e := range entries {
			if strings.HasPrefix(text[pos:], e.tok) {
				ids = append(ids, e.id)
				pos += len(e.tok)
				found = true
				break
			}
		}
		if !found {
			ids = append(ids, t.UNKID)
			pos++
		}
	}
	return ids
}
