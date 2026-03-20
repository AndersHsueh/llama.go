// Package flash provides flash (SSD) storage abstractions for LLM weight tensors.
// The core insight from "LLM in a Flash" is that flash memory has high sequential
// throughput but high latency for small random reads. This package provides the
// building blocks to read tensor data efficiently from flash.
package flash

import (
	"fmt"
	"io"
	"os"
	"sync"

	"llama.go/gguf"
	"llama.go/tensor"
)

// Storage provides random-access reads of tensor data from a GGUF file on flash.
// The file is kept open; reads are issued on-demand via ReadAt.
type Storage struct {
	f    *os.File
	mu   sync.Mutex // guards concurrent reads on the same fd
	path string
}

// Open opens the GGUF file at path for tensor data reads.
// On Darwin (macOS), F_NOCACHE is set so the OS does not buffer file pages in
// the unified buffer cache — this is the recommended "direct I/O" mode on Apple
// platforms and is essential for keeping flash reads out of RAM (see "LLM in a
// Flash", §4 "Direct I/O").
func Open(path string) (*Storage, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("flash.Open: %w", err)
	}
	// F_NOCACHE (macOS) — disable OS page caching for this fd.
	// This prevents the 19 GB model file from filling the page cache during
	// prefill and causing OOM kills.
	if err := setNocache(f); err != nil {
		// Non-fatal: fall back to cached I/O.
		_ = err
	}
	return &Storage{f: f, path: path}, nil
}

// Close releases the file handle.
func (s *Storage) Close() error {
	return s.f.Close()
}

// ReadRaw reads byteSize bytes starting at offset from flash into a new byte slice.
// ReadRaw reads byteSize bytes starting at offset from flash into a provided buffer.
// If rawBuf is nil or too small, a new one is allocated. Returns the used buffer.
func (s *Storage) ReadRawInto(offset int64, byteSize int64, rawBuf []byte) ([]byte, error) {
	if int64(cap(rawBuf)) < byteSize {
		rawBuf = make([]byte, byteSize)
	}
	buf := rawBuf[:byteSize]
	n, err := s.f.ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("flash.ReadRawInto offset=%d size=%d: %w", offset, byteSize, err)
	}
	if int64(n) < byteSize {
		return nil, fmt.Errorf("flash.ReadRawInto: short read %d/%d", n, byteSize)
	}
	return buf, nil
}

// ReadContiguousRows reads nRows contiguous rows starting at startRow into dst.
// dst must have length >= nRows * Dimensions[0]. This is a single sequential read
// (optimal for flash I/O) and avoids any extra allocation when dst is provided.
// Used by MoE expert loading to read all rows for one expert in one syscall.
func (s *Storage) ReadContiguousRows(ti *gguf.TensorInfo, startRow, nRows int, dst []float32, rawBuf []byte) ([]byte, error) {
	inDim := int(ti.Dimensions[0])
	rowByteSize, err := ti.Type.ByteSize(int64(inDim))
	if err != nil {
		return nil, err
	}
	totalBytes := rowByteSize * int64(nRows)
	offset := int64(ti.Offset) + int64(startRow)*rowByteSize

	rawBuf, err = s.ReadRawInto(offset, totalBytes, rawBuf)
	if err != nil {
		return nil, err
	}

	// Dequantize block-by-block into dst to avoid allocating a new slice.
	// We dequantize 'nRows' row-blocks worth of data.
	rowData, err := tensor.DequantizeInto(rawBuf, ti.Type, dst[:nRows*inDim])
	if err != nil {
		return nil, err
	}
	_ = rowData
	return rawBuf, nil
}

// This is the fundamental primitive — all higher-level reads are built on this.
func (s *Storage) ReadRaw(offset int64, byteSize int64) ([]byte, error) {
	buf := make([]byte, byteSize)
	n, err := s.f.ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("flash.ReadRaw offset=%d size=%d: %w", offset, byteSize, err)
	}
	if int64(n) < byteSize {
		return nil, fmt.Errorf("flash.ReadRaw: short read %d/%d", n, byteSize)
	}
	return buf, nil
}

// ReadTensor reads a tensor from flash, dequantizes it to float32, and returns it.
// Shape is inferred from ti.Dimensions (innermost first, GGUF convention).
func (s *Storage) ReadTensor(ti *gguf.TensorInfo) (*tensor.Tensor, error) {
	nElems := ti.NElements()
	byteSize, err := ti.Type.ByteSize(nElems)
	if err != nil {
		return nil, err
	}
	raw, err := s.ReadRaw(int64(ti.Offset), byteSize)
	if err != nil {
		return nil, err
	}
	data, err := tensor.Dequantize(raw, ti.Type)
	if err != nil {
		return nil, err
	}
	// GGUF stores dimensions innermost-first; convert to standard row-major shape.
	shape := ggufDimsToShape(ti.Dimensions)
	return &tensor.Tensor{Shape: shape, Data: data}, nil
}

// ReadRows reads only specific rows of a 2D weight matrix from flash.
// This is the core primitive for sparse FFN loading.
//
// ti must refer to a 2D tensor with shape [outDim, inDim] (row-major, innermost = inDim).
// rows is the set of row indices to load.
// Returns a *tensor.Tensor of shape [len(rows), inDim] with those rows dequantized.
func (s *Storage) ReadRows(ti *gguf.TensorInfo, rows []int) (*tensor.Tensor, error) {
	if len(ti.Dimensions) != 2 {
		return nil, fmt.Errorf("flash.ReadRows: tensor %q is not 2D", ti.Name)
	}
	// GGUF: Dimensions[0] = innermost = cols (inDim), Dimensions[1] = rows (outDim)
	inDim := int(ti.Dimensions[0])
	outDim := int(ti.Dimensions[1])

	rowByteSize, err := ti.Type.ByteSize(int64(inDim))
	if err != nil {
		return nil, err
	}

	out := tensor.New(len(rows), inDim)
	for destRow, srcRow := range rows {
		if srcRow >= outDim {
			return nil, fmt.Errorf("flash.ReadRows: row %d out of range [0, %d)", srcRow, outDim)
		}
		offset := int64(ti.Offset) + int64(srcRow)*rowByteSize
		raw, err := s.ReadRaw(offset, rowByteSize)
		if err != nil {
			return nil, fmt.Errorf("flash.ReadRows row %d: %w", srcRow, err)
		}
		rowData, err := tensor.Dequantize(raw, ti.Type)
		if err != nil {
			return nil, err
		}
		copy(out.Data[destRow*inDim:], rowData)
	}
	return out, nil
}

// ReadFlatRows reads specific rows from any tensor treated as a flat 2D matrix.
// The "columns" dimension is Dimensions[0] (innermost), and the total row count
// is the product of all remaining dimensions. This enables slicing 3D expert
// tensors (e.g. [hidden, expertFF, nExperts]) without modification.
func (s *Storage) ReadFlatRows(ti *gguf.TensorInfo, rows []int, nThreads int) (*tensor.Tensor, error) {
	inDim := int(ti.Dimensions[0])
	var totalRows int64 = 1
	for i := 1; i < len(ti.Dimensions); i++ {
		totalRows *= int64(ti.Dimensions[i])
	}

	rowByteSize, err := ti.Type.ByteSize(int64(inDim))
	if err != nil {
		return nil, err
	}

	out := tensor.New(len(rows), inDim)
	errs := make([]error, len(rows))

	type work struct {
		destRow int
		srcRow  int
	}
	jobs := make(chan work, len(rows))
	for i, r := range rows {
		jobs <- work{i, r}
	}
	close(jobs)

	if nThreads <= 0 {
		nThreads = 32
	}

	var wg sync.WaitGroup
	for t := 0; t < nThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f, ferr := os.Open(s.path)
			if ferr != nil {
				for job := range jobs {
					errs[job.destRow] = ferr
				}
				return
			}
			defer f.Close()
			for job := range jobs {
				if int64(job.srcRow) >= totalRows {
					errs[job.destRow] = fmt.Errorf("row %d out of range [0,%d)", job.srcRow, totalRows)
					continue
				}
				offset := int64(ti.Offset) + int64(job.srcRow)*rowByteSize
				buf := make([]byte, rowByteSize)
				if _, rerr := f.ReadAt(buf, offset); rerr != nil {
					errs[job.destRow] = rerr
					continue
				}
				rowData, derr := tensor.Dequantize(buf, ti.Type)
				if derr != nil {
					errs[job.destRow] = derr
					continue
				}
				copy(out.Data[job.destRow*inDim:], rowData)
			}
		}()
	}
	wg.Wait()
	for i, e := range errs {
		if e != nil {
			return nil, fmt.Errorf("flash.ReadFlatRows row[%d]: %w", i, e)
		}
	}
	return out, nil
}

// ReadRowsParallel reads specific rows in parallel using nThreads goroutines.
// For large batches of rows, this significantly reduces wall-clock latency
// by overlapping I/O (matching the paper's 32-thread strategy).
func (s *Storage) ReadRowsParallel(ti *gguf.TensorInfo, rows []int, nThreads int) (*tensor.Tensor, error) {
	if len(ti.Dimensions) != 2 {
		return nil, fmt.Errorf("flash.ReadRowsParallel: tensor %q is not 2D", ti.Name)
	}
	inDim := int(ti.Dimensions[0])
	outDim := int(ti.Dimensions[1])

	rowByteSize, err := ti.Type.ByteSize(int64(inDim))
	if err != nil {
		return nil, err
	}

	out := tensor.New(len(rows), inDim)
	errs := make([]error, len(rows))

	type work struct {
		destRow int
		srcRow  int
	}
	jobs := make(chan work, len(rows))
	for i, r := range rows {
		jobs <- work{i, r}
	}
	close(jobs)

	if nThreads <= 0 {
		nThreads = 32
	}

	var wg sync.WaitGroup
	for t := 0; t < nThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Each goroutine opens its own file descriptor to avoid locking.
			f, err := os.Open(s.path)
			if err != nil {
				// Store error for first failing job only.
				for job := range jobs {
					errs[job.destRow] = err
				}
				return
			}
			defer f.Close()

			for job := range jobs {
				if job.srcRow >= outDim {
					errs[job.destRow] = fmt.Errorf("row %d out of range [0,%d)", job.srcRow, outDim)
					continue
				}
				offset := int64(ti.Offset) + int64(job.srcRow)*rowByteSize
				buf := make([]byte, rowByteSize)
				if _, err := f.ReadAt(buf, offset); err != nil {
					errs[job.destRow] = err
					continue
				}
				rowData, err := tensor.Dequantize(buf, ti.Type)
				if err != nil {
					errs[job.destRow] = err
					continue
				}
				copy(out.Data[job.destRow*inDim:], rowData)
			}
		}()
	}
	wg.Wait()

	for i, e := range errs {
		if e != nil {
			return nil, fmt.Errorf("flash.ReadRowsParallel row[%d]: %w", i, e)
		}
	}
	return out, nil
}

// ggufDimsToShape converts GGUF dimension order (innermost first) to standard shape.
// A 2D weight matrix in GGUF is [cols, rows]; we return [rows, cols].
func ggufDimsToShape(dims []uint64) []int {
	shape := make([]int, len(dims))
	for i, d := range dims {
		shape[len(dims)-1-i] = int(d)
	}
	return shape
}
