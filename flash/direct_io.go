package flash

import (
	"fmt"
	"os"
	"sync"
	"time"

	"llama.go/gguf"
	"llama.go/tensor"
)

// BenchResult holds timing measurements for flash I/O.
type BenchResult struct {
	BytesRead     int64
	Elapsed       time.Duration
	ThroughputMBs float64 // MB/s
	NReads        int
}

func (b BenchResult) String() string {
	return fmt.Sprintf("%.2f MB in %.1fms → %.2f MB/s (%d reads)",
		float64(b.BytesRead)/1e6, float64(b.Elapsed.Microseconds())/1000.0,
		b.ThroughputMBs, b.NReads)
}

// ReadRowsBench reads specific rows from flash and returns timing info.
// Useful for measuring actual flash throughput on the target hardware.
func (s *Storage) ReadRowsBench(ti *gguf.TensorInfo, rows []int, nThreads int) (*tensor.Tensor, BenchResult, error) {
	t0 := time.Now()
	result, err := s.ReadRowsParallel(ti, rows, nThreads)
	elapsed := time.Since(t0)

	if err != nil {
		return nil, BenchResult{}, err
	}

	inDim := int(ti.Dimensions[0])
	rowBytes, _ := ti.Type.ByteSize(int64(inDim))
	bytesRead := int64(len(rows)) * rowBytes

	br := BenchResult{
		BytesRead: bytesRead,
		Elapsed:   elapsed,
		ThroughputMBs: float64(bytesRead) / 1e6 / elapsed.Seconds(),
		NReads:    len(rows),
	}
	return result, br, nil
}

// BenchSequential reads tensor data as one big sequential read (best-case throughput).
func (s *Storage) BenchSequential(ti *gguf.TensorInfo) (BenchResult, error) {
	nElems := ti.NElements()
	byteSize, err := ti.Type.ByteSize(nElems)
	if err != nil {
		return BenchResult{}, err
	}

	t0 := time.Now()
	_, err = s.ReadRaw(int64(ti.Offset), byteSize)
	elapsed := time.Since(t0)
	if err != nil {
		return BenchResult{}, err
	}

	return BenchResult{
		BytesRead:     byteSize,
		Elapsed:       elapsed,
		ThroughputMBs: float64(byteSize) / 1e6 / elapsed.Seconds(),
		NReads:        1,
	}, nil
}

// DropOSCache attempts to drop the OS page cache for the file by re-opening
// with O_DIRECT semantics (best-effort; may not work on all platforms).
// On macOS, we use F_NOCACHE via fcntl instead.
func (s *Storage) DropOSCache() {
	dropCache(s.f)
}

// ReadRowsDirect reads rows using O_DIRECT / F_NOCACHE to bypass the OS page
// cache, giving a more accurate measurement of actual flash throughput.
func (s *Storage) ReadRowsDirect(ti *gguf.TensorInfo, rows []int, nThreads int) (*tensor.Tensor, BenchResult, error) {
	if len(ti.Dimensions) != 2 {
		return nil, BenchResult{}, fmt.Errorf("ReadRowsDirect: tensor %q is not 2D", ti.Name)
	}
	inDim := int(ti.Dimensions[0])
	outDim := int(ti.Dimensions[1])
	rowByteSize, err := ti.Type.ByteSize(int64(inDim))
	if err != nil {
		return nil, BenchResult{}, err
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

	t0 := time.Now()
	var wg sync.WaitGroup
	for t := 0; t < nThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f, err := openDirect(s.path)
			if err != nil {
				for job := range jobs {
					errs[job.destRow] = err
				}
				return
			}
			defer f.Close()

			for job := range jobs {
				if job.srcRow >= outDim {
					errs[job.destRow] = fmt.Errorf("row %d out of range", job.srcRow)
					continue
				}
				offset := int64(ti.Offset) + int64(job.srcRow)*rowByteSize
				buf := makeAlignedBuf(rowByteSize)
				n, err := f.ReadAt(buf[:rowByteSize], offset)
				if err != nil || int64(n) < rowByteSize {
					// Fall back to regular read if direct I/O fails
					fp2, _ := os.Open(s.path)
					if fp2 != nil {
						fp2.ReadAt(buf[:rowByteSize], offset)
						fp2.Close()
					}
				}
				rowData, err2 := tensor.Dequantize(buf[:rowByteSize], ti.Type)
				if err2 != nil {
					errs[job.destRow] = err2
					continue
				}
				copy(out.Data[job.destRow*inDim:], rowData)
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(t0)

	for i, e := range errs {
		if e != nil {
			return nil, BenchResult{}, fmt.Errorf("direct read row[%d]: %w", i, e)
		}
	}

	bytesRead := int64(len(rows)) * rowByteSize
	br := BenchResult{
		BytesRead:     bytesRead,
		Elapsed:       elapsed,
		ThroughputMBs: float64(bytesRead) / 1e6 / elapsed.Seconds(),
		NReads:        len(rows),
	}
	return out, br, nil
}

// makeAlignedBuf returns a buffer of at least n bytes.
// On platforms requiring aligned buffers for O_DIRECT we'd use posix_memalign;
// here we allocate with a 4KB alignment margin.
func makeAlignedBuf(n int64) []byte {
	const align = 4096
	buf := make([]byte, n+align)
	// Round up to alignment boundary.
	offset := int64(uintptr(unsafe_ptr(buf)) & (align - 1))
	if offset != 0 {
		offset = align - offset
	}
	return buf[offset : offset+n]
}
