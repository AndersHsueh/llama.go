// Package flash provides the row-column bundling storage format.
// Section 3.2 of the paper: by storing the i-th column of the up-projection
// and the i-th row of the down-projection contiguously, we can load both with
// a single larger I/O operation, doubling the chunk size and improving throughput.
package flash

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sync"
	"unsafe"

	"llama.go/gguf"
	"llama.go/tensor"
)

// BundleHeader is written at the start of each .bundle file.
const BundleMagic = 0x424E444C // "BNDL"

// BundleFile is a pre-processed file where neuron i stores:
//   [up_row_i (inDim floats) | down_row_i (inDim floats)]
// as raw float32 little-endian, allowing a single ReadAt per neuron.
type BundleFile struct {
	path        string
	f           *os.File
	mu          sync.Mutex
	nNeurons    int
	hiddenSize  int // inDim (= d_model)
	bytesPerRow int64
}

// CreateBundle pre-processes two GGUF tensor (up, down) into a bundle file at outPath.
// outPath is typically the model path + ".layerN.bundle".
// This is a one-time offline operation.
func CreateBundle(outPath string, storage *Storage, upInfo, downInfo *gguf.TensorInfo) error {
	if len(upInfo.Dimensions) != 2 || len(downInfo.Dimensions) != 2 {
		return fmt.Errorf("bundle: up and down must be 2D tensors")
	}
	// up:   [hiddenSize, nNeurons]  → Dimensions[0]=hiddenSize, [1]=nNeurons
	// down: [nNeurons, hiddenSize]  → Dimensions[0]=hiddenSize, [1]=nNeurons
	hiddenSize := int(upInfo.Dimensions[0])
	nNeurons := int(upInfo.Dimensions[1])

	if int(downInfo.Dimensions[0]) != hiddenSize || int(downInfo.Dimensions[1]) != nNeurons {
		return fmt.Errorf("bundle: up/down dimension mismatch")
	}

	upRowBytes, err := upInfo.Type.ByteSize(int64(hiddenSize))
	if err != nil {
		return err
	}
	downRowBytes, err := downInfo.Type.ByteSize(int64(hiddenSize))
	if err != nil {
		return err
	}

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("bundle create %s: %w", outPath, err)
	}
	defer out.Close()

	// Write header.
	hdr := make([]byte, 16)
	binary.LittleEndian.PutUint32(hdr[0:], BundleMagic)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(nNeurons))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(hiddenSize))
	binary.LittleEndian.PutUint32(hdr[12:], 0) // reserved
	if _, err := out.Write(hdr); err != nil {
		return err
	}

	// For each neuron, write [dequantized up_row (f32) | dequantized down_row (f32)].
	rowF32Bytes := int64(hiddenSize) * 4 // float32 = 4 bytes
	for i := 0; i < nNeurons; i++ {
		// Read up row i.
		upOffset := int64(upInfo.Offset) + int64(i)*upRowBytes
		upRaw, err := storage.ReadRaw(upOffset, upRowBytes)
		if err != nil {
			return fmt.Errorf("bundle up row %d: %w", i, err)
		}
		upF32, err := tensor.Dequantize(upRaw, upInfo.Type)
		if err != nil {
			return err
		}

		// Read down row i.
		downOffset := int64(downInfo.Offset) + int64(i)*downRowBytes
		downRaw, err := storage.ReadRaw(downOffset, downRowBytes)
		if err != nil {
			return fmt.Errorf("bundle down row %d: %w", i, err)
		}
		downF32, err := tensor.Dequantize(downRaw, downInfo.Type)
		if err != nil {
			return err
		}

		// Write bundled row.
		buf := make([]byte, 2*rowF32Bytes)
		for j, v := range upF32 {
			binary.LittleEndian.PutUint32(buf[j*4:], *(*uint32)(unsafe.Pointer(&v)))
		}
		for j, v := range downF32 {
			binary.LittleEndian.PutUint32(buf[rowF32Bytes+int64(j)*4:], *(*uint32)(unsafe.Pointer(&v)))
		}
		if _, err := out.Write(buf); err != nil {
			return fmt.Errorf("bundle write row %d: %w", i, err)
		}
	}
	return nil
}

func ptrFloat32(f *float32) *uint32 {
	return (*uint32)(unsafe.Pointer(f))
}

// OpenBundle opens an existing bundle file for reading.
func OpenBundle(path string) (*BundleFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("bundle open %s: %w", path, err)
	}
	hdr := make([]byte, 16)
	if _, err := io.ReadFull(f, hdr); err != nil {
		f.Close()
		return nil, err
	}
	magic := binary.LittleEndian.Uint32(hdr[0:])
	if magic != BundleMagic {
		f.Close()
		return nil, fmt.Errorf("bundle: bad magic 0x%08x", magic)
	}
	nNeurons := int(binary.LittleEndian.Uint32(hdr[4:]))
	hiddenSize := int(binary.LittleEndian.Uint32(hdr[8:]))
	bytesPerRow := int64(2 * hiddenSize * 4) // 2 × f32 rows

	return &BundleFile{
		path:        path,
		f:           f,
		nNeurons:    nNeurons,
		hiddenSize:  hiddenSize,
		bytesPerRow: bytesPerRow,
	}, nil
}

// Close releases the file handle.
func (b *BundleFile) Close() error { return b.f.Close() }

// ReadNeurons reads the bundled [up_row | down_row] for each neuron in neurons.
// Returns (upRows, downRows) each of shape [len(neurons), hiddenSize].
func (b *BundleFile) ReadNeurons(neurons []int, nThreads int) (*tensor.Tensor, *tensor.Tensor, error) {
	h := b.hiddenSize
	upOut := tensor.New(len(neurons), h)
	downOut := tensor.New(len(neurons), h)
	errs := make([]error, len(neurons))

	type work struct {
		destIdx int
		neuron  int
	}
	jobs := make(chan work, len(neurons))
	for i, n := range neurons {
		jobs <- work{i, n}
	}
	close(jobs)

	if nThreads <= 0 {
		nThreads = 32
	}

	const headerBytes = 16

	var wg sync.WaitGroup
	for t := 0; t < nThreads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f, err := os.Open(b.path)
			if err != nil {
				for job := range jobs {
					errs[job.destIdx] = err
				}
				return
			}
			defer f.Close()

			rowBuf := make([]byte, b.bytesPerRow)
			for job := range jobs {
				offset := int64(headerBytes) + int64(job.neuron)*b.bytesPerRow
				if _, err := f.ReadAt(rowBuf, offset); err != nil {
					errs[job.destIdx] = err
					continue
				}
				// Deserialize float32 values.
				for i := 0; i < h; i++ {
					bits := binary.LittleEndian.Uint32(rowBuf[i*4:])
					upOut.Data[job.destIdx*h+i] = *(*float32)(unsafe.Pointer(&bits))
				}
				for i := 0; i < h; i++ {
					bits := binary.LittleEndian.Uint32(rowBuf[int64(h)*4+int64(i)*4:])
					downOut.Data[job.destIdx*h+i] = *(*float32)(unsafe.Pointer(&bits))
				}
			}
		}()
	}
	wg.Wait()

	for i, e := range errs {
		if e != nil {
			return nil, nil, fmt.Errorf("bundle read neuron[%d]: %w", i, e)
		}
	}
	return upOut, downOut, nil
}

// NNeurons returns the number of neurons in the bundle.
func (b *BundleFile) NNeurons() int { return b.nNeurons }

// HiddenSize returns the hidden dimension.
func (b *BundleFile) HiddenSize() int { return b.hiddenSize }
