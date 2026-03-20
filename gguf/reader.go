package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// File represents a parsed GGUF model file.
type File struct {
	Header   Header
	Meta     map[string]MetaValue // keyed by metadata key name
	Tensors  []*TensorInfo
	TensorsByName map[string]*TensorInfo

	// DataOffset is the byte offset in the file where tensor data begins.
	DataOffset int64

	path string
}

// Open parses the GGUF file at path and returns a File.
// Tensor data is NOT read into memory; only the metadata and tensor descriptors
// are loaded. Use ReadTensorData to fetch actual weight bytes.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %s: %w", path, err)
	}
	defer f.Close()

	r := &reader{r: f}

	// --- header ---
	magic, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("gguf: bad magic 0x%08x (want 0x%08x)", magic, Magic)
	}

	version, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version < VersionV1 || version > VersionV3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	tensorCount, err := r.readU64()
	if err != nil {
		return nil, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	metaKVCount, err := r.readU64()
	if err != nil {
		return nil, fmt.Errorf("gguf: read meta kv count: %w", err)
	}

	hdr := Header{
		Magic:       magic,
		Version:     version,
		TensorCount: tensorCount,
		MetaKVCount: metaKVCount,
	}

	// --- metadata key-value pairs ---
	meta := make(map[string]MetaValue, metaKVCount)
	for i := uint64(0); i < metaKVCount; i++ {
		kv, err := r.readMetaKV()
		if err != nil {
			return nil, fmt.Errorf("gguf: meta kv %d: %w", i, err)
		}
		meta[kv.Key] = kv
	}

	// --- tensor descriptors ---
	tensors := make([]*TensorInfo, 0, tensorCount)
	byName := make(map[string]*TensorInfo, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		ti, err := r.readTensorInfo()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %d: %w", i, err)
		}
		tensors = append(tensors, ti)
		byName[ti.Name] = ti
	}

	// Tensor data starts at the next ALIGNMENT boundary after the current position.
	const alignment = 32
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("gguf: seek current: %w", err)
	}
	dataOffset := align(pos, alignment)

	// Fix up tensor offsets — the offsets in the file are relative to dataOffset.
	// (In GGUF v3 they're already absolute; in v1/v2 they're relative.)
	// We normalise to absolute here.
	// Note: actually in all versions offsets are relative to the data section start.
	for _, ti := range tensors {
		ti.Offset += uint64(dataOffset)
	}

	return &File{
		Header:        hdr,
		Meta:          meta,
		Tensors:       tensors,
		TensorsByName: byName,
		DataOffset:    dataOffset,
		path:          path,
	}, nil
}

// align rounds x up to the nearest multiple of a.
func align(x, a int64) int64 {
	return (x + a - 1) &^ (a - 1)
}

// MetaString returns the string value for key, or an error if missing/wrong type.
func (f *File) MetaString(key string) (string, error) {
	v, ok := f.Meta[key]
	if !ok {
		return "", fmt.Errorf("gguf: metadata key %q not found", key)
	}
	s, ok := v.Value.(string)
	if !ok {
		return "", fmt.Errorf("gguf: metadata key %q is not a string", key)
	}
	return s, nil
}

// MetaUint32 returns the uint32 value for key.
func (f *File) MetaUint32(key string) (uint32, error) {
	v, ok := f.Meta[key]
	if !ok {
		return 0, fmt.Errorf("gguf: metadata key %q not found", key)
	}
	switch val := v.Value.(type) {
	case uint32:
		return val, nil
	case uint64:
		return uint32(val), nil
	case int32:
		return uint32(val), nil
	default:
		return 0, fmt.Errorf("gguf: metadata key %q is not uint32 (got %T)", key, v.Value)
	}
}

// MetaUint64 returns the uint64 value for key.
func (f *File) MetaUint64(key string) (uint64, error) {
	v, ok := f.Meta[key]
	if !ok {
		return 0, fmt.Errorf("gguf: metadata key %q not found", key)
	}
	switch val := v.Value.(type) {
	case uint64:
		return val, nil
	case uint32:
		return uint64(val), nil
	default:
		return 0, fmt.Errorf("gguf: metadata key %q is not uint64 (got %T)", key, v.Value)
	}
}

// MetaFloat32 returns the float32 value for key.
func (f *File) MetaFloat32(key string) (float32, error) {
	v, ok := f.Meta[key]
	if !ok {
		return 0, fmt.Errorf("gguf: metadata key %q not found", key)
	}
	val, ok := v.Value.(float32)
	if !ok {
		return 0, fmt.Errorf("gguf: metadata key %q is not float32 (got %T)", key, v.Value)
	}
	return val, nil
}

// MetaStringArray returns a []string for an array-typed metadata key.
func (f *File) MetaStringArray(key string) ([]string, error) {
	v, ok := f.Meta[key]
	if !ok {
		return nil, fmt.Errorf("gguf: metadata key %q not found", key)
	}
	arr, ok := v.Value.(ArrayValue)
	if !ok {
		return nil, fmt.Errorf("gguf: metadata key %q is not an array", key)
	}
	out := make([]string, len(arr.Values))
	for i, val := range arr.Values {
		s, ok := val.(string)
		if !ok {
			return nil, fmt.Errorf("gguf: array element %d is not string", i)
		}
		out[i] = s
	}
	return out, nil
}

// ReadTensorData reads the raw bytes for a tensor from the GGUF file.
func (f *File) ReadTensorData(ti *TensorInfo) ([]byte, error) {
	nElems := ti.NElements()
	byteSize, err := ti.Type.ByteSize(nElems)
	if err != nil {
		return nil, err
	}

	fp, err := os.Open(f.path)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	if _, err := fp.Seek(int64(ti.Offset), io.SeekStart); err != nil {
		return nil, err
	}
	buf := make([]byte, byteSize)
	if _, err := io.ReadFull(fp, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

// Path returns the file path.
func (f *File) Path() string { return f.path }

// --- internal binary reader ---

type reader struct {
	r io.Reader
}

func (r *reader) readU8() (uint8, error) {
	var b [1]byte
	_, err := io.ReadFull(r.r, b[:])
	return b[0], err
}

func (r *reader) readU16() (uint16, error) {
	var b [2]byte
	_, err := io.ReadFull(r.r, b[:])
	return binary.LittleEndian.Uint16(b[:]), err
}

func (r *reader) readU32() (uint32, error) {
	var b [4]byte
	_, err := io.ReadFull(r.r, b[:])
	return binary.LittleEndian.Uint32(b[:]), err
}

func (r *reader) readU64() (uint64, error) {
	var b [8]byte
	_, err := io.ReadFull(r.r, b[:])
	return binary.LittleEndian.Uint64(b[:]), err
}

func (r *reader) readI8() (int8, error) {
	v, err := r.readU8()
	return int8(v), err
}

func (r *reader) readI16() (int16, error) {
	v, err := r.readU16()
	return int16(v), err
}

func (r *reader) readI32() (int32, error) {
	v, err := r.readU32()
	return int32(v), err
}

func (r *reader) readI64() (int64, error) {
	v, err := r.readU64()
	return int64(v), err
}

func (r *reader) readF32() (float32, error) {
	v, err := r.readU32()
	return math.Float32frombits(v), err
}

func (r *reader) readF64() (float64, error) {
	v, err := r.readU64()
	return math.Float64frombits(v), err
}

func (r *reader) readBool() (bool, error) {
	v, err := r.readU8()
	return v != 0, err
}

// readString reads a GGUF length-prefixed UTF-8 string.
func (r *reader) readString() (string, error) {
	length, err := r.readU64()
	if err != nil {
		return "", err
	}
	if length > 1<<24 {
		return "", fmt.Errorf("gguf: string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func (r *reader) readMetaValue(t MetaValueType) (any, error) {
	switch t {
	case MetaValueTypeUint8:
		return r.readU8()
	case MetaValueTypeInt8:
		return r.readI8()
	case MetaValueTypeUint16:
		return r.readU16()
	case MetaValueTypeInt16:
		return r.readI16()
	case MetaValueTypeUint32:
		return r.readU32()
	case MetaValueTypeInt32:
		return r.readI32()
	case MetaValueTypeFloat32:
		return r.readF32()
	case MetaValueTypeBool:
		return r.readBool()
	case MetaValueTypeString:
		return r.readString()
	case MetaValueTypeUint64:
		return r.readU64()
	case MetaValueTypeInt64:
		return r.readI64()
	case MetaValueTypeFloat64:
		return r.readF64()
	case MetaValueTypeArray:
		return r.readArray()
	default:
		return nil, fmt.Errorf("unknown MetaValueType %d", t)
	}
}

func (r *reader) readArray() (ArrayValue, error) {
	elemTypeRaw, err := r.readU32()
	if err != nil {
		return ArrayValue{}, err
	}
	elemType := MetaValueType(elemTypeRaw)
	count, err := r.readU64()
	if err != nil {
		return ArrayValue{}, err
	}
	if count > 1<<20 {
		return ArrayValue{}, fmt.Errorf("gguf: array too large: %d", count)
	}
	values := make([]any, count)
	for i := uint64(0); i < count; i++ {
		v, err := r.readMetaValue(elemType)
		if err != nil {
			return ArrayValue{}, fmt.Errorf("array element %d: %w", i, err)
		}
		values[i] = v
	}
	return ArrayValue{ElemType: elemType, Values: values}, nil
}

func (r *reader) readMetaKV() (MetaValue, error) {
	key, err := r.readString()
	if err != nil {
		return MetaValue{}, fmt.Errorf("key: %w", err)
	}
	typeRaw, err := r.readU32()
	if err != nil {
		return MetaValue{}, fmt.Errorf("value type: %w", err)
	}
	t := MetaValueType(typeRaw)
	val, err := r.readMetaValue(t)
	if err != nil {
		return MetaValue{}, fmt.Errorf("value for key %q: %w", key, err)
	}
	return MetaValue{Key: key, Type: t, Value: val}, nil
}

func (r *reader) readTensorInfo() (*TensorInfo, error) {
	name, err := r.readString()
	if err != nil {
		return nil, fmt.Errorf("name: %w", err)
	}
	nDims, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("ndims: %w", err)
	}
	if nDims > 4 {
		return nil, fmt.Errorf("too many dimensions: %d", nDims)
	}
	dims := make([]uint64, nDims)
	for i := uint32(0); i < nDims; i++ {
		dims[i], err = r.readU64()
		if err != nil {
			return nil, fmt.Errorf("dim %d: %w", i, err)
		}
	}
	typeRaw, err := r.readU32()
	if err != nil {
		return nil, fmt.Errorf("type: %w", err)
	}
	offset, err := r.readU64()
	if err != nil {
		return nil, fmt.Errorf("offset: %w", err)
	}
	return &TensorInfo{
		Name:       name,
		Dimensions: dims,
		Type:       GGMLType(typeRaw),
		Offset:     offset,
	}, nil
}
