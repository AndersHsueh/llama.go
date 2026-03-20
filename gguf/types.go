// Package gguf implements parsing of the GGUF model file format.
// GGUF is the binary format used by llama.cpp for storing LLM weights and metadata.
package gguf

import "fmt"

// Magic bytes at the start of every GGUF file.
const Magic = 0x46554747 // "GGUF" in little-endian

// Supported GGUF versions.
const (
	VersionV1 = 1
	VersionV2 = 2
	VersionV3 = 3
)

// MetaValueType enumerates the types of metadata values.
type MetaValueType uint32

const (
	MetaValueTypeUint8   MetaValueType = 0
	MetaValueTypeInt8    MetaValueType = 1
	MetaValueTypeUint16  MetaValueType = 2
	MetaValueTypeInt16   MetaValueType = 3
	MetaValueTypeUint32  MetaValueType = 4
	MetaValueTypeInt32   MetaValueType = 5
	MetaValueTypeFloat32 MetaValueType = 6
	MetaValueTypeBool    MetaValueType = 7
	MetaValueTypeString  MetaValueType = 8
	MetaValueTypeArray   MetaValueType = 9
	MetaValueTypeUint64  MetaValueType = 10
	MetaValueTypeInt64   MetaValueType = 11
	MetaValueTypeFloat64 MetaValueType = 12
)

// GGMLType enumerates the data types for tensor elements.
type GGMLType uint32

const (
	GGMLTypeF32     GGMLType = 0
	GGMLTypeF16     GGMLType = 1
	GGMLTypeQ4_0    GGMLType = 2
	GGMLTypeQ4_1    GGMLType = 3
	GGMLTypeQ5_0    GGMLType = 6
	GGMLTypeQ5_1    GGMLType = 7
	GGMLTypeQ8_0    GGMLType = 8
	GGMLTypeQ8_1    GGMLType = 9
	GGMLTypeQ2_K    GGMLType = 10
	GGMLTypeQ3_K    GGMLType = 11
	GGMLTypeQ4_K    GGMLType = 12
	GGMLTypeQ5_K    GGMLType = 13
	GGMLTypeQ6_K    GGMLType = 14
	GGMLTypeQ8_K    GGMLType = 15
	GGMLTypeI8      GGMLType = 24
	GGMLTypeI16     GGMLType = 25
	GGMLTypeI32     GGMLType = 26
	GGMLTypeI64     GGMLType = 27
	GGMLTypeF64     GGMLType = 28
	GGMLTypeIQ4_NL  GGMLType = 20
	GGMLTypeIQ4_XS  GGMLType = 23
)

// GGMLTypeInfo holds block size and byte size for each GGML data type.
type GGMLTypeInfo struct {
	BlockSize int // number of elements per block
	TypeSize  int // bytes per block
}

// TypeInfos maps GGMLType to its size information.
var TypeInfos = map[GGMLType]GGMLTypeInfo{
	GGMLTypeF32:  {1, 4},
	GGMLTypeF16:  {1, 2},
	GGMLTypeQ4_0: {32, 18},
	GGMLTypeQ4_1: {32, 20},
	GGMLTypeQ5_0: {32, 22},
	GGMLTypeQ5_1: {32, 24},
	GGMLTypeQ8_0: {32, 34},
	GGMLTypeQ8_1: {32, 36},
	GGMLTypeQ2_K: {256, 84},
	GGMLTypeQ3_K: {256, 110},
	GGMLTypeQ4_K: {256, 144},
	GGMLTypeQ5_K: {256, 176},
	GGMLTypeQ6_K: {256, 210},
	GGMLTypeQ8_K: {256, 292},
	GGMLTypeI8:   {1, 1},
	GGMLTypeI16:  {1, 2},
	GGMLTypeI32:  {1, 4},
	GGMLTypeI64:  {1, 8},
	GGMLTypeF64:  {1, 8},
}

// ByteSize returns the total byte size for n elements of the given type.
func (t GGMLType) ByteSize(nElements int64) (int64, error) {
	info, ok := TypeInfos[t]
	if !ok {
		return 0, fmt.Errorf("unknown GGML type: %d", t)
	}
	if nElements%int64(info.BlockSize) != 0 {
		return 0, fmt.Errorf("element count %d not divisible by block size %d", nElements, info.BlockSize)
	}
	return (nElements / int64(info.BlockSize)) * int64(info.TypeSize), nil
}

func (t GGMLType) String() string {
	names := map[GGMLType]string{
		GGMLTypeF32:  "F32",
		GGMLTypeF16:  "F16",
		GGMLTypeQ4_0: "Q4_0",
		GGMLTypeQ4_1: "Q4_1",
		GGMLTypeQ5_0: "Q5_0",
		GGMLTypeQ5_1: "Q5_1",
		GGMLTypeQ8_0: "Q8_0",
		GGMLTypeQ8_1: "Q8_1",
		GGMLTypeQ2_K: "Q2_K",
		GGMLTypeQ3_K: "Q3_K",
		GGMLTypeQ4_K: "Q4_K",
		GGMLTypeQ5_K: "Q5_K",
		GGMLTypeQ6_K: "Q6_K",
		GGMLTypeQ8_K: "Q8_K",
		GGMLTypeI8:   "I8",
		GGMLTypeI16:  "I16",
		GGMLTypeI32:  "I32",
		GGMLTypeI64:  "I64",
		GGMLTypeF64:  "F64",
	}
	if s, ok := names[t]; ok {
		return s
	}
	return fmt.Sprintf("GGMLType(%d)", t)
}

// Header is the top-level GGUF file header.
type Header struct {
	Magic        uint32
	Version      uint32
	TensorCount  uint64
	MetaKVCount  uint64
}

// MetaValue holds a single metadata key-value entry.
type MetaValue struct {
	Key   string
	Type  MetaValueType
	Value any // uint8/int8/.../string/[]MetaValue
}

// ArrayValue represents an array metadata value.
type ArrayValue struct {
	ElemType MetaValueType
	Values   []any
}

// TensorInfo describes a tensor stored in the GGUF file.
type TensorInfo struct {
	Name       string
	Dimensions []uint64  // shape, from innermost to outermost
	Type       GGMLType
	Offset     uint64    // byte offset from start of tensor data section
}

// NElements returns the total number of elements in this tensor.
func (t *TensorInfo) NElements() int64 {
	n := int64(1)
	for _, d := range t.Dimensions {
		n *= int64(d)
	}
	return n
}
