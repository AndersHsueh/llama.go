package tensor

import (
	"encoding/binary"
	"fmt"
	"math"

	"llama.go/gguf"
)

// DequantizeF16 converts raw F16 bytes (little-endian) to []float32.
func DequantizeF16(raw []byte) ([]float32, error) {
	if len(raw)%2 != 0 {
		return nil, fmt.Errorf("dtype: F16 data length %d not divisible by 2", len(raw))
	}
	n := len(raw) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		h := binary.LittleEndian.Uint16(raw[i*2:])
		out[i] = float16ToFloat32(h)
	}
	return out, nil
}

// DequantizeF32 converts raw F32 bytes (little-endian) to []float32.
func DequantizeF32(raw []byte) ([]float32, error) {
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("dtype: F32 data length %d not divisible by 4", len(raw))
	}
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// DequantizeQ8_0 dequantizes Q8_0 blocks to float32.
// Each block: 2 bytes scale (f16) + 32 x int8 values.
func DequantizeQ8_0(raw []byte) ([]float32, error) {
	const blockSize = 32
	const bytesPerBlock = 2 + blockSize // 34
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q8_0 data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		scale := float16ToFloat32(binary.LittleEndian.Uint16(raw[base:]))
		for i := 0; i < blockSize; i++ {
			q := int8(raw[base+2+i])
			out[b*blockSize+i] = float32(q) * scale
		}
	}
	return out, nil
}

// DequantizeQ4_0 dequantizes Q4_0 blocks to float32.
// Each block: 2 bytes scale (f16) + 16 bytes (32 x 4-bit values, packed two per byte).
func DequantizeQ4_0(raw []byte) ([]float32, error) {
	const blockSize = 32
	const bytesPerBlock = 2 + blockSize/2 // 18
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q4_0 data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		scale := float16ToFloat32(binary.LittleEndian.Uint16(raw[base:]))
		for i := 0; i < blockSize/2; i++ {
			packed := raw[base+2+i]
			lo := int8((packed & 0x0F) - 8)
			hi := int8((packed >> 4) - 8)
			out[b*blockSize+i*2] = float32(lo) * scale
			out[b*blockSize+i*2+1] = float32(hi) * scale
		}
	}
	return out, nil
}

// DequantizeQ4_K dequantizes Q4_K blocks to float32.
// Block layout: 2 x F16 super-scales + 12 sub-scale bytes + 128 x 4-bit weights
// (256 elements per block, 144 bytes total).
func DequantizeQ4_K(raw []byte) ([]float32, error) {
	const blockSize = 256
	const bytesPerBlock = 144
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q4_K data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		dSuper := float16ToFloat32(binary.LittleEndian.Uint16(raw[base:]))
		dminSuper := float16ToFloat32(binary.LittleEndian.Uint16(raw[base+2:]))

		// 12 bytes of sub-scales (6-bit each, packed)
		scales := raw[base+4 : base+16]
		// 128 bytes of 4-bit weights
		qs := raw[base+16 : base+144]

		// 8 sub-blocks of 32 elements each
		for sb := 0; sb < 8; sb++ {
			// Extract 6-bit scale and min
			sc, mn := extractQ4KSubScale(scales, sb)
			d := dSuper * float32(sc)
			dmin := dminSuper * float32(mn)

			for i := 0; i < 32; i++ {
				idx := sb*32 + i
				byteIdx := idx / 2
				var q uint8
				if idx%2 == 0 {
					q = qs[byteIdx] & 0x0F
				} else {
					q = qs[byteIdx] >> 4
				}
				out[b*blockSize+idx] = d*float32(q) - dmin
			}
		}
	}
	return out, nil
}

// extractQ4KSubScale extracts the 6-bit scale and min for sub-block sb from
// the 12-byte packed scales array.
func extractQ4KSubScale(scales []byte, sb int) (uint8, uint8) {
	// Scales are packed as 6-bit values: first 8 = scales, next 8 = mins
	// but interleaved in a specific bit pattern.
	// Following the GGML Q4_K block layout:
	// bits [0..5] of byte sb*12/8 and following bytes for scale sb
	// This is a simplified extraction matching the ggml reference:
	switch sb {
	case 0:
		return scales[0] & 63, scales[4] & 15 | (scales[8]&3)<<4
	case 1:
		return scales[1] & 63, scales[5] & 15 | (scales[9]&3)<<4
	case 2:
		return scales[2] & 63, scales[6] & 15 | (scales[10]&3)<<4
	case 3:
		return scales[3] & 63, scales[7] & 15 | (scales[11]&3)<<4
	case 4:
		return scales[4] >> 4 | (scales[8]&0x0C)<<2, scales[0] >> 6 | (scales[4]>>4)<<2
	case 5:
		return scales[5] >> 4 | (scales[9]&0x0C)<<2, scales[1] >> 6 | (scales[5]>>4)<<2
	case 6:
		return scales[6] >> 4 | (scales[10]&0x0C)<<2, scales[2] >> 6 | (scales[6]>>4)<<2
	case 7:
		return scales[7] >> 4 | (scales[11]&0x0C)<<2, scales[3] >> 6 | (scales[7]>>4)<<2
	}
	return 0, 0
}

// Dequantize converts raw bytes in the given GGML type to []float32.
func Dequantize(raw []byte, dtype gguf.GGMLType) ([]float32, error) {
	switch dtype {
	case gguf.GGMLTypeF32:
		return DequantizeF32(raw)
	case gguf.GGMLTypeF16:
		return DequantizeF16(raw)
	case gguf.GGMLTypeQ8_0:
		return DequantizeQ8_0(raw)
	case gguf.GGMLTypeQ4_0:
		return DequantizeQ4_0(raw)
	case gguf.GGMLTypeQ4_K:
		return DequantizeQ4_K(raw)
	default:
		return nil, fmt.Errorf("dtype: dequantization not implemented for %s", dtype)
	}
}

// float16ToFloat32 converts a 16-bit IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &^= 0x400
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
	case 31:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000) // ±Inf
		}
		return math.Float32frombits(sign | 0x7FC00000) // NaN
	default:
		return math.Float32frombits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
	}
}
