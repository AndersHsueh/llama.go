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

// DequantizeQ5_K dequantizes Q5_K blocks to float32.
// Block layout (256 elements, 176 bytes):
//   d[2]       — F16 super-scale
//   dmin[2]    — F16 super-min
//   scales[12] — 6-bit sub-scales (same packing as Q4_K)
//   qh[32]     — upper (5th) bit for all 256 values (1 per bit, 8 per byte)
//   qs[128]    — lower 4 bits for all 256 values (2 per byte)
func DequantizeQ5_K(raw []byte) ([]float32, error) {
	const blockSize = 256
	const bytesPerBlock = 176
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q5_K data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		d := float16ToFloat32(binary.LittleEndian.Uint16(raw[base:]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(raw[base+2:]))
		scales := raw[base+4 : base+16]  // 12 bytes sub-scales
		qh := raw[base+16 : base+48]     // 32 bytes high bits
		qs := raw[base+48 : base+176]    // 128 bytes low nibbles

		for i := 0; i < blockSize; i++ {
			sb := i / 32
			sc, mn := extractQ4KSubScale(scales, sb)

			lo := qs[i/2]
			var nibble uint8
			if i%2 == 0 {
				nibble = lo & 0x0F
			} else {
				nibble = lo >> 4
			}
			hi := (qh[i/8] >> uint(i%8)) & 1
			q := uint8(nibble) | (hi << 4) // 5-bit value 0..31

			out[b*blockSize+i] = d*float32(sc)*float32(q) - dmin*float32(mn)
		}
	}
	return out, nil
}


// Block layout (256 elements, 210 bytes):
//   ql[128]    — lower 4 bits for all 256 values (2 per byte)
//   qh[64]     — upper 2 bits for all 256 values (4 per byte)
//   scales[16] — int8 scale per 16 elements
//   d[2]       — F16 super-scale (at end)
func DequantizeQ6_K(raw []byte) ([]float32, error) {
	const blockSize = 256
	const bytesPerBlock = 210
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q6_K data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		ql := raw[base : base+128]
		qh := raw[base+128 : base+192]
		sc := raw[base+192 : base+208] // int8 scales
		d := float16ToFloat32(binary.LittleEndian.Uint16(raw[base+208:]))

		outBase := b * blockSize
		// Two halves of 128 elements each (n = 0, then 128).
		for half := 0; half < 2; half++ {
			qlOff := half * 64
			qhOff := half * 32
			scOff := half * 8
			nOff := half * 128
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l]&0x0F)|((qh[qhOff+l]>>0&0x03)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0F)|((qh[qhOff+l]>>2&0x03)<<4)) - 32
				q3 := int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4&0x03)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6&0x03)<<4)) - 32
				s0 := float32(int8(sc[scOff+is+0]))
				s2 := float32(int8(sc[scOff+is+2]))
				s4 := float32(int8(sc[scOff+is+4]))
				s6 := float32(int8(sc[scOff+is+6]))
				out[outBase+nOff+l+0] = d * s0 * float32(q1)
				out[outBase+nOff+l+32] = d * s2 * float32(q2)
				out[outBase+nOff+l+64] = d * s4 * float32(q3)
				out[outBase+nOff+l+96] = d * s6 * float32(q4)
			}
		}
	}
	return out, nil
}


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
	case gguf.GGMLTypeQ5_K:
		return DequantizeQ5_K(raw)
	case gguf.GGMLTypeQ6_K:
		return DequantizeQ6_K(raw)
	default:
		return nil, fmt.Errorf("dtype: dequantization not implemented for %s", dtype)
	}
}

// DequantizeInto writes dequantized values directly into dst — zero heap allocation.
// dst must hold at least (len(raw) / bytesPerBlock) * 256 float32 values.
// Returns a slice of dst resliced to the actual output length.
func DequantizeInto(raw []byte, dtype gguf.GGMLType, dst []float32) ([]float32, error) {
	switch dtype {
	case gguf.GGMLTypeQ4_K:
		return dequantizeQ4KInto(raw, dst)
	case gguf.GGMLTypeQ5_K:
		return dequantizeQ5KInto(raw, dst)
	case gguf.GGMLTypeQ6_K:
		return dequantizeQ6KInto(raw, dst)
	default:
		// Fallback: allocate + copy for less-common types.
		out, err := Dequantize(raw, dtype)
		if err != nil {
			return nil, err
		}
		if len(dst) < len(out) {
			return nil, fmt.Errorf("DequantizeInto: dst too small (%d < %d)", len(dst), len(out))
		}
		n := copy(dst, out)
		return dst[:n], nil
	}
}

// dequantizeQ4KInto writes Q4_K blocks directly into dst (zero alloc).
func dequantizeQ4KInto(raw []byte, dst []float32) ([]float32, error) {
	const blockSize = 256
	const bytesPerBlock = 144
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q4_K data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	need := nBlocks * blockSize
	if len(dst) < need {
		return nil, fmt.Errorf("DequantizeInto Q4_K: dst too small (%d < %d)", len(dst), need)
	}
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		dSuper := float16ToFloat32(binary.LittleEndian.Uint16(raw[base:]))
		dminSuper := float16ToFloat32(binary.LittleEndian.Uint16(raw[base+2:]))
		scales := raw[base+4 : base+16]
		qs := raw[base+16 : base+144]
		outBase := b * blockSize
		for sb := 0; sb < 8; sb++ {
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
				dst[outBase+idx] = d*float32(q) - dmin
			}
		}
	}
	return dst[:need], nil
}

// dequantizeQ5KInto writes Q5_K blocks directly into dst (zero alloc).
func dequantizeQ5KInto(raw []byte, dst []float32) ([]float32, error) {
	const blockSize = 256
	const bytesPerBlock = 176
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q5_K data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	need := nBlocks * blockSize
	if len(dst) < need {
		return nil, fmt.Errorf("DequantizeInto Q5_K: dst too small (%d < %d)", len(dst), need)
	}
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		d := float16ToFloat32(binary.LittleEndian.Uint16(raw[base:]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(raw[base+2:]))
		scales := raw[base+4 : base+16]
		qh := raw[base+16 : base+48]
		qs := raw[base+48 : base+176]
		outBase := b * blockSize
		for i := 0; i < blockSize; i++ {
			sb := i / 32
			sc, mn := extractQ4KSubScale(scales, sb)
			lo := qs[i/2]
			var nibble uint8
			if i%2 == 0 {
				nibble = lo & 0x0F
			} else {
				nibble = lo >> 4
			}
			hi := (qh[i/8] >> uint(i%8)) & 1
			q := nibble | (hi << 4)
			dst[outBase+i] = d*float32(sc)*float32(q) - dmin*float32(mn)
		}
	}
	return dst[:need], nil
}

// dequantizeQ6KInto writes Q6_K blocks directly into dst (zero alloc).
func dequantizeQ6KInto(raw []byte, dst []float32) ([]float32, error) {
	const blockSize = 256
	const bytesPerBlock = 210
	if len(raw)%bytesPerBlock != 0 {
		return nil, fmt.Errorf("dtype: Q6_K data length %d not divisible by %d", len(raw), bytesPerBlock)
	}
	nBlocks := len(raw) / bytesPerBlock
	need := nBlocks * blockSize
	if len(dst) < need {
		return nil, fmt.Errorf("DequantizeInto Q6_K: dst too small (%d < %d)", len(dst), need)
	}
	for b := 0; b < nBlocks; b++ {
		base := b * bytesPerBlock
		ql := raw[base : base+128]
		qh := raw[base+128 : base+192]
		sc := raw[base+192 : base+208]
		dv := float16ToFloat32(binary.LittleEndian.Uint16(raw[base+208:]))
		outBase := b * blockSize
		for half := 0; half < 2; half++ {
			qlOff := half * 64
			qhOff := half * 32
			scOff := half * 8
			nOff := half * 128
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l]&0x0F)|((qh[qhOff+l]>>0&0x03)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0F)|((qh[qhOff+l]>>2&0x03)<<4)) - 32
				q3 := int8((ql[qlOff+l]>>4)|((qh[qhOff+l]>>4&0x03)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|((qh[qhOff+l]>>6&0x03)<<4)) - 32
				s0 := float32(int8(sc[scOff+is+0]))
				s2 := float32(int8(sc[scOff+is+2]))
				s4 := float32(int8(sc[scOff+is+4]))
				s6 := float32(int8(sc[scOff+is+6]))
				dst[outBase+nOff+l+0] = dv * s0 * float32(q1)
				dst[outBase+nOff+l+32] = dv * s2 * float32(q2)
				dst[outBase+nOff+l+64] = dv * s4 * float32(q3)
				dst[outBase+nOff+l+96] = dv * s6 * float32(q4)
			}
		}
	}
	return dst[:need], nil
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
