#pragma once
#include <stdint.h>

// Opaque handle to the Metal device context.
typedef void* LlamaMetalDevice;

// Init/free Metal context. Returns NULL if no GPU is available.
LlamaMetalDevice llama_metal_init(void);
void             llama_metal_free(LlamaMetalDevice dev);
int              llama_metal_available(void);

// Matrix-vector multiply: y[rows] = W[rows * cols] @ x[cols]  (F32)
int llama_metal_matvec(LlamaMetalDevice dev,
                       const float* w, int rows, int cols,
                       const float* x, float* y);

// RMS normalization: out[n] = (x[n] / rms(x)) * weight[n]
int llama_metal_rmsnorm(LlamaMetalDevice dev,
                        const float* x, const float* weight, float* out,
                        int n, float eps);
