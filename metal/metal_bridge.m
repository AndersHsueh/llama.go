// metal_bridge.m — Objective-C + Metal implementation.
// Compiled only on Darwin via CGo (see metal.go build constraint).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_bridge.h"
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// MSL shader source embedded as a string literal.
// Compiled at runtime on first llama_metal_init().
// ---------------------------------------------------------------------------
static const char* kShaderSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "// matvec_f32: each GPU thread handles one output row.\n"
    "// Parallelism = rows; each thread does a full dot-product over cols.\n"
    "kernel void matvec_f32(\n"
    "    device const float* w  [[buffer(0)]],\n"
    "    device const float* x  [[buffer(1)]],\n"
    "    device       float* y  [[buffer(2)]],\n"
    "    constant   uint&  cols [[buffer(3)]],\n"
    "    uint row [[thread_position_in_grid]])\n"
    "{\n"
    "    float sum = 0.0f;\n"
    "    uint base = row * cols;\n"
    "    for (uint j = 0; j < cols; j++) sum += w[base + j] * x[j];\n"
    "    y[row] = sum;\n"
    "}\n"
    "\n"
    "// rmsnorm: threadgroup reduction to compute rms, then normalise.\n"
    "// Dispatch with 1 threadgroup, threads_per_tg = 128 (or 256).\n"
    "kernel void rmsnorm(\n"
    "    device const float* x      [[buffer(0)]],\n"
    "    device const float* weight [[buffer(1)]],\n"
    "    device       float* out    [[buffer(2)]],\n"
    "    constant uint&  n          [[buffer(3)]],\n"
    "    constant float& eps        [[buffer(4)]],\n"
    "    uint tid    [[thread_position_in_threadgroup]],\n"
    "    uint tgSize [[threads_per_threadgroup]],\n"
    "    threadgroup float* sharedMem [[threadgroup(0)]])\n"
    "{\n"
    "    // Parallel sum-of-squares into shared memory.\n"
    "    float local_sq = 0.0f;\n"
    "    for (uint i = tid; i < n; i += tgSize) local_sq += x[i] * x[i];\n"
    "    sharedMem[tid] = local_sq;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    // Binary reduction.\n"
    "    for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {\n"
    "        if (tid < stride) sharedMem[tid] += sharedMem[tid + stride];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    float rms = sqrt(sharedMem[0] / float(n) + eps);\n"
    "    float inv_rms = 1.0f / rms;\n"
    "    for (uint i = tid; i < n; i += tgSize) out[i] = x[i] * inv_rms * weight[i];\n"
    "}\n";

// ---------------------------------------------------------------------------
// Context struct
// ---------------------------------------------------------------------------
typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLComputePipelineState> psoMatvec;
    id<MTLComputePipelineState> psoRMSNorm;
} LlamaMetalCtx;

// ---------------------------------------------------------------------------
// llama_metal_available — quick check before init
// ---------------------------------------------------------------------------
int llama_metal_available(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil ? 1 : 0;
}

// ---------------------------------------------------------------------------
// llama_metal_init
// ---------------------------------------------------------------------------
LlamaMetalDevice llama_metal_init(void) {
    LlamaMetalCtx* ctx = (LlamaMetalCtx*)calloc(1, sizeof(LlamaMetalCtx));
    if (!ctx) return NULL;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { free(ctx); return NULL; }

    ctx->queue = [ctx->device newCommandQueue];

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kShaderSource];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        NSLog(@"llama_metal: shader compile error: %@", err);
        free(ctx);
        return NULL;
    }

    id<MTLFunction> fnMatvec = [lib newFunctionWithName:@"matvec_f32"];
    id<MTLFunction> fnRMS    = [lib newFunctionWithName:@"rmsnorm"];

    ctx->psoMatvec  = [ctx->device newComputePipelineStateWithFunction:fnMatvec error:&err];
    ctx->psoRMSNorm = [ctx->device newComputePipelineStateWithFunction:fnRMS    error:&err];

    if (!ctx->psoMatvec || !ctx->psoRMSNorm) {
        NSLog(@"llama_metal: PSO creation error: %@", err);
        free(ctx);
        return NULL;
    }

    NSLog(@"llama_metal: initialised on %@", [ctx->device name]);
    return (LlamaMetalDevice)ctx;
}

// ---------------------------------------------------------------------------
// llama_metal_free
// ---------------------------------------------------------------------------
void llama_metal_free(LlamaMetalDevice dev) {
    if (dev) free(dev);
}

// ---------------------------------------------------------------------------
// llama_metal_matvec — y[rows] = W[rows*cols] @ x[cols]
// ---------------------------------------------------------------------------
int llama_metal_matvec(LlamaMetalDevice dev,
                       const float* w, int rows, int cols,
                       const float* x, float* y)
{
    LlamaMetalCtx* ctx = (LlamaMetalCtx*)dev;

    @autoreleasepool {
        // newBufferWithBytesNoCopy: wraps existing CPU memory without copying.
        // On Apple Silicon (unified memory) this is truly zero-copy CPU↔GPU.
        // The page_size alignment requirement is met by Go's mmap-based allocator
        // for all large tensors (Wq/Wk/Wv/Wo are all ≥ 4 KB).
        NSUInteger wLen = (NSUInteger)(rows * cols) * sizeof(float);
        NSUInteger xLen = (NSUInteger)cols * sizeof(float);
        NSUInteger yLen = (NSUInteger)rows * sizeof(float);

        id<MTLBuffer> wBuf = [ctx->device newBufferWithBytesNoCopy:(void*)w
                                                            length:wLen
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
        id<MTLBuffer> xBuf = [ctx->device newBufferWithBytesNoCopy:(void*)x
                                                            length:xLen
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
        // Fallback for small x (< page size): must copy since NoCopy requires page-aligned.
        if (xBuf == nil) {
            xBuf = [ctx->device newBufferWithBytes:x length:xLen options:MTLResourceStorageModeShared];
        }
        if (wBuf == nil) {
            wBuf = [ctx->device newBufferWithBytes:w length:wLen options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> yBuf = [ctx->device newBufferWithLength:yLen options:MTLResourceStorageModeShared];
        uint32_t cols32 = (uint32_t)cols;
        id<MTLBuffer> cBuf = [ctx->device newBufferWithBytes:&cols32
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer>         cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->psoMatvec];
        [enc setBuffer:wBuf offset:0 atIndex:0];
        [enc setBuffer:xBuf offset:0 atIndex:1];
        [enc setBuffer:yBuf offset:0 atIndex:2];
        [enc setBuffer:cBuf offset:0 atIndex:3];

        NSUInteger tgW = ctx->psoMatvec.maxTotalThreadsPerThreadgroup;
        if (tgW > (NSUInteger)rows) tgW = (NSUInteger)rows;
        [enc dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tgW, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(y, [yBuf contents], yLen);
    } // @autoreleasepool — releases all Metal buffers immediately
    return 0;
}

// ---------------------------------------------------------------------------
// llama_metal_rmsnorm — out[n] = (x[n] / rms(x)) * weight[n]
// ---------------------------------------------------------------------------
int llama_metal_rmsnorm(LlamaMetalDevice dev,
                        const float* x, const float* weight, float* out,
                        int n, float eps)
{
    LlamaMetalCtx* ctx = (LlamaMetalCtx*)dev;

    @autoreleasepool {
        size_t sz = (size_t)n * sizeof(float);
        id<MTLBuffer> xBuf = [ctx->device newBufferWithBytes:x      length:sz options:MTLResourceStorageModeShared];
        id<MTLBuffer> wBuf = [ctx->device newBufferWithBytes:weight  length:sz options:MTLResourceStorageModeShared];
        id<MTLBuffer> oBuf = [ctx->device newBufferWithLength:sz             options:MTLResourceStorageModeShared];

        uint32_t n32 = (uint32_t)n;
        id<MTLBuffer> nBuf = [ctx->device newBufferWithBytes:&n32  length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> eBuf = [ctx->device newBufferWithBytes:&eps  length:sizeof(float)    options:MTLResourceStorageModeShared];

        // Use 128 threads per threadgroup for the reduction.
        NSUInteger tgSize = 128;
        if (tgSize > ctx->psoRMSNorm.maxTotalThreadsPerThreadgroup)
            tgSize = ctx->psoRMSNorm.maxTotalThreadsPerThreadgroup;

        id<MTLCommandBuffer>         cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->psoRMSNorm];
        [enc setBuffer:xBuf offset:0 atIndex:0];
        [enc setBuffer:wBuf offset:0 atIndex:1];
        [enc setBuffer:oBuf offset:0 atIndex:2];
        [enc setBuffer:nBuf offset:0 atIndex:3];
        [enc setBuffer:eBuf offset:0 atIndex:4];
        // threadgroup(0) = shared memory for reduction.
        [enc setThreadgroupMemoryLength:tgSize * sizeof(float) atIndex:0];

        // 1 threadgroup of tgSize threads — enough for n≤4096.
        [enc dispatchThreads:MTLSizeMake(tgSize, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, [oBuf contents], sz);
    } // @autoreleasepool — releases all Metal buffers immediately
    return 0;
}
