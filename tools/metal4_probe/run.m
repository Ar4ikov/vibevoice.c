/* This runtime's tile GEMM against the same thing on Metal 4 tensor ops. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
static double ms(void) { return [NSDate timeIntervalSinceReferenceDate]*1000.0; }

int main(int argc, const char** argv) { @autoreleasepool {
    const int M = argc>1?atoi(argv[1]):512, N = argc>2?atoi(argv[2]):18944,
              K = argc>3?atoi(argv[3]):3584;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("%s  M=%d N=%d K=%d  (%.1f GFLOP a run)\n", dev.name.UTF8String,
           M, N, K, 2.0*M*N*K/1e9);
    NSError* e = nil;
    NSString* path = @"/private/tmp/claude-501/-Users-ar4ikov-vibevoice-c/daa7a3ba-db32-4b48-8d4c-c849230d0803/scratchpad/mtl4/gemm.metal";
    NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&e];
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = (MTLLanguageVersion)((4 << 16) + 0);
    o.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:&e];
    if (!lib) { printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }

    id<MTLBuffer> A = [dev newBufferWithLength:(NSUInteger)M*K*2 options:0];
    id<MTLBuffer> B = [dev newBufferWithLength:(NSUInteger)N*K*2 options:0];
    id<MTLBuffer> C = [dev newBufferWithLength:(NSUInteger)M*N*4 options:0];
    __fp16* a = (__fp16*)A.contents; __fp16* b = (__fp16*)B.contents;
    for (long i = 0; i < (long)M*K; i++) a[i] = (__fp16)(((i % 13) - 6) * 0.05f);
    for (long i = 0; i < (long)N*K; i++) b[i] = (__fp16)(((i % 7) - 3) * 0.02f);
    struct { int M, N, K; float alpha, beta; } pm = { M, N, K, 1.0f, 0.0f };
    struct { unsigned M, N, K; } p4 = { (unsigned)M, (unsigned)N, (unsigned)K };

    const char* names[] = { "vv_gemm_tn", "t4_64x32", "t4_64x64", "t4_32x32" };
    const int bm[] = { 64, 64, 64, 32 }, bn[] = { 64, 32, 64, 32 }, mine[] = { 1, 0, 0, 0 };
    id<MTLCommandQueue> q = [dev newCommandQueue];

    for (int v = 0; v < 4; v++) {
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:names[v]]];
        id<MTLComputePipelineState> ps = f ? [dev newComputePipelineStateWithFunction:f error:&e] : nil;
        if (!ps) { printf("  %-12s unavailable\n", names[v]); continue; }
        MTLSize tgs = mine[v] ? MTLSizeMake((N+bn[v]-1)/bn[v], (M+bm[v]-1)/bm[v], 1)
                              : MTLSizeMake((M+bm[v]-1)/bm[v], (N+bn[v]-1)/bn[v], 1);
        MTLSize tpt = MTLSizeMake(mine[v] ? 128 : ps.threadExecutionWidth*4, 1, 1);
        const int is_mine = mine[v];
        void (^dispatch)(void) = ^{
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            if (is_mine) {
                [enc setBytes:&pm length:sizeof(pm) atIndex:0];
                [enc setBuffer:A offset:0 atIndex:1]; [enc setBuffer:B offset:0 atIndex:2];
                [enc setBuffer:C offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:4*32*(32+4)*4 atIndex:0];
            } else {
                [enc setBuffer:A offset:0 atIndex:0]; [enc setBuffer:B offset:0 atIndex:1];
                [enc setBuffer:C offset:0 atIndex:2];
                [enc setBytes:&p4 length:sizeof(p4) atIndex:3];
            }
            [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpt];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        };
        /* correctness: one dispatch into a zeroed C, eight spot checks */
        memset(C.contents, 0, (size_t)M*N*4);
        dispatch();
        int bad = 0; double worst = 0;
        for (int t = 0; t < 8; t++) {
            const int m = (t*61) % M, n = (t*811) % N;
            double r = 0;
            for (int k = 0; k < K; k++) r += (double)a[(long)m*K+k] * (double)b[(long)n*K+k];
            const double g = is_mine ? (double)((const __fp16*)C.contents)[(long)m*N+n]
                                     : (double)((const float*)C.contents)[(long)m*N+n];
            const double err = fabs(g-r) / (fabs(r) + 1e-3);
            if (err > worst) worst = err;
            if (err > 0.02) bad++;
        }
        /* timing: best mean of three rounds of ten */
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            const double t0 = ms();
            for (int i = 0; i < 10; i++) dispatch();
            const double per = (ms()-t0)/10.0;
            if (per < best) best = per;
        }
        printf("  %-12s %8.3f ms  %6.2f TFLOP/s   %-16s (worst rel %.1e)\n",
               names[v], best, 2.0*M*N*K/(best*1e9),
               bad ? "WRONG" : "matches the CPU", worst);
    }
} return 0; }
