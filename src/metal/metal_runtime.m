/**
 * @file metal_runtime.m
 * @brief The device half of device.h on Apple Silicon: memory, streams,
 *        events, graph capture, copies, and the launch the op wrappers use.
 *
 * ## Memory
 *
 * Every allocation is a shared-storage MTLBuffer, and the "device pointer"
 * handed out is its CPU address. The GPU and the CPU see the same bytes, so
 * a host copy into device memory is a memcpy once the stream is idle, and a
 * device pointer can be offset, split and passed around exactly as the CUDA
 * path does. When a pointer reaches a kernel the runtime finds the buffer
 * that contains it (a sorted table, binary search) and binds (buffer,
 * offset). Pinned host memory is the same kind of buffer, so an H2D copy out
 * of it runs on the GPU in stream order, as a DMA would on CUDA.
 *
 * ## Streams
 *
 * A stream is a command queue plus the command buffer and compute encoder
 * currently being filled. Launches encode into it; a sync, an event record
 * or a device-to-host copy commits it. Serial dispatch keeps each launch
 * ordered after the previous one, which is the CUDA stream contract.
 * Command buffers retain what they use, so freeing a buffer that queued
 * work still reads is safe: its memory goes when the GPU is done with it,
 * where cudaFree would have synchronised the device instead.
 *
 * The NULL stream is CUDA's legacy default stream: work on it waits for
 * every other stream and completes before the call returns.
 *
 * ## Graphs
 *
 * A capture records the launches issued to a stream -- pipeline, buffers
 * resolved to (buffer, offset), parameter bytes, grid -- and a replay
 * encodes them again in one go. Anything that would need the host in the
 * middle (a sync, an event, a pageable copy) fails the capture, and the
 * caller falls back to launching each kernel, as it does on CUDA.
 *
 * ## Process-wide state
 *
 * The device, its compiled kernels and the buffer table are what the CUDA
 * runtime keeps inside the driver. Here they are one object created on
 * first use and never torn down.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <os/lock.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"
#include "metal_internal.h"

/* The kernels, embedded at build time (cmake/EmbedMetal.cmake). */
extern const char   vv_metal_source[];
extern const size_t vv_metal_source_len;

/*
 * When to commit a stream's command buffer. Early enough that the GPU
 * starts while the rest is still being encoded -- and, more importantly,
 * that no single buffer runs long enough for the system's GPU watchdog:
 * a speech encoder pass over 16 minutes of audio filled one with about a
 * minute of work and was killed ("Caused GPU Timeout Error"). The
 * threadgroup count is the cheap proxy for how much work that is; 64K of
 * them is well under a second even for the slowest kernels here (a
 * 2048-row NF4 GEMM runs 9.5K in 140 ms).
 */
#define MTL_FLUSH_EVERY 128
#define MTL_FLUSH_GROUPS (64u * 1024u)

/*
 * How much submitted work a stream may have outstanding. The watchdog that
 * killed the encoder counts from submission, not from when a buffer starts
 * running, so a queue seconds deep times out however short each buffer is.
 * The bound is on the work itself, not on the number of buffers: a decode
 * step is ~450 tiny launches across four buffers and must never stall on
 * one of them, while one encoder buffer alone can be a second of GPU.
 * Committing past the budget waits for the oldest buffer to finish, which
 * is also the backpressure that stops a long file's encoder from running
 * minutes ahead of the GPU.
 */
#define MTL_INFLIGHT_GROUPS (192u * 1024u)
#define MTL_MAX_IN_FLIGHT 8

/*
 * Allocations from this size up are anonymous mappings wrapped as buffers:
 * their pages are zero and cost no memory until something touches them.
 * A 32K-position KV cache is 1.8 GB, of which a 30 s file uses 2%; on a
 * machine whose GPU memory is its RAM that difference is the difference
 * between running and swapping.
 */
#define MTL_MMAP_MIN (64u * 1024u)
#define MTL_PAGE     16384u

/* ─── Allocation table ──────────────────────────────────────────────────── */

typedef struct mtl_alloc {
    uintptr_t base;
    size_t    len;
    void*     buf;      /* id<MTLBuffer>, retained by the table */
    int       pinned;
    int       fresh;    /* zero pages nothing has written: a zero fill
                           of them is already done                     */
} mtl_alloc_t;

/* ─── Recorded launches (graph capture) ─────────────────────────────────── */

typedef struct mtl_cmd {
    const char* name;                     /* the MSL function, for traces  */
    void*    pso;                         /* id<MTLComputePipelineState>, unretained */
    void*    buf[VV_MTL_MAX_BUFS];        /* id<MTLBuffer>, unretained               */
    size_t   off[VV_MTL_MAX_BUFS];
    int      nbufs;
    void*    use[VV_MTL_MAX_USES];
    int      nuses;
    uint8_t* params;                      /* vv_alloc'd copy                         */
    size_t   params_size;
    uint32_t grid[3], block[3], tg_mem;
} mtl_cmd_t;

typedef struct mtl_graph {
    mtl_cmd_t* cmds;
    int        n, cap;
    int        broken;      /* something uncapturable happened */
} mtl_graph_t;

/* ─── Kernel table ──────────────────────────────────────────────────────── */

#define MTL_PSO_SLOTS 1024

typedef struct mtl_pso_slot {
    const char* name;       /* vv_alloc'd copy */
    void*       pso;        /* id<MTLComputePipelineState>, retained */
} mtl_pso_slot_t;

/* ─── Streams and events ────────────────────────────────────────────────── */

@interface VVStream : NSObject {
@public
    id<MTLCommandQueue>          queue;
    id<MTLCommandBuffer>         cb;      /* being filled                 */
    id<MTLComputeCommandEncoder> enc;     /* open on cb, or nil           */
    id<MTLCommandBuffer>         last;    /* last committed               */
    id<MTLCommandBuffer>         queued[MTL_MAX_IN_FLIGHT];
    uint64_t                     queued_groups[MTL_MAX_IN_FLIGHT];
    int                          n_queued;
    uint64_t                     groups_out;/* groups in queued[]          */
    os_unfair_lock               lock;
    int                          n_disp;  /* launches in cb               */
    uint64_t                     n_groups;/* threadgroups in cb            */
    mtl_graph_t*                 rec;     /* capturing into, or NULL      */
    const char*                  last_kernel; /* for VV_METAL_TRACE        */
    vv_status_t                  error;   /* sticky, from a failed cb     */
}
@end
@implementation VVStream
@end

@interface VVEvent : NSObject {
@public
    id<MTLSharedEvent>   ev;
    uint64_t             value;     /* last recorded */
    id<MTLCommandBuffer> signaller; /* the cb that signals `value` */
    os_unfair_lock       lock;
}
@end
@implementation VVEvent
@end

/* ─── The device ────────────────────────────────────────────────────────── */

@interface VVMetal : NSObject {
@public
    id<MTLDevice>     device;       /* nil: no usable GPU, host memory only */
    id<MTLLibrary>    library;
    NSString*         lib_error;
    bool              lib_tried;
    int               cores;
    char              name[128];
    VVStream*         null_stream;
    NSHashTable*      streams;      /* weak, for device-wide syncs */
    os_unfair_lock    streams_lock;

    mtl_alloc_t*      allocs;
    int               n_allocs, cap_allocs;
    size_t            alloc_bytes;  /* live allocations, for the budget */
    os_unfair_lock    alloc_lock;

    mtl_pso_slot_t    pso[MTL_PSO_SLOTS];
    os_unfair_lock    pso_lock;
    bool              sync_each;    /* VV_METAL_SYNC=1: wait after every launch */
    double            trace_ms;     /* VV_METAL_TRACE=<ms>: report long buffers */
    int               family;       /* Apple<N> GPU family                      */
    bool              metal4;       /* Metal 4: tensor ops (M5 and up)          */
}
@end
@implementation VVMetal
@end

static VVMetal* g_mtl;

/** @brief GPU core count from the IORegistry (AGXAccelerator). */
static int query_core_count(void) {
    int cores = 0;
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"),
                                     &it) != KERN_SUCCESS)
        return 0;
    io_object_t svc;
    while (!cores && (svc = IOIteratorNext(it))) {
        CFTypeRef v = IORegistryEntrySearchCFProperty(
            svc, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
            kIORegistryIterateRecursively);
        if (v) {
            if (CFGetTypeID(v) == CFNumberGetTypeID())
                CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &cores);
            CFRelease(v);
        }
        IOObjectRelease(svc);
    }
    IOObjectRelease(it);
    return cores;
}

/**
 * @brief Memory the system could still give this process, in bytes.
 *
 * On unified memory the GPU's memory is the machine's, so what the device
 * can still take is bounded by what the OS has, not only by the working-set
 * limit: free, speculative and purgeable pages, plus the inactive ones it
 * would evict first. A 7B model that fits Metal's limit but not the machine
 * does not fail -- it thrashes, and the compressor turns a 3 tok/s decode
 * into the wrong answer about what fits.
 */
static size_t host_available_bytes(void) {
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) page = 16384;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    /* Unanswerable means "do not narrow the answer", not "nothing is free". */
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm, &count) != KERN_SUCCESS)
        return SIZE_MAX;
    const uint64_t pages = (uint64_t)vm.free_count + vm.inactive_count +
                           vm.purgeable_count + vm.speculative_count;
    return (size_t)(pages * (uint64_t)page);
}

static VVMetal* mtl(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        @autoreleasepool {
            VVMetal* m = [VVMetal new];
            m->streams_lock = OS_UNFAIR_LOCK_INIT;
            m->alloc_lock = OS_UNFAIR_LOCK_INIT;
            m->pso_lock = OS_UNFAIR_LOCK_INIT;
            m->streams = [NSHashTable weakObjectsHashTable];
            const char* e = getenv("VV_METAL_SYNC");
            m->sync_each = e && e[0] == '1';
            const char* tr = getenv("VV_METAL_TRACE");
            m->trace_ms = (tr && tr[0]) ? atof(tr) : 0.0;

            id<MTLDevice> d = MTLCreateSystemDefaultDevice();
            const char* off = getenv("VV_METAL_DISABLE");
            if (off && off[0] == '1') d = nil;
            /*
             * The kernels need simdgroup matrices and buffers that hold GPU
             * addresses: Apple7 (M1) and Metal 3. Anything older -- an Intel
             * Mac's AMD or Intel GPU -- runs the CPU path, as a machine with
             * no NVIDIA driver does in the CUDA build.
             */
            if (d && !([d supportsFamily:MTLGPUFamilyApple7] &&
                       [d supportsFamily:MTLGPUFamilyMetal3])) {
                VV_LOG_W("metal: %s lacks Apple7/Metal 3; using the CPU path",
                         d.name.UTF8String);
                d = nil;
            }
            m->device = d;
            if (d) {
                snprintf(m->name, sizeof(m->name), "%s", d.name.UTF8String);
                m->cores = query_core_count();
                if (m->cores <= 0) m->cores = 8;
                /*
                 * Which generation, and whether Metal 4's tensor operations
                 * are available. They are the door to M5's per-core neural
                 * accelerators, and they are offered on Apple9 too, where
                 * they reach the same band as the simdgroup-matrix kernels
                 * here because they run on the same units (docs/ANE.md has
                 * the measurements). Logged, not acted on: the hardware
                 * that would make the switch pay is not here to measure.
                 */
                int fam = 7;
                for (int f = 10; f >= 7; f--)
                    if ([d supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + f - 1)]) {
                        fam = f; break;
                    }
                m->family = fam;
                m->metal4 = [d supportsFamily:MTLGPUFamilyMetal4];
                VV_LOG_D("metal: %s, GPU family Apple%d, %d cores, Metal 4 %s",
                         m->name, fam, m->cores,
                         m->metal4 ? "yes (tensor operations available)" : "no");
                VVStream* s = [VVStream new];
                s->queue = [d newCommandQueue];
                s->lock = OS_UNFAIR_LOCK_INIT;
                s->queue.label = @"vv.null";
                m->null_stream = s;
            }
            g_mtl = m;
        }
    });
    return g_mtl;
}

static bool usable(void) { return mtl()->device != nil; }

int vv_mtl_core_count(void) { return usable() ? mtl()->cores : 0; }

/* ─── Allocation table ──────────────────────────────────────────────────── */

/** @brief Index of the allocation containing p, or -1. Caller holds the lock. */
static int alloc_find_locked(VVMetal* m, uintptr_t p) {
    int lo = 0, hi = m->n_allocs - 1, at = -1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (m->allocs[mid].base <= p) { at = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    if (at < 0) return -1;
    const mtl_alloc_t* a = &m->allocs[at];
    return p < a->base + a->len ? at : -1;
}

static vv_status_t alloc_insert(VVMetal* m, id<MTLBuffer> b, size_t len,
                                int pinned, int fresh) {
    const uintptr_t base = (uintptr_t)b.contents;
    os_unfair_lock_lock(&m->alloc_lock);
    if (m->n_allocs == m->cap_allocs) {
        const int cap = m->cap_allocs ? m->cap_allocs * 2 : 256;
        mtl_alloc_t* grown = (mtl_alloc_t*)vv_realloc(
            m->allocs, (size_t)cap * sizeof(mtl_alloc_t));
        if (!grown) { os_unfair_lock_unlock(&m->alloc_lock); return VV_ERR_OUT_OF_MEMORY; }
        m->allocs = grown;
        m->cap_allocs = cap;
    }
    int at = m->n_allocs;
    while (at > 0 && m->allocs[at - 1].base > base) {
        m->allocs[at] = m->allocs[at - 1];
        at--;
    }
    m->allocs[at].base = base;
    m->allocs[at].len = len;
    m->allocs[at].buf = (void*)CFBridgingRetain(b);
    m->allocs[at].pinned = pinned;
    m->allocs[at].fresh = fresh;
    m->n_allocs++;
    m->alloc_bytes += len;
    os_unfair_lock_unlock(&m->alloc_lock);
    return VV_OK;
}

/** @brief Drop the allocation starting exactly at p; false if there is none. */
static bool alloc_remove(VVMetal* m, void* p, int pinned) {
    void* buf = NULL;
    os_unfair_lock_lock(&m->alloc_lock);
    const int at = alloc_find_locked(m, (uintptr_t)p);
    if (at >= 0 && m->allocs[at].base == (uintptr_t)p &&
        m->allocs[at].pinned == pinned) {
        buf = m->allocs[at].buf;
        m->alloc_bytes -= m->allocs[at].len;
        memmove(&m->allocs[at], &m->allocs[at + 1],
                (size_t)(m->n_allocs - at - 1) * sizeof(mtl_alloc_t));
        m->n_allocs--;
    }
    os_unfair_lock_unlock(&m->alloc_lock);
    if (!buf) return false;
    /* Queued command buffers hold their own references; this is ours. */
    CFRelease(buf);
    return true;
}

/**
 * @brief (buffer, offset) of a device pointer; buffer nil if unknown.
 *
 * Resolving a pointer is how anything reaches a buffer -- a kernel binding
 * it, a copy into it -- so it also ends the buffer's "fresh" state.
 */
static id<MTLBuffer> resolve(const void* p, size_t* off) {
    VVMetal* m = mtl();
    id<MTLBuffer> b = nil;
    os_unfair_lock_lock(&m->alloc_lock);
    const int at = alloc_find_locked(m, (uintptr_t)p);
    if (at >= 0) {
        b = (__bridge id<MTLBuffer>)m->allocs[at].buf;
        *off = (uintptr_t)p - m->allocs[at].base;
        m->allocs[at].fresh = 0;
    }
    os_unfair_lock_unlock(&m->alloc_lock);
    return b;
}

/** @brief The host is about to write through p: the buffer is not fresh. */
static void touch(const void* p) {
    VVMetal* m = mtl();
    os_unfair_lock_lock(&m->alloc_lock);
    const int at = alloc_find_locked(m, (uintptr_t)p);
    if (at >= 0) m->allocs[at].fresh = 0;
    os_unfair_lock_unlock(&m->alloc_lock);
}

/** @brief Whether [p, p + n) lies in an allocation nothing has written. */
static bool still_zero(const void* p, size_t n) {
    VVMetal* m = mtl();
    bool zero = false;
    os_unfair_lock_lock(&m->alloc_lock);
    const int at = alloc_find_locked(m, (uintptr_t)p);
    if (at >= 0) {
        const mtl_alloc_t* a = &m->allocs[at];
        zero = a->fresh && (uintptr_t)p + n <= a->base + a->len;
    }
    os_unfair_lock_unlock(&m->alloc_lock);
    return zero;
}

int vv_mtl_is_device_ptr(const void* p) {
    if (!p || !usable()) return 0;
    VVMetal* m = mtl();
    os_unfair_lock_lock(&m->alloc_lock);
    const int at = alloc_find_locked(m, (uintptr_t)p);
    os_unfair_lock_unlock(&m->alloc_lock);
    return at >= 0;
}

uint64_t vv_mtl_addr(const void* p) {
    if (!p) return 0;
    size_t off = 0;
    id<MTLBuffer> b = resolve(p, &off);
    return b ? b.gpuAddress + off : 0;
}

/* ─── Kernels ───────────────────────────────────────────────────────────── */

static bool library_ready(VVMetal* m) {
    if (m->library) return true;
    os_unfair_lock_lock(&m->pso_lock);
    if (!m->library && !m->lib_tried) {
        m->lib_tried = true;
        @autoreleasepool {
            const double t0 = vv_time_ms();
            NSString* src = [[NSString alloc]
                initWithBytes:vv_metal_source length:vv_metal_source_len
                     encoding:NSUTF8StringEncoding];
            MTLCompileOptions* o = [MTLCompileOptions new];
            /*
             * Safe math: IEEE results, precise transcendentals, and a*b+c
             * fused only within one expression -- the contraction Apple
             * clang applies to the CPU kernels, so the two agree where the
             * tests compare them bit for bit. Kernels that want the fast
             * functions call fast:: explicitly.
             */
            if (@available(macOS 15.0, *)) {
                o.mathMode = MTLMathModeSafe;
                o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                o.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            o.languageVersion = MTLLanguageVersion3_0;
            NSError* err = nil;
            m->library = [m->device newLibraryWithSource:src options:o error:&err];
            if (!m->library) {
                m->lib_error = err.localizedDescription;
                VV_LOG_E("metal: kernel library failed to compile:\n%s",
                         err.localizedDescription.UTF8String);
            } else {
                VV_LOG_D("metal: kernel library compiled in %.0f ms",
                         vv_time_ms() - t0);
            }
        }
    }
    const bool ok = m->library != nil;
    os_unfair_lock_unlock(&m->pso_lock);
    return ok;
}

static uint32_t fnv1a(const char* s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}

/** @brief The pipeline for an MSL function, built on first use. */
static id<MTLComputePipelineState> pso_get(const char* name) {
    VVMetal* m = mtl();
    if (!library_ready(m)) return nil;
    uint32_t i = fnv1a(name) & (MTL_PSO_SLOTS - 1);
    os_unfair_lock_lock(&m->pso_lock);
    for (int probe = 0; probe < MTL_PSO_SLOTS; probe++) {
        mtl_pso_slot_t* s = &m->pso[i];
        if (!s->name) break;
        if (strcmp(s->name, name) == 0) {
            id<MTLComputePipelineState> p = (__bridge id<MTLComputePipelineState>)s->pso;
            os_unfair_lock_unlock(&m->pso_lock);
            return p;
        }
        i = (i + 1) & (MTL_PSO_SLOTS - 1);
    }
    id<MTLComputePipelineState> p = nil;
    @autoreleasepool {
        NSError* err = nil;
        id<MTLFunction> f = [m->library newFunctionWithName:
                             [NSString stringWithUTF8String:name]];
        if (f) p = [m->device newComputePipelineStateWithFunction:f error:&err];
        if (!p)
            VV_LOG_E("metal: no pipeline for kernel '%s'%s%s", name,
                     err ? ": " : " (not in the library)",
                     err ? err.localizedDescription.UTF8String : "");
    }
    if (p && !m->pso[i].name) {
        const size_t n = strlen(name) + 1;
        char* copy = (char*)vv_alloc(n);
        if (copy) {
            memcpy(copy, name, n);
            m->pso[i].pso = (void*)CFBridgingRetain(p);
            m->pso[i].name = copy;
        }
    }
    os_unfair_lock_unlock(&m->pso_lock);
    return p;
}

/* ─── Streams: encoding and flushing ────────────────────────────────────── */

static VVStream* stream_of(void* s) {
    return s ? (__bridge VVStream*)s : mtl()->null_stream;
}

static void end_encoder(VVStream* s) {
    if (s->enc) { [s->enc endEncoding]; s->enc = nil; }
}

static id<MTLCommandBuffer> current_cb(VVStream* s) {
    if (!s->cb) {
        s->cb = [s->queue commandBuffer];
        s->n_disp = 0;
        s->n_groups = 0;
    }
    return s->cb;
}

static id<MTLComputeCommandEncoder> encoder(VVStream* s) {
    if (!s->enc)
        s->enc = [current_cb(s) computeCommandEncoderWithDispatchType:
                  MTLDispatchTypeSerial];
    return s->enc;
}

/**
 * @brief Drop the queued buffers that have finished, oldest first.
 *
 * Status is a non-blocking read, so this is how the stream learns the GPU
 * caught up without ever waiting for it.
 */
static void retire_finished(VVStream* s) {
    int done = 0;
    while (done < s->n_queued) {
        const MTLCommandBufferStatus st = s->queued[done].status;
        if (st != MTLCommandBufferStatusCompleted &&
            st != MTLCommandBufferStatusError) break;
        if (st == MTLCommandBufferStatusError && s->error == VV_OK) {
            VV_LOG_E("metal: command buffer failed: %s",
                     s->queued[done].error.localizedDescription.UTF8String);
            s->error = VV_ERR_CUDA;
        }
        s->groups_out -= s->queued_groups[done];
        s->queued[done] = nil;
        done++;
    }
    if (!done) return;
    for (int i = done; i < s->n_queued; i++) {
        s->queued[i - done] = s->queued[i];
        s->queued_groups[i - done] = s->queued_groups[i];
        s->queued[i] = nil;
    }
    s->n_queued -= done;
}

/** @brief Commit whatever is being filled. Caller holds the stream lock. */
static void flush_locked(VVStream* s) {
    if (!s->cb) return;
    end_encoder(s);
    const double trace = mtl()->trace_ms;
    if (trace > 0.0) {
        const int nd = s->n_disp;
        const uint64_t ng = s->n_groups;
        const char* last = s->last_kernel;
        [s->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            if (ms >= trace)
                VV_LOG_I("metal: command buffer %8.1f ms, %d launches, "
                         "%llu groups, last '%s'", ms, nd,
                         (unsigned long long)ng, last ? last : "?");
        }];
    }
    [s->cb commit];
    s->last = s->cb;
    const uint64_t ng_cb = s->n_groups;
    /* Forget the ones that already finished; asking costs nothing. */
    retire_finished(s);
    /* Then wait, oldest first, until this one fits within the budget. */
    while (s->n_queued == MTL_MAX_IN_FLIGHT ||
           (s->n_queued > 0 && s->groups_out + ng_cb > MTL_INFLIGHT_GROUPS)) {
        [s->queued[0] waitUntilCompleted];
        retire_finished(s);
    }
    s->queued_groups[s->n_queued] = ng_cb;
    s->queued[s->n_queued++] = s->cb;
    s->groups_out += ng_cb;
    s->cb = nil;
    s->n_disp = 0;
    s->n_groups = 0;
}

/** @brief Commit and wait; reports a GPU-side failure once. */
static vv_status_t drain_locked(VVStream* s) {
    flush_locked(s);
    for (int i = 0; i < s->n_queued; i++) s->queued[i] = nil;
    s->n_queued = 0;
    s->groups_out = 0;
    id<MTLCommandBuffer> last = s->last;
    if (last) {
        [last waitUntilCompleted];
        if (last.status == MTLCommandBufferStatusError) {
            VV_LOG_E("metal: command buffer failed: %s",
                     last.error.localizedDescription.UTF8String);
            s->error = VV_ERR_CUDA;
        }
        s->last = nil;
    }
    const vv_status_t e = s->error;
    s->error = VV_OK;
    return e;
}

/** @brief Whether nothing is queued or running on the stream. */
static bool idle_locked(VVStream* s) {
    if (s->cb) return false;
    if (!s->last) return true;
    const MTLCommandBufferStatus st = s->last.status;
    return st == MTLCommandBufferStatusCompleted ||
           st == MTLCommandBufferStatusError;
}

/** @brief Every stream's queued work, finished: cudaDeviceSynchronize. */
static vv_status_t device_sync(void) {
    VVMetal* m = mtl();
    if (!m->device) return VV_OK;
    NSArray* all;
    os_unfair_lock_lock(&m->streams_lock);
    all = m->streams.allObjects;
    os_unfair_lock_unlock(&m->streams_lock);
    vv_status_t st = VV_OK;
    for (VVStream* s in all) {
        os_unfair_lock_lock(&s->lock);
        const vv_status_t e = drain_locked(s);
        os_unfair_lock_unlock(&s->lock);
        if (e != VV_OK) st = e;
    }
    VVStream* n = m->null_stream;
    os_unfair_lock_lock(&n->lock);
    const vv_status_t e = drain_locked(n);
    os_unfair_lock_unlock(&n->lock);
    return st != VV_OK ? st : e;
}

/* ─── Launch ────────────────────────────────────────────────────────────── */

static void encode_cmd(VVStream* s, const mtl_cmd_t* c) {
    id<MTLComputeCommandEncoder> e = encoder(s);
    [e setComputePipelineState:(__bridge id<MTLComputePipelineState>)c->pso];
    [e setBytes:c->params length:c->params_size atIndex:0];
    for (int i = 0; i < c->nbufs; i++)
        [e setBuffer:(__bridge id<MTLBuffer>)c->buf[i] offset:c->off[i]
             atIndex:(NSUInteger)i + 1];
    for (int i = 0; i < c->nuses; i++)
        [e useResource:(__bridge id<MTLBuffer>)c->use[i]
                 usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    if (c->tg_mem)
        [e setThreadgroupMemoryLength:(c->tg_mem + 15u) & ~15u atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(c->grid[0], c->grid[1], c->grid[2])
      threadsPerThreadgroup:MTLSizeMake(c->block[0], c->block[1], c->block[2])];
    s->n_disp++;
    s->n_groups += (uint64_t)c->grid[0] * c->grid[1] * c->grid[2];
    s->last_kernel = c->name;
}

static vv_status_t graph_append(mtl_graph_t* g, const mtl_cmd_t* c) {
    if (g->n == g->cap) {
        const int cap = g->cap ? g->cap * 2 : 512;
        mtl_cmd_t* grown = (mtl_cmd_t*)vv_realloc(g->cmds,
                                                   (size_t)cap * sizeof(mtl_cmd_t));
        if (!grown) return VV_ERR_OUT_OF_MEMORY;
        g->cmds = grown;
        g->cap = cap;
    }
    mtl_cmd_t* d = &g->cmds[g->n];
    *d = *c;
    d->params = (uint8_t*)vv_alloc(c->params_size ? c->params_size : 1);
    if (!d->params) return VV_ERR_OUT_OF_MEMORY;
    memcpy(d->params, c->params, c->params_size);
    /* A replay must not outlive what it binds, whatever the caller frees. */
    for (int i = 0; i < d->nbufs; i++) if (d->buf[i]) CFRetain(d->buf[i]);
    for (int i = 0; i < d->nuses; i++) CFRetain(d->use[i]);
    g->n++;
    return VV_OK;
}

vv_status_t vv_mtl_run(void* stream, const vv_mtl_launch_t* l) {
    if (!l || !l->kernel) return VV_ERR_NULL_PTR;
    if (!usable()) return VV_ERR_UNSUPPORTED;
    if (l->nbufs > VV_MTL_MAX_BUFS || l->nuses > VV_MTL_MAX_USES ||
        l->params_size > VV_MTL_MAX_PARAMS)
        return VV_ERR_INVALID_ARG;
    for (int i = 0; i < 3; i++)
        if (l->grid[i] == 0 || l->block[i] == 0) return VV_OK;   /* empty */

    @autoreleasepool {
        id<MTLComputePipelineState> pso = pso_get(l->kernel);
        if (!pso) return VV_ERR_CUDA_LAUNCH;
        const uint32_t threads = l->block[0] * l->block[1] * l->block[2];
        if (threads > pso.maxTotalThreadsPerThreadgroup) {
            VV_LOG_E("metal: '%s' launched with %u threads per group, the "
                     "pipeline allows %lu", l->kernel, threads,
                     (unsigned long)pso.maxTotalThreadsPerThreadgroup);
            return VV_ERR_CUDA_LAUNCH;
        }

        mtl_cmd_t c;
        memset(&c, 0, sizeof(c));
        c.name = l->kernel;
        c.pso = (__bridge void*)pso;
        c.nbufs = l->nbufs;
        for (int i = 0; i < l->nbufs; i++) {
            if (!l->bufs[i]) continue;
            size_t off = 0;
            id<MTLBuffer> b = resolve(l->bufs[i], &off);
            if (!b) {
                VV_LOG_E("metal: '%s' argument %d (%p) is not device memory",
                         l->kernel, i, l->bufs[i]);
                return VV_ERR_INVALID_ARG;
            }
            c.buf[i] = (__bridge void*)b;
            c.off[i] = off;
        }
        for (int i = 0; i < l->nuses; i++) {
            if (!l->uses[i]) continue;
            size_t off = 0;
            id<MTLBuffer> b = resolve(l->uses[i], &off);
            if (!b) {
                VV_LOG_E("metal: '%s' indirect argument %d (%p) is not device "
                         "memory", l->kernel, i, l->uses[i]);
                return VV_ERR_INVALID_ARG;
            }
            /* A table's items mostly share a few buffers: declare each once. */
            bool seen = false;
            for (int j = 0; j < c.nuses && !seen; j++)
                seen = c.use[j] == (__bridge void*)b;
            if (!seen) c.use[c.nuses++] = (__bridge void*)b;
        }
        static const uint8_t no_params[16] = {0};
        c.params = (uint8_t*)(l->params ? l->params : no_params);
        c.params_size = l->params ? l->params_size : sizeof(no_params);
        memcpy(c.grid, l->grid, sizeof(c.grid));
        memcpy(c.block, l->block, sizeof(c.block));
        c.tg_mem = l->tg_mem;

        VVStream* s = stream_of(stream);
        const bool null_stream = stream == NULL;
        if (null_stream) {
            /* Legacy default stream: after everything, before returning. */
            const vv_status_t e = device_sync();
            if (e != VV_OK) return e;
        }
        vv_status_t st = VV_OK;
        os_unfair_lock_lock(&s->lock);
        if (s->rec) {
            st = graph_append(s->rec, &c);
            if (st != VV_OK) s->rec->broken = 1;
        } else {
            encode_cmd(s, &c);
            if (null_stream || mtl()->sync_each) {
                const double t0 = mtl()->sync_each ? vv_time_ms() : 0.0;
                st = drain_locked(s);
                if (st != VV_OK)
                    VV_LOG_E("metal: '%s' failed on the GPU", l->kernel);
                else if (mtl()->sync_each)
                    VV_LOG_I("metal: %-28s %8.3f ms  grid %u x %u x %u",
                             l->kernel, vv_time_ms() - t0, l->grid[0],
                             l->grid[1], l->grid[2]);
            } else if (s->n_disp >= MTL_FLUSH_EVERY ||
                       s->n_groups >= MTL_FLUSH_GROUPS) {
                flush_locked(s);
            }
        }
        os_unfair_lock_unlock(&s->lock);
        return st;
    }
}

/* ─── Built-in copy kernels ─────────────────────────────────────────────── */

typedef struct { uint64_t n; uint32_t value; uint32_t pad; } mtl_fill_p;

/** @brief dst[0..n) = src[0..n) on the stream, 16 bytes at a time if aligned. */
static vv_status_t copy_kernel(void* dst, const void* src, size_t n,
                               void* stream) {
    if (n == 0) return VV_OK;
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    mtl_fill_p p = { n, 0, 0 };
    l.bufs[0] = dst; l.bufs[1] = src; l.nbufs = 2;
    l.params = &p; l.params_size = sizeof(p);
    if ((((uintptr_t)dst | (uintptr_t)src | n) & 15u) == 0) {
        l.kernel = "vv_copy16";
        p.n = n / 16;
        vv_mtl_grid1(&l, p.n, 256);
    } else if ((((uintptr_t)dst | (uintptr_t)src | n) & 3u) == 0) {
        l.kernel = "vv_copy4";
        p.n = n / 4;
        vv_mtl_grid1(&l, p.n, 256);
    } else {
        l.kernel = "vv_copy1";
        vv_mtl_grid1(&l, n, 256);
    }
    return vv_mtl_run(stream, &l);
}

/** @brief Small host data into device memory, carried in the launch itself. */
static vv_status_t write_inline(void* dst, const void* src, size_t n,
                                void* stream) {
    uint8_t buf[16 + 2048];
    uint64_t hdr[2] = { n, 0 };
    memcpy(buf, hdr, 16);
    memcpy(buf + 16, src, n);
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    l.kernel = "vv_write_bytes";
    l.bufs[0] = dst; l.nbufs = 1;
    l.params = buf; l.params_size = 16 + ((n + 15) & ~(size_t)15);
    vv_mtl_grid1(&l, n, 256);
    return vv_mtl_run(stream, &l);
}

/* ─── device.h: memory ──────────────────────────────────────────────────── */

static vv_status_t new_buffer(void** ptr, size_t size, int pinned) {
    if (!ptr) return VV_ERR_NULL_PTR;
    *ptr = NULL;
    VVMetal* m = mtl();
    if (!m->device) {
        /* No GPU: host memory, like device_none.c. */
        *ptr = vv_alloc(size ? size : 1);
        return *ptr ? VV_OK : VV_ERR_OUT_OF_MEMORY;
    }
    if (size == 0) size = 1;
    if (size > m->device.maxBufferLength) return VV_ERR_CUDA_OOM;
    @autoreleasepool {
        id<MTLBuffer> b = nil;
        int fresh = 0;
        if (size >= MTL_MMAP_MIN) {
            const size_t len = (size + MTL_PAGE - 1) & ~(size_t)(MTL_PAGE - 1);
            void* mem = mmap(NULL, len, PROT_READ | PROT_WRITE,
                             MAP_ANON | MAP_PRIVATE, -1, 0);
            if (mem == MAP_FAILED) return VV_ERR_CUDA_OOM;
            b = [m->device newBufferWithBytesNoCopy:mem length:len
                                            options:MTLResourceStorageModeShared
                                        deallocator:^(void* q, NSUInteger n) {
                                            munmap(q, n);
                                        }];
            if (!b) { munmap(mem, len); return VV_ERR_CUDA_OOM; }
            fresh = 1;
        } else {
            b = [m->device newBufferWithLength:size
                                       options:MTLResourceStorageModeShared];
        }
        if (!b) return VV_ERR_CUDA_OOM;
        const vv_status_t s = alloc_insert(m, b, size, pinned, fresh);
        if (s != VV_OK) return s;
        *ptr = b.contents;
    }
    return VV_OK;
}

vv_status_t vv_dev_alloc(void** ptr, size_t size) {
    return new_buffer(ptr, size, 0);
}

vv_status_t vv_dev_alloc_pinned(void** ptr, size_t size) {
    return new_buffer(ptr, size, 1);
}

static vv_status_t free_buffer(void* ptr, int pinned) {
    if (!ptr) return VV_OK;
    VVMetal* m = mtl();
    if (!m->device) { vv_free(ptr); return VV_OK; }
    if (!alloc_remove(m, ptr, pinned)) {
        VV_LOG_E("metal: free of %p, which is not a live %s allocation", ptr,
                 pinned ? "pinned" : "device");
        return VV_ERR_INVALID_ARG;
    }
    return VV_OK;
}

vv_status_t vv_dev_free(void* ptr) { return free_buffer(ptr, 0); }
vv_status_t vv_dev_free_pinned(void* ptr) { return free_buffer(ptr, 1); }

vv_status_t vv_dev_memcpy_h2d(void* dst, const void* src, size_t size,
                              void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    if (size == 0) return VV_OK;
    if (!usable()) { memcpy(dst, src, size); return VV_OK; }

    if (!stream) {
        const vv_status_t e = device_sync();
        touch(dst);
        memcpy(dst, src, size);
        return e;
    }
    VVStream* s = stream_of(stream);
    /* Out of device memory (pinned buffers included): a GPU copy in stream
     * order, the source read when the copy runs -- a DMA out of pinned
     * memory on CUDA. */
    if (vv_mtl_is_device_ptr(src)) return copy_kernel(dst, src, size, stream);

    os_unfair_lock_lock(&s->lock);
    const bool capturing = s->rec != NULL;
    const bool idle = !capturing && idle_locked(s);
    os_unfair_lock_unlock(&s->lock);
    if (idle) {
        /* Nothing ahead of it on this stream: the copy is simply now. */
        touch(dst);
        memcpy(dst, src, size);
        return VV_OK;
    }
    /* Pageable memory is consumed before the call returns, as on CUDA. */
    if (size <= 2048) return write_inline(dst, src, size, stream);
    if (capturing) {
        os_unfair_lock_lock(&s->lock);
        if (s->rec) s->rec->broken = 1;
        os_unfair_lock_unlock(&s->lock);
        return VV_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        id<MTLBuffer> stage = [mtl()->device newBufferWithBytes:src length:size
                                                        options:MTLResourceStorageModeShared];
        if (!stage) return VV_ERR_CUDA_OOM;
        if (alloc_insert(mtl(), stage, size, 1, 0) != VV_OK) return VV_ERR_OUT_OF_MEMORY;
        const vv_status_t st = copy_kernel(dst, stage.contents, size, stream);
        /* The encoded copy holds the buffer until it has run. */
        alloc_remove(mtl(), stage.contents, 1);
        return st;
    }
}

vv_status_t vv_dev_memcpy_d2h(void* dst, const void* src, size_t size,
                              void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    if (size == 0) return VV_OK;
    if (!usable()) { memcpy(dst, src, size); return VV_OK; }
    vv_status_t e;
    if (!stream) {
        e = device_sync();
    } else {
        VVStream* s = stream_of(stream);
        os_unfair_lock_lock(&s->lock);
        if (s->rec) {
            s->rec->broken = 1;
            os_unfair_lock_unlock(&s->lock);
            return VV_ERR_UNSUPPORTED;
        }
        e = drain_locked(s);
        os_unfair_lock_unlock(&s->lock);
    }
    memcpy(dst, src, size);
    return e;
}

vv_status_t vv_dev_memcpy_d2d(void* dst, const void* src, size_t size,
                              void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    if (size == 0) return VV_OK;
    if (!usable()) { memmove(dst, src, size); return VV_OK; }
    return copy_kernel(dst, src, size, stream);
}

typedef struct { uint64_t n; } mtl_copy_at_p;

vv_status_t vv_dev_memcpy_d2d_at(void* dst_base, const void* src, size_t size,
                                 const int* d_index, void* stream) {
    if (!dst_base || !src || !d_index) return VV_ERR_NULL_PTR;
    if (size == 0) return VV_OK;
    if (!usable()) {
        memcpy((uint8_t*)dst_base + (size_t)(*d_index) * size, src, size);
        return VV_OK;
    }
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    mtl_copy_at_p p = { size };
    l.bufs[0] = dst_base; l.bufs[1] = src; l.bufs[2] = d_index; l.nbufs = 3;
    l.params = &p; l.params_size = sizeof(p);
    if ((((uintptr_t)dst_base | (uintptr_t)src | size) & 15u) == 0) {
        l.kernel = "vv_copy_at16";
        vv_mtl_grid1(&l, size / 16, 128);
    } else {
        l.kernel = "vv_copy_at1";
        vv_mtl_grid1(&l, size, 128);
    }
    return vv_mtl_run(stream, &l);
}

vv_status_t vv_dev_memset(void* ptr, int value, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    if (!usable()) { memset(ptr, value, size); return VV_OK; }
    /* Zeroing pages nothing has written would only make them resident. */
    if (value == 0 && still_zero(ptr, size)) return VV_OK;
    const vv_status_t e = device_sync();
    touch(ptr);
    memset(ptr, value, size);
    return e;
}

vv_status_t vv_dev_memset_async(void* ptr, int value, size_t size,
                                void* stream) {
    if (!ptr) return VV_ERR_NULL_PTR;
    if (size == 0) return VV_OK;
    if (!usable()) { memset(ptr, value, size); return VV_OK; }
    if (!stream) return vv_dev_memset(ptr, value, size);
    if (value == 0 && still_zero(ptr, size)) return VV_OK;
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    const uint32_t b = (uint32_t)(value & 0xFF);
    mtl_fill_p p = { size, b | (b << 8) | (b << 16) | (b << 24), 0 };
    l.bufs[0] = ptr; l.nbufs = 1;
    l.params = &p; l.params_size = sizeof(p);
    if ((((uintptr_t)ptr | size) & 3u) == 0) {
        l.kernel = "vv_fill4";
        p.n = size / 4;
        vv_mtl_grid1(&l, p.n, 256);
    } else {
        l.kernel = "vv_fill1";
        vv_mtl_grid1(&l, size, 256);
    }
    return vv_mtl_run(stream, &l);
}

typedef struct { int32_t delta; } mtl_pos_add_p;

vv_status_t vv_pos_add_dev(int* dst, const int* src, int delta, void* stream) {
    if (!dst || !src) return VV_ERR_NULL_PTR;
    if (!usable()) { *dst = *src + delta; return VV_OK; }
    vv_mtl_launch_t l;
    memset(&l, 0, sizeof(l));
    mtl_pos_add_p p = { delta };
    l.kernel = "vv_pos_add";
    l.bufs[0] = dst; l.bufs[1] = src; l.nbufs = 2;
    l.params = &p; l.params_size = sizeof(p);
    vv_mtl_grid1(&l, 1, 1);
    return vv_mtl_run(stream, &l);
}

/* ─── device.h: streams ─────────────────────────────────────────────────── */

vv_status_t vv_dev_stream_create(void** stream) {
    if (!stream) return VV_ERR_NULL_PTR;
    *stream = NULL;
    VVMetal* m = mtl();
    if (!m->device) return VV_OK;
    @autoreleasepool {
        VVStream* s = [VVStream new];
        s->queue = [m->device newCommandQueue];
        if (!s->queue) return VV_ERR_CUDA;
        s->queue.label = @"vv.stream";
        s->lock = OS_UNFAIR_LOCK_INIT;
        os_unfair_lock_lock(&m->streams_lock);
        [m->streams addObject:s];
        os_unfair_lock_unlock(&m->streams_lock);
        *stream = (void*)CFBridgingRetain(s);
    }
    return VV_OK;
}

vv_status_t vv_dev_stream_create_background(void** stream) {
    /* Metal has no queue priorities; a background stream is a stream. */
    return vv_dev_stream_create(stream);
}

vv_status_t vv_dev_stream_destroy(void* stream) {
    if (!stream) return VV_OK;
    VVMetal* m = mtl();
    VVStream* s = (VVStream*)CFBridgingRelease(stream);
    os_unfair_lock_lock(&s->lock);
    const vv_status_t e = drain_locked(s);
    os_unfair_lock_unlock(&s->lock);
    os_unfair_lock_lock(&m->streams_lock);
    [m->streams removeObject:s];
    os_unfair_lock_unlock(&m->streams_lock);
    return e;
}

vv_status_t vv_dev_stream_sync(void* stream) {
    if (!usable()) return VV_OK;
    if (!stream) return device_sync();
    VVStream* s = stream_of(stream);
    os_unfair_lock_lock(&s->lock);
    vv_status_t e;
    if (s->rec) { s->rec->broken = 1; e = VV_ERR_UNSUPPORTED; }
    else e = drain_locked(s);
    os_unfair_lock_unlock(&s->lock);
    return e;
}

/* ─── device.h: devices ─────────────────────────────────────────────────── */

vv_status_t vv_dev_set_device(int device_id) {
    return (usable() && device_id == 0) ? VV_OK : VV_ERR_NOT_FOUND;
}

int vv_dev_device_count(void) { return usable() ? 1 : 0; }

vv_status_t vv_dev_get_device_info(int device_id, size_t* total_mem,
                                   size_t* free_mem, int* sm_count) {
    if (total_mem) *total_mem = 0;
    if (free_mem) *free_mem = 0;
    if (sm_count) *sm_count = 0;
    VVMetal* m = mtl();
    if (!m->device || device_id != 0) return VV_ERR_NOT_FOUND;
    /*
     * Unified memory: what the GPU may keep resident is the working-set
     * limit macOS recommends for it, and what is left of it is that minus
     * this process's allocations. Other processes' use shows up only as
     * memory pressure, which no count here can predict.
     */
    const size_t total = (size_t)m->device.recommendedMaxWorkingSetSize;
    os_unfair_lock_lock(&m->alloc_lock);
    size_t used = m->alloc_bytes;
    os_unfair_lock_unlock(&m->alloc_lock);
    /* Wrapped mappings are not in Metal's own count; ours has them all. */
    if ((size_t)m->device.currentAllocatedSize > used)
        used = (size_t)m->device.currentAllocatedSize;
    size_t freem = total > used ? total - used : 0;
    /*
     * ...but never more than the machine can still give, less a margin for
     * the system itself. What this process already holds is part of what
     * the OS reports as taken, so it is not subtracted twice.
     */
    const size_t margin = (size_t)1536 * 1024 * 1024;
    const size_t avail = host_available_bytes();
    if (avail != SIZE_MAX) {
        const size_t host_free = avail > margin ? avail - margin : 0;
        if (host_free < freem) freem = host_free;
    }
    if (total_mem) *total_mem = total;
    if (free_mem) *free_mem = freem;
    if (sm_count) *sm_count = m->cores;
    return VV_OK;
}

vv_status_t vv_dev_get_device_name(int device_id, char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return VV_ERR_NULL_PTR;
    VVMetal* m = mtl();
    if (!m->device || device_id != 0) { buf[0] = '\0'; return VV_ERR_NOT_FOUND; }
    snprintf(buf, buf_size, "%s", m->name);
    return VV_OK;
}

vv_status_t vv_dev_memcpy_peer(void* dst, int dst_device, const void* src,
                               int src_device, size_t size, void* stream) {
    (void)dst_device; (void)src_device;
    return vv_dev_memcpy_d2d(dst, src, size, stream);
}

vv_status_t vv_dev_enable_peer(int device, int peer) {
    return device == peer ? VV_OK : VV_ERR_UNSUPPORTED;
}

const char* vv_dev_backend_name(void) { return "Metal"; }

bool vv_dev_host_shares_memory(int device_id) {
    return usable() && device_id == 0;
}

/* ─── device.h: events ──────────────────────────────────────────────────── */

vv_status_t vv_dev_event_create(void** ev) {
    if (!ev) return VV_ERR_NULL_PTR;
    *ev = NULL;
    VVMetal* m = mtl();
    if (!m->device) return VV_OK;
    @autoreleasepool {
        VVEvent* e = [VVEvent new];
        e->ev = [m->device newSharedEvent];
        if (!e->ev) return VV_ERR_CUDA;
        e->lock = OS_UNFAIR_LOCK_INIT;
        *ev = (void*)CFBridgingRetain(e);
    }
    return VV_OK;
}

vv_status_t vv_dev_event_destroy(void* ev) {
    if (!ev) return VV_OK;
    (void)CFBridgingRelease(ev);
    return VV_OK;
}

vv_status_t vv_dev_event_record(void* ev, void* stream) {
    if (!ev) return VV_ERR_NULL_PTR;
    if (!usable()) return VV_OK;
    VVEvent* e = (__bridge VVEvent*)ev;
    if (!stream) {
        /* The default stream has finished everything once this returns. */
        const vv_status_t st = device_sync();
        os_unfair_lock_lock(&e->lock);
        e->value++;
        e->ev.signaledValue = e->value;
        e->signaller = nil;
        os_unfair_lock_unlock(&e->lock);
        return st;
    }
    VVStream* s = stream_of(stream);
    os_unfair_lock_lock(&s->lock);
    if (s->rec) {
        s->rec->broken = 1;
        os_unfair_lock_unlock(&s->lock);
        return VV_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = current_cb(s);
        end_encoder(s);
        os_unfair_lock_lock(&e->lock);
        const uint64_t v = ++e->value;
        [cb encodeSignalEvent:e->ev value:v];
        e->signaller = cb;
        os_unfair_lock_unlock(&e->lock);
        /* A waiter on another queue can only see a committed signal. */
        flush_locked(s);
    }
    os_unfair_lock_unlock(&s->lock);
    return VV_OK;
}

vv_status_t vv_dev_stream_wait_event(void* stream, void* ev) {
    if (!ev) return VV_ERR_NULL_PTR;
    if (!usable()) return VV_OK;
    VVEvent* e = (__bridge VVEvent*)ev;
    os_unfair_lock_lock(&e->lock);
    const uint64_t v = e->value;
    os_unfair_lock_unlock(&e->lock);
    if (v == 0) return VV_OK;                  /* never recorded: no-op */
    if (!stream) {
        /* The default stream runs synchronously; wait on the host. */
        return vv_dev_event_sync(ev);
    }
    VVStream* s = stream_of(stream);
    os_unfair_lock_lock(&s->lock);
    if (s->rec) {
        s->rec->broken = 1;
        os_unfair_lock_unlock(&s->lock);
        return VV_ERR_UNSUPPORTED;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = current_cb(s);
        end_encoder(s);
        [cb encodeWaitForEvent:e->ev value:v];
    }
    os_unfair_lock_unlock(&s->lock);
    return VV_OK;
}

vv_status_t vv_dev_event_sync(void* ev) {
    if (!ev) return VV_ERR_NULL_PTR;
    if (!usable()) return VV_OK;
    VVEvent* e = (__bridge VVEvent*)ev;
    os_unfair_lock_lock(&e->lock);
    id<MTLCommandBuffer> cb = e->signaller;
    os_unfair_lock_unlock(&e->lock);
    if (cb) {
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) return VV_ERR_CUDA;
    }
    return VV_OK;
}

/* Unified memory: there is nothing to page-lock. */
vv_status_t vv_dev_host_register(void* p, size_t n) {
    return (p && n) ? VV_OK : VV_ERR_NULL_PTR;
}
vv_status_t vv_dev_host_unregister(void* p) { (void)p; return VV_OK; }

/* ─── device.h: graphs ──────────────────────────────────────────────────── */

vv_status_t vv_dev_graph_begin(void* stream) {
    if (!stream) return VV_ERR_INVALID_ARG;   /* the default stream cannot */
    if (!usable()) return VV_ERR_UNSUPPORTED;
    VVStream* s = stream_of(stream);
    mtl_graph_t* g = (mtl_graph_t*)vv_alloc(sizeof(mtl_graph_t));
    if (!g) return VV_ERR_OUT_OF_MEMORY;
    memset(g, 0, sizeof(*g));
    os_unfair_lock_lock(&s->lock);
    if (s->rec) {
        os_unfair_lock_unlock(&s->lock);
        vv_free(g);
        return VV_ERR_INVALID_ARG;
    }
    s->rec = g;
    os_unfair_lock_unlock(&s->lock);
    return VV_OK;
}

static void graph_free(mtl_graph_t* g) {
    if (!g) return;
    for (int i = 0; i < g->n; i++) {
        mtl_cmd_t* c = &g->cmds[i];
        for (int j = 0; j < c->nbufs; j++) if (c->buf[j]) CFRelease(c->buf[j]);
        for (int j = 0; j < c->nuses; j++) CFRelease(c->use[j]);
        vv_free(c->params);
    }
    vv_free(g->cmds);
    vv_free(g);
}

vv_status_t vv_dev_graph_end(void* stream, void** graph_exec) {
    if (!stream || !graph_exec) return VV_ERR_NULL_PTR;
    *graph_exec = NULL;
    VVStream* s = stream_of(stream);
    os_unfair_lock_lock(&s->lock);
    mtl_graph_t* g = s->rec;
    s->rec = NULL;
    os_unfair_lock_unlock(&s->lock);
    if (!g) return VV_ERR_INVALID_ARG;
    if (g->broken) { graph_free(g); return VV_ERR_CUDA; }
    *graph_exec = g;
    return VV_OK;
}

vv_status_t vv_dev_graph_launch(void* graph_exec, void* stream) {
    if (!graph_exec) return VV_ERR_NULL_PTR;
    if (!usable()) return VV_ERR_UNSUPPORTED;
    mtl_graph_t* g = (mtl_graph_t*)graph_exec;
    VVStream* s = stream_of(stream);
    const bool null_stream = stream == NULL;
    if (null_stream) {
        const vv_status_t e = device_sync();
        if (e != VV_OK) return e;
    }
    vv_status_t st = VV_OK;
    os_unfair_lock_lock(&s->lock);
    if (s->rec) {
        s->rec->broken = 1;
        st = VV_ERR_UNSUPPORTED;
    } else {
        @autoreleasepool {
            for (int i = 0; i < g->n; i++) {
                encode_cmd(s, &g->cmds[i]);
                if (s->n_groups >= MTL_FLUSH_GROUPS) flush_locked(s);
            }
            /* The rest of the step in one submission. */
            flush_locked(s);
            if (null_stream || mtl()->sync_each) st = drain_locked(s);
        }
    }
    os_unfair_lock_unlock(&s->lock);
    return st;
}

void vv_dev_graph_destroy(void* graph_exec) {
    graph_free((mtl_graph_t*)graph_exec);
}

void vv_gemm_cleanup(void) {}
