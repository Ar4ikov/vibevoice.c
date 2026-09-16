/**
 * @file attention_mma.cuh
 * @brief Flash prefill attention on tensor cores.
 *
 * The scalar prefill kernel gives one warp one query row and walks head_dim
 * with FP32 FMAs. That is fine to about 15K tokens and hopeless past it: both
 * matmuls in attention are O(S²) and neither touches a tensor core, so a
 * 32-minute file spent 39 s of its 231 s here.
 *
 * Same algorithm — FlashAttention-2's online softmax over KV tiles — with both
 * matmuls issued as `mma.sync.m16n8k16`.
 *
 * ## Why raw PTX rather than the WMMA API
 *
 * The online softmax rescales the output accumulator by a per-row factor every
 * time the running maximum moves, which means knowing which row of the tile
 * each accumulator register holds. The WMMA API deliberately does not say;
 * `mma.sync` does, in the PTX ISA:
 *
 *   accumulator reg i  ->  row = lane/4 + (i < 2 ? 0 : 8)
 *                          col = (lane%4)*2 + (i%2)
 *
 * so the rescale is two multiplies per register with no round trip through
 * shared memory.
 *
 * It buys a second thing. The A operand of `m16n8k16` is laid out as
 *
 *   reg 0 -> row = lane/4,     col = (lane%4)*2 + {0,1}
 *   reg 1 -> row = lane/4 + 8, col = (lane%4)*2 + {0,1}
 *   reg 2 -> row = lane/4,     col = (lane%4)*2 + {0,1} + 8
 *   reg 3 -> row = lane/4 + 8, col = (lane%4)*2 + {0,1} + 8
 *
 * which is exactly the accumulator layout of two adjacent n-tiles. The softmax
 * probabilities therefore reach the tensor cores from the registers the
 * softmax left them in, with nothing but a float-to-half pack in between.
 *
 * ## Shape
 *
 * Block: 4 warps, 64 query rows (16 per warp), 64 KV positions per tile,
 * head_dim 128. Per warp per tile: 64 mma for Q·Kᵀ and 64 for P·V, which is
 * exactly the useful MAC count — no padding waste.
 *
 * Shared memory holds K and V as [kv][head_dim], 34 KB, under the 48 KB a
 * block gets without opting in. Q needs no tile at all: each warp owns 16 rows
 * nobody else reads, so its fragments load straight from global into registers
 * once per block.
 *
 * ## The transpose
 *
 * The B operand is column-major, so P·V wants V read down a column while the
 * tile has head_dim contiguous. Keeping a transposed copy in shared memory
 * does not work: a conflict-free read of the B fragments needs a row stride
 * that is a multiple of 8 halves, and a conflict-free write of the transpose
 * needs one that is not. `ldmatrix.x2.trans` settles it in hardware — it reads
 * two 8x8 tiles and hands each thread exactly the two registers the B operand
 * wants — and with the 136-half row stride its eight row addresses walk all 32
 * banks once.
 *
 * Needs sm_80. The caller keeps the scalar kernel for older cards and for
 * head_dim != 128.
 */
#ifndef VV_CUDA_ATTENTION_MMA_CUH
#define VV_CUDA_ATTENTION_MMA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

/* ─── Tile geometry ──────────────────────────────────────────────────────── */

#define MMA_WARPS     4
#define MMA_BR        (MMA_WARPS * 16)   /* 64 query rows per block  */
#define MMA_BC        64                 /* KV positions per tile    */
#define MMA_D         128                /* head_dim this path takes */
#define MMA_THREADS   (MMA_WARPS * 32)

/*
 * Row stride of the shared K and V tiles, in halves. A quad of lanes reads one
 * row at column (lane%4)*2 and the eight quads of a warp read eight rows, so
 * the word index comes out as 4*(lane/4) + (lane%4) plus a constant, which
 * walks all 32 banks exactly once. The same stride makes `ldmatrix`'s eight
 * 16-byte row reads conflict-free.
 */
#define MMA_LD        (MMA_D + 8)        /* 136 */

#define MMA_SMEM_BYTES ((size_t)2 * MMA_BC * MMA_LD * sizeof(half))

/** @brief D = A[16x16] * B[16x8] + C, f16 in, f32 accumulate. */
__device__ __forceinline__ void vv_mma_m16n8k16(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)a; (void)b;
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

/**
 * @brief Two 8x8 half tiles from shared memory, transposed, as a B operand.
 *
 * `src` is this lane's row of the source: rows 0-7 of the first tile come from
 * lanes 0-7 and rows 8-15 of the second from lanes 8-15. Everyone gets two
 * registers back.
 */
__device__ __forceinline__ void vv_ldmatrix_x2_trans(
    uint32_t b[2], const half* src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
#else
    (void)src;
    b[0] = 0u; b[1] = 0u;
#endif
}

/** @brief Max across the four lanes that share a row of the fragment. */
__device__ __forceinline__ float vv_quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    return v;
}

/** @brief Sum across the same four lanes. */
__device__ __forceinline__ float vv_quad_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    return v;
}

/** @brief Two halves at `p` as one 32-bit register operand. */
__device__ __forceinline__ uint32_t vv_pack2(const half* p) {
    return *(const uint32_t*)p;
}

__device__ __forceinline__ uint32_t vv_pack2f(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t*)&h;
}

/**
 * @brief Flash prefill: `q_len` rows at `q_offset` against a KV cache.
 *
 * Grid  (n_q_heads, ceil(q_len / MMA_BR)), block (32, MMA_WARPS).
 * Q, K, V and O are [seq, heads, head_dim]. K and V are the cache, so their
 * rows are absolute positions and `q_offset` says where the queries sit in it.
 */
__global__ __launch_bounds__(MMA_THREADS)
void flash_attn_prefill_mma_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int n_q_heads, int n_kv_heads,
    int q_len, int q_offset, int kv_len, float scale, bool causal)
{
    const int head    = blockIdx.x;
    const int warp    = threadIdx.y;
    const int lane    = threadIdx.x;
    const int group   = lane >> 2;          /* 0..7, the fragment's row  */
    const int quad    = lane & 3;           /* 0..3, its column pair     */
    const int kv_head = head / (n_q_heads / n_kv_heads);

    const int q_stride  = n_q_heads  * MMA_D;
    const int kv_stride = n_kv_heads * MMA_D;

    /* The 16 query rows this warp owns, and the two a lane touches. */
    const int q_tile0 = blockIdx.y * MMA_BR + warp * 16;
    const int row_lo  = q_tile0 + group;
    const int row_hi  = row_lo + 8;

    extern __shared__ char mma_smem[];
    half* K_tile = (half*)mma_smem;
    half* V_tile = K_tile + MMA_BC * MMA_LD;

    /*
     * Q fragments, straight from global: two halves per (row, k) pair, which
     * is one 32-bit load. Rows past the end of the sequence read as zero and
     * are dropped at the store.
     */
    uint32_t a_q[MMA_D / 16][4];
    {
        const bool lo_ok = row_lo < q_len;
        const bool hi_ok = row_hi < q_len;
        const half* q_lo = Q + (size_t)row_lo * q_stride + head * MMA_D;
        const half* q_hi = Q + (size_t)row_hi * q_stride + head * MMA_D;
        #pragma unroll
        for (int s = 0; s < MMA_D / 16; ++s) {
            const int c0 = s * 16 + quad * 2;
            a_q[s][0] = lo_ok ? vv_pack2(q_lo + c0)     : 0u;
            a_q[s][1] = hi_ok ? vv_pack2(q_hi + c0)     : 0u;
            a_q[s][2] = lo_ok ? vv_pack2(q_lo + c0 + 8) : 0u;
            a_q[s][3] = hi_ok ? vv_pack2(q_hi + c0 + 8) : 0u;
        }
    }

    float o_acc[MMA_D / 8][4];
    #pragma unroll
    for (int n = 0; n < MMA_D / 8; ++n)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[n][i] = 0.0f;

    float m_lo = -FLT_MAX, m_hi = -FLT_MAX;
    float l_lo = 0.0f,     l_hi = 0.0f;

    /*
     * The KV bound is uniform across the block: every warp must reach the same
     * number of __syncthreads(). Per-row causality is a score mask instead —
     * deriving the loop bound per row is what corrupted the shared tiles in an
     * earlier version of the scalar kernel.
     */
    const int block_last = blockIdx.y * MMA_BR + MMA_BR - 1;
    int kv_bound = causal ? (q_offset + block_last + 1) : kv_len;
    if (kv_bound > kv_len) kv_bound = kv_len;

    const int tid = warp * 32 + lane;

    for (int kv0 = 0; kv0 < kv_bound; kv0 += MMA_BC) {
        /* Eight halves per thread per pass keeps the global reads at 16 bytes. */
        #pragma unroll
        for (int p = 0; p < (MMA_BC * MMA_D) / (MMA_THREADS * 8); ++p) {
            const int idx = (p * MMA_THREADS + tid) * 8;
            const int r   = idx / MMA_D;
            const int c   = idx % MMA_D;
            const int pos = kv0 + r;
            const size_t sh = (size_t)r * MMA_LD + c;

            if (pos < kv_len) {
                const size_t off = (size_t)pos * kv_stride + kv_head * MMA_D + c;
                *(float4*)&K_tile[sh] = *(const float4*)(K + off);
                *(float4*)&V_tile[sh] = *(const float4*)(V + off);
            } else {
                const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
                *(float4*)&K_tile[sh] = z;
                *(float4*)&V_tile[sh] = z;
            }
        }
        __syncthreads();

        /* S = Q Kᵀ, [16 x MMA_BC]. K is the B operand: B[k][n] = K[n][k]. */
        float s_acc[MMA_BC / 8][4];
        #pragma unroll
        for (int n = 0; n < MMA_BC / 8; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) s_acc[n][i] = 0.0f;
            const half* krow = K_tile + (n * 8 + group) * MMA_LD + quad * 2;
            #pragma unroll
            for (int s = 0; s < MMA_D / 16; ++s) {
                const uint32_t b[2] = { vv_pack2(krow + s * 16),
                                        vv_pack2(krow + s * 16 + 8) };
                vv_mma_m16n8k16(s_acc[n], a_q[s], b, s_acc[n]);
            }
        }

        /* Scale, mask, and take the row maxima. */
        float mr_lo = -FLT_MAX, mr_hi = -FLT_MAX;
        #pragma unroll
        for (int n = 0; n < MMA_BC / 8; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kv_pos = kv0 + n * 8 + quad * 2 + (i & 1);
                const int q_abs  = q_offset + ((i < 2) ? row_lo : row_hi);
                const bool ok = kv_pos < kv_len &&
                                (!causal || kv_pos <= q_abs);
                s_acc[n][i] = ok ? s_acc[n][i] * scale : -FLT_MAX;
                if (i < 2) mr_lo = fmaxf(mr_lo, s_acc[n][i]);
                else       mr_hi = fmaxf(mr_hi, s_acc[n][i]);
            }
        }
        mr_lo = vv_quad_max(mr_lo);
        mr_hi = vv_quad_max(mr_hi);

        const float mn_lo = fmaxf(m_lo, mr_lo);
        const float mn_hi = fmaxf(m_hi, mr_hi);
        const float a_lo = (mn_lo > -FLT_MAX && m_lo > -FLT_MAX)
                           ? __expf(m_lo - mn_lo) : 1.0f;
        const float a_hi = (mn_hi > -FLT_MAX && m_hi > -FLT_MAX)
                           ? __expf(m_hi - mn_hi) : 1.0f;

        /* exp in place; a row with nothing unmasked contributes zero. */
        float sum_lo = 0.0f, sum_hi = 0.0f;
        #pragma unroll
        for (int n = 0; n < MMA_BC / 8; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float m = (i < 2) ? mn_lo : mn_hi;
                const float p = (s_acc[n][i] > -FLT_MAX && m > -FLT_MAX)
                                ? __expf(s_acc[n][i] - m) : 0.0f;
                s_acc[n][i] = p;
                if (i < 2) sum_lo += p; else sum_hi += p;
            }
        }
        l_lo = l_lo * a_lo + vv_quad_sum(sum_lo);
        l_hi = l_hi * a_hi + vv_quad_sum(sum_hi);

        /* Rescale the output: regs 0,1 are the low row, 2,3 the high one. */
        #pragma unroll
        for (int n = 0; n < MMA_D / 8; ++n) {
            o_acc[n][0] *= a_lo; o_acc[n][1] *= a_lo;
            o_acc[n][2] *= a_hi; o_acc[n][3] *= a_hi;
        }
        m_lo = mn_lo;
        m_hi = mn_hi;

        /*
         * O += P V. Two adjacent n-tiles of S are exactly one A fragment, so
         * the probabilities go straight in; V comes through ldmatrix.trans.
         */
        #pragma unroll
        for (int s = 0; s < MMA_BC / 16; ++s) {
            const uint32_t a_p[4] = {
                vv_pack2f(s_acc[2 * s][0],     s_acc[2 * s][1]),
                vv_pack2f(s_acc[2 * s][2],     s_acc[2 * s][3]),
                vv_pack2f(s_acc[2 * s + 1][0], s_acc[2 * s + 1][1]),
                vv_pack2f(s_acc[2 * s + 1][2], s_acc[2 * s + 1][3]),
            };
            const half* vbase = V_tile + (size_t)(s * 16 + (lane & 15)) * MMA_LD;
            #pragma unroll
            for (int n = 0; n < MMA_D / 8; ++n) {
                uint32_t b[2];
                vv_ldmatrix_x2_trans(b, vbase + n * 8);
                vv_mma_m16n8k16(o_acc[n], a_p, b, o_acc[n]);
            }
        }
        __syncthreads();
    }

    const float inv_lo = (l_lo > 1e-20f) ? 1.0f / l_lo : 0.0f;
    const float inv_hi = (l_hi > 1e-20f) ? 1.0f / l_hi : 0.0f;

    half* o_lo = O + (size_t)row_lo * q_stride + head * MMA_D;
    half* o_hi = O + (size_t)row_hi * q_stride + head * MMA_D;
    #pragma unroll
    for (int n = 0; n < MMA_D / 8; ++n) {
        const int c0 = n * 8 + quad * 2;
        if (row_lo < q_len)
            *(uint32_t*)(o_lo + c0) =
                vv_pack2f(o_acc[n][0] * inv_lo, o_acc[n][1] * inv_lo);
        if (row_hi < q_len)
            *(uint32_t*)(o_hi + c0) =
                vv_pack2f(o_acc[n][2] * inv_hi, o_acc[n][3] * inv_hi);
    }
}

#endif /* VV_CUDA_ATTENTION_MMA_CUH */
