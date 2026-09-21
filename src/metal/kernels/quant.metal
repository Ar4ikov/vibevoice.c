/*
 * quant.metal -- weights on FP16 activations in the formats that are read
 * as they lie: per-channel INT8 (int8_gemv.cu) and the row-major INT4
 * group-affine layout of the legacy AWQ path (awq_gemv.cu). One simdgroup
 * per output row, the lanes' partial sums in the CUDA kernels' order.
 */

/* ─── INT8, w = q * scale[row] ──────────────────────────────────────────── */

static inline void i8_unpack16(uint4 bits, thread float w[16]) {
    const uint words[4] = { bits.x, bits.y, bits.z, bits.w };
    for (int j = 0; j < 4; j++) {
        const char4 c = as_type<char4>(words[j]);
        w[4 * j + 0] = (float)c.x;
        w[4 * j + 1] = (float)c.y;
        w[4 * j + 2] = (float)c.z;
        w[4 * j + 3] = (float)c.w;
    }
}

static inline float i8_dot16(thread const float w[16], device const half* x) {
    const half4 a0 = *(device const half4*)(x);
    const half4 a1 = *(device const half4*)(x + 4);
    const half4 b0 = *(device const half4*)(x + 8);
    const half4 b1 = *(device const half4*)(x + 12);
    const float a[8] = { (float)a0.x, (float)a0.y, (float)a0.z, (float)a0.w,
                         (float)a1.x, (float)a1.y, (float)a1.z, (float)a1.w };
    const float b[8] = { (float)b0.x, (float)b0.y, (float)b0.z, (float)b0.w,
                         (float)b1.x, (float)b1.y, (float)b1.z, (float)b1.w };
    float acc = 0.0f;
    for (int j = 0; j < 4; j++) {
        acc = fma(w[2 * j + 0], a[2 * j], acc);
        acc = fma(w[2 * j + 1], a[2 * j + 1], acc);
        acc = fma(w[8 + 2 * j + 0], b[2 * j], acc);
        acc = fma(w[8 + 2 * j + 1], b[2 * j + 1], acc);
    }
    return acc;
}

struct vv_i8w_p { int M; int N; int K; int has_bias; };

kernel void vv_int8_gemv(constant vv_i8w_p& p [[buffer(0)]],
                         device const half* x [[buffer(1)]],
                         device const char* q [[buffer(2)]],
                         device const float* scales [[buffer(3)]],
                         device const half* bias [[buffer(4)]],
                         device half* y [[buffer(5)]],
                         VV_GRID_ARGS) {
    const int row = (int)blockIdx.x * 8 + (int)warp;
    if (row >= p.N) return;
    device const char* wrow = q + (ulong)row * (ulong)p.K;
    float acc = 0.0f;
    for (int e0 = (int)lane * 16; e0 < p.K; e0 += 512) {
        float w[16];
        i8_unpack16(*(device const uint4*)(wrow + e0), w);
        acc += i8_dot16(w, x + e0);
    }
    acc = vv_warp_sum(acc);
    if (lane == 0) {
        acc *= scales[row];
        if (p.has_bias) acc += (float)bias[row];
        y[row] = (half)acc;
    }
}

kernel void vv_int8_gemm_small(constant vv_i8w_p& p [[buffer(0)]],
                               device const half* x [[buffer(1)]],
                               device const char* q [[buffer(2)]],
                               device const float* scales [[buffer(3)]],
                               device half* y [[buffer(4)]],
                               VV_GRID_ARGS) {
    const int row = (int)blockIdx.x * 8 + (int)warp;
    if (row >= p.N) return;
    device const char* wrow = q + (ulong)row * (ulong)p.K;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int e0 = (int)lane * 16; e0 < p.K; e0 += 512) {
        float w[16];
        i8_unpack16(*(device const uint4*)(wrow + e0), w);
        for (int m = 0; m < 8; m++)
            if (m < p.M) acc[m] += i8_dot16(w, x + (ulong)m * (ulong)p.K + e0);
    }
    const float s = scales[row];
    for (int m = 0; m < 8; m++) {
        const float v = vv_warp_sum(acc[m]);
        if (m < p.M && lane == 0) y[(ulong)m * (ulong)p.N + row] = (half)(v * s);
    }
}

kernel void vv_int8_dequant(constant vv_i8w_p& p [[buffer(0)]],
                            device const char* q [[buffer(1)]],
                            device const float* scales [[buffer(2)]],
                            device half* out [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
    const int per_row = p.K >> 4;
    if ((ulong)tid >= (ulong)p.N * (ulong)per_row) return;
    const int row = (int)(tid / (uint)per_row);
    const int e0 = (int)(tid % (uint)per_row) * 16;
    float w[16];
    i8_unpack16(*(device const uint4*)(q + (ulong)row * (ulong)p.K + e0), w);
    const float s = scales[row];
    device half4* dst = (device half4*)(out + (ulong)row * (ulong)p.K + e0);
    for (int j = 0; j < 4; j++)
        dst[j] = half4((half)(w[4 * j] * s), (half)(w[4 * j + 1] * s),
                       (half)(w[4 * j + 2] * s), (half)(w[4 * j + 3] * s));
}

/* ─── INT4 row-major (legacy AWQ layout), w = q * s + m ─────────────────── */

struct vv_awq_p { int M; int N; int K; int gshift; int has_bias; };

kernel void vv_awq_gemv(constant vv_awq_p& p [[buffer(0)]],
                        device const half* x [[buffer(1)]],
                        device const uint* w [[buffer(2)]],
                        device const half* scales [[buffer(3)]],
                        device const half* mins [[buffer(4)]],
                        device const half* bias [[buffer(5)]],
                        device half* y [[buffer(6)]],
                        VV_GRID_ARGS) {
    const int row = (int)blockIdx.x * 8 + (int)warp;
    if (row >= p.N) return;
    const int groups = p.K >> p.gshift;
    device const uint* wrow = w + (ulong)row * (ulong)(p.K >> 3);
    device const half* srow = scales + (ulong)row * (ulong)groups;
    device const half* mrow = mins + (ulong)row * (ulong)groups;
    float acc = 0.0f;
    for (int base = 0; base < p.K; base += 256) {
        const int e0 = base + (int)lane * 8;
        const uint bits = wrow[e0 >> 3];
        const int g = e0 >> p.gshift;
        const float sc = (float)srow[g], mn = (float)mrow[g];
        const half4 xa = *(device const half4*)(x + e0);
        const half4 xb = *(device const half4*)(x + e0 + 4);
        const float xs[8] = { (float)xa.x, (float)xa.y, (float)xa.z, (float)xa.w,
                              (float)xb.x, (float)xb.y, (float)xb.z, (float)xb.w };
        for (int b = 0; b < 4; b++) {
            const uint byte = (bits >> (b * 8)) & 0xFFu;
            acc = fma(fma((float)(byte >> 4), sc, mn), xs[2 * b], acc);
            acc = fma(fma((float)(byte & 0xFu), sc, mn), xs[2 * b + 1], acc);
        }
    }
    acc = vv_warp_sum(acc);
    if (lane == 0) {
        if (p.has_bias) acc += (float)bias[row];
        y[row] = (half)acc;
    }
}

kernel void vv_awq_gemm_small(constant vv_awq_p& p [[buffer(0)]],
                              device const half* x [[buffer(1)]],
                              device const uint* w [[buffer(2)]],
                              device const half* scales [[buffer(3)]],
                              device const half* mins [[buffer(4)]],
                              device half* y [[buffer(5)]],
                              VV_GRID_ARGS) {
    const int row = (int)blockIdx.x * 8 + (int)warp;
    if (row >= p.N) return;
    const int groups = p.K >> p.gshift;
    device const uint* wrow = w + (ulong)row * (ulong)(p.K >> 3);
    device const half* srow = scales + (ulong)row * (ulong)groups;
    device const half* mrow = mins + (ulong)row * (ulong)groups;
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int base = 0; base < p.K; base += 256) {
        const int e0 = base + (int)lane * 8;
        const uint bits = wrow[e0 >> 3];
        const int g = e0 >> p.gshift;
        const float sc = (float)srow[g], mn = (float)mrow[g];
        float wv[8];
        for (int b = 0; b < 4; b++) {
            const uint byte = (bits >> (b * 8)) & 0xFFu;
            wv[2 * b + 0] = fma((float)(byte >> 4), sc, mn);
            wv[2 * b + 1] = fma((float)(byte & 0xFu), sc, mn);
        }
        for (int m = 0; m < 8; m++) {
            if (m >= p.M) break;
            device const half* xr = x + (ulong)m * (ulong)p.K + e0;
            const half4 xa = *(device const half4*)(xr);
            const half4 xb = *(device const half4*)(xr + 4);
            const float xs[8] = { (float)xa.x, (float)xa.y, (float)xa.z, (float)xa.w,
                                  (float)xb.x, (float)xb.y, (float)xb.z, (float)xb.w };
            for (int b = 0; b < 8; b++) acc[m] = fma(wv[b], xs[b], acc[m]);
        }
    }
    for (int m = 0; m < 8; m++) {
        const float v = vv_warp_sum(acc[m]);
        if (m < p.M && lane == 0) y[(ulong)m * (ulong)p.N + row] = (half)v;
    }
}

kernel void vv_awq_dequant(constant vv_awq_p& p [[buffer(0)]],
                           device const uint* w [[buffer(1)]],
                           device const half* scales [[buffer(2)]],
                           device const half* mins [[buffer(3)]],
                           device half* out [[buffer(4)]],
                           uint tid [[thread_position_in_grid]]) {
    const int per_row = p.K >> 3;
    if ((ulong)tid >= (ulong)p.N * (ulong)per_row) return;
    const int row = (int)(tid / (uint)per_row);
    const int wi = (int)(tid % (uint)per_row);
    const int e0 = wi * 8;
    const int groups = p.K >> p.gshift;
    const uint bits = w[(ulong)row * (ulong)per_row + wi];
    const float sc = (float)scales[(ulong)row * (ulong)groups + (e0 >> p.gshift)];
    const float mn = (float)mins[(ulong)row * (ulong)groups + (e0 >> p.gshift)];
    half v[8];
    for (int b = 0; b < 4; b++) {
        const uint byte = (bits >> (b * 8)) & 0xFFu;
        v[2 * b + 0] = (half)fma((float)(byte >> 4), sc, mn);
        v[2 * b + 1] = (half)fma((float)(byte & 0xFu), sc, mn);
    }
    device half4* dst = (device half4*)(out + (ulong)row * (ulong)p.K + e0);
    dst[0] = half4(v[0], v[1], v[2], v[3]);
    dst[1] = half4(v[4], v[5], v[6], v[7]);
}
