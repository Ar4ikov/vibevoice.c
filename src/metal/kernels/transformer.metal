/*
 * transformer.metal -- the element-wise and per-row ops of a Qwen2 layer:
 * embedding, RMSNorm, RoPE, SwiGLU, residual and bias adds, conversions.
 * Ports of src/cuda/{embedding,rmsnorm,rope,swiglu,conv_vae}.cu, with the
 * same reduction order where there is a reduction.
 */

struct vv_embed_p { int seq_len; int hidden; };

kernel void vv_embedding(constant vv_embed_p& p [[buffer(0)]],
                         device const half* table [[buffer(1)]],
                         device const int* ids [[buffer(2)]],
                         device half* out [[buffer(3)]],
                         VV_GRID_ARGS) {
    const int pos = (int)blockIdx.x;
    if (pos >= p.seq_len) return;
    const ulong tok = (ulong)ids[pos];
    device const half* row = table + tok * (ulong)p.hidden;
    device half* o = out + (ulong)pos * (ulong)p.hidden;
    for (int i = (int)threadIdx.x; i < p.hidden; i += (int)blockDim.x) o[i] = row[i];
}

struct vv_rmsnorm_p { int hidden; float eps; };

/* One threadgroup of 256 per row; the partial sums reduce as a tree in
 * threadgroup memory, in the order rmsnorm.cu uses. */
kernel void vv_rmsnorm(constant vv_rmsnorm_p& p [[buffer(0)]],
                       device const half* x [[buffer(1)]],
                       device const half* w [[buffer(2)]],
                       device half* y [[buffer(3)]],
                       VV_GRID_ARGS) {
    threadgroup float sdata[256];
    const int row = (int)blockIdx.x, tid = (int)threadIdx.x, nt = (int)blockDim.x;
    device const half* xr = x + (ulong)row * (ulong)p.hidden;
    device half* yr = y + (ulong)row * (ulong)p.hidden;
    float sum_sq = 0.0f;
    for (int i = tid; i < p.hidden; i += nt) {
        const float v = (float)xr[i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms = rsqrt(sdata[0] / (float)p.hidden + p.eps);
    for (int i = tid; i < p.hidden; i += nt) {
        const float v = (float)xr[i];
        const float wv = (float)w[i];
        yr[i] = (half)(v * rms * wv);
    }
}

struct vv_rope_p { int seq_len; int n_heads; int head_dim; int position_offset;
                   int use_dev_pos; float theta; };

/* x[seq][head][d] in place, half-split pairs (d, d + head_dim/2). Grid
 * (seq_len, n_heads), head_dim/2 threads. The position is read from device
 * memory in decode so the launch replays. */
kernel void vv_rope(constant vv_rope_p& p [[buffer(0)]],
                    device half* x [[buffer(1)]],
                    device const int* d_pos [[buffer(2)]],
                    VV_GRID_ARGS) {
    const int seq = (int)blockIdx.x, head = (int)blockIdx.y, d = (int)threadIdx.x;
    const int half_dim = p.head_dim / 2;
    if (d >= half_dim || seq >= p.seq_len) return;
    const int pos = seq + (p.use_dev_pos ? *d_pos : p.position_offset);
    const float freq = pow(p.theta, -2.0f * (float)d / (float)p.head_dim);
    const float angle = (float)pos * freq;
    const float c = cos(angle), s = sin(angle);
    const ulong base = ((ulong)seq * (ulong)p.n_heads + (ulong)head) * (ulong)p.head_dim;
    const float x0 = (float)x[base + d];
    const float x1 = (float)x[base + d + half_dim];
    x[base + d] = (half)(x0 * c - x1 * s);
    x[base + d + half_dim] = (half)(x0 * s + x1 * c);
}

struct vv_n_p { ulong n; };

static inline float vv_silu_mul(float g, float u) {
    return g / (1.0f + exp(-g)) * u;
}

kernel void vv_swiglu(constant vv_n_p& p [[buffer(0)]],
                      device const half* gate [[buffer(1)]],
                      device const half* up [[buffer(2)]],
                      device half* out [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    out[i] = (half)vv_silu_mul((float)gate[i], (float)up[i]);
}

struct vv_swiglu_fused_p { int rows; int inter; };

/* gate_up[r] = [gate(inter) | up(inter)] -> out[r][inter] */
kernel void vv_swiglu_fused(constant vv_swiglu_fused_p& p [[buffer(0)]],
                            device const half* gu [[buffer(1)]],
                            device half* out [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    const ulong n = (ulong)p.rows * (ulong)p.inter;
    if (i >= n) return;
    const ulong r = i / (ulong)p.inter, c = i % (ulong)p.inter;
    device const half* row = gu + r * 2 * (ulong)p.inter;
    out[i] = (half)vv_silu_mul((float)row[c], (float)row[p.inter + c]);
}

kernel void vv_residual_add(constant vv_n_p& p [[buffer(0)]],
                            device half* x [[buffer(1)]],
                            device const half* y [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    x[i] = (half)((float)x[i] + (float)y[i]);
}

struct vv_bias_p { int M; int N; };

kernel void vv_bias_add(constant vv_bias_p& p [[buffer(0)]],
                        device half* out [[buffer(1)]],
                        device const half* bias [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    const ulong total = (ulong)p.M * (ulong)p.N;
    if (i >= total) return;
    out[i] = (half)((float)out[i] + (float)bias[i % (ulong)p.N]);
}

kernel void vv_f32_to_f16(constant vv_n_p& p [[buffer(0)]],
                          device const float* in [[buffer(1)]],
                          device half* out [[buffer(2)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint3 t [[thread_position_in_threadgroup]],
                          uint3 ntg [[threadgroups_per_grid]]) {
    /* 2-D grid so counts past 2^32 / 256 threads still fit */
    const ulong i = ((ulong)tg.y * (ulong)ntg.x + (ulong)tg.x) * 256ul + t.x;
    if (i < p.n) out[i] = (half)in[i];
}

kernel void vv_f16_to_f32(constant vv_n_p& p [[buffer(0)]],
                          device const half* in [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < p.n) out[i] = (float)in[i];
}
