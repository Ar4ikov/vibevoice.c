/**
 * @file metal_stubs.c
 * @brief Device ops the Metal backend does not implement yet.
 *
 * Same contract as device_none.c: VV_ERR_UNSUPPORTED, which the callers
 * that have a fallback take. Each op leaves this file as its kernel lands.
 */

#include "vibevoice/device.h"
#include "vibevoice/kv_quant.h"
#include "vibevoice/vibevoice.h"

/* ─── Compute: not available ────────────────────────────────────────────── */

#define VV_STUB(sig, args) sig { args return VV_ERR_UNSUPPORTED; }
#define U(x) (void)(x);



/* BitNet integer ops */

#undef U
#undef VV_STUB

vv_status_t vv_skinny_linear_dev(const void* A, int f,
                                 const vv_skinny_proj_t* p, int n, int M,
                                 int K, float al, void* s) {
    (void)A; (void)f; (void)p; (void)n; (void)M; (void)K; (void)al; (void)s;
    return VV_ERR_UNSUPPORTED;
}

#define U(x) (void)(x);

/* Speculative decoding (dflash.cu): no kernels here, so no drafter. */
vv_status_t vv_lm_head_rows_dev(const void* x, const void* W, void* l, int M,
                                int V, int K, void* s) {
    U(x) U(W) U(l) U(M) U(V) U(K) U(s) return VV_ERR_UNSUPPORTED;
}
size_t vv_argmax_rows_scratch_bytes(int M) { U(M) return 0; }
vv_status_t vv_argmax_rows_dev(const void* l, int M, int V, void* scr,
                               int32_t* t, void* s) {
    U(l) U(M) U(V) U(scr) U(t) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_topk_rows_dev(const float* l, int M, int V, int k, float* v,
                             int32_t* i, void* s) {
    U(l) U(M) U(V) U(k) U(v) U(i) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dflash_conv_dev(const void* x, const void* d, int ld,
                               const float* b, void* o, int M, int H, int t,
                               int g, int bl, void* s) {
    U(x) U(d) U(ld) U(b) U(o) U(M) U(H) U(t) U(g) U(bl) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dflash_walk_dev(const void* h, const float* u,
                               const int32_t* c, const void* p,
                               const void* q, const int32_t* a, int M, int k,
                               int r, int32_t* o, void* s) {
    U(h) U(u) U(c) U(p) U(q) U(a) U(M) U(k) U(r) U(o) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a16_gemv_rows_dev(const void* x, int M,
                                   const vv_w4a16_proj_t* p, int n, int K,
                                   int g, void* s) {
    U(x) U(M) U(p) U(n) U(K) U(g) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_attn_decode_rows(int b, const void* q, const vv_kv_view_t* kv,
                                void* o, int h, int r, int c, const int* dc,
                                void* scr, void* s) {
    U(b) U(q) U(kv) U(o) U(h) U(r) U(c) U(dc) U(scr) U(s)
    return VV_ERR_UNSUPPORTED;
}
size_t vv_attn_rows_scratch_bytes(int h, int d, int r) {
    U(h) U(d) U(r) return 0;
}
vv_status_t vv_rmsnorm_f32in_dev(const float* x, const void* w, void* y,
                                 int r, int H, float e, void* s) {
    U(x) U(w) U(y) U(r) U(H) U(e) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_add_scaled_f16_dev(float* h, const void* y, float sc, int n,
                                  bool set, void* s) {
    U(h) U(y) U(sc) U(n) U(set) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gather_i32_dev(const int32_t* m, int nm, int32_t* i, int n,
                              void* s) {
    U(m) U(nm) U(i) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dflash_block_ids_dev(const int32_t* a, int m, int n,
                                    int32_t* i, void* s) {
    U(a) U(m) U(n) U(i) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dflash_accept_dev(const int32_t* d, const int32_t* p, int n,
                                 int32_t* o, int32_t* c, void* s) {
    U(d) U(p) U(n) U(o) U(c) U(s) return VV_ERR_UNSUPPORTED;
}
#undef U
