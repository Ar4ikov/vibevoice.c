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

vv_status_t vv_awq_gemv_dev(const void* x, const uint32_t* p, const uint32_t* m,
                            const void* s, const void* b, void* y,
                            int N, int K, int g, void* st) {
    U(x) U(p) U(m) U(s) U(b) U(y) U(N) U(K) U(g) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_awq_gemm_dev(const void* a, const uint32_t* p, const uint32_t* m,
                            const void* s, void* o, void* t,
                            int M, int N, int K, int g, void* st) {
    U(a) U(p) U(m) U(s) U(o) U(t) U(M) U(N) U(K) U(g) U(st)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_col_absmax_dev(const void* x, int M, int K, float* a,
                              void* st) {
    U(x) U(M) U(K) U(a) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_act_quant_dev(const void* x, int M, int K, int l, int8_t* q,
                             float* sx, int32_t* xs, void* st) {
    U(x) U(M) U(K) U(l) U(q) U(sx) U(xs) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_rmsnorm_q8_dev(const void* x, const void* w, int M, int K,
                              float e, int l, int8_t* q, float* sx,
                              int32_t* xs, void* st) {
    U(x) U(w) U(M) U(K) U(e) U(l) U(q) U(sx) U(xs) U(st)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_swiglu_q8_dev(const void* g, const void* u, int M, int K,
                             int l, int8_t* q, float* sx, int32_t* xs,
                             void* st) {
    U(g) U(u) U(M) U(K) U(l) U(q) U(sx) U(xs) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w8a8_linear_dev(const int8_t* q, const float* sx,
                               const int8_t* w, const float* sw, const void* b,
                               const void* r, void* y, int f, int M, int N,
                               int K, int pa, void* st) {
    U(q) U(sx) U(w) U(sw) U(b) U(r) U(y) U(f) U(M) U(N) U(K) U(pa) U(st)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a8_linear_dev(const int8_t* q, int l, const float* sx,
                               const int32_t* xs, const void* w,
                               const void* sz, int g,
                               const void* b, const void* r, void* y, int f,
                               int M, int N, int K, int pa, void* st) {
    U(q) U(l) U(sx) U(xs) U(w) U(sz) U(g) U(b) U(r) U(y) U(f) U(M) U(N)
    U(K) U(pa) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_i8_linear_multi_dev(const int8_t* q, int l, const float* sx,
                                   const int32_t* xs, int w4,
                                   const vv_i8_proj_t* p, int n, int g,
                                   int M, int K, void* st) {
    U(q) U(l) U(sx) U(xs) U(w4) U(p) U(n) U(g) U(M) U(K) U(st)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dequant_awq_dev(const uint32_t* p, const uint32_t* m,
                               const void* s, void* o, int N, int K, int g,
                               void* st) {
    U(p) U(m) U(s) U(o) U(N) U(K) U(g) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_int8_gemv_dev(const void* x, const int8_t* q, const float* s,
                             const void* b, void* y, int N, int K, void* st) {
    U(x) U(q) U(s) U(b) U(y) U(N) U(K) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_int8_gemm_dev(const void* a, const int8_t* q, const float* s,
                             void* o, void* t, int M, int N, int K, void* st) {
    U(a) U(q) U(s) U(o) U(t) U(M) U(N) U(K) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dequant_int8_dev(const int8_t* q, const float* s, void* o,
                                int N, int K, void* st) {
    U(q) U(s) U(o) U(N) U(K) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a16_gemv_dev(const void* x, const void* p, const void* sz,
                              const void* b, void* y, int N, int K, int g,
                              void* st) {
    U(x) U(p) U(sz) U(b) U(y) U(N) U(K) U(g) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a16_gemv_multi_dev(const void* x, const vv_w4a16_proj_t* p,
                                    int n, int K, int g, void* st) {
    U(x) U(p) U(n) U(K) U(g) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a16_gather_dev(const void* x, const int32_t* p, void* o,
                                int M, int K, void* st) {
    U(x) U(p) U(o) U(M) U(K) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a16_gemm_dev(const void* a, const void* p, const void* sz,
                              const void* b, void* o, void* t, size_t tb,
                              int M, int N, int K, int g, void* st) {
    U(a) U(p) U(sz) U(b) U(o) U(t) U(tb) U(M) U(N) U(K) U(g) U(st)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_w4a16_dequant_dev(const void* p, const void* sz, void* o,
                                 int N, int K, int g, void* st) {
    U(p) U(sz) U(o) U(N) U(K) U(g) U(st) return VV_ERR_UNSUPPORTED;
}


/* BitNet integer ops */
vv_status_t vv_act_quant_i8_dev(const void* x, int f16, int M, int K,
                                int8_t* q, float* sc, int32_t* sum, void* s) {
    U(x) U(f16) U(M) U(K) U(q) U(sc) U(sum) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_ternary_gemm_dev(const int8_t* q, const int32_t* xs,
                                const float* xsc, const uint8_t* c, float ws,
                                const float* b, int32_t* acc, void* y,
                                int f16, int M, int N, int K, void* s) {
    U(q) U(xs) U(xsc) U(c) U(ws) U(b) U(acc) U(y) U(f16) U(M) U(N) U(K) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_i8_gemm_dev(const int8_t* a, const float* as, const int8_t* w,
                           float ws, const float* b, int32_t* acc, float* y,
                           float* am, int M, int N, int K, void* s) {
    U(a) U(as) U(w) U(ws) U(b) U(acc) U(y) U(am) U(M) U(N) U(K) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_i8s_requant_dev(const float* y, int64_t n, const float* am,
                               int relu, int8_t* q, float* os, void* s) {
    U(y) U(n) U(am) U(relu) U(q) U(os) U(s) return VV_ERR_UNSUPPORTED;
}
size_t vv_i8_head_argmax_scratch_bytes(int V) { U(V) return 0; }
vv_status_t vv_i8_head_argmax_dev(const int8_t* q, const float* sc,
                                  const int8_t* w, const float* ws, int V,
                                  int K, int32_t* tok, float* val, void* scr,
                                  void* s) {
    U(q) U(sc) U(w) U(ws) U(V) U(K) U(tok) U(val) U(scr) U(s)
    return VV_ERR_UNSUPPORTED;
}

#undef U
#undef VV_STUB
