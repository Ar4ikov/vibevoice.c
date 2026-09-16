/**
 * @file device_none.c
 * @brief The op set when no accelerator backend is compiled in.
 *
 * Built instead of src/cuda or src/metal for a CPU-only target. The pipeline
 * already decides at runtime whether a device is usable, but the symbols
 * still have to exist for the link, and the memory ops still have to work:
 * the speech tokenizer allocates through them before it knows which path it
 * is on.
 *
 * So allocation, copies and streams are implemented against host memory, and
 * every compute op returns VV_ERR_UNSUPPORTED. vv_dev_get_device_info
 * reports no device, which is what steers the pipeline to VV_PLACE_CPU_ONLY.
 */

#include "vibevoice/device.h"
#include "vibevoice/vibevoice.h"

#include <stdlib.h>
#include <string.h>

/* ─── Memory and streams: real, on the host ─────────────────────────────── */

vv_status_t vv_dev_alloc(void** ptr, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    *ptr = vv_alloc(size);
    return *ptr ? VV_OK : VV_ERR_OUT_OF_MEMORY;
}

vv_status_t vv_dev_alloc_pinned(void** ptr, size_t size) {
    return vv_dev_alloc(ptr, size);
}

vv_status_t vv_dev_free(void* ptr) { vv_free(ptr); return VV_OK; }
vv_status_t vv_dev_free_pinned(void* ptr) { vv_free(ptr); return VV_OK; }

vv_status_t vv_dev_memcpy_h2d(void* dst, const void* src, size_t size, void* s) {
    (void)s;
    if (!dst || !src) return VV_ERR_NULL_PTR;
    memcpy(dst, src, size);
    return VV_OK;
}
vv_status_t vv_dev_memcpy_d2h(void* dst, const void* src, size_t size, void* s) {
    return vv_dev_memcpy_h2d(dst, src, size, s);
}
vv_status_t vv_dev_memcpy_d2d(void* dst, const void* src, size_t size, void* s) {
    return vv_dev_memcpy_h2d(dst, src, size, s);
}
vv_status_t vv_dev_memset_async(void* p, int v, size_t n, void* s) {
    (void)s;
    if (!p) return VV_ERR_NULL_PTR;
    memset(p, v, n);
    return VV_OK;
}
vv_status_t vv_dev_memcpy_d2d_at(void* dst_base, const void* src, size_t size,
                                 const int* d_index, void* s) {
    (void)s;
    if (!dst_base || !src || !d_index) return VV_ERR_NULL_PTR;
    memcpy((unsigned char*)dst_base + (size_t)(*d_index) * size, src, size);
    return VV_OK;
}
vv_status_t vv_pos_add_dev(int* dst, const int* src, int delta, void* s) {
    (void)s;
    if (!dst || !src) return VV_ERR_NULL_PTR;
    *dst = *src + delta;
    return VV_OK;
}
vv_status_t vv_dev_memset(void* ptr, int value, size_t size) {
    if (!ptr) return VV_ERR_NULL_PTR;
    memset(ptr, value, size);
    return VV_OK;
}

vv_status_t vv_dev_stream_create(void** stream) {
    if (stream) *stream = NULL;
    return VV_OK;
}
vv_status_t vv_dev_stream_destroy(void* s) { (void)s; return VV_OK; }
vv_status_t vv_dev_stream_sync(void* s) { (void)s; return VV_OK; }
vv_status_t vv_dev_set_device(int id) { (void)id; return VV_ERR_NOT_FOUND; }

vv_status_t vv_dev_get_device_info(int id, size_t* total, size_t* freem,
                                   int* sm) {
    (void)id;
    if (total) *total = 0;
    if (freem) *freem = 0;
    if (sm) *sm = 0;
    return VV_ERR_NOT_FOUND;
}

int vv_dev_device_count(void) { return 0; }

vv_status_t vv_dev_get_device_name(int id, char* buf, size_t n) {
    (void)id;
    if (buf && n) buf[0] = '\0';
    return VV_ERR_NOT_FOUND;
}

vv_status_t vv_dev_event_create(void** ev) { if (ev) *ev = NULL; return VV_OK; }
vv_status_t vv_dev_event_destroy(void* ev) { (void)ev; return VV_OK; }
vv_status_t vv_dev_event_record(void* ev, void* s) { (void)ev; (void)s; return VV_OK; }
vv_status_t vv_dev_stream_wait_event(void* s, void* ev) { (void)s; (void)ev; return VV_OK; }
vv_status_t vv_dev_host_register(void* p, size_t n) { (void)p; (void)n; return VV_OK; }
vv_status_t vv_dev_host_unregister(void* p) { (void)p; return VV_OK; }

const char* vv_dev_backend_name(void) { return "none"; }

void vv_gemm_cleanup(void) {}

/* ─── Compute: not available ────────────────────────────────────────────── */

#define VV_STUB(sig, args) sig { args return VV_ERR_UNSUPPORTED; }
#define U(x) (void)(x);

vv_status_t vv_nf4_gemv_dev(const void* x, const uint8_t* p, const void* s,
                            const void* b, void* y, int N, int K, void* st) {
    U(x) U(p) U(s) U(b) U(y) U(N) U(K) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_nf4_gemm_dev(const void* a, const uint8_t* p, const void* s,
                            void* o, void* t, int M, int N, int K, int bs,
                            void* st) {
    U(a) U(p) U(s) U(o) U(t) U(M) U(N) U(K) U(bs) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dequant_nf4_dev(const uint8_t* p, const void* s, void* o,
                               int n, int bs, void* st) {
    U(p) U(s) U(o) U(n) U(bs) U(st) return VV_ERR_UNSUPPORTED;
}
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
vv_status_t vv_dequant_awq_dev(const uint32_t* p, const uint32_t* m,
                               const void* s, void* o, int N, int K, int g,
                               void* st) {
    U(p) U(m) U(s) U(o) U(N) U(K) U(g) U(st) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gemm_fp16_dev(const void* A, const void* B, void* C,
                             int M, int N, int K, float al, float be, void* s) {
    U(A) U(B) U(C) U(M) U(N) U(K) U(al) U(be) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gemm_fp16_nn_dev(const void* A, const void* B, void* C,
                                int M, int K, int P, float al, float be,
                                void* s) {
    U(A) U(B) U(C) U(M) U(K) U(P) U(al) U(be) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_embedding_dev(const void* t, const int32_t* i, void* o,
                             int n, int h, void* s) {
    U(t) U(i) U(o) U(n) U(h) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_rmsnorm_dev(const void* i, const void* w, void* o,
                           int n, int h, float e, void* s) {
    U(i) U(w) U(o) U(n) U(h) U(e) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_rope_dev(void* x, int n, int h, int d, int p,
                        const int* dp, float t, void* s) {
    U(x) U(n) U(h) U(d) U(p) U(dp) U(t) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_swiglu_dev(const void* g, const void* u, void* o, int n, void* s) {
    U(g) U(u) U(o) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_swiglu_fused_dev(const void* gu, void* o, int r, int i, void* s) {
    U(gu) U(o) U(r) U(i) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_residual_add_dev(void* x, const void* y, int n, void* s) {
    U(x) U(y) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_bias_add_dev(void* o, const void* b, int M, int N, void* s) {
    U(o) U(b) U(M) U(N) U(s) return VV_ERR_UNSUPPORTED;
}
size_t vv_gqa_decode_scratch_bytes(int nq, int d) { U(nq) U(d) return 0; }
vv_status_t vv_dev_graph_begin(void* s) { U(s) return VV_ERR_UNSUPPORTED; }
vv_status_t vv_dev_graph_end(void* s, void** g) {
    U(s) U(g) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_dev_graph_launch(void* g, void* s) {
    U(g) U(s) return VV_ERR_UNSUPPORTED;
}
void vv_dev_graph_destroy(void* g) { U(g) }

vv_status_t vv_gqa_attention_decode_dev(const void* q, const void* k,
                                        const void* v, void* o,
                                        int nq, int nkv, int d, int c,
                                        const int* dc, void* w, void* s) {
    U(q) U(k) U(v) U(o) U(nq) U(nkv) U(d) U(c) U(dc) U(w) U(s)
    return VV_ERR_UNSUPPORTED;
}
int vv_gqa_decode_shape(int c) { U(c) return 0; }
vv_status_t vv_gqa_attention_prefill_cached_dev(
    const void* q, const void* k, const void* v, void* o,
    int nq, int nkv, int d, int ql, int qo, int kl, bool c, void* s) {
    U(q) U(k) U(v) U(o) U(nq) U(nkv) U(d) U(ql) U(qo) U(kl) U(c) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gqa_attention_prefill_dev(const void* q, const void* k,
                                         const void* v, void* o,
                                         int nq, int nkv, int d, int sl,
                                         bool c, void* s) {
    U(q) U(k) U(v) U(o) U(nq) U(nkv) U(d) U(sl) U(c) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gqa_attention_decode_q_dev(
    const void* q, const void* ks, const void* vs, const void* km,
    const void* vm, void* o, int nq, int nkv, int d, int c,
    const int* dc, int f, void* w, void* s) {
    U(q) U(ks) U(vs) U(km) U(vm) U(o) U(nq) U(nkv) U(d) U(c) U(dc) U(f)
    U(w) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gqa_attention_prefill_q_dev(
    const void* q, const void* ks, const void* vs, const void* km,
    const void* vm, void* o, int nq, int nkv, int d, int ql, int qo, int kl,
    bool c, int f, void* s) {
    U(q) U(ks) U(vs) U(km) U(vm) U(o) U(nq) U(nkv) U(d) U(ql) U(qo) U(kl)
    U(c) U(f) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_kv_quant_store_dev(const void* k, const void* v, void* ks,
                                  void* vs, void* km, void* vm, void* kr,
                                  bool br, int nkv, int d, int p,
                                  const int* dp, int n, int f, void* s) {
    U(k) U(v) U(ks) U(vs) U(km) U(vm) U(kr) U(br) U(nkv) U(d) U(p) U(dp)
    U(n) U(f) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_kv_dequant_dev(const void* st, const void* m, const void* r,
                              void* o, int nkv, int d, int n, int f, void* s) {
    U(st) U(m) U(r) U(o) U(nkv) U(d) U(n) U(f) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_kv_rotate_dev(void* x, int h, int d, int r, void* s) {
    U(x) U(h) U(d) U(r) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_kv_unrotate_dev(void* x, int h, int d, int r, void* s) {
    U(x) U(h) U(d) U(r) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_lm_head_gemv_dev(const void* x, const void* W, void* l,
                                int V, int K, void* s) {
    U(x) U(W) U(l) U(V) U(K) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_argmax_dev(const void* l, int V, void* sv, void* si,
                          void* ot, void* ov, void* s) {
    U(l) U(V) U(sv) U(si) U(ot) U(ov) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_conv1d_dev(const void* i, const void* w, const void* b, void* o,
                          int ic, int il, int oc, int k, int st, int g,
                          bool c, int* ol, void* s) {
    U(i) U(w) U(b) U(o) U(ic) U(il) U(oc) U(k) U(st) U(g) U(c) U(ol) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_conv1d_raw_dev(const void* i, const void* w, const void* b,
                              void* o, int ic, int il, int oc, int k, int st,
                              int g, int pl, int ol, void* s) {
    U(i) U(w) U(b) U(o) U(ic) U(il) U(oc) U(k) U(st) U(g) U(pl) U(ol) U(s)
    return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_rmsnorm_channel_first_dev(const void* i, const void* w, void* o,
                                         int c, int l, float e, void* s) {
    U(i) U(w) U(o) U(c) U(l) U(e) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_residual_add_scaled_dev(void* x, const void* y, const void* sc,
                                       int c, int l, void* s) {
    U(x) U(y) U(sc) U(c) U(l) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_channel_bias_add_dev(void* o, const void* b, int c, int l,
                                    void* s) {
    U(o) U(b) U(c) U(l) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_silu_dev(void* d, int n, void* s) {
    U(d) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gelu_dev(void* d, int n, void* s) {
    U(d) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_fp32_to_fp16_dev(const void* i, void* o, int n, void* s) {
    U(i) U(o) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_fp16_to_fp32_dev(const void* i, void* o, int n, void* s) {
    U(i) U(o) U(n) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_gather_tile_dev(const void* src, void* dst, int c, int fl,
                               int to, int tl, void* s) {
    U(src) U(dst) U(c) U(fl) U(to) U(tl) U(s) return VV_ERR_UNSUPPORTED;
}
vv_status_t vv_scatter_tile_dev(const void* src, void* dst, int c, int fl,
                                int to, int tl, void* s) {
    U(src) U(dst) U(c) U(fl) U(to) U(tl) U(s) return VV_ERR_UNSUPPORTED;
}

#undef U
#undef VV_STUB
