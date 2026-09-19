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
