/*
 * runtime.metal -- the copies, fills and device-side scalars behind the
 * memory ops of metal_runtime.m.
 */

struct vv_fill_p { ulong n; uint value; uint pad; };

kernel void vv_copy16(constant vv_fill_p& p [[buffer(0)]],
                      device uint4* dst [[buffer(1)]],
                      device const uint4* src [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = src[i];
}

kernel void vv_copy4(constant vv_fill_p& p [[buffer(0)]],
                     device uint* dst [[buffer(1)]],
                     device const uint* src [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = src[i];
}

kernel void vv_copy1(constant vv_fill_p& p [[buffer(0)]],
                     device uchar* dst [[buffer(1)]],
                     device const uchar* src [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = src[i];
}

kernel void vv_fill4(constant vv_fill_p& p [[buffer(0)]],
                     device uint* dst [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = p.value;
}

kernel void vv_fill1(constant vv_fill_p& p [[buffer(0)]],
                     device uchar* dst [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = (uchar)(p.value & 0xFFu);
}

/* Host bytes carried in the launch: a 16-byte header {n}, then the data. */
kernel void vv_write_bytes(constant uchar* p [[buffer(0)]],
                           device uchar* dst [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    const ulong n = *(constant ulong*)p;
    if (i < n) dst[i] = p[16 + i];
}

struct vv_copy_at_p { ulong n; };

/* dst_base + (*index) * n <- src: the destination is read on the device, so
 * a replayed graph writes where this token belongs. */
kernel void vv_copy_at16(constant vv_copy_at_p& p [[buffer(0)]],
                         device uint4* dst [[buffer(1)]],
                         device const uint4* src [[buffer(2)]],
                         device const int* index [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    const ulong vecs = p.n / 16;
    if (i < vecs) dst[(ulong)(*index) * vecs + i] = src[i];
}

kernel void vv_copy_at1(constant vv_copy_at_p& p [[buffer(0)]],
                        device uchar* dst [[buffer(1)]],
                        device const uchar* src [[buffer(2)]],
                        device const int* index [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[(ulong)(*index) * p.n + i] = src[i];
}

struct vv_pos_add_p { int delta; };

kernel void vv_pos_add(constant vv_pos_add_p& p [[buffer(0)]],
                       device int* dst [[buffer(1)]],
                       device const int* src [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i == 0) *dst = *src + p.delta;
}
