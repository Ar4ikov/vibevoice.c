/**
 * @file gguf.c
 * @brief GGUF reader. See gguf.h for the format and the contract.
 *
 * Everything that comes out of the file is a length or an offset that the
 * next read trusts, so the parser goes through one bounds-checked cursor and
 * never dereferences the mapping directly. A tensor pointer is handed out
 * only after its whole extent has been checked against the file size.
 */

#include "vibevoice/gguf.h"
#include "vibevoice/vibevoice.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define GGUF_MAGIC 0x46554747u /* "GGUF" little-endian */
#define GGUF_DEFAULT_ALIGNMENT 32u
/* Nested arrays are legal GGUF; nothing real nests more than once. */
#define GGUF_MAX_ARRAY_DEPTH 4

struct vv_gguf {
    /* mapping (NULL handles for vv_gguf_open_memory) */
#ifdef _WIN32
    HANDLE file_handle;
    HANDLE map_handle;
#else
    int fd;
#endif
    int            owns_mapping;
    const uint8_t* base;
    size_t         size;

    uint32_t version;
    uint32_t alignment;
    uint64_t data_offset;

    vv_gguf_kv_t*     kvs;
    int               n_kv;
    vv_gguf_tensor_t* tensors;
    int               n_tensors;

    /* one allocation holding every key, tensor name and string value */
    char*  strings;
    size_t strings_used;
    size_t strings_cap;
};

/* ─── Type traits ───────────────────────────────────────────────────────── */

typedef struct {
    uint32_t    type;
    const char* name;
    int64_t     blck;
    size_t      size;
} ggml_traits_t;

static const ggml_traits_t k_traits[] = {
    { VV_GGML_F32,  "f32",  1,   4   }, { VV_GGML_F16,  "f16",  1,   2   },
    { VV_GGML_Q4_0, "q4_0", 32,  18  }, { VV_GGML_Q4_1, "q4_1", 32,  20  },
    { VV_GGML_Q5_0, "q5_0", 32,  22  }, { VV_GGML_Q5_1, "q5_1", 32,  24  },
    { VV_GGML_Q8_0, "q8_0", 32,  34  }, { VV_GGML_Q8_1, "q8_1", 32,  36  },
    { VV_GGML_Q2_K, "q2_K", 256, 84  }, { VV_GGML_Q3_K, "q3_K", 256, 110 },
    { VV_GGML_Q4_K, "q4_K", 256, 144 }, { VV_GGML_Q5_K, "q5_K", 256, 176 },
    { VV_GGML_Q6_K, "q6_K", 256, 210 }, { VV_GGML_Q8_K, "q8_K", 256, 292 },
    { VV_GGML_I8,   "i8",   1,   1   }, { VV_GGML_I16,  "i16",  1,   2   },
    { VV_GGML_I32,  "i32",  1,   4   }, { VV_GGML_BF16, "bf16", 1,   2   },
    { VV_GGML_I2_S, "i2_s", 1,   1   }, { VV_GGML_I8_S, "i8_s", 1,   1   },
};

static const ggml_traits_t* traits(uint32_t type) {
    for (size_t i = 0; i < sizeof(k_traits) / sizeof(k_traits[0]); i++)
        if (k_traits[i].type == type) return &k_traits[i];
    return NULL;
}

const char* vv_ggml_type_name(uint32_t type) {
    const ggml_traits_t* t = traits(type);
    return t ? t->name : "unknown";
}

size_t vv_ggml_nbytes(uint32_t type, const int64_t ne[4]) {
    const ggml_traits_t* t = traits(type);
    if (!t) return 0;
    for (int i = 0; i < 4; i++)
        if (ne[i] < 0) return 0;
    if (ne[0] % t->blck != 0) return 0;
    /* overflow-safe product: every factor is bounded by the file size later,
     * but reject anything that would not fit in 2^62 first */
    uint64_t n = 1;
    for (int i = 0; i < 4; i++) {
        if (ne[i] != 0 && n > (UINT64_C(1) << 62) / (uint64_t)ne[i]) return 0;
        n *= (uint64_t)ne[i];
    }
    if (type == VV_GGML_I2_S) {
        /* four codes per byte, then the FP32 scale padded to 32 bytes */
        if (n % 4 != 0) return 0;
        return (size_t)(n / 4 + 32);
    }
    if (type == VV_GGML_I8_S) return (size_t)(n + 32);
    return (size_t)(n / (uint64_t)t->blck * t->size);
}

int64_t vv_gguf_nelements(const vv_gguf_tensor_t* t) {
    return t ? t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] : 0;
}

/* ─── Bounds-checked cursor ─────────────────────────────────────────────── */

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    int            bad;
} cursor_t;

static int cur_take(cursor_t* c, void* dst, size_t n) {
    if (c->bad || (size_t)(c->end - c->p) < n) { c->bad = 1; return 0; }
    if (dst) memcpy(dst, c->p, n);
    c->p += n;
    return 1;
}

static uint32_t cur_u32(cursor_t* c) { uint32_t v = 0; cur_take(c, &v, 4); return v; }
static uint64_t cur_u64(cursor_t* c) { uint64_t v = 0; cur_take(c, &v, 8); return v; }

/** Copy a GGUF string into the handle's string arena. */
static const char* cur_str(cursor_t* c, vv_gguf_t* g, int v1) {
    uint64_t len = v1 ? (uint64_t)cur_u32(c) : cur_u64(c);
    if (c->bad || len > (uint64_t)(c->end - c->p)) { c->bad = 1; return NULL; }
    if (g->strings_used + len + 1 > g->strings_cap) {
        size_t cap = g->strings_cap ? g->strings_cap * 2 : 65536;
        while (cap < g->strings_used + len + 1) cap *= 2;
        /* The arena moves on growth, so strings are stored as offsets until
         * parsing is done and resolved to pointers at the end. */
        char* grown = (char*)vv_alloc(cap);
        if (!grown) { c->bad = 1; return NULL; }
        if (g->strings) {
            memcpy(grown, g->strings, g->strings_used);
            vv_free(g->strings);
        }
        g->strings = grown;
        g->strings_cap = cap;
    }
    const size_t off = g->strings_used;
    memcpy(g->strings + off, c->p, (size_t)len);
    g->strings[off + len] = '\0';
    g->strings_used += (size_t)len + 1;
    c->p += len;
    /* encoded as an offset + 1 so that 0 stays "no string" */
    return (const char*)(uintptr_t)(off + 1);
}

static const char* resolve(const vv_gguf_t* g, const char* enc) {
    const uintptr_t off = (uintptr_t)enc;
    return off ? g->strings + (off - 1) : NULL;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case VV_GGUF_U8: case VV_GGUF_I8: case VV_GGUF_BOOL: return 1;
        case VV_GGUF_U16: case VV_GGUF_I16: return 2;
        case VV_GGUF_U32: case VV_GGUF_I32: case VV_GGUF_F32: return 4;
        case VV_GGUF_U64: case VV_GGUF_I64: case VV_GGUF_F64: return 8;
        default: return 0;
    }
}

/** Skip one value of type @p t (used for array elements). */
static void skip_value(cursor_t* c, uint32_t t, int v1, int depth) {
    if (depth > GGUF_MAX_ARRAY_DEPTH) { c->bad = 1; return; }
    const size_t sz = scalar_size(t);
    if (sz) { cur_take(c, NULL, sz); return; }
    if (t == VV_GGUF_STRING) {
        uint64_t len = v1 ? (uint64_t)cur_u32(c) : cur_u64(c);
        if (c->bad || len > (uint64_t)(c->end - c->p)) { c->bad = 1; return; }
        c->p += len;
        return;
    }
    if (t == VV_GGUF_ARRAY) {
        const uint32_t et = cur_u32(c);
        const uint64_t n = v1 ? (uint64_t)cur_u32(c) : cur_u64(c);
        const size_t esz = scalar_size(et);
        if (esz) {
            if (c->bad || n > (uint64_t)(c->end - c->p) / esz) { c->bad = 1; return; }
            c->p += n * esz;
            return;
        }
        for (uint64_t i = 0; i < n && !c->bad; i++) skip_value(c, et, v1, depth + 1);
        return;
    }
    c->bad = 1;
}

static void read_scalar(cursor_t* c, uint32_t t, vv_gguf_kv_t* kv) {
    uint8_t buf[8] = {0};
    const size_t sz = scalar_size(t);
    cur_take(c, buf, sz);
    switch (t) {
        case VV_GGUF_U8:   kv->v.u = buf[0]; break;
        case VV_GGUF_BOOL: kv->v.u = buf[0] != 0; break;
        case VV_GGUF_I8:   kv->v.i = (int8_t)buf[0]; break;
        case VV_GGUF_U16:  { uint16_t x; memcpy(&x, buf, 2); kv->v.u = x; } break;
        case VV_GGUF_I16:  { int16_t x;  memcpy(&x, buf, 2); kv->v.i = x; } break;
        case VV_GGUF_U32:  { uint32_t x; memcpy(&x, buf, 4); kv->v.u = x; } break;
        case VV_GGUF_I32:  { int32_t x;  memcpy(&x, buf, 4); kv->v.i = x; } break;
        case VV_GGUF_F32:  { float x;    memcpy(&x, buf, 4); kv->v.f = x; } break;
        case VV_GGUF_U64:  { uint64_t x; memcpy(&x, buf, 8); kv->v.u = x; } break;
        case VV_GGUF_I64:  { int64_t x;  memcpy(&x, buf, 8); kv->v.i = x; } break;
        case VV_GGUF_F64:  { double x;   memcpy(&x, buf, 8); kv->v.f = x; } break;
        default: c->bad = 1; break;
    }
}

/* ─── Parse ─────────────────────────────────────────────────────────────── */

static vv_status_t parse(vv_gguf_t* g) {
    cursor_t c = { g->base, g->base + g->size, 0 };

    const uint32_t magic = cur_u32(&c);
    if (c.bad || magic != GGUF_MAGIC) {
        VV_LOG_E("gguf: bad magic");
        return VV_ERR_PARSE;
    }
    g->version = cur_u32(&c);
    if (g->version < 1 || g->version > 3) {
        VV_LOG_E("gguf: unsupported version %u", g->version);
        return VV_ERR_MODEL_VERSION;
    }
    /* v1 used 32-bit counts and lengths */
    const int v1 = g->version == 1;
    const uint64_t n_tensors = v1 ? cur_u32(&c) : cur_u64(&c);
    const uint64_t n_kv = v1 ? cur_u32(&c) : cur_u64(&c);
    if (c.bad) return VV_ERR_PARSE;
    /* every kv needs at least 12 bytes and every tensor info 28, so a count
     * larger than that is a lie and would only make us allocate */
    const uint64_t left = (uint64_t)(c.end - c.p);
    if (n_kv > left / 12 || n_tensors > left / 28 || n_tensors > 1u << 20) {
        VV_LOG_E("gguf: implausible counts (%llu kv, %llu tensors)",
                 (unsigned long long)n_kv, (unsigned long long)n_tensors);
        return VV_ERR_PARSE;
    }

    g->kvs = (vv_gguf_kv_t*)vv_alloc(sizeof(vv_gguf_kv_t) * (n_kv ? n_kv : 1));
    g->tensors = (vv_gguf_tensor_t*)vv_alloc(sizeof(vv_gguf_tensor_t) *
                                             (n_tensors ? n_tensors : 1));
    if (!g->kvs || !g->tensors) return VV_ERR_OUT_OF_MEMORY;
    memset(g->kvs, 0, sizeof(vv_gguf_kv_t) * (n_kv ? n_kv : 1));
    memset(g->tensors, 0, sizeof(vv_gguf_tensor_t) * (n_tensors ? n_tensors : 1));

    g->alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < n_kv && !c.bad; i++) {
        vv_gguf_kv_t* kv = &g->kvs[i];
        kv->key = cur_str(&c, g, v1);
        kv->type = cur_u32(&c);
        if (c.bad) break;
        if (kv->type == VV_GGUF_STRING) {
            kv->str = cur_str(&c, g, v1);
        } else if (kv->type == VV_GGUF_ARRAY) {
            kv->arr_type = cur_u32(&c);
            kv->arr_n = v1 ? cur_u32(&c) : cur_u64(&c);
            kv->arr_data = c.p;
            const size_t esz = scalar_size(kv->arr_type);
            if (esz) {
                if (kv->arr_n > (uint64_t)(c.end - c.p) / esz) { c.bad = 1; break; }
                c.p += kv->arr_n * esz;
            } else {
                for (uint64_t j = 0; j < kv->arr_n && !c.bad; j++)
                    skip_value(&c, kv->arr_type, v1, 1);
            }
        } else if (scalar_size(kv->type)) {
            read_scalar(&c, kv->type, kv);
        } else {
            c.bad = 1;
        }
        g->n_kv = (int)(i + 1);
    }
    if (c.bad) {
        VV_LOG_E("gguf: truncated or malformed metadata");
        return VV_ERR_PARSE;
    }

    for (uint64_t i = 0; i < n_tensors && !c.bad; i++) {
        vv_gguf_tensor_t* t = &g->tensors[i];
        t->name = cur_str(&c, g, v1);
        const uint32_t nd = cur_u32(&c);
        if (c.bad || nd < 1 || nd > 4) { c.bad = 1; break; }
        t->n_dims = (int)nd;
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < nd; d++) {
            const uint64_t ne = v1 ? cur_u32(&c) : cur_u64(&c);
            if (ne > (UINT64_C(1) << 40)) { c.bad = 1; break; }
            t->ne[d] = (int64_t)ne;
        }
        t->type = cur_u32(&c);
        t->offset = cur_u64(&c);
        g->n_tensors = (int)(i + 1);
    }
    if (c.bad) {
        VV_LOG_E("gguf: truncated or malformed tensor table");
        return VV_ERR_PARSE;
    }

    /* Keys and names were stored as arena offsets; resolve them now that the
     * arena no longer moves. */
    for (int i = 0; i < g->n_kv; i++) {
        g->kvs[i].key = resolve(g, g->kvs[i].key);
        g->kvs[i].str = resolve(g, g->kvs[i].str);
        if (!g->kvs[i].key) return VV_ERR_PARSE;
    }
    for (int i = 0; i < g->n_tensors; i++) {
        g->tensors[i].name = resolve(g, g->tensors[i].name);
        if (!g->tensors[i].name) return VV_ERR_PARSE;
    }

    const vv_gguf_kv_t* al = vv_gguf_find_kv(g, "general.alignment");
    if (al) {
        const uint64_t a = al->v.u;
        if (al->type != VV_GGUF_U32 || a == 0 || (a & (a - 1)) != 0 || a > 65536) {
            VV_LOG_E("gguf: bad general.alignment");
            return VV_ERR_PARSE;
        }
        g->alignment = (uint32_t)a;
    }

    const uint64_t pos = (uint64_t)(c.p - g->base);
    g->data_offset = (pos + g->alignment - 1) / g->alignment * g->alignment;
    if (g->data_offset > g->size) {
        VV_LOG_E("gguf: data block starts past the end of the file");
        return VV_ERR_PARSE;
    }
    const uint64_t data_size = g->size - g->data_offset;

    for (int i = 0; i < g->n_tensors; i++) {
        vv_gguf_tensor_t* t = &g->tensors[i];
        const size_t nb = vv_ggml_nbytes(t->type, t->ne);
        if (!traits(t->type)) {
            /* Unknown types are listed but carry no data pointer: the caller
             * cannot decode them anyway, and we cannot bound them. */
            VV_LOG_W("gguf: tensor '%s' has unknown type %u", t->name, t->type);
            continue;
        }
        if (nb == 0 && vv_gguf_nelements(t) != 0) {
            VV_LOG_E("gguf: tensor '%s' has a shape %s cannot hold", t->name,
                     vv_ggml_type_name(t->type));
            return VV_ERR_PARSE;
        }
        if (t->offset % g->alignment != 0 || t->offset > data_size ||
            nb > data_size - t->offset) {
            VV_LOG_E("gguf: tensor '%s' lies outside the data block", t->name);
            return VV_ERR_PARSE;
        }
        t->nbytes = nb;
        t->data = g->base + g->data_offset + t->offset;
    }
    return VV_OK;
}

/* ─── Open / close ──────────────────────────────────────────────────────── */

static vv_gguf_t* new_handle(void) {
    vv_gguf_t* g = (vv_gguf_t*)vv_alloc(sizeof(vv_gguf_t));
    if (!g) return NULL;
    memset(g, 0, sizeof(*g));
#ifdef _WIN32
    g->file_handle = INVALID_HANDLE_VALUE;
#else
    g->fd = -1;
#endif
    return g;
}

vv_status_t vv_gguf_open_memory(const void* data, size_t size, vv_gguf_t** out) {
    if (!data || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    vv_gguf_t* g = new_handle();
    if (!g) return VV_ERR_OUT_OF_MEMORY;
    g->base = (const uint8_t*)data;
    g->size = size;
    vv_status_t s = parse(g);
    if (s != VV_OK) { vv_gguf_close(g); return s; }
    *out = g;
    return VV_OK;
}

vv_status_t vv_gguf_open(const char* path, vv_gguf_t** out) {
    if (!path || !out) return VV_ERR_NULL_PTR;
    *out = NULL;
    vv_gguf_t* g = new_handle();
    if (!g) return VV_ERR_OUT_OF_MEMORY;

#ifdef _WIN32
    g->file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g->file_handle == INVALID_HANDLE_VALUE) { vv_gguf_close(g); return VV_ERR_IO; }
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(g->file_handle, &fsize) || fsize.QuadPart == 0) {
        vv_gguf_close(g);
        return VV_ERR_IO;
    }
    g->map_handle = CreateFileMappingA(g->file_handle, NULL, PAGE_READONLY,
                                       0, 0, NULL);
    if (!g->map_handle) { vv_gguf_close(g); return VV_ERR_IO; }
    g->base = (const uint8_t*)MapViewOfFile(g->map_handle, FILE_MAP_READ, 0, 0, 0);
    if (!g->base) { vv_gguf_close(g); return VV_ERR_IO; }
    g->size = (size_t)fsize.QuadPart;
#else
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { vv_gguf_close(g); return VV_ERR_IO; }
    struct stat st;
    if (fstat(g->fd, &st) != 0 || st.st_size <= 0) { vv_gguf_close(g); return VV_ERR_IO; }
    void* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (m == MAP_FAILED) { vv_gguf_close(g); return VV_ERR_IO; }
    g->base = (const uint8_t*)m;
    g->size = (size_t)st.st_size;
#endif
    g->owns_mapping = 1;

    vv_status_t s = parse(g);
    if (s != VV_OK) {
        VV_LOG_E("gguf: failed to parse %s", path);
        vv_gguf_close(g);
        return s;
    }
    VV_LOG_D("gguf: %s v%u, %d kv, %d tensors, data at %llu", path, g->version,
             g->n_kv, g->n_tensors, (unsigned long long)g->data_offset);
    *out = g;
    return VV_OK;
}

void vv_gguf_close(vv_gguf_t* g) {
    if (!g) return;
    if (g->owns_mapping) {
#ifdef _WIN32
        if (g->base) UnmapViewOfFile(g->base);
#else
        if (g->base) munmap((void*)g->base, g->size);
#endif
    }
#ifdef _WIN32
    if (g->map_handle) CloseHandle(g->map_handle);
    if (g->file_handle != INVALID_HANDLE_VALUE) CloseHandle(g->file_handle);
#else
    if (g->fd >= 0) close(g->fd);
#endif
    vv_free(g->kvs);
    vv_free(g->tensors);
    vv_free(g->strings);
    vv_free(g);
}

/* ─── Accessors ─────────────────────────────────────────────────────────── */

uint32_t vv_gguf_version(const vv_gguf_t* g) { return g ? g->version : 0; }
uint32_t vv_gguf_alignment(const vv_gguf_t* g) { return g ? g->alignment : 0; }
uint64_t vv_gguf_data_offset(const vv_gguf_t* g) { return g ? g->data_offset : 0; }
int vv_gguf_n_tensors(const vv_gguf_t* g) { return g ? g->n_tensors : 0; }
int vv_gguf_n_kv(const vv_gguf_t* g) { return g ? g->n_kv : 0; }

const vv_gguf_tensor_t* vv_gguf_tensor(const vv_gguf_t* g, int i) {
    return (g && i >= 0 && i < g->n_tensors) ? &g->tensors[i] : NULL;
}

const vv_gguf_kv_t* vv_gguf_kv(const vv_gguf_t* g, int i) {
    return (g && i >= 0 && i < g->n_kv) ? &g->kvs[i] : NULL;
}

const vv_gguf_tensor_t* vv_gguf_find_tensor(const vv_gguf_t* g, const char* name) {
    if (!g || !name) return NULL;
    for (int i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}

const vv_gguf_kv_t* vv_gguf_find_kv(const vv_gguf_t* g, const char* key) {
    if (!g || !key) return NULL;
    for (int i = 0; i < g->n_kv; i++)
        if (g->kvs[i].key && strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    return NULL;
}

int64_t vv_gguf_get_int(const vv_gguf_t* g, const char* key, int64_t def) {
    const vv_gguf_kv_t* kv = vv_gguf_find_kv(g, key);
    if (!kv) return def;
    switch (kv->type) {
        case VV_GGUF_U8: case VV_GGUF_U16: case VV_GGUF_U32: case VV_GGUF_BOOL:
        case VV_GGUF_U64:
            return (int64_t)kv->v.u;
        case VV_GGUF_I8: case VV_GGUF_I16: case VV_GGUF_I32: case VV_GGUF_I64:
            return kv->v.i;
        default:
            return def;
    }
}

double vv_gguf_get_float(const vv_gguf_t* g, const char* key, double def) {
    const vv_gguf_kv_t* kv = vv_gguf_find_kv(g, key);
    if (!kv) return def;
    if (kv->type == VV_GGUF_F32 || kv->type == VV_GGUF_F64) return kv->v.f;
    const int64_t sentinel = INT64_MIN;
    const int64_t i = vv_gguf_get_int(g, key, sentinel);
    return i == sentinel ? def : (double)i;
}

const char* vv_gguf_get_str(const vv_gguf_t* g, const char* key) {
    const vv_gguf_kv_t* kv = vv_gguf_find_kv(g, key);
    return (kv && kv->type == VV_GGUF_STRING) ? kv->str : NULL;
}

const char* vv_gguf_arr_str(const vv_gguf_kv_t* kv, uint64_t idx, size_t* len) {
    /* The array was validated at parse time, so walking it cannot run off
     * the mapping; it is O(idx), callers that need random access build an
     * index once. v1 files are not supported here (32-bit lengths). */
    if (!kv || kv->type != VV_GGUF_ARRAY || kv->arr_type != VV_GGUF_STRING ||
        idx >= kv->arr_n)
        return NULL;
    const uint8_t* p = kv->arr_data;
    for (uint64_t i = 0;; i++) {
        uint64_t n;
        memcpy(&n, p, 8);
        if (i == idx) {
            if (len) *len = (size_t)n;
            return (const char*)(p + 8);
        }
        p += 8 + n;
    }
}

/* ─── Decoding ──────────────────────────────────────────────────────────── */

vv_status_t vv_gguf_i2s_view(const vv_gguf_tensor_t* t, const uint8_t** codes,
                             float* scale) {
    if (!t || !codes || !scale) return VV_ERR_NULL_PTR;
    if (t->type != VV_GGML_I2_S || !t->data) return VV_ERR_INVALID_ARG;
    const int64_t n = vv_gguf_nelements(t);
    *codes = t->data;
    memcpy(scale, t->data + n / 4, sizeof(float));
    return VV_OK;
}

vv_status_t vv_gguf_i8s_view(const vv_gguf_tensor_t* t, const int8_t** q,
                             float* scale) {
    if (!t || !q || !scale) return VV_ERR_NULL_PTR;
    if (t->type != VV_GGML_I8_S || !t->data) return VV_ERR_INVALID_ARG;
    const int64_t n = vv_gguf_nelements(t);
    *q = (const int8_t*)t->data;
    memcpy(scale, t->data + n, sizeof(float));
    return VV_OK;
}

static float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void vv_q6k_dequant_row(const uint8_t* blocks, float* y, int64_t k) {
    /* block_q6_K: ql[128] qh[64] scales[16] (int8) d (fp16) = 210 bytes.
     * Same operation order as ggml's dequantize_row_q6_K: (d * sc) * q. */
    const int64_t nb = k / 256;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* b = blocks + i * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = (const int8_t*)(b + 192);
        uint16_t dh;
        memcpy(&dh, b + 208, 2);
        const float d = vv_half_to_float(dh);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

vv_status_t vv_gguf_dequant_rows_f32(const vv_gguf_tensor_t* t, int64_t row0,
                                     int64_t nrows, float* out) {
    if (!t || !out) return VV_ERR_NULL_PTR;
    if (!t->data) return VV_ERR_UNSUPPORTED;
    const int64_t k = t->ne[0];
    const int64_t rows = vv_gguf_nelements(t) / (k ? k : 1);
    if (row0 < 0 || nrows < 0 || row0 + nrows > rows) return VV_ERR_INVALID_ARG;

    switch (t->type) {
        case VV_GGML_F32:
            memcpy(out, t->data + row0 * k * 4, (size_t)(nrows * k) * 4);
            return VV_OK;
        case VV_GGML_F16: {
            const uint8_t* p = t->data + row0 * k * 2;
            for (int64_t i = 0; i < nrows * k; i++) {
                uint16_t h;
                memcpy(&h, p + i * 2, 2);
                out[i] = vv_half_to_float(h);
            }
            return VV_OK;
        }
        case VV_GGML_BF16: {
            const uint8_t* p = t->data + row0 * k * 2;
            for (int64_t i = 0; i < nrows * k; i++) {
                uint16_t h;
                memcpy(&h, p + i * 2, 2);
                out[i] = bf16_to_f32(h);
            }
            return VV_OK;
        }
        case VV_GGML_Q8_0: {
            const int64_t bpr = k / 32;
            for (int64_t r = 0; r < nrows; r++) {
                const uint8_t* b = t->data + (row0 + r) * bpr * 34;
                for (int64_t j = 0; j < bpr; j++, b += 34) {
                    uint16_t dh;
                    memcpy(&dh, b, 2);
                    const float d = vv_half_to_float(dh);
                    for (int l = 0; l < 32; l++)
                        out[r * k + j * 32 + l] = (float)(int8_t)b[2 + l] * d;
                }
            }
            return VV_OK;
        }
        case VV_GGML_Q6_K: {
            const int64_t bpr = k / 256;
            for (int64_t r = 0; r < nrows; r++)
                vv_q6k_dequant_row(t->data + (row0 + r) * bpr * 210, out + r * k, k);
            return VV_OK;
        }
        case VV_GGML_I8_S: {
            const int8_t* q;
            float s;
            vv_gguf_i8s_view(t, &q, &s);
            for (int64_t i = 0; i < nrows * k; i++)
                out[i] = (float)q[row0 * k + i] * s;
            return VV_OK;
        }
        case VV_GGML_I2_S: {
            if (k % 128 != 0) return VV_ERR_UNSUPPORTED;
            const uint8_t* codes;
            float s;
            vv_gguf_i2s_view(t, &codes, &s);
            static const float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };
            for (int64_t r = 0; r < nrows; r++) {
                const uint8_t* rb = codes + (row0 + r) * (k / 4);
                float* y = out + r * k;
                for (int64_t blk = 0; blk < k / 128; blk++) {
                    for (int j = 0; j < 32; j++) {
                        const uint8_t b = rb[blk * 32 + j];
                        y[blk * 128 + j + 0] = s * map2bit[(b >> 6) & 3];
                        y[blk * 128 + j + 32] = s * map2bit[(b >> 4) & 3];
                        y[blk * 128 + j + 64] = s * map2bit[(b >> 2) & 3];
                        y[blk * 128 + j + 96] = s * map2bit[(b >> 0) & 3];
                    }
                }
            }
            return VV_OK;
        }
        default:
            return VV_ERR_UNSUPPORTED;
    }
}

vv_status_t vv_gguf_dequant_rows_f16(const vv_gguf_tensor_t* t, int64_t row0,
                                     int64_t nrows, uint16_t* out) {
    if (!t || !out) return VV_ERR_NULL_PTR;
    const int64_t k = t->ne[0];
    if (t->type == VV_GGML_F16) {
        const int64_t rows = vv_gguf_nelements(t) / (k ? k : 1);
        if (row0 < 0 || nrows < 0 || row0 + nrows > rows) return VV_ERR_INVALID_ARG;
        memcpy(out, t->data + row0 * k * 2, (size_t)(nrows * k) * 2);
        return VV_OK;
    }
    float row[4096];
    float* buf = k <= 4096 ? row : (float*)vv_alloc((size_t)k * sizeof(float));
    if (!buf) return VV_ERR_OUT_OF_MEMORY;
    vv_status_t s = VV_OK;
    for (int64_t r = 0; r < nrows && s == VV_OK; r++) {
        s = vv_gguf_dequant_rows_f32(t, row0 + r, 1, buf);
        for (int64_t i = 0; s == VV_OK && i < k; i++)
            out[r * k + i] = vv_float_to_half(buf[i]);
    }
    if (buf != row) vv_free(buf);
    return s;
}
