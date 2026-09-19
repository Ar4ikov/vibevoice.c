/**
 * @file ws.c
 * @brief SSE framing and the socket-free half of WebSocket (RFC 6455).
 */

#include "vv_ws.h"

#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <string.h>

/* ─── SHA-1 and base64 ──────────────────────────────────────────────────── */

static uint32_t rol32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

static void sha1_block(uint32_t h[5], const uint8_t* b) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)b[4 * i] << 24 | (uint32_t)b[4 * i + 1] << 16 |
               (uint32_t)b[4 * i + 2] << 8 | (uint32_t)b[4 * i + 3];
    for (int i = 16; i < 80; i++)
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20)      { f = (bb & c) | (~bb & d);           k = 0x5A827999u; }
        else if (i < 40) { f = bb ^ c ^ d;                     k = 0x6ED9EBA1u; }
        else if (i < 60) { f = (bb & c) | (bb & d) | (c & d);  k = 0x8F1BBCDCu; }
        else             { f = bb ^ c ^ d;                     k = 0xCA62C1D6u; }
        const uint32_t t = rol32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rol32(bb, 30); bb = a; a = t;
    }
    h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e;
}

void vv_sha1(const void* data, size_t n, uint8_t out[20]) {
    uint32_t h[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u,
                      0xC3D2E1F0u };
    const uint8_t* p = (const uint8_t*)data;
    size_t left = n;
    while (left >= 64) { sha1_block(h, p); p += 64; left -= 64; }
    uint8_t tail[128];
    memset(tail, 0, sizeof(tail));
    if (left) memcpy(tail, p, left);
    tail[left] = 0x80;
    const size_t tl = left + 9 <= 64 ? 64 : 128;
    const uint64_t bits = (uint64_t)n * 8;
    for (int i = 0; i < 8; i++) tail[tl - 1 - i] = (uint8_t)(bits >> (8 * i));
    sha1_block(h, tail);
    if (tl == 128) sha1_block(h, tail + 64);
    for (int i = 0; i < 5; i++) {
        out[4 * i] = (uint8_t)(h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(h[i] >> 8);
        out[4 * i + 3] = (uint8_t)h[i];
    }
}

size_t vv_base64_encode(const uint8_t* in, size_t n, char* out) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)in[i] << 16 |
                           (i + 1 < n ? (uint32_t)in[i + 1] << 8 : 0) |
                           (i + 2 < n ? (uint32_t)in[i + 2] : 0);
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = i + 1 < n ? tab[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < n ? tab[v & 63] : '=';
    }
    out[o] = '\0';
    return o;
}

bool vv_ws_accept_key(const char* client_key, char out[29]) {
    static const char guid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    if (!client_key || !out) return false;
    /* The key is base64 of 16 random bytes: 24 characters. */
    const size_t kl = strlen(client_key);
    if (kl == 0 || kl > 64) return false;
    char buf[64 + sizeof(guid)];
    memcpy(buf, client_key, kl);
    memcpy(buf + kl, guid, sizeof(guid) - 1);
    uint8_t digest[20];
    vv_sha1(buf, kl + sizeof(guid) - 1, digest);
    vv_base64_encode(digest, 20, out);
    return true;
}

/* ─── Frames and events ─────────────────────────────────────────────────── */

size_t vv_ws_frame_header(int opcode, bool fin, uint64_t len, uint8_t out[10]) {
    out[0] = (uint8_t)((fin ? 0x80 : 0) | (opcode & 0x0F));
    if (len < 126) {
        out[1] = (uint8_t)len;
        return 2;
    }
    if (len <= 0xFFFF) {
        out[1] = 126;
        out[2] = (uint8_t)(len >> 8);
        out[3] = (uint8_t)len;
        return 4;
    }
    out[1] = 127;
    for (int i = 0; i < 8; i++) out[2 + i] = (uint8_t)(len >> (56 - 8 * i));
    return 10;
}

size_t vv_sse_format(const char* event, const char* data, char* buf,
                     size_t cap) {
    size_t need = 0;
#define PUT(s, n) do {                                                     \
        const size_t n_ = (n);                                             \
        if (buf && need + n_ < cap) memcpy(buf + need, (s), n_);           \
        else if (buf && need < cap) memcpy(buf + need, (s), cap - 1 - need); \
        need += n_;                                                        \
    } while (0)
    if (event && event[0]) {
        PUT("event: ", 7);
        PUT(event, strlen(event));
        PUT("\n", 1);
    }
    /* Every line of data is its own "data:" field; the client joins them
     * with "\n". A trailing "\r" of a CRLF line is dropped. */
    const char* p = data ? data : "";
    for (;;) {
        const char* nl = strchr(p, '\n');
        size_t n = nl ? (size_t)(nl - p) : strlen(p);
        const size_t cut = n;
        if (n && p[n - 1] == '\r') n--;
        PUT("data: ", 6);
        PUT(p, n);
        PUT("\n", 1);
        if (!nl) break;
        p += cut + 1;
    }
    PUT("\n", 1);
#undef PUT
    if (buf && cap) buf[need < cap ? need : cap - 1] = '\0';
    return need;
}

/* ─── Parser ────────────────────────────────────────────────────────────── */

void vv_ws_parser_init(vv_ws_parser_t* p, size_t max_message) {
    memset(p, 0, sizeof(*p));
    p->max_message = max_message ? max_message : (size_t)16 << 20;
    p->require_mask = true;
}

void vv_ws_parser_free(vv_ws_parser_t* p) {
    if (!p) return;
    vv_free(p->in);
    vv_free(p->msg);
    memset(p, 0, sizeof(*p));
}

static bool grow(uint8_t** buf, size_t* cap, size_t need) {
    if (need <= *cap) return true;
    size_t c = *cap ? *cap : 4096;
    while (c < need) c *= 2;
    uint8_t* nb = (uint8_t*)vv_realloc(*buf, c);
    if (!nb) return false;
    *buf = nb;
    *cap = c;
    return true;
}

static vv_status_t fail(vv_ws_parser_t* p, int code) {
    p->failed = true;
    p->close_code = code;
    return code == VV_WS_CLOSE_TOO_BIG ? VV_ERR_OVERFLOW : VV_ERR_PARSE;
}

vv_status_t vv_ws_parser_feed(vv_ws_parser_t* p, const uint8_t* data,
                              size_t n, vv_ws_message_fn cb, void* user) {
    if (!p) return VV_ERR_NULL_PTR;
    if (p->failed)
        return p->close_code == VV_WS_CLOSE_TOO_BIG ? VV_ERR_OVERFLOW
                                                    : VV_ERR_PARSE;
    if (n) {
        if (!data) return VV_ERR_NULL_PTR;
        if (!grow(&p->in, &p->in_cap, p->in_len + n)) return VV_ERR_OUT_OF_MEMORY;
        memcpy(p->in + p->in_len, data, n);
        p->in_len += n;
    }

    size_t off = 0;
    for (;;) {
        const uint8_t* f = p->in + off;
        const size_t avail = p->in_len - off;
        if (avail < 2) break;
        const bool fin = (f[0] & 0x80) != 0;
        const int rsv = f[0] & 0x70;
        const int op = f[0] & 0x0F;
        const bool masked = (f[1] & 0x80) != 0;
        uint64_t len = f[1] & 0x7F;
        size_t hl = 2;
        if (rsv) return fail(p, VV_WS_CLOSE_PROTOCOL);   /* no extensions */
        if (!(op <= 2 || (op >= 8 && op <= 10))) return fail(p, VV_WS_CLOSE_PROTOCOL);
        if (p->require_mask && !masked) return fail(p, VV_WS_CLOSE_PROTOCOL);
        const bool control = op >= 8;
        if (control && (!fin || len > 125)) return fail(p, VV_WS_CLOSE_PROTOCOL);
        if (len == 126) {
            if (avail < 4) break;
            len = (uint64_t)f[2] << 8 | f[3];
            hl = 4;
        } else if (len == 127) {
            if (avail < 10) break;
            len = 0;
            for (int i = 0; i < 8; i++) len = len << 8 | f[2 + i];
            if (len >> 63) return fail(p, VV_WS_CLOSE_PROTOCOL);
            hl = 10;
        }
        /* Bound the frame before waiting for it, and the message too. */
        if (len > p->max_message) return fail(p, VV_WS_CLOSE_TOO_BIG);
        if (!control && op == VV_WS_OP_CONT && p->msg_len + len > p->max_message)
            return fail(p, VV_WS_CLOSE_TOO_BIG);
        if (masked) hl += 4;
        if (avail < hl + len) break;

        uint8_t* payload = p->in + off + hl;
        if (masked) {
            const uint8_t* key = p->in + off + hl - 4;
            for (uint64_t i = 0; i < len; i++) payload[i] ^= key[i & 3];
        }

        if (control) {
            if (cb) cb(user, op, payload, (size_t)len);
        } else if (op == VV_WS_OP_CONT) {
            if (!p->msg_opcode) return fail(p, VV_WS_CLOSE_PROTOCOL);
            if (!grow(&p->msg, &p->msg_cap, p->msg_len + (size_t)len + 1))
                return VV_ERR_OUT_OF_MEMORY;
            memcpy(p->msg + p->msg_len, payload, (size_t)len);
            p->msg_len += (size_t)len;
            if (fin) {
                p->msg[p->msg_len] = 0;
                if (cb) cb(user, p->msg_opcode, p->msg, p->msg_len);
                p->msg_opcode = 0;
                p->msg_len = 0;
            }
        } else {
            if (p->msg_opcode) return fail(p, VV_WS_CLOSE_PROTOCOL);
            if (fin) {
                if (cb) cb(user, op, payload, (size_t)len);
            } else {
                if (!grow(&p->msg, &p->msg_cap, (size_t)len + 1))
                    return VV_ERR_OUT_OF_MEMORY;
                memcpy(p->msg, payload, (size_t)len);
                p->msg_len = (size_t)len;
                p->msg_opcode = op;
            }
        }
        off += hl + (size_t)len;
    }
    if (off) {
        memmove(p->in, p->in + off, p->in_len - off);
        p->in_len -= off;
    }
    return VV_OK;
}
