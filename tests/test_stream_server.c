/**
 * @file test_stream_server.c
 * @brief The WebSocket endpoint of `serve` over loopback, on a real
 *        Streaming-7B engine.
 *
 * What test_ws.c cannot reach without weights -- the session logic in
 * api.c:
 *
 *  - a session.start split over two text frames with a ping between them
 *    (the pong comes first, then session.started);
 *  - s16 PCM in odd 4801-byte frames, samples split across frames, gives
 *    the transcript of the same audio in 96000-byte frames;
 *  - audio before session.start is refused with an error and close 1008;
 *  - a client that holds the only slot and sends nothing but pings is
 *    closed once the idle time runs out, and meanwhile a second client
 *    gets 503 server_overloaded after the slot wait instead of blocking;
 *    one that sends a trickle of audio is closed too;
 *  - a client that drops mid-stream gives its slot back.
 *
 * Needs VV_TEST_STREAM_E2E=1, VV_TEST_STREAM_MODEL (the Streaming-7B
 * directory) and VV_TEST_STREAM_REF (a `compare_ref.py dump-stream`
 * directory; its audio24k.f32 is the clip). VV_TEST_STREAM_QUANT picks the
 * load-time quantization (default int4). POSIX only; skipped otherwise.
 */

#include "vibevoice/vibevoice.h"
#include "vibevoice/engine.h"
#include "vibevoice/server.h"
#include "vibevoice/quant.h"

#include "vv_thread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
    g_fail++; } } while (0)

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

typedef struct {
    vv_engine_t* engine;
    vv_server_params_t p;
    vv_server_t* handle;
    vv_status_t st;
} srv_t;

static VV_THREAD_RET serve_thread(void* a) {
    srv_t* s = (srv_t*)a;
    s->st = vv_server_run(s->engine, &s->p, &s->handle);
    VV_THREAD_RETURN;
}

/* ─── a minimal WebSocket client ────────────────────────────────────────── */

/* Connect and ask for the upgrade. Returns the socket once the server
 * answered 101; otherwise -1, with the status line and body of whatever it
 * answered instead in `refusal` (when given). */
static int ws_connect_ex(int port, char* refusal, size_t refusal_cap) {
    if (refusal && refusal_cap) refusal[0] = 0;
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr*)&a, sizeof(a)) != 0) {
        close(fd);
        return -1;
    }
    const char* req =
        "GET /v1/audio/stream HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n";
    if (send(fd, req, strlen(req), 0) != (ssize_t)strlen(req)) {
        close(fd);
        return -1;
    }
    /* The 101 response, byte by byte so no frame is swallowed. */
    char h[1024];
    size_t n = 0;
    while (n + 1 < sizeof(h)) {
        if (recv(fd, h + n, 1, 0) != 1) break;
        n++;
        if (n >= 4 && memcmp(h + n - 4, "\r\n\r\n", 4) == 0) break;
    }
    h[n] = 0;
    if (strstr(h, " 101 ") == NULL) {
        if (refusal && refusal_cap) {
            /* A refusal has a body: read what is there. */
            struct timeval tv = { 1, 0 };
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            while (n + 1 < sizeof(h)) {
                const ssize_t k = recv(fd, h + n, sizeof(h) - 1 - n, 0);
                if (k <= 0) break;
                n += (size_t)k;
            }
            h[n] = 0;
            snprintf(refusal, refusal_cap, "%s", h);
        }
        close(fd);
        return -1;
    }
    return fd;
}

static int ws_connect(int port) { return ws_connect_ex(port, NULL, 0); }

static bool send_all(int fd, const void* d, size_t n) {
    const char* p = (const char*)d;
    while (n) {
        const ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return false;
        p += k;
        n -= (size_t)k;
    }
    return true;
}

/* One masked frame; `fin` false leaves the message open. */
static bool ws_send(int fd, int opcode, bool fin, const void* data,
                    size_t len) {
    uint8_t h[14];
    size_t hn = 0;
    h[hn++] = (uint8_t)((fin ? 0x80 : 0) | opcode);
    if (len < 126) {
        h[hn++] = (uint8_t)(0x80 | len);
    } else if (len < 65536) {
        h[hn++] = 0x80 | 126;
        h[hn++] = (uint8_t)(len >> 8);
        h[hn++] = (uint8_t)len;
    } else {
        h[hn++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) h[hn++] = (uint8_t)((uint64_t)len >> (8 * i));
    }
    const uint8_t mask[4] = { 0x12, 0x34, 0x56, 0x78 };
    memcpy(h + hn, mask, 4);
    hn += 4;
    uint8_t* m = (uint8_t*)malloc(len ? len : 1);
    for (size_t i = 0; i < len; i++)
        m[i] = ((const uint8_t*)data)[i] ^ mask[i & 3];
    const bool ok = send_all(fd, h, hn) && send_all(fd, m, len);
    free(m);
    return ok;
}

static bool ws_text(int fd, const char* s) {
    return ws_send(fd, 0x1, true, s, strlen(s));
}

static bool recv_n(int fd, uint8_t* d, size_t n, int timeout_ms) {
    size_t got = 0;
    while (got < n) {
        struct pollfd p;
        p.fd = fd;
        p.events = POLLIN;
        p.revents = 0;
        if (poll(&p, 1, timeout_ms) <= 0) return false;
        const ssize_t k = recv(fd, d + got, n - got, 0);
        if (k <= 0) return false;
        got += (size_t)k;
    }
    return true;
}

/* One server frame (unmasked). Returns its opcode, -1 on timeout or EOF;
 * the payload is NUL-terminated in a malloc'd *out. */
static int ws_recv(int fd, char** out, size_t* len, int timeout_ms) {
    uint8_t h[2];
    *out = NULL;
    *len = 0;
    if (!recv_n(fd, h, 2, timeout_ms)) return -1;
    uint64_t n = h[1] & 0x7f;
    if (n == 126) {
        uint8_t e[2];
        if (!recv_n(fd, e, 2, timeout_ms)) return -1;
        n = ((uint64_t)e[0] << 8) | e[1];
    } else if (n == 127) {
        uint8_t e[8];
        if (!recv_n(fd, e, 8, timeout_ms)) return -1;
        n = 0;
        for (int i = 0; i < 8; i++) n = (n << 8) | e[i];
    }
    char* p = (char*)malloc((size_t)n + 1);
    if (n && !recv_n(fd, (uint8_t*)p, (size_t)n, timeout_ms)) {
        free(p);
        return -1;
    }
    p[n] = 0;
    *out = p;
    *len = (size_t)n;
    return h[0] & 0x0f;
}

/* Read until a text message containing `want`, or a close. Returns the
 * text message (malloc'd), or NULL; *close_code gets a close's code. */
static char* wait_for(int fd, const char* want, int timeout_ms,
                      int* close_code, bool* saw_pong) {
    if (close_code) *close_code = 0;
    const double end = vv_time_ms() + timeout_ms;
    for (;;) {
        const int left = (int)(end - vv_time_ms());
        if (left <= 0) return NULL;
        char* d = NULL;
        size_t n = 0;
        const int op = ws_recv(fd, &d, &n, left);
        if (op < 0) { free(d); return NULL; }
        if (op == 0xA && saw_pong) *saw_pong = true;
        if (op == 0x8) {
            if (close_code && n >= 2)
                *close_code = ((uint8_t)d[0] << 8) | (uint8_t)d[1];
            free(d);
            return NULL;
        }
        if (op == 0x1 && strstr(d, want)) return d;
        free(d);
    }
}

/* Stream `pcm16` in frames of `frame` bytes and return the done text. */
static char* run_session(int port, const int16_t* pcm16, size_t n,
                         size_t frame, bool split_start) {
    const int fd = ws_connect(port);
    CHECK(fd >= 0);
    if (fd < 0) return NULL;
    const char* start = "{\"type\":\"session.start\",\"sample_rate\":24000,"
                        "\"format\":\"pcm_s16le\"}";
    bool pong = false;
    if (split_start) {
        const size_t half = strlen(start) / 2;
        CHECK(ws_send(fd, 0x1, false, start, half));
        CHECK(ws_send(fd, 0x9, true, "hi", 2));          /* ping between */
        CHECK(ws_send(fd, 0x0, true, start + half, strlen(start) - half));
    } else {
        CHECK(ws_text(fd, start));
    }
    char* m = wait_for(fd, "session.started", 60000, NULL, &pong);
    CHECK(m != NULL);
    if (split_start) CHECK(pong);
    free(m);

    const uint8_t* b = (const uint8_t*)pcm16;
    const size_t bytes = n * 2;
    for (size_t off = 0; off < bytes; off += frame) {
        const size_t k = bytes - off < frame ? bytes - off : frame;
        if (!ws_send(fd, 0x2, true, b + off, k)) { CHECK(false); break; }
    }
    CHECK(ws_text(fd, "{\"type\":\"session.finish\"}"));
    int code = 0;
    char* done = wait_for(fd, "transcript.text.done", 120000, &code, NULL);
    CHECK(done != NULL);
    close(fd);
    return done;
}

static void run_tests(int port, const int16_t* pcm16, size_t n) {
    /* 1, 2: frame sizes and a fragmented start give the same transcript. */
    char* big = run_session(port, pcm16, n, 96000, true);
    char* odd = run_session(port, pcm16, n, 4801, false);
    CHECK(big && odd && strcmp(big, odd) == 0);
    CHECK(big && strlen(big) > 40);
    if (big) printf("server: %s\n", big);
    free(big);
    free(odd);

    /* 3: audio before session.start. */
    int fd = ws_connect(port);
    CHECK(fd >= 0);
    const uint8_t zeros[64] = { 0 };
    CHECK(ws_send(fd, 0x2, true, zeros, sizeof(zeros)));
    int code = 0;
    char* m = wait_for(fd, "\"error\"", 5000, &code, NULL);
    CHECK(m && strstr(m, "session.start"));
    free(m);
    if (!code) m = wait_for(fd, "\x01never", 5000, &code, NULL);
    CHECK(code == 1008);
    close(fd);

    /* 4: a slot held on pings alone is not held for ever, and a second
     *    session is told the server is busy (503 before the upgrade)
     *    instead of waiting behind it. */
    const int holder = ws_connect(port);
    CHECK(holder >= 0);
    CHECK(ws_text(holder, "{\"type\":\"session.start\"}"));
    m = wait_for(holder, "session.started", 60000, NULL, NULL);
    CHECK(m != NULL);
    free(m);
    const double t_held = vv_time_ms();

    char refusal[1024];
    const double t_ask = vv_time_ms();
    fd = ws_connect_ex(port, refusal, sizeof(refusal));
    const double asked = vv_time_ms() - t_ask;
    CHECK(fd < 0);
    if (fd >= 0) close(fd);
    CHECK(strstr(refusal, " 503 ") != NULL);
    CHECK(strstr(refusal, "server_overloaded") != NULL);
    CHECK(asked < 5000);
    printf("server: a second session was refused in %.0f ms while the slot "
           "was held\n", asked);

    /* The holder pings every 300 ms and is closed all the same. */
    int hcode = 0;
    bool closed = false;
    while (vv_time_ms() - t_held < 20000 && !closed) {
        if (!ws_send(holder, 0x9, true, "p", 1)) { closed = true; break; }
        char* d = NULL;
        size_t dn = 0;
        const int op = ws_recv(holder, &d, &dn, 300);
        if (op == 0x8) {
            closed = true;
            if (dn >= 2) hcode = ((uint8_t)d[0] << 8) | (uint8_t)d[1];
        }
        free(d);
    }
    const double held = vv_time_ms() - t_held;
    CHECK(closed);
    CHECK(hcode == 1008);
    CHECK(held < 10000);
    printf("server: a ping-only session was closed after %.1f s\n",
           held / 1000.0);
    close(holder);

    /* 4b: a trickle of audio (2 bytes every 300 ms) does not hold it
     *     either: under a tenth of real time is dropped. */
    const int trickle = ws_connect(port);
    CHECK(trickle >= 0);
    CHECK(ws_text(trickle, "{\"type\":\"session.start\"}"));
    m = wait_for(trickle, "session.started", 60000, NULL, NULL);
    CHECK(m != NULL);
    free(m);
    const double t_tr = vv_time_ms();
    bool dropped = false;
    bool slow_msg = false;
    while (vv_time_ms() - t_tr < 20000 && !dropped) {
        const uint8_t two[2] = { 0, 0 };
        if (!ws_send(trickle, 0x2, true, two, 2)) { dropped = true; break; }
        char* d = NULL;
        size_t dn = 0;
        const int op = ws_recv(trickle, &d, &dn, 300);
        if (op == 0x1 && d && strstr(d, "real time")) slow_msg = true;
        if (op == 0x8) dropped = true;
        free(d);
    }
    CHECK(dropped);
    CHECK(slow_msg);
    printf("server: a trickle session was closed after %.1f s\n",
           (vv_time_ms() - t_tr) / 1000.0);
    close(trickle);

    /* 5: a client that drops mid-stream gives the slot back. */
    fd = ws_connect(port);
    CHECK(fd >= 0);
    CHECK(ws_text(fd, "{\"type\":\"session.start\"}"));
    m = wait_for(fd, "session.started", 60000, NULL, NULL);
    CHECK(m != NULL);
    free(m);
    const size_t part = n / 2 * 2;
    CHECK(ws_send(fd, 0x2, true, pcm16, part));
    close(fd);
    bool got = false;
    const double t_drop = vv_time_ms();
    while (!got && vv_time_ms() - t_drop < 30000) {
        fd = ws_connect(port);
        if (fd < 0) break;
        CHECK(ws_text(fd, "{\"type\":\"session.start\"}"));
        m = wait_for(fd, "session.", 10000, &code, NULL);
        got = m && strstr(m, "session.started");
        free(m);
        close(fd);
    }
    CHECK(got);
    printf("server: the slot came back %.1f s after the client dropped\n",
           (vv_time_ms() - t_drop) / 1000.0);
}

static void test_server(void) {
    const char* dir = getenv("VV_TEST_STREAM_MODEL");
    const char* refs = getenv("VV_TEST_STREAM_REF");
    const char* on = getenv("VV_TEST_STREAM_E2E");
    if (!dir || !dir[0] || !refs || !refs[0] || !on || on[0] != '1') {
        puts("server: SKIP (set VV_TEST_STREAM_E2E=1, VV_TEST_STREAM_MODEL "
             "and VV_TEST_STREAM_REF)");
        return;
    }
    char ref[1024];
    snprintf(ref, sizeof(ref), "%s", refs);
    char* colon = strchr(ref, ':');
    if (colon) *colon = 0;
    char path[1100];
    snprintf(path, sizeof(path), "%s/audio24k.f32", ref);
    FILE* f = fopen(path, "rb");
    CHECK(f != NULL);
    if (!f) return;
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    const size_t n = (size_t)bytes / 4;
    float* pcm = (float*)malloc((size_t)bytes);
    CHECK(fread(pcm, 1, (size_t)bytes, f) == (size_t)bytes);
    fclose(f);
    int16_t* pcm16 = (int16_t*)malloc(n * 2);
    for (size_t i = 0; i < n; i++) {
        float v = pcm[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm16[i] = (int16_t)v;
    }
    free(pcm);

    vv_engine_params_t ep = vv_engine_params_default();
    ep.model_dir = dir;
    ep.n_slots = 1;
    const char* q = getenv("VV_TEST_STREAM_QUANT");
    ep.weight_quant = (int)vv_load_quant_parse(q && q[0] ? q : "int4");
    vv_engine_t* engine = NULL;
    CHECK(vv_engine_create(&ep, &engine) == VV_OK);
    if (!engine) { free(pcm16); return; }

    srv_t s;
    memset(&s, 0, sizeof(s));
    s.engine = engine;
    s.p = vv_server_params_default();
    s.p.host = "127.0.0.1";
    s.p.port = 41000 + (int)(getpid() % 2000);
    s.p.stream_idle_ms = 1500;
    s.p.stream_slot_wait_ms = 500;
    vv_thread_t th;
    CHECK(vv_thread_start(&th, serve_thread, &s));
    int probe = -1;
    for (int i = 0; i < 500 && probe < 0; i++) {
        probe = ws_connect(s.p.port);
        if (probe < 0) vv_sleep_ms(20);
    }
    CHECK(probe >= 0);
    if (probe >= 0) {
        close(probe);
        run_tests(s.p.port, pcm16, n);
    }
    for (int i = 0; i < 100 && !s.handle; i++) vv_sleep_ms(10);
    if (s.handle) vv_server_stop(s.handle);
    vv_thread_join(th);
    vv_engine_free(engine);
    free(pcm16);
}
#else
static void test_server(void) { puts("server: SKIP on Windows"); }
#endif

int main(void) {
    test_server();
    if (g_fail) {
        fprintf(stderr, "server: %d check(s) failed\n", g_fail);
        return 1;
    }
    puts("server: WebSocket session checks passed (or skipped)");
    return 0;
}
