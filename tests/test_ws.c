/**
 * @file test_ws.c
 * @brief SSE framing, WebSocket handshake keys, frame headers and the frame
 *        parser (RFC 6455 examples), plus an SSE and WebSocket round trip
 *        over loopback on POSIX.
 */

#include "vv_ws.h"
#include "vv_http.h"
#include "vv_thread.h"

#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
    g_fail++; } } while (0)

static void hex(const uint8_t* d, size_t n, char* out) {
    for (size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", d[i]);
}

static void test_sha1_base64(void) {
    uint8_t d[20];
    char h[41];
    vv_sha1("abc", 3, d);
    hex(d, 20, h);
    CHECK(strcmp(h, "a9993e364706816aba3e25717850c26c9cd0d89d") == 0);
    vv_sha1("", 0, d);
    hex(d, 20, h);
    CHECK(strcmp(h, "da39a3ee5e6b4b0d3255bfef95601890afd80709") == 0);
    const char* two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    vv_sha1(two, strlen(two), d);
    hex(d, 20, h);
    CHECK(strcmp(h, "84983e441c3bd26ebaae4aa1f95129e5e54670f1") == 0);
    /* 55/56/64-byte edges of the padding. */
    char m[200];
    memset(m, 'a', sizeof(m));
    vv_sha1(m, 55, d); hex(d, 20, h);
    CHECK(strcmp(h, "c1c8bbdc22796e28c0e15163d20899b65621d65a") == 0);
    vv_sha1(m, 56, d); hex(d, 20, h);
    CHECK(strcmp(h, "c2db330f6083854c99d4b5bfb6e8f29f201be699") == 0);
    vv_sha1(m, 64, d); hex(d, 20, h);
    CHECK(strcmp(h, "0098ba824b5c16427bd7a1122a5a442a25ec644d") == 0);

    char b[16];
    CHECK(vv_base64_encode((const uint8_t*)"", 0, b) == 0 && b[0] == 0);
    vv_base64_encode((const uint8_t*)"f", 1, b);   CHECK(strcmp(b, "Zg==") == 0);
    vv_base64_encode((const uint8_t*)"fo", 2, b);  CHECK(strcmp(b, "Zm8=") == 0);
    vv_base64_encode((const uint8_t*)"foo", 3, b); CHECK(strcmp(b, "Zm9v") == 0);
    vv_base64_encode((const uint8_t*)"foob", 4, b); CHECK(strcmp(b, "Zm9vYg==") == 0);

    /* RFC 6455 section 1.3. */
    char acc[29];
    CHECK(vv_ws_accept_key("dGhlIHNhbXBsZSBub25jZQ==", acc));
    CHECK(strcmp(acc, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") == 0);
    CHECK(!vv_ws_accept_key(NULL, acc));
    CHECK(!vv_ws_accept_key("", acc));
}

static void test_frame_header(void) {
    uint8_t h[10];
    CHECK(vv_ws_frame_header(VV_WS_OP_TEXT, true, 5, h) == 2);
    CHECK(h[0] == 0x81 && h[1] == 0x05);
    CHECK(vv_ws_frame_header(VV_WS_OP_BINARY, true, 256, h) == 4);
    CHECK(h[0] == 0x82 && h[1] == 0x7E && h[2] == 0x01 && h[3] == 0x00);
    CHECK(vv_ws_frame_header(VV_WS_OP_BINARY, true, 65536, h) == 10);
    CHECK(h[1] == 0x7F && h[7] == 0x01 && h[8] == 0 && h[9] == 0);
    CHECK(vv_ws_frame_header(VV_WS_OP_TEXT, false, 125, h) == 2);
    CHECK(h[0] == 0x01 && h[1] == 125);
}

static void test_sse(void) {
    char buf[256];
    size_t n = vv_sse_format("transcript.text.delta",
                             "{\"type\":\"transcript.text.delta\",\"delta\":\"Hi\"}",
                             buf, sizeof(buf));
    CHECK(strcmp(buf, "event: transcript.text.delta\n"
                      "data: {\"type\":\"transcript.text.delta\",\"delta\":\"Hi\"}\n\n") == 0);
    CHECK(n == strlen(buf));
    n = vv_sse_format(NULL, "a\nb\r\nc", buf, sizeof(buf));
    CHECK(strcmp(buf, "data: a\ndata: b\ndata: c\n\n") == 0 && n == strlen(buf));
    n = vv_sse_format(NULL, "[DONE]", buf, sizeof(buf));
    CHECK(strcmp(buf, "data: [DONE]\n\n") == 0);
    CHECK(vv_sse_format("e", "x", NULL, 0) == strlen("event: e\ndata: x\n\n"));
    char small[6];
    CHECK(vv_sse_format("e", "x", small, sizeof(small)) == 18);
    CHECK(strcmp(small, "event") == 0);
}

typedef struct {
    int     n;
    int     ops[16];
    size_t  lens[16];
    char    data[16][300];
} got_t;

static void on_msg(void* user, int op, const uint8_t* d, size_t len) {
    got_t* g = (got_t*)user;
    if (g->n >= 16) return;
    g->ops[g->n] = op;
    g->lens[g->n] = len;
    memcpy(g->data[g->n], d, len < 299 ? len : 299);
    g->data[g->n][len < 299 ? len : 299] = 0;
    g->n++;
}

/* Feed `bytes` whole and then byte by byte; both must give `want_n`
 * messages and the same status. */
static vv_status_t feed_both(const uint8_t* bytes, size_t n, got_t* g,
                             size_t max_msg, bool require_mask) {
    vv_ws_parser_t p;
    vv_ws_parser_init(&p, max_msg);
    p.require_mask = require_mask;
    memset(g, 0, sizeof(*g));
    vv_status_t whole = vv_ws_parser_feed(&p, bytes, n, on_msg, g);
    const int whole_n = g->n;
    vv_ws_parser_free(&p);

    got_t g2;
    memset(&g2, 0, sizeof(g2));
    vv_ws_parser_init(&p, max_msg);
    p.require_mask = require_mask;
    vv_status_t st = VV_OK;
    for (size_t i = 0; i < n && st == VV_OK; i++)
        st = vv_ws_parser_feed(&p, bytes + i, 1, on_msg, &g2);
    vv_ws_parser_free(&p);
    CHECK(st == whole);
    CHECK(g2.n == whole_n);
    for (int i = 0; i < whole_n && i < g2.n; i++)
        CHECK(g2.ops[i] == g->ops[i] && g2.lens[i] == g->lens[i] &&
              memcmp(g2.data[i], g->data[i], g->lens[i] < 299 ? g->lens[i] : 299) == 0);
    return whole;
}

static void test_parser(void) {
    got_t g;
    /* RFC 6455 5.7: a single-frame masked text "Hello". */
    static const uint8_t masked[] = { 0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d,
                                      0x7f, 0x9f, 0x4d, 0x51, 0x58 };
    CHECK(feed_both(masked, sizeof(masked), &g, 1024, true) == VV_OK);
    CHECK(g.n == 1 && g.ops[0] == VV_WS_OP_TEXT && strcmp(g.data[0], "Hello") == 0);

    /* Unmasked frames from a client are refused... */
    static const uint8_t unmasked[] = { 0x81, 0x05, 'H', 'e', 'l', 'l', 'o' };
    vv_ws_parser_t p;
    vv_ws_parser_init(&p, 1024);
    memset(&g, 0, sizeof(g));
    CHECK(vv_ws_parser_feed(&p, unmasked, sizeof(unmasked), on_msg, &g) == VV_ERR_PARSE);
    CHECK(p.close_code == VV_WS_CLOSE_PROTOCOL && g.n == 0);
    CHECK(vv_ws_parser_feed(&p, masked, sizeof(masked), on_msg, &g) == VV_ERR_PARSE);
    vv_ws_parser_free(&p);

    /* ...and parse when masking is not required (the RFC's other examples):
     * a fragmented "Hel" + "lo" with a ping in the middle, then a pong. */
    static const uint8_t frag[] = {
        0x01, 0x03, 'H', 'e', 'l',
        0x89, 0x05, 'H', 'e', 'l', 'l', 'o',
        0x80, 0x02, 'l', 'o',
        0x8A, 0x00 };
    CHECK(feed_both(frag, sizeof(frag), &g, 1024, false) == VV_OK);
    CHECK(g.n == 3);
    CHECK(g.ops[0] == VV_WS_OP_PING && strcmp(g.data[0], "Hello") == 0);
    CHECK(g.ops[1] == VV_WS_OP_TEXT && strcmp(g.data[1], "Hello") == 0);
    CHECK(g.ops[2] == VV_WS_OP_PONG && g.lens[2] == 0);

    /* 16-bit and 64-bit lengths, masked binary. */
    uint8_t big[4 + 4 + 300];
    big[0] = 0x82; big[1] = 0x80 | 126; big[2] = 0x01; big[3] = 0x2C;   /* 300 */
    big[4] = 1; big[5] = 2; big[6] = 3; big[7] = 4;
    for (int i = 0; i < 300; i++) big[8 + i] = (uint8_t)(i ^ big[4 + (i & 3)]);
    CHECK(feed_both(big, sizeof(big), &g, 1024, true) == VV_OK);
    CHECK(g.n == 1 && g.ops[0] == VV_WS_OP_BINARY && g.lens[0] == 300);
    int ok = 1;
    for (int i = 0; i < 299; i++) ok &= (uint8_t)g.data[0][i] == (uint8_t)i;
    CHECK(ok);

    uint8_t b64[10 + 4 + 5] = { 0x82, 0x80 | 127, 0, 0, 0, 0, 0, 0, 0, 5,
                                0, 0, 0, 0, 'a', 'b', 'c', 'd', 'e' };
    CHECK(feed_both(b64, sizeof(b64), &g, 1024, true) == VV_OK);
    CHECK(g.n == 1 && strcmp(g.data[0], "abcde") == 0);

    /* Violations. */
    static const uint8_t rsv[] = { 0xC1, 0x80, 0, 0, 0, 0 };
    CHECK(feed_both(rsv, sizeof(rsv), &g, 1024, true) == VV_ERR_PARSE);
    static const uint8_t badop[] = { 0x83, 0x80, 0, 0, 0, 0 };
    CHECK(feed_both(badop, sizeof(badop), &g, 1024, true) == VV_ERR_PARSE);
    static const uint8_t ctl_frag[] = { 0x09, 0x80, 0, 0, 0, 0 };
    CHECK(feed_both(ctl_frag, sizeof(ctl_frag), &g, 1024, true) == VV_ERR_PARSE);
    static const uint8_t ctl_long[] = { 0x89, 0x80 | 126, 0, 126 };
    CHECK(feed_both(ctl_long, sizeof(ctl_long), &g, 1024, true) == VV_ERR_PARSE);
    static const uint8_t lone_cont[] = { 0x80, 0x80, 0, 0, 0, 0 };
    CHECK(feed_both(lone_cont, sizeof(lone_cont), &g, 1024, true) == VV_ERR_PARSE);
    static const uint8_t nested[] = { 0x01, 0x80, 0, 0, 0, 0, 0x01, 0x80, 0, 0, 0, 0 };
    CHECK(feed_both(nested, sizeof(nested), &g, 1024, true) == VV_ERR_PARSE);
    static const uint8_t neg[] = { 0x82, 0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0 };
    CHECK(feed_both(neg, sizeof(neg), &g, 1024, true) == VV_ERR_PARSE);
    /* Too big: refused from the header, before the payload arrives. */
    static const uint8_t huge[] = { 0x82, 0xFF, 0, 0, 0, 1, 0, 0, 0, 0 };
    vv_ws_parser_init(&p, 1024);
    CHECK(vv_ws_parser_feed(&p, huge, sizeof(huge), on_msg, &g) == VV_ERR_OVERFLOW);
    CHECK(p.close_code == VV_WS_CLOSE_TOO_BIG);
    vv_ws_parser_free(&p);
    /* ...and so is a fragmented message that grows past the limit. */
    static const uint8_t grow[] = { 0x02, 0x88, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                    0x80, 0x88, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    CHECK(feed_both(grow, sizeof(grow), &g, 12, true) == VV_ERR_OVERFLOW);
}

/* ─── Loopback round trip (POSIX) ───────────────────────────────────────── */

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct {
    vv_http_t* srv;
    int        port;
    vv_status_t st;
} srv_arg_t;

static void handler(const vv_http_req_t* req, vv_http_res_t* res, void* user) {
    (void)user;
    if (strcmp(req->path, "/sse") == 0) {
        vv_http_respond_begin(res, 200, "text/event-stream");
        vv_http_sse_send(res, "transcript.text.delta", "{\"delta\":\"Hel\"}");
        vv_http_sse_send(res, "transcript.text.delta", "{\"delta\":\"lo\"}");
        vv_http_sse_send(res, NULL, "[DONE]");
        return;
    }
    if (strcmp(req->path, "/ws") == 0) {
        if (!vv_http_ws_accept(req, res)) return;
        vv_ws_parser_t p;
        vv_ws_parser_init(&p, 1 << 16);
        got_t g;
        memset(&g, 0, sizeof(g));
        if (req->body_len)
            vv_ws_parser_feed(&p, (const uint8_t*)req->body, req->body_len, on_msg, &g);
        uint8_t buf[512];
        while (g.n == 0) {
            const int n = vv_http_read(res, buf, sizeof(buf), 2000);
            if (n <= 0) break;
            if (vv_ws_parser_feed(&p, buf, (size_t)n, on_msg, &g) != VV_OK) break;
        }
        if (g.n == 1) {
            /* Echo it back with a prefix, then close. */
            char out[320];
            const int on = snprintf(out, sizeof(out), "echo:%s", g.data[0]);
            vv_http_ws_send(res, VV_WS_OP_TEXT, out, (size_t)on);
            const uint8_t code[2] = { 0x03, 0xE8 };
            vv_http_ws_send(res, VV_WS_OP_CLOSE, code, 2);
        }
        vv_ws_parser_free(&p);
        return;
    }
    vv_http_error(res, 404, "not_found", "no");
}

static VV_THREAD_RET serve_thread(void* a) {
    srv_arg_t* s = (srv_arg_t*)a;
    s->st = vv_http_serve("127.0.0.1", s->port, 4, 1 << 20, handler, NULL, &s->srv);
    VV_THREAD_RETURN;
}

static int connect_to(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((unsigned short)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr*)&a, sizeof(a)) != 0) { close(fd); return -1; }
    return fd;
}

static size_t read_all(int fd, char* buf, size_t cap) {
    size_t n = 0;
    for (;;) {
        const ssize_t r = recv(fd, buf + n, cap - 1 - n, 0);
        if (r <= 0) break;
        n += (size_t)r;
        if (n + 1 >= cap) break;
    }
    buf[n] = 0;
    return n;
}

static void test_loopback(void) {
    srv_arg_t s;
    memset(&s, 0, sizeof(s));
    s.port = 39000 + (int)(getpid() % 2000);
    vv_thread_t th;
    CHECK(vv_thread_start(&th, serve_thread, &s));
    int fd = -1;
    for (int i = 0; i < 200 && fd < 0; i++) {
        fd = connect_to(s.port);
        if (fd < 0) vv_sleep_ms(10);
    }
    if (fd < 0) {
        puts("ws: loopback SKIP (cannot bind/connect)");
        if (s.srv) vv_http_shutdown(s.srv);
        vv_thread_join(th);
        return;
    }
    const char* get = "GET /sse HTTP/1.1\r\nHost: x\r\n\r\n";
    CHECK(send(fd, get, strlen(get), 0) == (ssize_t)strlen(get));
    char buf[2048];
    read_all(fd, buf, sizeof(buf));
    close(fd);
    CHECK(strstr(buf, "HTTP/1.1 200 OK\r\n") == buf);
    CHECK(strstr(buf, "Content-Type: text/event-stream\r\n") != NULL);
    CHECK(strstr(buf, "Content-Length") == NULL);
    const char* body = strstr(buf, "\r\n\r\n");
    CHECK(body && strcmp(body + 4,
        "event: transcript.text.delta\ndata: {\"delta\":\"Hel\"}\n\n"
        "event: transcript.text.delta\ndata: {\"delta\":\"lo\"}\n\n"
        "data: [DONE]\n\n") == 0);

    /* WebSocket: handshake plus a masked frame in the same packet (it lands
     * in req->body), echo, close. */
    fd = connect_to(s.port);
    CHECK(fd >= 0);
    char req[512];
    int rn = snprintf(req, sizeof(req),
        "GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
        "Connection: keep-alive, Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n");
    static const uint8_t hello[] = { 0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d,
                                     0x7f, 0x9f, 0x4d, 0x51, 0x58 };
    memcpy(req + rn, hello, sizeof(hello));
    rn += (int)sizeof(hello);
    CHECK(send(fd, req, (size_t)rn, 0) == rn);
    const size_t n = read_all(fd, buf, sizeof(buf));
    close(fd);
    CHECK(strstr(buf, "HTTP/1.1 101 Switching Protocols\r\n") == buf);
    CHECK(strstr(buf, "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n") != NULL);
    const char* fr = strstr(buf, "\r\n\r\n");
    CHECK(fr != NULL);
    if (fr) {
        const uint8_t* f = (const uint8_t*)fr + 4;
        const size_t left = n - (size_t)((const char*)f - buf);
        CHECK(left == 2 + 10 + 4);
        CHECK(f[0] == 0x81 && f[1] == 10 && memcmp(f + 2, "echo:Hello", 10) == 0);
        CHECK(f[12] == 0x88 && f[13] == 2 && f[14] == 0x03 && f[15] == 0xE8);
    }

    /* A plain GET to the WebSocket path is refused. */
    fd = connect_to(s.port);
    const char* plain = "GET /ws HTTP/1.1\r\nHost: x\r\n\r\n";
    CHECK(send(fd, plain, strlen(plain), 0) == (ssize_t)strlen(plain));
    read_all(fd, buf, sizeof(buf));
    close(fd);
    CHECK(strstr(buf, "HTTP/1.1 400") == buf);

    vv_http_shutdown(s.srv);
    vv_thread_join(th);
    CHECK(s.st == VV_OK);
    puts("ws: SSE and WebSocket loopback round trip passed");
}
#else
static void test_loopback(void) { puts("ws: loopback SKIP on Windows"); }
#endif

int main(void) {
    test_sha1_base64();
    test_frame_header();
    test_sse();
    test_parser();
    test_loopback();
    if (g_fail) {
        fprintf(stderr, "ws: %d check(s) failed\n", g_fail);
        return 1;
    }
    puts("ws: SHA-1, base64, accept key, frame headers, parser and SSE passed");
    return 0;
}
