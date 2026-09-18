/**
 * @file test_untrusted_input.c
 * @brief Input from outside the process stays data: arguments to helper
 *        programs, uploaded audio, HTTP request heads.
 *
 * Regression tests for issue #14 (CodeQL cpp/command-line-injection and the
 * server-side problems found next to it):
 *
 *  - vv_spawn passes every argument through byte for byte -- quotes, `$(...)`,
 *    backticks, `;`, `&`, `|`, `%VAR%`, backslash runs, empty strings,
 *    newlines -- and no shell ever sees them;
 *  - exit status, stderr routing and termination behave;
 *  - concurrent uploads never share a temporary file (they used to, through
 *    an unsynchronised counter);
 *  - the HTTP reader honours Content-Length in any capitalisation, only as a
 *    header of its own, refuses Transfer-Encoding, and error bodies are
 *    valid JSON whatever the message holds.
 */
#include "vibevoice/vibevoice.h"
#include "vibevoice/audio.h"

#include "vv_spawn.h"
#include "vv_thread.h"
#include "vv_http.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET sock_t;
#define close_sock closesocket
static void sleep_ms(int ms) { Sleep((DWORD)ms); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define close_sock close
static void sleep_ms(int ms) { usleep((useconds_t)ms * 1000); }
#endif

#ifndef VV_SPAWN_CHILD
#error "VV_SPAWN_CHILD must name the spawn_child helper"
#endif

static int failures = 0;
#define CHECK(cond, msg) do {                                             \
    if (cond) printf("  PASS: %s\n", msg);                                \
    else { fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__);     \
           failures++; }                                                  \
} while (0)

static size_t read_all(FILE* f, char* buf, size_t cap) {
    size_t len = 0, got;
    while (len < cap && (got = fread(buf + len, 1, cap - len, f)) > 0)
        len += got;
    return len;
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

/* ─── vv_spawn ─────────────────────────────────────────────────────────── */

#define CANARY "vv_spawn_canary"

static void test_spawn_arguments(void) {
    printf("test_spawn_arguments:\n");
    remove(CANARY);

    const char* const argv[] = {
        VV_SPAWN_CHILD,
        "plain",
        "two words",
        "q\"uote",
        "$(touch " CANARY ")",
        "`touch " CANARY "`",
        "; touch " CANARY,
        "& echo " CANARY " > " CANARY,
        "| touch " CANARY,
        "%PATH%",
        "back\\slash\\",
        "trailing\\\\",
        "\\\"",
        "\"",
        "",
        "new\nline",
        "tab\there",
        "a&b|c<d>e^f",
        NULL
    };

    FILE* out = NULL;
    vv_child_t* c = vv_spawn_read(argv, VV_SPAWN_STDERR_NULL, &out);
    CHECK(c && out, "child starts");
    if (!c) return;

    char got[4096];
    const size_t n = read_all(out, got, sizeof(got));
    const int rc = vv_spawn_wait(c, false);
    CHECK(rc == 0, "child exits 0");

    char want[4096];
    size_t wn = 0;
    for (int i = 1; argv[i]; i++) {
        const size_t l = strlen(argv[i]);
        memcpy(want + wn, argv[i], l);
        wn += l;
        want[wn++] = '\0';
    }
    CHECK(n == wn && memcmp(got, want, wn) == 0,
          "every argument arrives byte for byte");
    if (n != wn || memcmp(got, want, wn) != 0) {
        size_t off = 0;
        for (int i = 1; argv[i] && off < n; i++) {
            const size_t l = strlen(got + off);
            if (strcmp(got + off, argv[i]) != 0)
                fprintf(stderr, "    arg %d: sent [%s] got [%s]\n",
                        i, argv[i], got + off);
            off += l + 1;
        }
    }
    CHECK(!file_exists(CANARY), "nothing in an argument was executed");
    remove(CANARY);
}

static void test_spawn_status_and_stderr(void) {
    printf("test_spawn_status_and_stderr:\n");

    const char* const exit7[] = { VV_SPAWN_CHILD, "--exit7", NULL };
    FILE* out = NULL;
    vv_child_t* c = vv_spawn_read(exit7, VV_SPAWN_STDERR_NULL, &out);
    CHECK(c != NULL, "child starts");
    if (c) {
        char buf[64];
        read_all(out, buf, sizeof(buf));
        CHECK(vv_spawn_wait(c, false) == 7, "exit status comes back");
    }

    const char* const err[] = { VV_SPAWN_CHILD, "--stderr", "x", NULL };
    c = vv_spawn_read(err, VV_SPAWN_STDERR_MERGE, &out);
    if (c) {
        char buf[256];
        const size_t n = read_all(out, buf, sizeof(buf) - 1);
        buf[n] = '\0';
        vv_spawn_wait(c, false);
        /* "on-stderr" then "x\0": the stream is text up to the NUL. */
        CHECK(strstr(buf, "on-stderr") != NULL, "merged stderr reaches the pipe");
    }
    c = vv_spawn_read(err, VV_SPAWN_STDERR_NULL, &out);
    if (c) {
        char buf[256];
        const size_t n = read_all(out, buf, sizeof(buf));
        vv_spawn_wait(c, false);
        CHECK(n == 2 && buf[0] == 'x' && buf[1] == '\0',
              "discarded stderr stays out of the pipe");
    }

    const char* const missing[] = { "vv-no-such-program-0x5eed", NULL };
    c = vv_spawn_read(missing, VV_SPAWN_STDERR_NULL, &out);
    CHECK(c == NULL, "a missing program is reported, not run");
    if (c) vv_spawn_wait(c, true);
}

static void test_spawn_terminate(void) {
    printf("test_spawn_terminate:\n");
    const char* const argv[] = { VV_SPAWN_CHILD, "--sleep", NULL };
    FILE* out = NULL;
    vv_child_t* c = vv_spawn_read(argv, VV_SPAWN_STDERR_NULL, &out);
    CHECK(c != NULL, "child starts");
    if (!c) return;
    char buf[16];
    const size_t n = fread(buf, 1, 9, out);
    CHECK(n == 9 && memcmp(buf, "sleeping", 8) == 0, "child is running");

    const double t0 = vv_time_ms();
    vv_spawn_terminate(c);
    const size_t rest = read_all(out, buf, sizeof(buf));  /* EOF once it dies */
    const int rc = vv_spawn_wait(c, false);
    const double dt = vv_time_ms() - t0;
    CHECK(rest == 0 && rc != 0 && dt < 10000.0,
          "terminate ends a blocked reader promptly");
}

/* ─── Concurrent uploads ───────────────────────────────────────────────── */

#define UPLOAD_THREADS 8
#define UPLOAD_ROUNDS  25

/* A 16-bit mono 24 kHz WAV of `n` samples, all equal to `v`. */
static unsigned char* make_wav(int n, int16_t v, size_t* size) {
    const size_t data = (size_t)n * 2;
    *size = 44 + data;
    unsigned char* w = (unsigned char*)vv_alloc(*size);
    if (!w) return NULL;
    const uint32_t riff = (uint32_t)(36 + data), sr = 24000, br = 48000;
    const uint32_t d32 = (uint32_t)data, fmt_len = 16;
    const uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
    memcpy(w, "RIFF", 4);      memcpy(w + 4, &riff, 4);
    memcpy(w + 8, "WAVEfmt ", 8); memcpy(w + 16, &fmt_len, 4);
    memcpy(w + 20, &pcm, 2);   memcpy(w + 22, &ch, 2);
    memcpy(w + 24, &sr, 4);    memcpy(w + 28, &br, 4);
    memcpy(w + 32, &align, 2); memcpy(w + 34, &bits, 2);
    memcpy(w + 36, "data", 4); memcpy(w + 40, &d32, 4);
    for (int i = 0; i < n; i++) memcpy(w + 44 + 2 * i, &v, 2);
    return w;
}

typedef struct { int id; int bad; } upload_arg_t;

static VV_THREAD_RET upload_thread(void* p) {
    upload_arg_t* a = (upload_arg_t*)p;
    const int n = 2400 + 37 * a->id;           /* a length of its own */
    const int16_t v = (int16_t)(1000 * (a->id + 1));
    size_t size = 0;
    unsigned char* wav = make_wav(n, v, &size);
    for (int r = 0; wav && r < UPLOAD_ROUNDS; r++) {
        float* pcm = NULL;
        int got = 0, sr = 0;
        if (vv_audio_load_memory(wav, size, &pcm, &got, &sr) != VV_OK) {
            a->bad++;
            continue;
        }
        const float want = (float)v / 32768.0f;
        if (got != n || sr != 24000 || !pcm ||
            pcm[0] < want - 1e-3f || pcm[0] > want + 1e-3f ||
            pcm[got - 1] < want - 1e-3f || pcm[got - 1] > want + 1e-3f)
            a->bad++;
        vv_free(pcm);
    }
    vv_free(wav);
    VV_THREAD_RETURN;
}

static void test_concurrent_uploads(void) {
    printf("test_concurrent_uploads:\n");
    vv_thread_t th[UPLOAD_THREADS];
    upload_arg_t args[UPLOAD_THREADS];
    int started = 0;
    for (int i = 0; i < UPLOAD_THREADS; i++) {
        args[i].id = i;
        args[i].bad = 0;
        if (vv_thread_start(&th[i], (vv_thread_fn)upload_thread, &args[i]))
            started++;
        else
            args[i].bad = -1;
    }
    for (int i = 0; i < UPLOAD_THREADS; i++)
        if (args[i].bad >= 0) vv_thread_join(th[i]);
    int bad = 0;
    for (int i = 0; i < UPLOAD_THREADS; i++) bad += args[i].bad > 0 ? args[i].bad : 0;
    CHECK(started == UPLOAD_THREADS, "upload threads start");
    CHECK(bad == 0, "every upload decodes to its own audio");
}

/* ─── HTTP request heads ───────────────────────────────────────────────── */

typedef struct {
    int           port;
    vv_http_t*    srv;
    volatile int  done;
    vv_status_t   status;
} server_arg_t;

static void handler(const vv_http_req_t* req, vv_http_res_t* res, void* user) {
    (void)user;
    if (strcmp(req->path, "/err") == 0) {
        vv_http_error(res, 400, "bad \"type\"", "quote \" backslash \\ newline \n end");
        return;
    }
    char body[64];
    const int n = snprintf(body, sizeof(body), "%zu", req->body_len);
    vv_http_respond(res, 200, "text/plain", body, (size_t)n);
}

static VV_THREAD_RET server_thread(void* p) {
    server_arg_t* a = (server_arg_t*)p;
    a->status = vv_http_serve("127.0.0.1", a->port, 8, 1 << 20, handler, NULL,
                              &a->srv);
    a->done = 1;
    VV_THREAD_RETURN;
}

/*
 * Send `parts` with a pause between them and return what came back, or an
 * empty string if the server closed without answering or went quiet.
 */
static void exchange(int port, const char* const* parts, char* out, size_t cap) {
    out[0] = '\0';
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return;
#ifdef _WIN32
    DWORD tv = 5000;
#else
    struct timeval tv = { 5, 0 };
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = htonl(0x7f000001);
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close_sock(s);
        return;
    }
    for (int i = 0; parts[i]; i++) {
        if (i) sleep_ms(150);
        send(s, parts[i], (int)strlen(parts[i]), 0);
    }
    size_t len = 0;
    for (;;) {
        const int n = (int)recv(s, out + len, (int)(cap - 1 - len), 0);
        if (n <= 0) break;
        len += (size_t)n;
        if (len >= cap - 1) break;
    }
    out[len] = '\0';
    close_sock(s);
}

static const char* body_of(const char* resp) {
    const char* b = strstr(resp, "\r\n\r\n");
    return b ? b + 4 : "";
}

static void test_http_heads(void) {
    printf("test_http_heads:\n");
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    server_arg_t a;
    memset(&a, 0, sizeof(a));
    vv_thread_t th;
    int up = 0;
    for (int attempt = 0; attempt < 8 && !up; attempt++) {
        memset(&a, 0, sizeof(a));
        a.port = 39000 + (int)((unsigned)(vv_time_ms()) % 2000) + attempt * 7;
        if (!vv_thread_start(&th, (vv_thread_fn)server_thread, &a)) break;
        for (int w = 0; w < 200 && !a.srv && !a.done; w++) sleep_ms(10);
        if (a.srv && !a.done) up = 1;
        else vv_thread_join(th);
    }
    CHECK(up, "server listens");
    if (!up) return;
    sleep_ms(50);

    char resp[4096];

    /* Any capitalisation, body arriving after the head. */
    const char* const upper[] = {
        "POST /x HTTP/1.1\r\nHost: t\r\nCONTENT-LENGTH: 5\r\n\r\n", "hello", NULL };
    exchange(a.port, upper, resp, sizeof(resp));
    CHECK(strcmp(body_of(resp), "5") == 0,
          "CONTENT-LENGTH is honoured, and the body is waited for");

    const char* const mixed[] = {
        "POST /x HTTP/1.1\r\ncOnTeNt-LeNgTh:3\r\n\r\n", "abc", NULL };
    exchange(a.port, mixed, resp, sizeof(resp));
    CHECK(strcmp(body_of(resp), "3") == 0, "cOnTeNt-LeNgTh is honoured");

    /* The length is a header of its own, not text inside another one. */
    const char* const inner[] = {
        "POST /x HTTP/1.1\r\nX-Note: Content-Length: 900\r\n"
        "Content-Length: 3\r\n\r\nabc", NULL };
    exchange(a.port, inner, resp, sizeof(resp));
    CHECK(strcmp(body_of(resp), "3") == 0,
          "a Content-Length inside another header's value is ignored");

    const char* const conflict[] = {
        "POST /x HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabcd",
        NULL };
    exchange(a.port, conflict, resp, sizeof(resp));
    CHECK(resp[0] == '\0', "conflicting Content-Length is refused");

    const char* const chunked[] = {
        "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n",
        NULL };
    exchange(a.port, chunked, resp, sizeof(resp));
    CHECK(resp[0] == '\0', "Transfer-Encoding is refused, not misread");

    const char* const err[] = { "GET /err HTTP/1.1\r\nHost: t\r\n\r\n", NULL };
    exchange(a.port, err, resp, sizeof(resp));
    const char* eb = body_of(resp);
    CHECK(strstr(eb, "quote \\\" backslash \\\\ newline \\u000a end") != NULL &&
          strstr(eb, "bad \\\"type\\\"") != NULL,
          "error bodies escape quotes, backslashes and control characters");

    vv_http_shutdown(a.srv);
    vv_thread_join(th);
}

int main(void) {
    printf("=== Untrusted input ===\n\n");
    test_spawn_arguments();
    test_spawn_status_and_stderr();
    test_spawn_terminate();
    test_concurrent_uploads();
    test_http_heads();
    printf("\n=== %s ===\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
