/**
 * @file http.c
 * @brief A small HTTP/1.1 server: sockets, request parsing, multipart.
 *
 * Deliberately minimal — enough to be a correct OpenAI audio endpoint and
 * nothing more. One thread per connection, bounded by max_conns; requests are
 * read whole into memory (capped by max_body) because the handler needs the
 * complete upload anyway. No TLS: put it behind a reverse proxy, which is
 * also how GPUStack runs its backends.
 */

#include "vv_http.h"
#include "vv_ws.h"

#include "vibevoice/vibevoice.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
typedef int socklen_t;
#define close_socket closesocket
#define SHUT_RDWR SD_BOTH
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define close_socket close
#endif

#include "vv_thread.h"

struct vv_http {
    SOCKET        listen_fd;
    volatile int  running;
    int           max_conns;
    size_t        max_body;
    vv_http_handler_fn handler;
    void*         user;

    vv_mutex_t    lock;
    vv_cond_t     conn_done;
    int           n_conns;
};

/* ─── Small helpers ──────────────────────────────────────────────────────── */

static int ci_equal(const char* a, const char* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        char ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca += 32;
        if (cb >= 'A' && cb <= 'Z') cb += 32;
        if (ca != cb) return 0;
        if (!ca) break;
    }
    return 1;
}

const char* vv_http_header(const vv_http_req_t* r, const char* name) {
    const size_t n = strlen(name);
    for (int i = 0; i < r->n_headers; i++)
        if (strlen(r->headers[i].name) == n &&
            ci_equal(r->headers[i].name, name, n))
            return r->headers[i].value;
    return NULL;
}

/** @brief Case-insensitive substring search over a bounded buffer. */
static const char* mem_find(const char* hay, size_t hlen,
                            const char* needle, size_t nlen) {
    if (nlen == 0 || hlen < nlen) return NULL;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, nlen) == 0) return hay + i;
    return NULL;
}

/* ─── Multipart/form-data ────────────────────────────────────────────────── */

/**
 * @brief Split a multipart body into parts.
 *
 * Only what the audio endpoint needs: a flat list of parts, each with a name,
 * an optional filename and a value. Nested multiparts and transfer encodings
 * are not supported, and no client sends them here.
 */
int vv_http_parse_multipart(const char* body, size_t len,
                            const char* content_type,
                            vv_http_part_t* parts, int max_parts) {
    const char* bpos = content_type ? strstr(content_type, "boundary=") : NULL;
    if (!bpos) return 0;
    bpos += strlen("boundary=");

    char boundary[256];
    size_t bi = 0;
    if (*bpos == '"') {
        bpos++;
        while (*bpos && *bpos != '"' && bi < sizeof(boundary) - 1)
            boundary[bi++] = *bpos++;
    } else {
        while (*bpos && *bpos != ';' && *bpos != ' ' && *bpos != '\r' &&
               bi < sizeof(boundary) - 1)
            boundary[bi++] = *bpos++;
    }
    boundary[bi] = '\0';
    if (bi == 0) return 0;

    char sep[268];
    const int seplen = snprintf(sep, sizeof(sep), "--%s", boundary);

    int n = 0;
    const char* p = mem_find(body, len, sep, (size_t)seplen);
    while (p && n < max_parts) {
        p += seplen;
        if ((size_t)(p - body) + 2 <= len && p[0] == '-' && p[1] == '-') break;
        if ((size_t)(p - body) + 2 <= len && p[0] == '\r' && p[1] == '\n') p += 2;

        const char* hdr_end = mem_find(p, len - (size_t)(p - body), "\r\n\r\n", 4);
        if (!hdr_end) break;

        vv_http_part_t* part = &parts[n];
        memset(part, 0, sizeof(*part));

        /* Content-Disposition: form-data; name="x"; filename="y" */
        const size_t hlen = (size_t)(hdr_end - p);
        const char* np = mem_find(p, hlen, "name=\"", 6);
        if (np) {
            np += 6;
            size_t i = 0;
            while (np < hdr_end && *np != '"' && i < sizeof(part->name) - 1)
                part->name[i++] = *np++;
            part->name[i] = '\0';
        }
        const char* fp = mem_find(p, hlen, "filename=\"", 10);
        if (fp) {
            fp += 10;
            size_t i = 0;
            while (fp < hdr_end && *fp != '"' && i < sizeof(part->filename) - 1)
                part->filename[i++] = *fp++;
            part->filename[i] = '\0';
        }

        const char* data = hdr_end + 4;
        const char* next = mem_find(data, len - (size_t)(data - body),
                                    sep, (size_t)seplen);
        const char* end = next ? next : body + len;
        /* Strip the CRLF that precedes the next boundary. */
        if (end - data >= 2 && end[-2] == '\r' && end[-1] == '\n') end -= 2;

        part->data = data;
        part->size = (size_t)(end - data);
        n++;
        p = next;
    }
    return n;
}

const vv_http_part_t* vv_http_part(const vv_http_part_t* parts, int n,
                                   const char* name) {
    for (int i = 0; i < n; i++)
        if (strcmp(parts[i].name, name) == 0) return &parts[i];
    return NULL;
}

/* ─── Responses ──────────────────────────────────────────────────────────── */

static void send_all(SOCKET fd, const char* buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        const int n = (int)send(fd, buf + off, (int)(len - off), 0);
        if (n <= 0) return;
        off += (size_t)n;
    }
}

void vv_http_respond(vv_http_res_t* res, int status, const char* content_type,
                     const void* body, size_t len) {
    static const struct { int code; const char* text; } reasons[] = {
        { 200, "OK" }, { 400, "Bad Request" }, { 401, "Unauthorized" },
        { 404, "Not Found" }, { 405, "Method Not Allowed" },
        { 413, "Payload Too Large" }, { 415, "Unsupported Media Type" },
        { 500, "Internal Server Error" }, { 503, "Service Unavailable" },
    };
    const char* reason = "OK";
    for (size_t i = 0; i < sizeof(reasons) / sizeof(reasons[0]); i++)
        if (reasons[i].code == status) { reason = reasons[i].text; break; }

    char head[512];
    const int hn = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, reason, content_type, len);

    send_all((SOCKET)(intptr_t)res->fd, head, (size_t)hn);
    if (body && len) send_all((SOCKET)(intptr_t)res->fd, (const char*)body, len);
    res->sent = true;
}

void vv_http_respond_json(vv_http_res_t* res, int status, const char* json) {
    vv_http_respond(res, status, "application/json", json, strlen(json));
}

/*
 * Copy `s` into `out` as the body of a JSON string. Messages can carry text
 * that came from outside (a status string, a file name); a quote or a
 * control character in one must not end the string it sits in.
 */
static void json_escape_into(char* out, size_t cap, const char* s) {
    size_t o = 0;
    for (const unsigned char* p = (const unsigned char*)(s ? s : ""); *p; p++) {
        char tmp[8];
        const char* piece = tmp;
        if (*p == '"')       piece = "\\\"";
        else if (*p == '\\') piece = "\\\\";
        else if (*p < 0x20)  snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
        else { tmp[0] = (char)*p; tmp[1] = '\0'; }
        const size_t n = strlen(piece);
        if (o + n + 1 > cap) break;
        memcpy(out + o, piece, n);
        o += n;
    }
    out[o] = '\0';
}

void vv_http_error(vv_http_res_t* res, int status, const char* type,
                   const char* message) {
    char msg[512], typ[64], buf[1024];
    json_escape_into(msg, sizeof(msg), message);
    json_escape_into(typ, sizeof(typ), type);
    /* OpenAI's error envelope, which clients and GPUStack both understand. */
    snprintf(buf, sizeof(buf),
             "{\"error\":{\"message\":\"%s\",\"type\":\"%s\",\"code\":%d}}",
             msg, typ, status);
    vv_http_respond_json(res, status, buf);
}

/* ─── Connection handling ────────────────────────────────────────────────── */

typedef struct {
    vv_http_t* srv;
    SOCKET     fd;
} conn_arg_t;

/*
 * Content-Length of a request head, found as a header line of its own (not as
 * text inside some other header's value), in any capitalisation. Returns -1
 * for a head that must be refused: a malformed or repeated-and-different
 * length, or a Transfer-Encoding, which this server does not implement (a
 * chunked body read as raw bytes would be parsed as garbage).
 */
static int head_content_length(const char* head, size_t head_len,
                               size_t* want) {
    *want = 0;
    int seen = 0;
    const char* end = head + head_len;
    const char* line = memchr(head, '\n', head_len);
    while (line && ++line < end) {
        const char* eol = memchr(line, '\n', (size_t)(end - line));
        if (!eol) eol = end;
        const size_t n = (size_t)(eol - line);
        if (n >= 17 && ci_equal(line, "transfer-encoding", 17)) return -1;
        if (n >= 15 && ci_equal(line, "content-length:", 15)) {
            const char* v = line + 15;
            while (v < eol && (*v == ' ' || *v == '\t')) v++;
            size_t value = 0;
            const char* d = v;
            for (; d < eol && *d >= '0' && *d <= '9'; d++) {
                if (value > ((size_t)-1 - 9) / 10) return -1;   /* overflow */
                value = value * 10 + (size_t)(*d - '0');
            }
            if (d == v) return -1;
            while (d < eol && (*d == ' ' || *d == '\t' || *d == '\r')) d++;
            if (d != eol) return -1;
            if (seen && value != *want) return -1;
            *want = value;
            seen = 1;
        }
        line = eol < end ? eol : NULL;
    }
    return seen;
}

/* How long a client may take to send its request head, and to go quiet. */
#define VV_HTTP_HEAD_TIMEOUT_MS  30000
#define VV_HTTP_IO_TIMEOUT_MS    60000

static void set_io_timeouts(SOCKET fd, int ms) {
#ifdef _WIN32
    const DWORD tv = (DWORD)ms;
#else
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
#endif
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
}

static bool read_request(SOCKET fd, size_t max_body, vv_http_req_t* req,
                         char** out_buf) {
    /*
     * One spare byte past `cap` keeps the buffer NUL-terminated after every
     * recv(), so no string function can run past what was received.
     */
    size_t cap = 64 * 1024, len = 0;
    char* buf = (char*)vv_alloc(cap + 1);
    if (!buf) return false;
    buf[0] = '\0';

    /*
     * Every recv() is bounded by the socket timeout, and the head as a whole
     * by a deadline, so a client that trickles one byte at a time cannot
     * keep a connection thread forever.
     */
    const double head_deadline = vv_time_ms() + VV_HTTP_HEAD_TIMEOUT_MS;
    size_t head_end = 0, want = 0;
    for (;;) {
        if (len == cap) {
            if (cap >= max_body + (1 << 20)) { vv_free(buf); return false; }
            size_t ncap = cap * 2;
            char* nb = (char*)vv_alloc(ncap + 1);
            if (!nb) { vv_free(buf); return false; }
            memcpy(nb, buf, len);
            vv_free(buf);
            buf = nb; cap = ncap;
        }
        const int n = (int)recv(fd, buf + len, (int)(cap - len), 0);
        if (n <= 0) break;
        len += (size_t)n;
        buf[len] = '\0';

        if (!head_end) {
            const char* e = mem_find(buf, len, "\r\n\r\n", 4);
            if (e) {
                head_end = (size_t)(e - buf) + 4;
                if (head_content_length(buf, head_end, &want) < 0 ||
                    want > max_body) {
                    vv_free(buf);
                    return false;
                }
            } else if (vv_time_ms() > head_deadline || len > (64u << 10)) {
                vv_free(buf);          /* head too slow, or too large */
                return false;
            }
        }
        /* Stop once Content-Length worth of body has arrived. */
        if (head_end && len >= head_end + want) break;
    }
    if (!head_end) { vv_free(buf); return false; }

    memset(req, 0, sizeof(*req));

    /*
     * Parse only the head. The blank line that ends it becomes a NUL, so
     * every search below stops there and never reaches into the body.
     */
    buf[head_end - 2] = '\0';

    /* Request line */
    char* p = buf;
    char* sp1 = memchr(p, ' ', head_end);
    if (!sp1) { vv_free(buf); return false; }
    *sp1 = '\0';
    snprintf(req->method, sizeof(req->method), "%s", p);

    char* sp2 = memchr(sp1 + 1, ' ', head_end - (size_t)(sp1 + 1 - buf));
    if (!sp2) { vv_free(buf); return false; }
    *sp2 = '\0';
    snprintf(req->path, sizeof(req->path), "%s", sp1 + 1);

    char* q = strchr(req->path, '?');
    if (q) { *q = '\0'; snprintf(req->query, sizeof(req->query), "%s", q + 1); }

    /* Headers */
    char* line = strchr(sp2 + 1, '\n');
    while (line && req->n_headers < VV_HTTP_MAX_HEADERS) {
        line++;
        if (line[0] == '\0' || line[0] == '\r' || line[0] == '\n') break;
        char* eol = strchr(line, '\n');
        char* colon = strchr(line, ':');
        if (!colon || (eol && colon > eol)) break;
        if (eol) *eol = '\0';
        *colon = '\0';
        char* val = colon + 1;
        while (*val == ' ' || *val == '\t') val++;
        size_t vlen = strlen(val);
        while (vlen && (val[vlen - 1] == '\r' || val[vlen - 1] == ' ' ||
                        val[vlen - 1] == '\t'))
            val[--vlen] = '\0';
        snprintf(req->headers[req->n_headers].name,
                 sizeof(req->headers[0].name), "%s", line);
        snprintf(req->headers[req->n_headers].value,
                 sizeof(req->headers[0].value), "%s", val);
        req->n_headers++;
        if (!eol) break;
        line = eol;
    }

    req->body = buf + head_end;
    req->body_len = len - head_end;
    *out_buf = buf;
    return true;
}

static VV_THREAD_RET conn_thread(void* arg) {
    conn_arg_t* ca = (conn_arg_t*)arg;
    vv_http_t* srv = ca->srv;
    const SOCKET fd = ca->fd;
    vv_free(ca);

    vv_http_req_t req;
    char* buf = NULL;
    if (read_request(fd, srv->max_body, &req, &buf)) {
        vv_http_res_t res;
        res.fd = (void*)(intptr_t)fd;
        res.sent = false;

        if (strcmp(req.method, "OPTIONS") == 0) {
            vv_http_respond(&res, 200, "text/plain", "", 0);
        } else {
            srv->handler(&req, &res, srv->user);
            if (!res.sent)
                vv_http_error(&res, 500, "internal_error", "no response");
        }
        vv_free(buf);
    }

    shutdown(fd, SHUT_RDWR);
    close_socket(fd);

    vv_mutex_lock(&srv->lock);
    srv->n_conns--;
    vv_cond_signal(&srv->conn_done);
    vv_mutex_unlock(&srv->lock);
    VV_THREAD_RETURN;
}

vv_status_t vv_http_serve(const char* host, int port, int max_conns,
                          size_t max_body, vv_http_handler_fn handler,
                          void* user, vv_http_t** out) {
    if (!handler || !out) return VV_ERR_NULL_PTR;

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return VV_ERR_IO;
#else
    signal(SIGPIPE, SIG_IGN);
#endif

    vv_http_t* s = (vv_http_t*)vv_alloc(sizeof(vv_http_t));
    if (!s) return VV_ERR_OUT_OF_MEMORY;
    memset(s, 0, sizeof(*s));
    s->max_conns = max_conns > 0 ? max_conns : 8;
    s->max_body  = max_body ? max_body : (size_t)512 * 1024 * 1024;
    s->handler   = handler;
    s->user      = user;
    s->running   = 1;
    vv_mutex_init(&s->lock);
    vv_cond_init(&s->conn_done);

    s->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (s->listen_fd == INVALID_SOCKET) { vv_free(s); return VV_ERR_IO; }

    int one = 1;
    setsockopt(s->listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one,
               sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = (!host || strcmp(host, "0.0.0.0") == 0)
                           ? INADDR_ANY : inet_addr(host);

    if (bind(s->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        VV_LOG_E("server: cannot bind %s:%d", host ? host : "0.0.0.0", port);
        close_socket(s->listen_fd);
        vv_free(s);
        return VV_ERR_IO;
    }
    if (listen(s->listen_fd, 64) != 0) {
        close_socket(s->listen_fd);
        vv_free(s);
        return VV_ERR_IO;
    }

    *out = s;
    VV_LOG_I("server: listening on %s:%d (max %d connections)",
             host ? host : "0.0.0.0", port, s->max_conns);

    while (s->running) {
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        const SOCKET fd = accept(s->listen_fd, (struct sockaddr*)&peer, &plen);
        if (fd == INVALID_SOCKET) {
            if (!s->running) break;
            continue;
        }

        int nod = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&nod, sizeof(nod));
        set_io_timeouts(fd, VV_HTTP_IO_TIMEOUT_MS);

        vv_mutex_lock(&s->lock);
        if (s->n_conns >= s->max_conns || !s->running) {
            vv_mutex_unlock(&s->lock);
            vv_http_res_t res = {0};
            res.fd = (void*)(intptr_t)fd;
            vv_http_error(&res, 503, "server_overloaded", "connection limit reached");
            close_socket(fd);
            continue;
        }
        s->n_conns++;
        vv_mutex_unlock(&s->lock);

        conn_arg_t* ca = (conn_arg_t*)vv_alloc(sizeof(conn_arg_t));
        if (!ca) {
            close_socket(fd);
            vv_mutex_lock(&s->lock); s->n_conns--; vv_mutex_unlock(&s->lock);
            continue;
        }
        ca->srv = s;
        ca->fd = fd;

        vv_thread_t th;
        if (!vv_thread_start(&th, (vv_thread_fn)conn_thread, ca)) {
            close_socket(fd);
            vv_free(ca);
            vv_mutex_lock(&s->lock);
            s->n_conns--;
            vv_mutex_unlock(&s->lock);
            continue;
        }
#ifndef _WIN32
        pthread_detach(th);
#else
        CloseHandle(th);
#endif
    }

    /* Let in-flight requests finish before the handler's state goes away. */
    vv_mutex_lock(&s->lock);
    while (s->n_conns > 0) vv_cond_wait(&s->conn_done, &s->lock);
    vv_mutex_unlock(&s->lock);

    close_socket(s->listen_fd);
    vv_cond_destroy(&s->conn_done);
    vv_mutex_destroy(&s->lock);
    vv_free(s);
    return VV_OK;
}

void vv_http_shutdown(vv_http_t* s) {
    if (!s) return;
    s->running = 0;
    /* Wake the accept() loop. */
    shutdown(s->listen_fd, SHUT_RDWR);
    close_socket(s->listen_fd);
    vv_cond_broadcast(&s->conn_done);
}

/* ─── Streaming responses and WebSocket ──────────────────────────────────── */
/*
 * Additive on purpose: nothing above changes, so request parsing can evolve
 * on its own. A streaming handler runs on its connection's thread, writes as
 * much as it likes and returns; the connection closes after it, so the body
 * is delimited by the close and needs neither Content-Length nor chunked
 * encoding. docs/STREAMING.md describes the endpoints built on this.
 */

static bool send_all_ok(SOCKET fd, const char* buf, size_t len) {
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    size_t off = 0;
    while (off < len) {
        const int n = (int)send(fd, buf + off, (int)(len - off), flags);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

bool vv_http_respond_begin(vv_http_res_t* res, int status,
                           const char* content_type) {
    if (!res || res->sent) return false;
    char head[512];
    const int hn = snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Cache-Control: no-cache\r\n"
        "X-Accel-Buffering: no\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status == 200 ? "OK" : "Error", content_type);
    res->sent = true;
    return hn > 0 && send_all_ok((SOCKET)(intptr_t)res->fd, head, (size_t)hn);
}

bool vv_http_write(vv_http_res_t* res, const void* buf, size_t n) {
    if (!res || !res->sent) return false;
    return n == 0 || send_all_ok((SOCKET)(intptr_t)res->fd, (const char*)buf, n);
}

bool vv_http_sse_send(vv_http_res_t* res, const char* event, const char* data) {
    char small[1024];
    const size_t need = vv_sse_format(event, data, NULL, 0);
    char* buf = need < sizeof(small) ? small : (char*)vv_alloc(need + 1);
    if (!buf) return false;
    vv_sse_format(event, data, buf, need + 1);
    const bool ok = vv_http_write(res, buf, need);
    if (buf != small) vv_free(buf);
    return ok;
}

int vv_http_read(vv_http_res_t* res, void* buf, size_t cap, int timeout_ms) {
    if (!res || !buf || cap == 0) return -1;
    const SOCKET fd = (SOCKET)(intptr_t)res->fd;
    if (timeout_ms >= 0) {
        fd_set rd;
        FD_ZERO(&rd);
        FD_SET(fd, &rd);
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        const int r = select((int)fd + 1, &rd, NULL, NULL, &tv);
        if (r == 0) return -2;
        if (r < 0) return -1;
    }
    const size_t want = cap > ((size_t)1 << 30) ? ((size_t)1 << 30) : cap;
    const int n = (int)recv(fd, (char*)buf, (int)want, 0);
    return n < 0 ? -1 : n;
}

bool vv_http_peer_gone(vv_http_res_t* res) {
    if (!res) return true;
    const SOCKET fd = (SOCKET)(intptr_t)res->fd;
    fd_set rd;
    FD_ZERO(&rd);
    FD_SET(fd, &rd);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    const int r = select((int)fd + 1, &rd, NULL, NULL, &tv);
    if (r == 0) return false;          /* nothing to read: still there */
    if (r < 0) return true;
    char c;
    const int n = (int)recv(fd, &c, 1, MSG_PEEK);
    return n <= 0;                     /* EOF or a reset */
}

static bool header_has_token(const char* value, const char* token) {
    /* Comma-separated, case-insensitive: "keep-alive, Upgrade". */
    const size_t tl = strlen(token);
    for (const char* p = value; p && *p; ) {
        while (*p == ' ' || *p == ',') p++;
        const char* e = p;
        while (*e && *e != ',') e++;
        const char* t = e;
        while (t > p && t[-1] == ' ') t--;
        if ((size_t)(t - p) == tl && ci_equal(p, token, tl)) return true;
        p = e;
    }
    return false;
}

bool vv_http_ws_accept(const vv_http_req_t* req, vv_http_res_t* res) {
    if (!req || !res || res->sent) return false;
    const char* up = vv_http_header(req, "Upgrade");
    const char* conn = vv_http_header(req, "Connection");
    const char* ver = vv_http_header(req, "Sec-WebSocket-Version");
    const char* key = vv_http_header(req, "Sec-WebSocket-Key");
    char accept[29];
    if (strcmp(req->method, "GET") != 0 || !up ||
        !header_has_token(up, "websocket") || !conn ||
        !header_has_token(conn, "upgrade") || !ver || strcmp(ver, "13") != 0 ||
        !vv_ws_accept_key(key, accept)) {
        vv_http_error(res, 400, "invalid_request_error",
                      "expected a WebSocket upgrade (version 13)");
        return false;
    }
    char head[256];
    const int hn = snprintf(head, sizeof(head),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n"
        "\r\n", accept);
    res->sent = true;
    return hn > 0 && send_all_ok((SOCKET)(intptr_t)res->fd, head, (size_t)hn);
}

bool vv_http_ws_send(vv_http_res_t* res, int opcode, const void* data,
                     size_t n) {
    uint8_t hdr[10];
    const size_t hl = vv_ws_frame_header(opcode, true, n, hdr);
    return vv_http_write(res, hdr, hl) && vv_http_write(res, data, n);
}
