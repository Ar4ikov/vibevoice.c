/**
 * @file vv_http.h
 * @brief Internal HTTP plumbing used by src/server/api.c.
 */
#ifndef VV_HTTP_H
#define VV_HTTP_H

#include "vibevoice/types.h"

#include <stddef.h>

#define VV_HTTP_MAX_HEADERS 48
#define VV_HTTP_MAX_PARTS   16

typedef struct vv_http vv_http_t;

typedef struct {
    char name[64];
    char value[512];
} vv_http_hdr_t;

typedef struct {
    char        method[8];
    char        path[512];
    char        query[512];
    vv_http_hdr_t headers[VV_HTTP_MAX_HEADERS];
    int         n_headers;
    const char* body;
    size_t      body_len;
} vv_http_req_t;

typedef struct {
    void* fd;
    bool  sent;
} vv_http_res_t;

/** @brief One part of a multipart/form-data body; `data` points into it. */
typedef struct {
    char        name[128];
    char        filename[256];
    const char* data;
    size_t      size;
} vv_http_part_t;

typedef void (*vv_http_handler_fn)(const vv_http_req_t* req,
                                   vv_http_res_t* res, void* user);

/** @brief Bind and accept until vv_http_shutdown(). Blocks the caller. */
vv_status_t vv_http_serve(const char* host, int port, int max_conns,
                          size_t max_body, vv_http_handler_fn handler,
                          void* user, vv_http_t** out);

/** @brief Stop the accept loop and drain in-flight connections. */
void vv_http_shutdown(vv_http_t* s);

const char* vv_http_header(const vv_http_req_t* r, const char* name);

int vv_http_parse_multipart(const char* body, size_t len,
                            const char* content_type,
                            vv_http_part_t* parts, int max_parts);
const vv_http_part_t* vv_http_part(const vv_http_part_t* parts, int n,
                                   const char* name);

void vv_http_respond(vv_http_res_t* res, int status, const char* content_type,
                     const void* body, size_t len);
void vv_http_respond_json(vv_http_res_t* res, int status, const char* json);
void vv_http_error(vv_http_res_t* res, int status, const char* type,
                   const char* message);

/* ─── Streaming responses and WebSocket (end of http.c) ─────────────────── */

/**
 * @brief Start a streaming response: status and headers, no Content-Length.
 *
 * The body ends when the connection closes (every response here is
 * `Connection: close`). Follow with vv_http_write() / vv_http_sse_send().
 * @return false when the client is gone
 */
bool vv_http_respond_begin(vv_http_res_t* res, int status,
                           const char* content_type);

/** @brief Write body bytes; false once the client has disconnected. */
bool vv_http_write(vv_http_res_t* res, const void* buf, size_t n);

/** @brief One Server-Sent Event (see vv_sse_format() in vv_ws.h). */
bool vv_http_sse_send(vv_http_res_t* res, const char* event, const char* data);

/**
 * @brief Read from the connection after the request (WebSocket frames).
 *
 * Bytes the client sent right after the upgrade request may already be in
 * `req->body`; feed those first.
 * @param timeout_ms  < 0 blocks
 * @return bytes read, 0 on close, -1 on error, -2 on timeout
 */
int vv_http_read(vv_http_res_t* res, void* buf, size_t cap, int timeout_ms);

/**
 * @brief Whether the client has closed its end (or the socket failed),
 *        without waiting and without consuming anything it sent.
 *
 * Bytes still unread count as "not gone": what the client sent before
 * closing is read first, and its EOF shows up after.
 */
bool vv_http_peer_gone(vv_http_res_t* res);

/**
 * @brief Validate a WebSocket upgrade and answer 101, or 400 if invalid.
 * @return true when the connection is now a WebSocket
 */
bool vv_http_ws_accept(const vv_http_req_t* req, vv_http_res_t* res);

/** @brief Send one unfragmented, unmasked server frame. */
bool vv_http_ws_send(vv_http_res_t* res, int opcode, const void* data,
                     size_t n);

#endif /* VV_HTTP_H */
