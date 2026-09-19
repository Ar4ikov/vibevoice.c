/**
 * @file vv_ws.h
 * @brief Socket-free pieces of the streaming server: SSE event framing and
 *        WebSocket (RFC 6455) handshake keys, frame headers and a frame
 *        parser.
 *
 * Kept apart from http.c so they can be tested without sockets and so the
 * request-parsing changes on other branches do not collide with them. The
 * socket side (vv_http_respond_begin / vv_http_write / vv_http_ws_accept /
 * vv_http_ws_send / vv_http_read) lives at the end of http.c.
 * docs/STREAMING.md describes the endpoints built on top.
 */
#ifndef VV_WS_H
#define VV_WS_H

#include "vibevoice/types.h"

#include <stddef.h>
#include <stdint.h>

#define VV_WS_OP_CONT   0x0
#define VV_WS_OP_TEXT   0x1
#define VV_WS_OP_BINARY 0x2
#define VV_WS_OP_CLOSE  0x8
#define VV_WS_OP_PING   0x9
#define VV_WS_OP_PONG   0xA

/** Close codes the parser reports (RFC 6455 section 7.4.1). */
#define VV_WS_CLOSE_NORMAL    1000
#define VV_WS_CLOSE_PROTOCOL  1002
#define VV_WS_CLOSE_TOO_BIG   1009

/** @brief SHA-1 (FIPS 180-4). Only for the handshake key, not security. */
void vv_sha1(const void* data, size_t n, uint8_t out[20]);

/** @brief Standard base64 with padding; `out` needs 4*ceil(n/3)+1 bytes. */
size_t vv_base64_encode(const uint8_t* in, size_t n, char* out);

/**
 * @brief Sec-WebSocket-Accept for a client's Sec-WebSocket-Key.
 *
 * base64(SHA-1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")).
 * @return false if the key is missing or longer than a sane client sends
 */
bool vv_ws_accept_key(const char* client_key, char out[29]);

/**
 * @brief Header of a server-to-client frame (never masked).
 * @return header length, 2 to 10 bytes
 */
size_t vv_ws_frame_header(int opcode, bool fin, uint64_t payload_len,
                          uint8_t out[10]);

/**
 * @brief One Server-Sent Event: "event: <event>\n" (when given), a
 *        "data: " line per line of `data`, and the blank line.
 *
 * @return bytes the event needs, like snprintf; `buf` is truncated and
 *         NUL-terminated when it does not fit
 */
size_t vv_sse_format(const char* event, const char* data, char* buf,
                     size_t cap);

/** @brief A complete message or control frame from the parser. */
typedef void (*vv_ws_message_fn)(void* user, int opcode,
                                 const uint8_t* data, size_t len);

/**
 * @brief Incremental client-frame parser.
 *
 * Feed it whatever recv() returned. It unmasks, reassembles fragmented
 * messages and delivers each text/binary message once complete; control
 * frames (ping, pong, close) are delivered as they arrive, even in the
 * middle of a fragmented message. Protocol violations stop it with a close
 * code for the caller to send.
 */
typedef struct {
    size_t   max_message;   /**< largest message (and frame) accepted */
    bool     require_mask;  /**< clients must mask (RFC 6455 5.1) */
    uint8_t* in;            /**< unparsed input */
    size_t   in_len, in_cap;
    uint8_t* msg;           /**< fragments of the message in progress */
    size_t   msg_len, msg_cap;
    int      msg_opcode;    /**< 0 when no message is in progress */
    int      close_code;    /**< set on error */
    bool     failed;
} vv_ws_parser_t;

void vv_ws_parser_init(vv_ws_parser_t* p, size_t max_message);
void vv_ws_parser_free(vv_ws_parser_t* p);

/**
 * @brief Consume bytes, delivering every message they complete.
 * @return VV_OK, VV_ERR_PARSE (protocol error, see close_code),
 *         VV_ERR_OVERFLOW (message too big) or VV_ERR_OUT_OF_MEMORY
 */
vv_status_t vv_ws_parser_feed(vv_ws_parser_t* p, const uint8_t* data,
                              size_t n, vv_ws_message_fn cb, void* user);

#endif /* VV_WS_H */
