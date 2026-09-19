#!/usr/bin/env python3
"""Live-streaming client for `vv_cli serve` with a streaming model.

Streams a WAV file to the WebSocket endpoint (GET /v1/audio/stream) the way
a microphone would -- in small blocks, paced to real time -- prints the text
deltas as they come back, and measures what a live caller waits: for each
chunk, the time from sending the last sample of its window to receiving its
text.

    python tools/stream_client.py --url ws://127.0.0.1:8080/v1/audio/stream \
        --audio test30.wav [--speed 1] [--sessions 4] [--hotwords "A,B"]

--sessions N opens N connections at once, each streaming the same file
(staggered by --stagger seconds), which is how "sessions per GPU" is
measured: every session must keep up with real time.

Also --sse posts the whole file with stream=true and prints the SSE events.

Standard library only; not part of the runtime.
"""
import argparse
import base64
import bisect
import json
import os
import socket
import struct
import sys
import threading
import time
import urllib.parse

CHUNK = 70400      # 22 frames of 3200 samples at 24 kHz
WINDOW = 83200     # + 4 frames of lookahead


def read_wav(path):
    """Mono PCM (16-bit or float32) -> (rate, fmt, raw bytes)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit(f"{path}: not a WAV file")
    pos, fmt, rate, ch, bits, pcm = 12, None, None, None, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if ch != 1:
        raise SystemExit(f"{path}: {ch} channels; send mono")
    if fmt == 1 and bits == 16:
        return rate, "pcm_s16le", 2, pcm
    if fmt == 3 and bits == 32:
        return rate, "pcm_f32le", 4, pcm
    raise SystemExit(f"{path}: format {fmt}/{bits} bits; want PCM16 or float32")


class WebSocket:
    def __init__(self, url, api_key=None):
        u = urllib.parse.urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port or 80))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        path = u.path + (("?" + u.query) if u.query else "")
        req = (f"GET {path} HTTP/1.1\r\nHost: {u.hostname}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n")
        if api_key:
            req += f"Authorization: Bearer {api_key}\r\n"
        self.sock.sendall((req + "\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            b = self.sock.recv(4096)
            if not b:
                raise ConnectionError("closed during the handshake")
            head += b
        head, self.buf = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError(head.decode(errors="replace"))
        self.lock = threading.Lock()

    def send(self, opcode, payload):
        n = len(payload)
        hdr = bytes([0x80 | opcode])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        # XOR with the mask via int arithmetic: fast enough for 20 ms blocks
        m = (mask * (n // 4 + 1))[:n]
        body = (int.from_bytes(payload, "little") ^
                int.from_bytes(m, "little")).to_bytes(n, "little") if n else b""
        with self.lock:
            self.sock.sendall(hdr + mask + body)

    def recv(self):
        """-> (opcode, payload) or (None, None) on close."""
        def need(k):
            while len(self.buf) < k:
                b = self.sock.recv(65536)
                if not b:
                    return False
                self.buf += b
            return True
        if not need(2):
            return None, None
        op, ln = self.buf[0] & 0x0F, self.buf[1] & 0x7F
        off = 2
        if ln == 126:
            if not need(4):
                return None, None
            ln, off = struct.unpack(">H", self.buf[2:4])[0], 4
        elif ln == 127:
            if not need(10):
                return None, None
            ln, off = struct.unpack(">Q", self.buf[2:10])[0], 10
        if not need(off + ln):
            return None, None
        payload, self.buf = self.buf[off:off + ln], self.buf[off + ln:]
        return op, payload


def run_session(idx, args, rate, fmt, bps, pcm, out, echo):
    res = {"session": idx, "chunks": [], "text": None, "error": None}
    out[idx] = res
    try:
        ws = WebSocket(args.url, args.api_key)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"connect: {e}"
        return
    start_msg = {"type": "session.start", "sample_rate": rate, "format": fmt}
    if args.hotwords:
        start_msg["hotwords"] = args.hotwords
    ws.send(0x1, json.dumps(start_msg).encode())

    n_samples = len(pcm) // bps
    sent_pos, sent_t = [], []   # end of each block sent, and when
    t_open = time.time()
    started = threading.Event()

    def reader():
        while True:
            op, p = ws.recv()
            if op is None or op == 0x8:
                break
            if op != 0x1:
                continue
            now = time.time()
            m = json.loads(p)
            if m["type"] == "session.started":
                res["open_ms"] = (now - t_open) * 1000.0
                started.set()
            elif m["type"] == "transcript.text.delta":
                k = m["chunk"]
                # the window's last real input sample, in the sender's rate
                end24 = min(k * CHUNK + WINDOW, int(n_samples * 24000 / rate))
                end_in = min(int(end24 * rate / 24000), n_samples)
                # the block that carried that sample
                j = bisect.bisect_left(sent_pos, end_in)
                t_sent = sent_t[j] if j < len(sent_t) else now
                res["chunks"].append({"chunk": k, "latency_ms": (now - t_sent) * 1000.0,
                                      "text": m["delta"]})
                if echo:
                    sys.stdout.write(m["delta"])
                    sys.stdout.flush()
            elif m["type"] == "transcript.text.done":
                res["text"] = m["text"]
            elif m["type"] == "error":
                res["error"] = m["error"]
                started.set()
        started.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    started.wait(600)
    if res["error"]:
        return
    block = max(1, int(rate * args.block_ms / 1000.0))
    t0 = time.time()
    pos = 0
    while pos < n_samples:
        n = min(block, n_samples - pos)
        ws.send(0x2, pcm[pos * bps:(pos + n) * bps])
        pos += n
        now = time.time()
        sent_pos.append(pos)
        sent_t.append(now)
        if args.speed > 0:
            due = t0 + pos / rate / args.speed
            if due > now:
                time.sleep(due - now)
    t_fin = time.time()
    ws.send(0x1, b'{"type":"session.finish"}')
    th.join(3600)
    res["finish_ms"] = (time.time() - t_fin) * 1000.0
    res["audio_s"] = n_samples / rate
    res["wall_s"] = time.time() - t0
    try:
        ws.send(0x8, struct.pack(">H", 1000))
        ws.sock.close()
    except OSError:
        pass


def sse(args):
    rate, fmt, bps, pcm = read_wav(args.audio)
    with open(args.audio, "rb") as fh:
        wav = fh.read()
    u = urllib.parse.urlparse(args.url)
    boundary = "----vvstream" + base64.b16encode(os.urandom(6)).decode()
    parts = []
    for k, v in (("stream", "true"), ("prompt", args.hotwords or "")):
        if v:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                         f"name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
                 + wav + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    s = socket.create_connection((u.hostname, u.port or 80))
    head = (f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {u.hostname}\r\n"
            f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
            f"Content-Length: {len(body)}\r\n")
    if args.api_key:
        head += f"Authorization: Bearer {args.api_key}\r\n"
    t0 = time.time()
    s.sendall((head + "\r\n").encode() + body)
    buf = b""
    while True:
        b = s.recv(65536)
        if not b:
            break
        buf += b
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            print(f"[{(time.time() - t0) * 1000:7.0f} ms] {ev.decode(errors='replace')}")
            if args.close_after and b"text.delta" in ev:
                args.close_after -= 1
                if args.close_after == 0:
                    print("closing the connection early")
                    s.close()
                    return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8080/v1/audio/stream")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1 = real time, 0 = as fast as the socket takes it")
    ap.add_argument("--block-ms", type=float, default=20.0)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="seconds between session starts")
    ap.add_argument("--hotwords")
    ap.add_argument("--api-key")
    ap.add_argument("--json", help="write per-session results here")
    ap.add_argument("--sse", action="store_true",
                    help="POST the file with stream=true instead")
    ap.add_argument("--close-after", type=int, default=0,
                    help="--sse: drop the connection after N deltas")
    args = ap.parse_args()

    if args.sse:
        if args.url.startswith("ws://"):
            args.url = "http://" + args.url[5:]
        return sse(args)

    rate, fmt, bps, pcm = read_wav(args.audio)
    out = [None] * args.sessions
    ths = []
    for i in range(args.sessions):
        t = threading.Thread(target=run_session,
                             args=(i, args, rate, fmt, bps, pcm, out,
                                   args.sessions == 1))
        t.start()
        ths.append(t)
        if args.stagger:
            time.sleep(args.stagger)
    for t in ths:
        t.join()
    if args.sessions == 1:
        print()

    lat = sorted(c["latency_ms"] for r in out if r for c in r["chunks"])
    errors = [r["error"] for r in out if r and r["error"]]
    summary = {
        "sessions": args.sessions,
        "speed": args.speed,
        "chunks": len(lat),
        "errors": errors,
        "latency_ms_p50": lat[len(lat) // 2] if lat else None,
        "latency_ms_p95": lat[min(len(lat) - 1, len(lat) * 95 // 100)] if lat else None,
        "latency_ms_max": lat[-1] if lat else None,
        "open_ms_max": max((r.get("open_ms", 0) for r in out if r), default=None),
        "finish_ms_max": max((r.get("finish_ms", 0) for r in out if r), default=None),
        "wall_s_max": max((r.get("wall_s", 0) for r in out if r), default=None),
        "audio_s": out[0].get("audio_s") if out[0] else None,
        "texts_identical": len({r["text"] for r in out if r}) == 1,
    }
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"summary": summary, "sessions": out}, fh, indent=1)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
