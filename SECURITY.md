# Security

## Reporting a vulnerability

Report it privately through
[GitHub's advisory form](https://github.com/Ar4ikov/TurboQwen/security/advisories/new),
not as a public issue. Include what you ran, what happened, and the commit or
release you saw it on; a reproduction is worth more than a description.

Expect a first reply within a few days. A fix ships in the next release, with
the advisory published once it is out.

## What is in scope

The runtime reads files and network requests that it does not control, so the
parts worth looking at are:

- **`serve`**: the HTTP reader and multipart parser (`src/server/http.c`),
  the WebSocket framing (`src/server/ws.c`), the JSON built for responses,
  the admission queue, and the API key comparison.
- **Model loading**: the safetensors and GGUF readers (`src/model/`) and the
  tokenizer (`src/text_tokenizer/bpe.c`) — a malformed or hostile checkpoint
  must fail the load, not read past a buffer.
- **Audio decoding**: WAV parsing, and how `ffmpeg` is started for everything
  else (`src/audio/decode.c`, `src/core/vv_spawn.c`). Helpers are started
  from an argument vector, never through a shell.
- **Temporary files**: uploads are created exclusively, mode 0600.

Out of scope: what the model says, the NVIDIA driver and CUDA, and anything
that needs an operator to set an environment variable on the server process
(those are debugging aids, and the dangerous ones are behind build flags).

## Hardening already in place

`serve` bounds the request head, honours `Content-Length` in any spelling,
refuses `Transfer-Encoding`, times out slow clients, compares the API key in
constant time, escapes every string it puts in JSON, and caps both
connections and queued uploads. CodeQL runs on every pull request
(`c-cpp`, `actions`, `python`).
