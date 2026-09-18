# GPUStack 2.2.2

Import [vibevoice.yaml](vibevoice.yaml) in **Inference Backends → Add Backend → YAML**.
The backend name must end in `-custom`. The worker pulls the OCI image from GHCR,
mounts the model cache, replaces the command placeholders and starts its configured
container runtime workload (a Pod with Kubernetes, a container with Docker).
No weights, Python or Hugging Face credentials are embedded in the image.

## Deploy

1. Wait for a release to publish `ghcr.io/ar4ikov/vibevoice.c:latest` (the
   newest release; `master` is the head of the branch). On first publication, make the GHCR package
   public in GitHub's package settings, or configure authenticated registry pulls
   in GPUStack. A public Git repository does not automatically make its package public.
2. Import the backend YAML. Create/edit a deployment, choose `vibevoice-custom`,
   a version (`latest`, a release line such as `0.2`, or `master` -- see
   [RELEASING](../../docs/RELEASING.md)), category **Speech-to-Text**, and Hugging Face repository
   `Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM`. Use the complete model directory, not
   an individual safetensors shard. Its tokenizer files must be present.
3. Assign **one NVIDIA GPU per replica**. Start with one replica and one slot.
   This runtime does not shard one model over several GPUs. GPUStack selects and
   exposes the GPU; leave the runtime's GPU index at its container-local default 0.
4. In **Advanced → Backend Parameters**, use one entry per option:

   ```text
   --slots 1
   --queue-size 16
   --kv-cache tq4
   --max-seq-len 32768
   ```

   Remove vLLM flags such as `--gpu-memory-utilization`, `--max-num-seqs`, tool
   parsers, etc. They are not vv_cli options. `parameter_format: space` lets
   GPUStack normalize UI entries such as `--slots=2` into `--slots 2`.
   For two simultaneous transcriptions set `--slots 2`; KV/workspace memory
   grows per slot. `--vram-budget 0.9` and `--gpu-layers N` are also supported.
5. `/health` becomes available after model loading. `/v1/models` reports
   `{{model_name}}`, which is the GPUStack deployment name, not necessarily its
   Hugging Face repository ID. Send requests through GPUStack's OpenAI API.

The effective container command is:

```text
/usr/local/bin/vv_cli serve --model <mounted-model-directory> --host 0.0.0.0 --port <assigned-port> --model-name <deployment-name> --slots 1 --queue-size 16 --kv-cache tq4 --max-seq-len 32768
```

Entrypoint is the binary; run_command contains its arguments. No `/bin/sh -c` is
needed. For reproducible deployments replace the version and image tag with a
published `sha-...` tag (or use an image digest). The workflow publishes on master
pushes, `v*` tags and manual runs; pull requests build/test without publishing.
It uses the repository's `GITHUB_TOKEN`, with `packages: write`.

## Queue semantics

`--slots` limits active transcriptions; `--queue-size` limits additional waiting
requests. Admission is FIFO after upload/authentication, before audio decoding.
The HTTP request stays open and receives the usual OpenAI response when complete.
Full queues return HTTP 503 with an OpenAI error envelope; clients should retry
with backoff. `--queue-size 0` rejects when all slots are occupied.

This is an in-memory synchronous queue, not a durable job broker: no job IDs,
polling API, persistence, or cancellation of work when a client disconnects.
Shutdown drains admitted requests. Set proxy/client timeouts and container
termination grace periods for the longest expected audio plus queue wait.
For durable background jobs, put a persistent broker and job API in front of the
OpenAI endpoint; do not change GPUStack's synchronous transcription contract.

`/metrics` includes `vibevoice_queue_waiting`, `vibevoice_queue_capacity`, and
`vibevoice_queue_rejected_total`. Health/models/metrics bypass the transcription
queue. The default connection limit is `slots + queue-size + 8`, reserving room
for those requests under normal inference load. `--max-conns` overrides it;
all connections, including uploads, count toward this separate limit.
Queued uploads remain in RAM (up to 512 MiB each); choose a modest queue size.

## Standalone smoke test

```sh
docker build -t vibevoice:test .
docker run --rm --gpus 'device=0' -p 8080:8080 \
  -v /path/to/model:/model:ro vibevoice:test \
  serve --model /model --model-name vibevoice-asr --slots 1 --queue-size 16 --kv-cache tq4
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=vibevoice-asr -F response_format=verbose_json
```

FFmpeg is installed for MP3/M4A/FLAC/etc.; WAV decoding is built into the runtime.
The image targets Linux amd64 and CUDA 12.4 (Turing through Hopper cubins plus
forward-compatible PTX). The host supplies the NVIDIA driver/container runtime.

References: [GPUStack backend management](https://docs.gpustack.ai/2.1/user-guide/inference-backend-management/),
[2.2.2 custom backend implementation](https://github.com/gpustack/gpustack/blob/v2.2.2/gpustack/worker/backends/custom.py).
