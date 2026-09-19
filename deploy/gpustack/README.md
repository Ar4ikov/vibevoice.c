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
3. Start with **one NVIDIA GPU per replica**, one replica and one slot. The run
   command passes `--gpus all`, so the runtime uses exactly the GPUs GPUStack
   assigned to the replica, however many that is. A replica may also get
   several; see [GPUs](#gpus).
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
   grows per slot. `--split-mode` and `--gpu-memory` are described under
   [GPUs](#gpus) and [Memory](#memory). `--acoustic-sampling gaussian|fix`
   draws the acoustic latent the way the reference does, a fresh draw per
   request, so the same upload no longer gives the same transcript twice; the
   default `mode` is deterministic. `--vram-budget 0.9` and `--gpu-layers N`
   are also supported.
5. `/health` becomes available after model loading. `/v1/models` reports
   `{{model_name}}`, which is the GPUStack deployment name, not necessarily its
   Hugging Face repository ID. Send requests through GPUStack's OpenAI API.

The effective container command is:

```text
/usr/local/bin/vv_cli serve --model <mounted-model-directory> --host 0.0.0.0 --port <assigned-port> --model-name <deployment-name> --gpus all --slots 1 --queue-size 16 --kv-cache tq4 --max-seq-len 32768
```

`--gpus` needs image 0.2.0 or newer, which every tag in `version_configs` is.
A deployment pinned to an older `sha-...` image has to drop it from its run
command.

Entrypoint is the binary; run_command contains its arguments. No `/bin/sh -c` is
needed. For reproducible deployments replace the version and image tag with a
published `sha-...` tag (or use an image digest). The workflow publishes on master
pushes, `v*` tags and manual runs; pull requests build/test without publishing.
It uses the repository's `GITHUB_TOKEN`, with `packages: write`.

## GPUs

The GPUStack worker exposes the GPUs it assigned to a replica through
`CUDA_VISIBLE_DEVICES`, and CUDA renumbers them from 0 inside the container.
`--gpus all` therefore means those cards and no others. Do not pass
`--gpus {{gpu_ids}}`: the placeholder expands to host indexes, and a replica
placed on host GPU 1 would ask for device 1 when it can only see device 0,
then exit with `gpus: device 1 asked for, but only 1 is visible`.

A replica gets **several GPUs** (all on one worker, because custom backends
cannot span workers) in two cases. You can pick them yourself: set
**Scheduling → Schedule Mode** to Manual and fill in **GPU Selector** and
**GPUs per Replica**. GPUStack also assigns them by itself when no single
card has room for its estimate (see [Memory](#memory)); in that case it gives
the replica every GPU on the worker. What the runtime does with several GPUs
is set by `--split-mode`:

| `--split-mode` | what it does | measured on two RTX 3090s ([#9](https://github.com/Ar4ikov/vibevoice.c/pull/9), [#10](https://github.com/Ar4ikov/vibevoice.c/pull/10)) |
|---|---|---|
| `replica` | a full copy on every card, the slots spread over them behind one FIFO queue | 8 × 30 s clips through 4 slots: 13.2 s on one card, 7.6 s on two |
| `layer` | one model, its layers split over the cards; each card holds the KV cache of its own layers and only the hidden state crosses | both capped at 6 GiB: 7 of 28 layers resident and RTF 0.595 on one card, 28 of 28 and RTF 0.059 on two |
| `auto` (default) | `replica` when `--slots` ≥ GPUs, otherwise `layer` | |

If GPUStack spread the replica because the model did not fit on one card, a
copy per card will not fit either. Keep `--slots` below the number of GPUs or
set `--split-mode layer`. With several GPUs and fewer slots, 0.2.0 still logs
`raise --slots to use them all`. The warning is wrong under `auto` and
`layer`: the layer split already uses every card.

One two-GPU replica with `--slots 2` holds the same per card as two one-GPU
replicas. The difference is the queue: the two-GPU replica has one, and the
next request goes to whichever slot frees up first. Two replicas each have
their own, so a request can wait behind one replica while the other is idle.

**CPU only.** GPUStack starts a replica without a GPU only when
**Advanced → Allow CPU Offloading** is on and no card fits. That replica must
not keep `--gpus all`. If no device is visible, it exits with
`gpus: 'all' asked for, but no device is visible`, and `--cpu` does not help.
On a Docker host whose default runtime is `nvidia`, the image's
`NVIDIA_VISIBLE_DEVICES=all` shows the replica every card on the host
instead. Set the deployment's run command to the default without
`--gpus all`, and add `--cpu` to the backend parameters.

## Memory

GPUStack does not measure a custom backend. It reserves an estimate:
safetensors size × 1.2 + 0.5 GiB for a speech-to-text model. For
`Ar4ikov/VibeVoice-ASR-AWQ-W4A16-ASYM` that is 8.3 GiB, divided evenly over
the GPUs of a multi-GPU replica. The estimate only decides placement, and
nothing enforces it. Measured with image 0.2.0 and the parameters above:

| replica | holds per card |
|---|---|
| one GPU | 8.1 GiB |
| two GPUs, `--slots 1` (layer split) | 5.9 GiB on the first, 2.9 GiB on the second |
| two GPUs, `--slots 2` (a copy per card) | 8.1 GiB on each |

The estimate is close for one card. It is not close for a layer split: the
first card also holds the embeddings, the head and the speech encoder, so it
takes more than the even half GPUStack reserved on it. To replace the
estimate, set `GPUSTACK_MODEL_VRAM_CLAIM` (in bytes) under
**Advanced → Environment Variables**.

`--gpu-memory 80%|18GiB|8192M|<bytes>` caps each card. Give one value for all
cards, or one per GPU in container order. The cap counts everything on the
card when the replica starts: the CUDA context and other processes as well as
this one. So `18GiB` means "this card stays under 18 GiB", not "this replica
may take 18 GiB". On a card shared with another deployment, size the cap for
both. When the budget is tight, the runtime first shortens the KV window, down
to 8192 tokens, and then streams layers from host memory. The `budget:` and
`engine:` lines in the log show what it decided.

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
  serve --model /model --model-name vibevoice-asr --gpus all --slots 1 --queue-size 16 --kv-cache tq4
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=vibevoice-asr -F response_format=verbose_json
```

Docker's `--gpus '"device=0,1"'` together with vv_cli's `--gpus all` shows the
container two cards, numbered 0 and 1, which is what a two-GPU replica sees
under GPUStack.

FFmpeg is installed for MP3/M4A/FLAC/etc.; WAV decoding is built into the runtime.
The image targets Linux amd64 and CUDA 12.4 (Turing through Hopper cubins plus
forward-compatible PTX). The host supplies the NVIDIA driver/container runtime.

References: [GPUStack backend management](https://docs.gpustack.ai/2.1/user-guide/inference-backend-management/),
[2.2.2 custom backend implementation](https://github.com/gpustack/gpustack/blob/v2.2.2/gpustack/worker/backends/custom.py),
[its GPU placement](https://github.com/gpustack/gpustack/blob/v2.2.2/gpustack/policies/candidate_selectors/custom_backend_resource_fit_selector.py)
and [VRAM estimate](https://github.com/gpustack/gpustack/blob/v2.2.2/gpustack/policies/utils.py).
