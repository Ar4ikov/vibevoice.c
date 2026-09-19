#!/usr/bin/env python3
"""Diff the C runtime's intermediate tensors against a PyTorch reference.

Two halves:

  dump   run the HuggingFace model on a wav and save every stage as .npy
  cmp    load those, load the raw blobs the C runtime wrote to VV_DUMP_DIR,
         and report max/mean/relative error and cosine similarity per stage

The C side writes its blobs when VV_DUMP_DIR is set:

    VV_DUMP_DIR=./cdump ./build/vv_cli --model ./model_hf --audio clip.wav

Both halves use the latent *mean* for the acoustic tokenizer rather than the
gaussian draw the reference makes at inference time, so the comparison is
deterministic.

Usage:
    python tools/compare_ref.py dump --model ./model_hf --audio clip.wav \
                                     --out ./refdump --repo /path/to/VibeVoice
    python tools/compare_ref.py cmp  --ref ./refdump --c ./cdump

The streaming checkpoint (microsoft/VibeVoice-ASR-Streaming-7B) has its own
pair, which works chunk by chunk (docs/STREAMING.md has the protocol):

    python tools/compare_ref.py dump-stream --model ./streaming-7b \
        --audio clip.wav --out ./refstream --repo /path/to/VibeVoice
    python tools/compare_ref.py cmp-stream --ref ./refstream --c ./cdump
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SEGMENT_SAMPLES = 60 * 24000

# What upstream strips from each chunk's text after
# tokenizer.decode(skip_special_tokens=True) (streaming_generate).
STREAM_STRIP = ['<|text_chunk_end|>', '<|object_ref_start|>',
                '<|object_ref_end|>', '<|box_start|>', '<|speech_start|>',
                '<|speech_end|>', '<|speech_pad|>']

# (label, reference .npy, C blob written by src/inference/pipeline.c)
STAGES = [
    ("audio24k", "speech", "c_audio24k"),
    ("ac_mean", "ac_mean", "c_ac_mean"),
    ("sem_mean", "sem_mean", "c_sem_mean"),
    ("ac_feat", "ac_feat", "c_ac_feat"),
    ("sem_feat", "sem_feat", "c_sem_feat"),
    ("combined", "combined", "c_combined"),
    ("embeds", "inputs_embeds", "c_embeds"),
    ("logits", "prefill_last_logits", "c_prefill_logits"),
]


def _patch_transformers():
    """Make the upstream VibeVoice package importable on transformers 4.51."""
    from transformers.configuration_utils import PretrainedConfig
    from transformers.models.auto import AutoModel, AutoModelForCausalLM

    # config.json stores torch dtypes that the default __repr__ cannot
    # serialise, and from_dict() logs the config eagerly.
    PretrainedConfig.__repr__ = lambda self: "<%s>" % self.__class__.__name__

    # The package registers its Auto* classes at import time; re-imports and
    # newer transformers make that a hard error instead of a no-op.
    for cls in (AutoModel, AutoModelForCausalLM):
        original = cls.register

        def tolerant(cfg, mdl, exist_ok=True, _o=original, **kw):
            try:
                return _o(cfg, mdl, exist_ok=True)
            except Exception:
                return None

        cls.register = tolerant


def cmd_dump(args):
    import torch

    torch.set_grad_enabled(False)
    sys.path.insert(0, args.repo)
    _patch_transformers()

    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    os.makedirs(args.out, exist_ok=True)
    proc = VibeVoiceASRProcessor.from_pretrained(args.model)
    # `dtype` only from transformers 4.56 on; older releases silently ignore
    # it and load the unquantised checkpoint in its config dtype (float32,
    # 35 GB for the 7B), which does not fit a 24 GB card.
    import transformers
    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = ({"dtype": torch.bfloat16} if major_minor >= (4, 56)
                else {"torch_dtype": torch.bfloat16})
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, device_map=args.device,
        attn_implementation="sdpa", **dtype_kw).eval()

    # The speech path is not quantised; run it in fp32 so the reference is a
    # clean target rather than another approximation.
    mm = model.model
    for module in (mm.acoustic_tokenizer, mm.semantic_tokenizer,
                   mm.acoustic_connector, mm.semantic_connector):
        module.to(torch.float32)

    inputs = proc(args.audio, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    speech = inputs["speech_tensors"][0].numpy()
    np.save(f"{args.out}/input_ids.npy", np.asarray(ids, dtype=np.int64))
    np.save(f"{args.out}/speech.npy", speech)
    print(f"prompt {len(ids)} tokens, {speech.shape[0]} samples")

    st = torch.tensor(speech, dtype=torch.float32, device=args.device).unsqueeze(0)

    if st.shape[1] <= SEGMENT_SAMPLES:
        ac = mm.acoustic_tokenizer.encode(st.unsqueeze(1)).mean
        se = mm.semantic_tokenizer.encode(st.unsqueeze(1)).mean
    else:
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerStreamingCache,
        )
        cache_a = VibeVoiceTokenizerStreamingCache()
        cache_s = VibeVoiceTokenizerStreamingCache()
        idx = torch.arange(1, device=st.device)
        bounds = [(a, min(a + SEGMENT_SAMPLES, st.shape[1]))
                  for a in range(0, st.shape[1], SEGMENT_SAMPLES)]
        parts_a, parts_s = [], []
        for i, (a, b) in enumerate(bounds):
            chunk = st[:, a:b].contiguous().unsqueeze(1)
            final = i == len(bounds) - 1
            parts_a.append(mm.acoustic_tokenizer.encode(
                chunk, cache=cache_a, sample_indices=idx,
                use_cache=True, is_final_chunk=final).mean)
            parts_s.append(mm.semantic_tokenizer.encode(
                chunk, cache=cache_s, sample_indices=idx,
                use_cache=True, is_final_chunk=final).mean)
        ac = torch.cat(parts_a, dim=1).contiguous()
        se = torch.cat(parts_s, dim=1).contiguous()

    ac_feat = mm.acoustic_connector(ac)
    sem_feat = mm.semantic_connector(se)
    combined = ac_feat + sem_feat

    for name, tensor in (("ac_mean", ac), ("sem_mean", se),
                         ("ac_feat", ac_feat), ("sem_feat", sem_feat),
                         ("combined", combined)):
        arr = tensor.float().cpu().numpy()[0]
        np.save(f"{args.out}/{name}.npy", arr)
        print(f"{name:10s} {arr.shape} "
              f"min={arr.min():.4f} max={arr.max():.4f} std={arr.std():.4f}")

    embed = model.get_input_embeddings()
    inp = embed(torch.tensor([ids], device=args.device)).clone()
    inp[inputs["acoustic_input_mask"].to(args.device)] = combined[0].to(inp.dtype)
    np.save(f"{args.out}/inputs_embeds.npy", inp.float().cpu().numpy()[0])

    out = model(inputs_embeds=inp, use_cache=True, return_dict=True,
                output_hidden_states=True)
    np.save(f"{args.out}/prefill_last_logits.npy",
            out.logits[0, -1].float().cpu().numpy())
    for i, h in enumerate(out.hidden_states):
        np.save(f"{args.out}/hidden_{i}.npy", h[0].float().cpu().numpy())

    # Greedy decode so the transcript can be diffed against the CLI's output.
    past = out.past_key_values
    eos = proc.tokenizer.eos_token_id
    tok = int(out.logits[0, -1].argmax())
    gen = [tok]
    while len(gen) < args.max_new_tokens and tok != eos:
        step = model(inputs_embeds=embed(torch.tensor([[tok]], device=args.device)),
                     past_key_values=past, use_cache=True, return_dict=True)
        past = step.past_key_values
        tok = int(step.logits[0, -1].argmax())
        if tok == eos:
            break
        gen.append(tok)
    text = proc.tokenizer.decode(gen, skip_special_tokens=True)
    with open(f"{args.out}/transcript.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
    print("=== reference transcript ===")
    print(text)


def _report(label, ref, got):
    if ref is None:
        print(f"{label:16s} no reference")
        return
    if got is None:
        print(f"{label:16s} no C dump")
        return
    a = ref.reshape(-1).astype(np.float64)
    b = got.reshape(-1).astype(np.float64)
    note = ""
    if a.size != b.size:
        note = f"  SIZE ref={a.size} c={b.size}"
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
    d = np.abs(a - b)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = d.mean() / (np.abs(a).mean() + 1e-9)
    print(f"{label:16s} n={a.size:9d} max={d.max():10.4g} "
          f"rel={rel:8.3%} cos={cos:.6f}{note}")


def cmd_cmp(args):
    def load_c(name, dtype=np.float32):
        path = os.path.join(args.c, name + ".bin")
        return np.fromfile(path, dtype=dtype) if os.path.exists(path) else None

    def load_ref(name):
        path = os.path.join(args.ref, name + ".npy")
        return np.load(path) if os.path.exists(path) else None

    for label, ref_name, c_name in STAGES:
        _report(label, load_ref(ref_name), load_c(c_name))

    # hidden_states[i] is the input to layer i, i.e. the output of layer i-1,
    # except the last entry, which transformers appends *after* the final
    # RMSNorm. The C dump is always pre-norm, so that one is not comparable.
    layers = [i for i in range(64) if load_ref(f"hidden_{i}") is not None]
    last = max(layers) if layers else -1
    for i in layers:
        got = load_c(f"c_layer{i:02d}")
        if got is None:
            continue
        if i == last:
            print(f"layer{i:02d}           skipped "
                  f"(reference is post-final-norm, C dump is pre-norm)")
            continue
        _report(f"layer{i:02d}", load_ref(f"hidden_{i}"), got)

    ref_ids = load_ref("input_ids")
    c_ids = load_c("c_input_ids", np.int32)
    if ref_ids is not None and c_ids is not None:
        same = len(ref_ids) == len(c_ids) and bool((ref_ids == c_ids).all())
        print(f"input_ids        ref={len(ref_ids)} c={len(c_ids)} "
              f"{'identical' if same else 'DIFFER'}")
        if not same:
            n = min(len(ref_ids), len(c_ids))
            bad = [i for i in range(n) if ref_ids[i] != c_ids[i]]
            print("  first mismatches:",
                  [(i, int(ref_ids[i]), int(c_ids[i])) for i in bad[:10]])


# ─── Streaming-7B ────────────────────────────────────────────────────────────


def stream_geometry(model_dir):
    """Chunk geometry exactly as streaming_generate computes it.

    The demo turns preprocessor_config.json's frame counts into seconds and
    streaming_generate turns them back into samples with float math and
    int() truncation; replaying the same expressions keeps us honest if a
    future checkpoint picks counts where that rounding matters.
    """
    with open(os.path.join(model_dir, "preprocessor_config.json")) as fh:
        cfg = json.load(fh)
    sr = cfg["target_sample_rate"]
    ratio = cfg["speech_tok_compress_ratio"]
    frame_s = ratio / sr
    chunk_duration = cfg["chunk_frames"] * frame_s
    delay = cfg["lookahead_frames"] * frame_s
    chunk_samples = int(chunk_duration * sr)
    frame_dur = 3200 / sr                      # hardcoded upstream
    lookahead_sec = round(delay / frame_dur) * frame_dur
    lookahead_samples = int(lookahead_sec * sr)
    return {
        "sample_rate": sr,
        "frame_samples": ratio,
        "chunk_frames": cfg["chunk_frames"],
        "lookahead_frames": cfg["lookahead_frames"],
        "chunk_samples": chunk_samples,
        "lookahead_samples": lookahead_samples,
        "window_samples": chunk_samples + lookahead_samples,
        "normalize_audio": bool(cfg.get("normalize_audio", True)),
    }


def stream_windows(total, chunk, lookahead):
    """[start, end) of every window, as the split_then_encode loop yields."""
    out = []
    start = 0
    while start < total:
        end = min(start + chunk + lookahead, total)
        if end > start:
            out.append((start, end))
        start = min(start + chunk, total)
    return out


def stream_prompt_text(context_info=None):
    keys = "speaker, content"
    head = ("You are a helpful assistant that transcribes audio input into "
            "text output. Please transcribe the following audios streamingly "
            f"with these keys: {keys}")
    if context_info:
        return f"{head} and extra info: {context_info}\n"
    return f"{head}\n"


def stream_clean(text):
    for s in STREAM_STRIP:
        text = text.replace(s, "")
    return text


def cmd_dump_stream(args):
    import torch
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    sys.path.insert(0, args.repo)
    _patch_transformers()

    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    geo = stream_geometry(args.model)
    os.makedirs(args.out, exist_ok=True)

    proc = VibeVoiceASRProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    if tok.text_chunk_end_id is None:
        raise SystemExit(f"{args.model}: no <|text_chunk_end|>, "
                         "not a streaming checkpoint")
    t0 = time.time()
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(args.device).eval()
    # transformers 4.51 knows torch_dtype only; a silently ignored dtype
    # would load the 7B in fp32 (config.json says float32) and not fit.
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    print(f"model loaded in {time.time() - t0:.1f}s")
    mm = model.model
    dev = args.device

    # Deterministic acoustic latent: sample() returns the mean for any
    # dist_type other than 'fix' / 'gaussian'.
    mm.acoustic_tokenizer.std_dist_type = "none"

    if args.speech_dtype == "fp32":
        for m in (mm.acoustic_tokenizer, mm.semantic_tokenizer,
                  mm.acoustic_connector, mm.semantic_connector):
            m.to(torch.float32)

    # The demo's loader: ffmpeg to 24 kHz mono float, no normalisation.
    audio, _ = load_audio_use_ffmpeg(args.audio, resample=True,
                                     target_sr=geo["sample_rate"])
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if args.start or args.duration:
        a = int(round(args.start * geo["sample_rate"]))
        b = len(audio) if not args.duration else \
            a + int(round(args.duration * geo["sample_rate"]))
        audio = audio[a:b].copy()
    audio.tofile(os.path.join(args.out, "audio24k.f32"))
    total = len(audio)
    at = torch.from_numpy(audio).unsqueeze(0).to(dev)

    wins = stream_windows(total, geo["chunk_samples"], geo["lookahead_samples"])
    target = geo["window_samples"]
    print(f"{total} samples ({total / geo['sample_rate']:.2f}s), "
          f"{len(wins)} windows of {target}")

    embed = model.get_input_embeddings()
    prompt_text = stream_prompt_text(args.context_info)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    np.save(os.path.join(args.out, "prompt_ids.npy"),
            np.asarray(prompt_ids, dtype=np.int64))

    ids = {
        "speech_start": tok.speech_start_id,
        "speech_end": tok.speech_end_id,
        "text_chunk_end": tok.text_chunk_end_id,
        "eos": tok.eos_token_id,
    }

    def encode_window(seg):
        if args.speech_dtype == "bf16":
            # Exactly what streaming_generate calls.
            return model.encode_speech(seg)
        x = seg.to(torch.float32).unsqueeze(1)
        ac = mm.acoustic_tokenizer.encode(x).mean
        se = mm.semantic_tokenizer.encode(x).mean
        return mm.acoustic_connector(ac) + mm.semantic_connector(se)

    def fwd(emb, past):
        o = model(inputs_embeds=emb, past_key_values=past, use_cache=True,
                  return_dict=True)
        return o.logits, o.past_key_values

    t0 = time.time()
    logits, past = fwd(embed(torch.tensor([prompt_ids], device=dev)), None)
    sp_start = embed(torch.tensor([[ids["speech_start"]]], device=dev))
    sp_end = embed(torch.tensor([[ids["speech_end"]]], device=dev))
    tce = embed(torch.tensor([[ids["text_chunk_end"]]], device=dev))

    chunks = []
    for ci, (a, b) in enumerate(wins):
        seg = at[:, a:b]
        pad = 0
        if seg.shape[1] < target:
            pad = target - seg.shape[1]
            seg = F.pad(seg, (0, pad))
        feats = encode_window(seg)
        np.save(os.path.join(args.out, f"chunk{ci:04d}_feats.npy"),
                feats[0].float().cpu().numpy())
        kv0 = past.get_seq_length()
        emb = torch.cat([sp_start, feats.to(sp_start.dtype), sp_end], dim=1)
        logits, past = fwd(emb, past)
        np.save(os.path.join(args.out, f"chunk{ci:04d}_logits.npy"),
                logits[0, -1].float().cpu().numpy())
        gen, stop = [], "max_tokens"
        for _ in range(args.max_new_tokens):
            t = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if t == ids["text_chunk_end"]:
                stop = "text_chunk_end"
                break
            if t == ids["eos"]:
                stop = "eos"
                break
            gen.append(t)
            logits, past = fwd(embed(torch.tensor([[t]], device=dev)), past)
        _, past = fwd(tce, past)
        text = stream_clean(tok.decode(gen, skip_special_tokens=True))
        chunks.append({
            "index": ci, "start": a, "end": b, "pad": pad,
            "n_frames": int(feats.shape[1]), "ids": gen, "stop": stop,
            "text": text, "kv_before": kv0,
            "kv_after": past.get_seq_length(),
        })
        print(f"[{ci + 1}/{len(wins)}] kv={kv0}+{emb.shape[1]}+{len(gen)}+1 "
              f"{stop:14s} {text!r}", flush=True)
    elapsed = time.time() - t0

    meta = {
        "model": os.path.abspath(args.model),
        "audio": os.path.abspath(args.audio),
        "start_s": args.start, "duration_s": args.duration,
        "speech_dtype": args.speech_dtype,
        "llm_dtype": "bf16",
        "acoustic_latent": "mean",
        "context_info": args.context_info,
        "geometry": geo,
        "total_samples": total,
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids,
        "token_ids": ids,
        "max_new_tokens_per_chunk": args.max_new_tokens,
        "chunks": chunks,
        "transcript": "".join(c["text"] for c in chunks),
        "kv_final": past.get_seq_length(),
        "elapsed_s": elapsed,
    }

    if args.check_upstream:
        # Run upstream streaming_generate itself and diff chunk texts. With
        # --speech-dtype bf16 this must match exactly; it proves the loop
        # above is the reference and not a paraphrase of it.
        if args.speech_dtype == "fp32":
            for m in (mm.acoustic_tokenizer, mm.semantic_tokenizer,
                      mm.acoustic_connector, mm.semantic_connector):
                m.to(torch.bfloat16)
        up = [t for _, _, t in model.streaming_generate(
            audio_tensor=torch.from_numpy(audio), tokenizer=tok,
            chunk_duration=geo["chunk_frames"] * geo["frame_samples"] / geo["sample_rate"],
            text_audio_delay=geo["lookahead_frames"] * geo["frame_samples"] / geo["sample_rate"],
            sample_rate=geo["sample_rate"],
            max_new_tokens_per_chunk=args.max_new_tokens,
            temperature=0.0, context_info=args.context_info)]
        same = sum(1 for c, u in zip(chunks, up) if c["text"] == u)
        meta["upstream_texts"] = up
        meta["upstream_same_chunks"] = same
        print(f"upstream streaming_generate: {same}/{len(chunks)} chunk texts "
              f"identical (n_upstream={len(up)})")
        for c, u in zip(chunks, up):
            if c["text"] != u:
                print(f"  chunk {c['index']}: ours={c['text']!r} upstream={u!r}")
                break

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "transcript.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(meta["transcript"])
    n_gen = sum(len(c["ids"]) for c in chunks)
    print(f"=== {len(chunks)} chunks, {n_gen} text tokens, KV {meta['kv_final']} "
          f"positions, {elapsed:.1f}s ===")
    print(meta["transcript"])


def cmd_cmp_stream(args):
    """Compare a C-side streaming dump against dump-stream, chunk by chunk.

    C side (VV_DUMP_DIR, written by the streaming session):
      stream_prompt_ids.bin        int32
      stream_cNNNN_feats.bin       float32 [n_frames, hidden]
      stream_cNNNN_logits.bin      float32 [vocab], first generated token
      stream_cNNNN_ids.bin         int32 [count, ids...], generated ids
                                   without the stop id
      stream_cNNNN_text.bin        UTF-8 chunk text after stripping
                                   (absent when the chunk text is empty)
    """
    with open(os.path.join(args.ref, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    def c_path(name):
        return os.path.join(args.c, name + ".bin")

    def load_c(name, dtype=np.float32):
        p = c_path(name)
        return np.fromfile(p, dtype=dtype) if os.path.exists(p) else None

    ref_prompt = np.asarray(meta["prompt_ids"], dtype=np.int64)
    c_prompt = load_c("stream_prompt_ids", np.int32)
    if c_prompt is None:
        print("prompt_ids       no C dump")
    else:
        same = len(c_prompt) == len(ref_prompt) and bool(
            (c_prompt.astype(np.int64) == ref_prompt).all())
        print(f"prompt_ids       ref={len(ref_prompt)} c={len(c_prompt)} "
              f"{'identical' if same else 'DIFFER'}")

    n_same_ids = n_same_text = n_seen = 0
    first_bad = None
    for ch in meta["chunks"]:
        i = ch["index"]
        tag = f"c{i:04d}"
        c_ids = load_c(f"stream_{tag}_ids", np.int32)
        if c_ids is None:
            if n_seen == 0:
                print(f"chunk {i}: no C dump")
            break
        # [count, ids...]: an empty chunk still leaves a file behind.
        c_ids = c_ids[1:1 + int(c_ids[0])] if c_ids.size else c_ids
        n_seen += 1
        ref_ids = ch["ids"]
        same_ids = list(map(int, c_ids)) == ref_ids
        n_same_ids += same_ids
        if not same_ids and first_bad is None:
            first_bad = i
        tp = c_path(f"stream_{tag}_text")
        # vv_debug_dump writes nothing for an empty chunk text.
        c_text = (open(tp, "rb").read().decode("utf-8", "replace")
                  if os.path.exists(tp) else "")
        same_text = c_text == ch["text"]
        n_same_text += same_text
        if args.verbose or not same_ids or not same_text:
            print(f"chunk {i:4d} [{ch['start']},{ch['end']}) ids "
                  f"{'same' if same_ids else 'DIFFER'} "
                  f"(ref {len(ref_ids)}, c {len(c_ids)}) text "
                  f"{'same' if same_text else 'DIFFER'}")
            if not same_ids:
                n = min(len(ref_ids), len(c_ids))
                k = next((j for j in range(n) if ref_ids[j] != c_ids[j]), n)
                print(f"    first divergence at token {k}: "
                      f"ref={ref_ids[k:k + 5]} c={list(map(int, c_ids[k:k + 5]))}")
            if not same_text:
                print(f"    ref: {ch['text']!r}\n    c:   {c_text!r}")
        if args.verbose or not same_ids:
            rf = os.path.join(args.ref, f"chunk{i:04d}_feats.npy")
            _report(f"  feats {i}", np.load(rf) if os.path.exists(rf) else None,
                    load_c(f"stream_{tag}_feats"))
            rl = os.path.join(args.ref, f"chunk{i:04d}_logits.npy")
            ref_l = np.load(rl) if os.path.exists(rl) else None
            c_l = load_c(f"stream_{tag}_logits")
            _report(f"  logits {i}", ref_l, c_l)
            if ref_l is not None and c_l is not None and c_l.size == ref_l.size:
                top_r = np.argsort(-ref_l)[:5]
                top_c = np.argsort(-c_l)[:5]
                print(f"    top1 {'same' if top_r[0] == top_c[0] else 'DIFFER'} "
                      f"top5 overlap {len(set(top_r) & set(top_c))}/5")
        if first_bad is not None and not args.all:
            print("stopping at the first diverging chunk (--all to go on; "
                  "later chunks run on a different history)")
            break

    # The whole transcript, every chunk the C side wrote: a word that moves
    # to the neighbouring chunk diverges the ids but not the text.
    c_texts = []
    for ch in meta["chunks"]:
        tag = f"c{ch['index']:04d}"
        if not os.path.exists(c_path(f"stream_{tag}_ids")):
            break
        tp = c_path(f"stream_{tag}_text")
        c_texts.append(open(tp, "rb").read().decode("utf-8", "replace")
                       if os.path.exists(tp) else "")
    if len(c_texts) == len(meta["chunks"]):
        rw, cw = meta["transcript"].split(), "".join(c_texts).split()
        # word-level edit distance
        prev = list(range(len(cw) + 1))
        for a, x in enumerate(rw, 1):
            cur = [a] + [0] * len(cw)
            for b, y in enumerate(cw, 1):
                cur[b] = min(prev[b] + 1, cur[b - 1] + 1,
                             prev[b - 1] + (x != y))
            prev = cur
        d = prev[-1]
        same_tr = "".join(c_texts) == meta["transcript"]
        print(f"transcript       {'identical' if same_tr else 'DIFFER'} "
              f"(word edits {d}/{len(rw)}, WER {100.0 * d / max(len(rw), 1):.2f}%)")

    total = len(meta["chunks"])
    print(f"summary: ids identical {n_same_ids}/{n_seen}, text identical "
          f"{n_same_text}/{n_seen}, reference has {total} chunks"
          + (f", first divergence in chunk {first_bad}" if first_bad is not None
             else ""))
    return 0 if (first_bad is None and n_seen == total) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="write the PyTorch reference tensors")
    d.add_argument("--model", required=True, help="model directory")
    d.add_argument("--audio", required=True, help="input wav")
    d.add_argument("--out", default="./refdump")
    d.add_argument("--repo", default="./VibeVoice",
                   help="checkout of github.com/microsoft/VibeVoice")
    d.add_argument("--device", default="cuda:0")
    d.add_argument("--max-new-tokens", type=int, default=2048)
    d.set_defaults(func=cmd_dump)

    c = sub.add_parser("cmp", help="compare the C dump against the reference")
    c.add_argument("--ref", default="./refdump")
    c.add_argument("--c", default="./cdump")
    c.set_defaults(func=cmd_cmp)

    ds = sub.add_parser("dump-stream",
                        help="run the Streaming-7B reference chunk by chunk")
    ds.add_argument("--model", required=True, help="streaming model directory")
    ds.add_argument("--audio", required=True, help="input audio file")
    ds.add_argument("--out", default="./refstream")
    ds.add_argument("--repo", default="./VibeVoice",
                    help="checkout of github.com/microsoft/VibeVoice")
    ds.add_argument("--device", default="cuda:0")
    ds.add_argument("--max-new-tokens", type=int, default=256,
                    help="per chunk, as streaming_generate's default")
    ds.add_argument("--context-info", default=None, help="hotwords")
    ds.add_argument("--speech-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="encoder + connectors precision: fp32 is the clean "
                         "target the C encoder is held to, bf16 is exactly "
                         "what upstream runs")
    ds.add_argument("--start", type=float, default=0.0,
                    help="excerpt start in seconds (after resampling)")
    ds.add_argument("--duration", type=float, default=0.0,
                    help="excerpt length in seconds, 0 = to the end")
    ds.add_argument("--check-upstream", action="store_true",
                    help="also run upstream streaming_generate and diff texts")
    ds.set_defaults(func=cmd_dump_stream)

    cs = sub.add_parser("cmp-stream",
                        help="compare a C streaming dump chunk by chunk")
    cs.add_argument("--ref", default="./refstream")
    cs.add_argument("--c", default="./cdump")
    cs.add_argument("--all", action="store_true",
                    help="keep going after the first diverging chunk")
    cs.add_argument("-v", "--verbose", action="store_true",
                    help="print tensor stats for every chunk")
    cs.set_defaults(func=cmd_cmp_stream)

    args = ap.parse_args()
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
