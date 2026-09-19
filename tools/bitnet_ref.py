#!/usr/bin/env python3
"""
Fake-quant reference for VibeVoice-ASR-BitNet, reproducing the numerics of
microsoft/VibeASR.cpp (the reference runtime) from the F32 safetensors.

This is a tool, not runtime code: numpy + safetensors only. It exists so the
C runtime can be compared against the reference tensor by tensor without
building ggml, and so the reference's own approximations (int8 VAE with ReLU,
FP16 KV, ...) can be measured against the full-precision model.
docs/BITNET.md explains every step and cites the reference source.

Subcommands
-----------
  check-gguf  Ternarize every projection with the converter's formula and
              compare with the I2_S tensors in the GGUF; quantize every
              encoder weight with quantize_i8_s and compare with the I8_S
              tensors. Pure Python GGUF parsing, independent of the C reader.

  lm-layer    Run one Qwen2 decoder layer three ways on the same input:
                ref      W1.58A8 as VibeASR.cpp computes it (ternary weights,
                         per-token int8 activations, same FP32 epilogue
                         order, optional FP16 K/V/Q rounding like llama.cpp)
                w158a16  the same ternary weights with float activations
                f32      the latent F32 weights (not what the model was
                         trained to run: BitNet trains through the ternary
                         forward, so this one is expected to be far off)
              and report per-tensor cosine and max error. --dump writes every
              intermediate to an .npz for tools/compare against C dumps.

  vae         Run the acoustic or semantic encoder + connector on a WAV
              three ways: F32 with exact GELU (the HF model), the reference's
              int8 pipeline (I8_S weights, one scale per activation tensor,
              ReLU in the FFN), and the int8 pipeline with GELU. Reports the
              cosine of each against F32 at the connector output.

Examples
--------
  python tools/bitnet_ref.py check-gguf --model /path/VibeVoice-ASR-BitNet
  python tools/bitnet_ref.py lm-layer --model DIR --layer 0 --tokens 64
  python tools/bitnet_ref.py vae --model DIR --audio jfk.wav --seconds 4
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

F32 = np.float32


# ─── Loading ─────────────────────────────────────────────────────────────────

class Weights:
    """Lazy access to the sharded F32 safetensors, by name."""

    def __init__(self, model_dir):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.map = idx["weight_map"]
        self.dir = model_dir
        self.files = {}
        self._open = safe_open

    def __call__(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = self._open(os.path.join(self.dir, f), framework="np")
        return self.files[f].get_tensor(name).astype(F32, copy=False)


def read_gguf(path):
    """Minimal GGUF v3 parser: returns (kv dict, {name: (ne, type, bytes)})."""
    with open(path, "rb") as fh:
        data = fh.read()
    off = 0

    def rd(fmt):
        nonlocal off
        v = struct.unpack_from("<" + fmt, data, off)
        off += struct.calcsize(fmt)
        return v

    def rstr():
        (n,) = rd("Q")
        nonlocal off
        s = data[off:off + n].decode("utf-8", "replace")
        off += n
        return s

    sc = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t in sc:
            return rd(sc[t])[0]
        if t == 8:
            return rstr()
        if t == 9:
            (et,) = rd("I")
            (n,) = rd("Q")
            return [rval(et) for _ in range(n)]
        raise ValueError(t)

    magic, ver, nt, nkv = rd("IIQQ")
    assert magic == 0x46554747, "not a GGUF file"
    kv = {}
    for _ in range(nkv):
        k = rstr()
        (t,) = rd("I")
        kv[k] = rval(t) if not k.startswith("tokenizer.ggml.") else (rval(t) and None)
    infos = []
    for _ in range(nt):
        name = rstr()
        (nd,) = rd("I")
        ne = rd("Q" * nd)
        (ty,) = rd("I")
        (o,) = rd("Q")
        infos.append((name, ne, ty, o))
    align = kv.get("general.alignment", 32)
    base = (off + align - 1) // align * align
    tensors = {}
    for name, ne, ty, o in infos:
        n = int(np.prod(ne))
        if ty == 36:      # I2_S
            nb = n // 4 + 32
        elif ty == 37:    # I8_S
            nb = n + 32
        elif ty == 0:
            nb = 4 * n
        elif ty == 1:
            nb = 2 * n
        elif ty == 14:    # Q6_K
            nb = n // 256 * 210
        else:
            nb = 0
        tensors[name] = (ne, ty, memoryview(data)[base + o: base + o + nb])
    return kv, tensors


# ─── The reference's quantizers ──────────────────────────────────────────────

def ternarize(w, mean_mode="f64"):
    """convert_lm_to_gguf.py quant_weight_fp16 + ggml quantize_i2_s.

    Returns (t in {-1,0,1} int8 [N,K], scale float32).
    mean_mode: 'f64' (what the C runtime does) or 'f32' (numpy pairwise FP32,
    closer in spirit to torch's FP32 reduction)."""
    a = np.abs(w)
    mean = F32(a.mean(dtype=np.float64)) if mean_mode == "f64" else a.mean(dtype=F32)
    s = F32(1.0) / max(mean, F32(1e-5))
    r = np.clip(np.rint(w * s), -1, 1).astype(F32)       # torch.round: half to even
    wq = r / s
    t = np.where(np.abs(wq) < 1e-6, 0, np.where(wq > 0, 1, -1)).astype(np.int8)
    scale = F32(np.abs(wq).max())
    return t, scale


def pack_i2s(t):
    """[N,K] {-1,0,1} -> I2_S bytes (row-major, 128-element blocks)."""
    c = (t.reshape(-1, 4, 32) + 1).astype(np.uint8)       # [blocks, group, j]
    return (c[:, 0] << 6 | c[:, 1] << 4 | c[:, 2] << 2 | c[:, 3]).reshape(-1)


def unpack_i2s(b, n):
    b = np.frombuffer(bytes(b[: n // 4]), np.uint8).reshape(-1, 32)
    g = np.stack([(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3], axis=1)
    return (g.astype(np.int8) - 1).reshape(-1)


def quantize_i8s(w):
    """ggml quantize_i8_s: one scale per tensor, roundf (half away), +-127."""
    amax = F32(np.abs(w).max())
    s = F32(127.0) / amax if amax > 0 else F32(1.0)
    v = w * s
    q = np.clip(np.sign(v) * np.floor(np.abs(v) + F32(0.5)), -127, 127).astype(np.int8)
    return q, F32(F32(1.0) / s)


def act_quant(x):
    """ggml quantize_row_i8_s per row: s = 127/max(|x|,1e-5), rint, [-128,127].

    The reference build fuses x * s + 1.5 * 2^23 into one FMA, so the product
    is rounded to an integer exactly (FP64 holds it), not first to FP32."""
    amax = np.maximum(np.abs(x).max(axis=-1).astype(np.float64), 0.00001)
    s = (127.0 / amax).astype(F32)
    q = np.clip(np.rint(x.astype(np.float64) * s[:, None].astype(np.float64)),
                -128, 127).astype(np.int8)
    return q, s


def ternary_linear(xq, s, t, ws, bias=None):
    """y = (float)acc / s * ws (+ bias), in that order, FP32."""
    acc = xq.astype(np.int64) @ t.astype(np.int64).T
    assert np.abs(acc).max() < 2 ** 24
    y = (acc.astype(F32) / s[:, None]) * ws
    return y + bias if bias is not None else y


# ─── LM layer ────────────────────────────────────────────────────────────────

def rmsnorm_ggml(x, w, eps):
    ss = (x.astype(np.float64) ** 2).sum(-1)
    mean = (ss / x.shape[-1]).astype(F32)
    scale = F32(1.0) / np.sqrt(mean + F32(eps))
    return (x * scale[:, None]) * w


def rope_neox(x, pos, theta, hd):
    half = hd // 2
    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2 / hd)
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    c, s = np.cos(ang).astype(F32), np.sin(ang).astype(F32)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c[:, None] - x2 * s[:, None], x2 * c[:, None] + x1 * s[:, None]], -1)


def attention(q, k, v, nh, nkv, hd, f16):
    T = q.shape[0]
    q = q.reshape(T, nh, hd)
    k = k.reshape(T, nkv, hd)
    v = v.reshape(T, nkv, hd)
    if f16:  # llama.cpp: F16 cache, and q converted to F16 for the KQ product
        k, v, q = k.astype(np.float16).astype(F32), v.astype(np.float16).astype(F32), q.astype(np.float16).astype(F32)
    g = nh // nkv
    out = np.zeros((T, nh, hd), F32)
    mask = np.triu(np.full((T, T), -np.inf, F32), 1)
    for h in range(nh):
        sc = (q[:, h] @ k[:, h // g].T) * F32(1.0 / math.sqrt(hd)) + mask
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        p = sc / sc.sum(-1, keepdims=True)
        if f16:
            p = p.astype(np.float16).astype(F32)
        out[:, h] = p @ v[:, h // g]
    return out.reshape(T, nh * hd)


def lm_layer(W, L, x, cfg, mode, f16_attn):
    p = f"model.language_model.layers.{L}."
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // nh
    pos = np.arange(x.shape[0])
    d = {"in": x}

    def lin(name, h, bias=True):
        w = W(p + name + ".weight")
        b = W(p + name + ".bias") if bias else None
        if mode == "f32":
            y = h @ w.T
            return y + b if b is not None else y
        t, ws = ternarize(w)
        if mode == "w158a16":            # ternary weights, float activations
            y = h @ (t.astype(F32) * ws).T
            return y + b if b is not None else y
        xq, s = act_quant(h)
        return ternary_linear(xq, s, t, ws, b)

    h = rmsnorm_ggml(x, W(p + "input_layernorm.weight"), eps)
    d["attn_norm"] = h
    q = lin("self_attn.q_proj", h)
    k = lin("self_attn.k_proj", h)
    v = lin("self_attn.v_proj", h)
    d["q"], d["k"], d["v"] = q, k, v
    q = rope_neox(q.reshape(-1, nh, hd), pos, theta, hd).reshape(q.shape)
    k = rope_neox(k.reshape(-1, nkv, hd), pos, theta, hd).reshape(k.shape)
    a = attention(q, k, v, nh, nkv, hd, f16_attn and mode != "f32")
    d["attn"] = a
    o = lin("self_attn.o_proj", a, bias=False)
    x = x + o
    d["attn_out"] = x
    h = rmsnorm_ggml(x, W(p + "post_attention_layernorm.weight"), eps)
    g = lin("mlp.gate_proj", h, bias=False)
    u = lin("mlp.up_proj", h, bias=False)
    act = (g / (F32(1.0) + np.exp(-g))) * u
    d["ffn_act"] = act
    x = x + lin("mlp.down_proj", act, bias=False)
    d["out"] = x
    return d


# ─── VAE encoder ─────────────────────────────────────────────────────────────

def gelu(x):
    from math import sqrt
    import numpy as _np
    try:
        from scipy.special import erf
        return (0.5 * x * (1.0 + erf(x / sqrt(2.0)))).astype(F32)
    except ImportError:
        v = _np.vectorize(math.erf)
        return (0.5 * x * (1.0 + v(x / sqrt(2.0)))).astype(F32)


def conv1d(x, w, b, stride, pad_left, out_len, groups=1):
    """x [C, T], w [O, C/groups, k]; causal with explicit padding."""
    C, T = x.shape
    O, Cg, k = w.shape
    need = (out_len - 1) * stride + k
    xp = np.zeros((C, max(need, pad_left + T)), F32)
    xp[:, pad_left:pad_left + T] = x
    idx = np.arange(out_len)[:, None] * stride + np.arange(k)[None, :]
    cols = xp[:, idx]                                    # [C, out, k]
    if groups == 1:
        y = np.einsum("ctk,ock->ot", cols, w, optimize=True)
    else:                                                # depthwise
        y = np.einsum("ctk,ck->ct", cols, w[:, 0, :], optimize=True)
    return y + b[:, None]


class I8:
    """An I8_S activation: int8 values and the 127/amax multiplier."""

    def __init__(self, y, relu=False):
        amax = F32(np.abs(y).max())
        self.s = F32(127.0) / amax if amax != 0 else F32(0.0)
        lo = 0.0 if relu else -127.0
        self.q = np.rint(np.clip(y * self.s, lo, 127.0)).astype(np.int8)

    def f(self):
        return self.q.astype(F32) / self.s


def vae_encode(W, tower, audio, mode):
    """mode: f32 (HF), i8relu (VibeASR.cpp), i8gelu. audio at 24 kHz."""
    p = f"model.{tower}_tokenizer.encoder."
    ratios = [8, 5, 5, 4, 2, 2][::-1]
    depths = [3, 3, 3, 3, 3, 3, 8]
    q = mode != "f32"
    if q:  # the reference quantizes the input audio itself (roundf, clamp)
        amax = max(float(np.abs(audio).max()), 1e-5)
        s = F32(127.0 / amax)
        a8 = np.clip(np.sign(audio * s) * np.floor(np.abs(audio * s) + 0.5), -128, 127)
        x = (a8 / s).astype(F32)[None, :]
    else:
        x = audio[None, :].astype(F32)

    def wq(name):
        w = W(name)
        if not q:
            return w
        qq, sc = quantize_i8s(w)
        return qq.astype(F32) * sc

    def norm(x, w):
        # the reference's I8_S RMSNorm uses eps 1e-5 over channels (ggml_nn_rms_norm)
        ms = (x.astype(np.float64) ** 2).mean(0).astype(F32)
        return x / np.sqrt(ms + F32(1e-5)) * w[:, None]

    def rq(y, relu=False):
        return I8(y, relu).f() if q else (np.maximum(y, 0) if relu else y)

    for i in range(7):
        w = wq(p + f"downsample_layers.{i}.0.conv.conv.weight")
        b = W(p + f"downsample_layers.{i}.0.conv.conv.bias")
        k = w.shape[-1]
        st = 1 if i == 0 else ratios[i - 1]
        T = x.shape[1]
        out = -(-T // st) if not q else (T + (k - st) - k) // st + 1
        x = rq(conv1d(x, w, b, st, (k - 1) - (st - 1), out))
        for j in range(depths[i]):
            bp = p + f"stages.{i}.{j}."
            res = x
            h = rq(norm(x, W(bp + "norm.weight")))
            dw = wq(bp + "mixer.conv.conv.conv.weight")
            h = rq(conv1d(h, dw, W(bp + "mixer.conv.conv.conv.bias"), 1, dw.shape[-1] - 1, h.shape[1], groups=h.shape[0]))
            x = rq(res + W(bp + "gamma")[:, None] * h)
            res = x
            h = rq(norm(x, W(bp + "ffn_norm.weight")))
            h = wq(bp + "ffn.linear1.weight") @ h + W(bp + "ffn.linear1.bias")[:, None]
            h = rq(h, relu=True) if mode == "i8relu" else rq(gelu(h))
            h = rq(wq(bp + "ffn.linear2.weight") @ h + W(bp + "ffn.linear2.bias")[:, None])
            x = rq(res + W(bp + "ffn_gamma")[:, None] * h)
    w = wq(p + "head.conv.conv.weight")
    x = rq(conv1d(x, w, W(p + "head.conv.conv.bias"), 1, w.shape[-1] - 1, x.shape[1]))
    lat = x.T                                            # [frames, vae_dim]
    c = f"model.{tower}_connector."
    h = rq(lat @ wq(c + "fc1.weight").T + W(c + "fc1.bias"))
    eps = 1e-5 if q else 1e-6                            # the reference hardcodes 1e-5
    ms = (h.astype(np.float64) ** 2).mean(-1).astype(F32)
    h = rq(h / np.sqrt(ms + F32(eps))[:, None] * W(c + "norm.weight"))
    return lat, rq(h @ wq(c + "fc2.weight").T + W(c + "fc2.bias"))


def load_wav_24k(path, seconds):
    import wave
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    assert sw == 2, "16-bit PCM only"
    a = np.frombuffer(raw, np.int16).astype(F32).reshape(-1, ch).mean(1) / 32768.0
    if sr != 24000:  # linear, as VibeASR.cpp's audio_io does
        t = np.arange(int(len(a) * 24000 / sr)) * (sr / 24000.0)
        a = np.interp(t, np.arange(len(a)), a).astype(F32)
    a = a[: int(seconds * 24000)] if seconds else a
    rms = math.sqrt(float((a.astype(np.float64) ** 2).mean()))
    if rms > 1e-6:  # -25 dBFS
        a = (a * F32(10 ** (-25 / 20) / (rms + 1e-6))).astype(F32)
    return a


# ─── Commands ────────────────────────────────────────────────────────────────

def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def cmd_check_gguf(args):
    W = Weights(args.model)
    _, lm = read_gguf(os.path.join(args.model, "vibeasr-lm-i2_s-embed-q6_k.gguf"))
    names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
             "ffn_down": "mlp.down_proj"}
    layers = range(28) if args.all else [0, 27]
    for mean_mode in ("f64", "f32"):
        tot = diff = seq = 0
        for L in layers:
            for g, h in names.items():
                w = W(f"model.language_model.layers.{L}.{h}.weight")
                ne, ty, b = lm[f"blk.{L}.{g}.weight"]
                assert ty == 36
                t, sc = ternarize(w, mean_mode)
                gt = unpack_i2s(b, w.size)
                gs = np.frombuffer(bytes(b[w.size // 4: w.size // 4 + 4]), F32)[0]
                diff += int((gt != t.reshape(-1)).sum())
                tot += w.size
                seq += int(gs == sc)
        print(f"I2_S ({mean_mode} mean): {diff} of {tot} codes differ, {seq} of {len(layers) * 7} scales equal")
    _, vae = read_gguf(os.path.join(args.model, "vibeasr-vae-encoder-i8_s.gguf"))
    bad = n = 0
    for name, (ne, ty, b) in vae.items():
        if ty != 37:
            continue
        hf = ("model." + name.replace("acoustic.", "acoustic_tokenizer.encoder.", 1)
              .replace("semantic.", "semantic_tokenizer.encoder.", 1)) if "_connector" not in name else "model." + name
        w = W(hf)
        k = w.shape[-1]
        if w.ndim == 3 and ne[0] != k:           # kernel padded with zeros in front
            w = np.concatenate([np.zeros(w.shape[:-1] + (ne[0] - k,), F32), w], -1)
        qq, sc = quantize_i8s(w)
        gq = np.frombuffer(bytes(b[: w.size]), np.int8)
        gs = np.frombuffer(bytes(b[w.size: w.size + 4]), F32)[0]
        n += 1
        if not (np.array_equal(gq, qq.reshape(-1)) and gs == sc):
            bad += 1
            print("  I8_S mismatch:", name, int((gq != qq.reshape(-1)).sum()), sc, gs)
    print(f"I8_S: {n - bad} of {n} tensors identical")


def cmd_lm_layer(args):
    W = Weights(args.model)
    cfg = json.load(open(os.path.join(args.model, "config.json")))["decoder_config"]
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, 151643, args.tokens)
    emb = W("model.language_model.embed_tokens.weight")
    x = emb[ids]
    ref = lm_layer(W, args.layer, x, cfg, "ref", args.f16_attn)
    a16 = lm_layer(W, args.layer, x, cfg, "w158a16", False)
    f32 = lm_layer(W, args.layer, x, cfg, "f32", False)
    print(f"layer {args.layer}, {args.tokens} tokens: the reference's W1.58A8 against W1.58A16 "
          f"(same ternary weights, float activations) and against the latent F32 weights")
    for k in ref:
        print(f"  {k:10s} vs A16 cos {cos(ref[k], a16[k]):.6f} max|d| {np.abs(ref[k] - a16[k]).max():9.4g}"
              f"   vs F32 cos {cos(ref[k], f32[k]):.6f}")
    if args.dump:
        np.savez(args.dump, ids=ids, **{"ref_" + k: v for k, v in ref.items()},
                 **{"a16_" + k: v for k, v in a16.items()},
                 **{"f32_" + k: v for k, v in f32.items()})
        print("wrote", args.dump)


def cmd_vae(args):
    W = Weights(args.model)
    a = load_wav_24k(args.audio, args.seconds)
    out = {}
    for mode in ("f32", "i8relu", "i8gelu"):
        out[mode] = vae_encode(W, args.tower, a, mode)
    n = min(o[1].shape[0] for o in out.values())
    print(f"{args.tower} encoder on {len(a) / 24000:.2f} s: frames {[o[1].shape[0] for o in out.values()]}")
    for mode in ("i8relu", "i8gelu"):
        print(f"  {mode:7s} vs f32: latent cos {cos(out[mode][0][:n], out['f32'][0][:n]):.5f}, "
              f"connector cos {cos(out[mode][1][:n], out['f32'][1][:n]):.5f}")
    if args.dump:
        np.savez(args.dump, **{f"{m}_latent": o[0] for m, o in out.items()},
                 **{f"{m}_connector": o[1] for m, o in out.items()})
        print("wrote", args.dump)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-gguf")
    c.add_argument("--model", required=True)
    c.add_argument("--all", action="store_true", help="all 28 layers (default: first and last)")
    c = sub.add_parser("lm-layer")
    c.add_argument("--model", required=True)
    c.add_argument("--layer", type=int, default=0)
    c.add_argument("--tokens", type=int, default=64)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--f16-attn", action="store_true", help="round q/k/v and P to FP16 like llama.cpp")
    c.add_argument("--dump")
    c = sub.add_parser("vae")
    c.add_argument("--model", required=True)
    c.add_argument("--audio", required=True)
    c.add_argument("--tower", default="acoustic", choices=["acoustic", "semantic"])
    c.add_argument("--seconds", type=float, default=4.0)
    c.add_argument("--dump")
    args = ap.parse_args()
    {"check-gguf": cmd_check_gguf, "lm-layer": cmd_lm_layer, "vae": cmd_vae}[args.cmd](args)


if __name__ == "__main__":
    main()
