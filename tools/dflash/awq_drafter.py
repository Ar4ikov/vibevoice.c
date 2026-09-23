#!/usr/bin/env python3
"""INT4 AWQ for a DFlash 2 drafter, stored the way compressed-tensors stores a
pack-quantized W4A16 checkpoint -- which is what the runtime's drafter loader
reads as it is (no quantization at load, `--draft-quant` has nothing to do).

    awq_drafter.py --target M --draft D --calib traces/train --eval traces/eval \\
        --out D-awq [--n 48] [--anchors 64]

Activation-aware scales where they fold into the drafter without changing
what it computes:

  post_attention_layernorm -> gate, up    (the MLP conv's kernel projection
                                          reads the same normed input and is
                                          compensated column for column)
  up -> down
  v -> o                                  (per KV channel, shared by the
                                          query heads of its group)

The attention input does not fold: the context's keys and values go through
the same k/v projections from hidden_norm, which every layer shares. Every
quantized projection -- q, k, v, o, gate, up, down, fc and the two kernel
projections per layer -- then gets AWQ's clip search, one ratio per (row,
group). Asymmetric INT4 in groups of 128, the runtime quantizer's rule: the
range holds 0, FP16 scale, integer zero point, codes against the FP16 scale.

Scales are searched on inputs captured from the drafter as it drafts over
--calib traces; the result is checked on --eval traces against the drafter
itself and against the runtime's own round-to-nearest INT4.
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

G = 128                     # the runtime's drafter group size (DRAFT_Q_GROUP)
RATIOS = [1.0 - 0.05 * i for i in range(10)]     # clip search: 1.0 .. 0.55


def load_train_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("train", os.path.join(here, "train.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T


def int4_codes(w, ratio=None):
    """[N, K] -> codes [N, K] (0..15), scale [N, K/G] (FP16 values, as
    float32), zero [N, K/G]; `ratio` [N, K/G] shrinks each group's range."""
    N, K = w.shape
    g = w.view(N, K // G, G)
    hi = g.amax(-1).clamp_min(0)
    lo = g.amin(-1).clamp_max(0)
    if ratio is not None:
        hi, lo = hi * ratio, lo * ratio
    s = ((hi - lo) / 15).half().float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    z = torch.floor(-lo * inv + 0.5).clamp(0, 15)
    q = torch.floor(g * inv[..., None] + z[..., None] + 0.5).clamp(0, 15)
    return q.view(N, K), s, z


def dequant(q, s, z):
    N, K = q.shape
    return ((q.view(N, K // G, G) - z[..., None]) * s[..., None]).view(N, K)


def clip_search(w, x):
    """AWQ's auto_clip: per (row, group), the range ratio whose quantized
    group best reproduces x_g . w_g over the captured inputs."""
    N, K = w.shape
    best_r = torch.ones(N, K // G, device=w.device)
    for gi in range(K // G):
        wg = w[:, gi * G:(gi + 1) * G]
        xg = x[:, gi * G:(gi + 1) * G]
        best = None
        for r in RATIOS:
            rr = torch.full((N, 1), r, device=w.device)
            q, s, z = int4_codes(wg, rr)
            err = ((xg @ (dequant(q, s, z) - wg).t()) ** 2).mean(0)
            if best is None:
                best = err
                best_r[:, gi] = r
            else:
                better = err < best
                best = torch.where(better, err, best)
                best_r[better, gi] = r
    return best_r


def scale_search(x, ws, grid=20):
    """AWQ's auto_scale: s = mean|x|^a, the a whose quantized W.diag(s) with
    x / s best reproduces x W^T, summed over the projections that share x."""
    xm = x.abs().mean(0)
    refs = [x @ w.t() for w in ws]
    best, best_s = None, None
    for i in range(grid + 1):
        a = i / grid
        s = xm.pow(a).clamp_min(1e-4)
        s = s / (s.max() * s.min()).sqrt()
        err = 0.0
        for w, y in zip(ws, refs):
            q, sc, z = int4_codes(w * s[None, :])
            err += (((x / s[None, :]) @ dequant(q, sc, z).t() - y) ** 2).mean().item()
        if best is None or err < best:
            best, best_s = err, s
    return best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--calib", required=True, help="trace dir to capture inputs on")
    ap.add_argument("--eval", required=True, help="trace dir to check the result on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48, help="calibration traces")
    ap.add_argument("--anchors", type=int, default=64, help="blocks per trace")
    ap.add_argument("--rows", type=int, default=4096, help="captured rows per projection")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    T = load_train_module()
    from safetensors.torch import load_file, save_file
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(a.draft, "config.json")))
    sd = load_file(os.path.join(a.draft, "model.safetensors"))
    vocab = sd.pop("draft_vocab", None)
    model = T.Drafter(cfg).to(dev)
    model.load_state_dict({k.replace(".self_attn.", ".").replace(".mlp.", "."): v.float()
                           for k, v in sd.items()})
    model.eval()
    emb, head = T.load_target(a.target, dev)
    sub = vocab.long().to(dev) if vocab is not None else None
    vocab_pair = None
    if sub is not None:
        head = head[sub].contiguous()
        full2sub = torch.full((cfg["vocab_size"],), -1, dtype=torch.long, device=dev)
        full2sub[sub] = torch.arange(len(sub), device=dev)
        vocab_pair = (sub, full2sub)
    dc = cfg["dflash_config"]
    B, mask_id = dc["block_size"], dc["mask_token_id"]

    # ── the projections that get quantized, by name ──────────────────────────
    lin = {"fc": model.fc}
    for i, l in enumerate(model.layers):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                  "down_proj"):
            lin[f"layers.{i}.{n}"] = getattr(l, n)
        lin[f"layers.{i}.attention_conv.kernel_projection"] = l.attention_conv.kernel_projection
        lin[f"layers.{i}.mlp_conv.kernel_projection"] = l.mlp_conv.kernel_projection

    # ── capture their inputs while drafting over the calibration traces ──────
    files = sorted(glob.glob(os.path.join(a.calib, "*.vvdt")))
    random.Random(a.seed).shuffle(files)
    files = files[:a.n]
    per_call = max(8, a.rows // (2 * len(files)))
    caps = {k: [] for k in lin}

    def hook(name):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            idx = torch.randperm(x.shape[0], device=x.device)[:per_call]
            caps[name].append(x[idx].float().cpu())
        return f

    handles = [m.register_forward_hook(hook(k)) for k, m in lin.items()]
    rng = np.random.default_rng(a.seed)
    with torch.no_grad():
        for f in files:
            t = T.read_vvdt(f)
            blk = T.make_blocks(t, B, a.anchors, rng)
            if blk is None:
                continue
            anchors = blk[0]
            S = int(anchors.max())
            feats = torch.from_numpy(t["feats"][:S]).to(dev).float()
            N = len(anchors)
            noise_ids = np.full((N, B), mask_id, np.int64)
            noise_ids[:, 0] = t["ids"][anchors]
            noise = torch.nn.functional.embedding(
                torch.from_numpy(noise_ids).to(dev).view(-1), emb).float()
            model(feats, noise, torch.from_numpy(anchors).to(dev), ckpt=False)
    for h in handles:
        h.remove()
    X = {k: torch.cat(v)[:a.rows].to(dev) for k, v in caps.items() if v}
    print(f"captured {len(files)} traces, {min(x.shape[0] for x in X.values())}.."
          f"{max(x.shape[0] for x in X.values())} rows per projection", flush=True)

    # ── the runtime's own INT4 (round to nearest), for comparison ───────────
    orig = {k: m.weight.detach().clone() for k, m in lin.items()}

    def set_weights(ws):
        with torch.no_grad():
            for k, m in lin.items():
                m.weight.copy_(ws[k])

    def evaluate(tag):
        efiles = sorted(glob.glob(os.path.join(a.eval, "*.vvdt")))
        erng = np.random.default_rng(1234)
        tot = {"base_acc": 0.0, "sel_acc": 0.0}
        nb = 0
        with torch.no_grad():
            for f in efiles:
                r = T.step_loss(model, emb, head, T.read_vvdt(f), dev, B, dc.get(
                    "loss_decay_gamma", 4.0), 256, erng, mask_id=mask_id, vocab=vocab_pair)
                if r is None:
                    continue
                st = r[3]
                n = st["n_blocks"]
                for k in tot:
                    tot[k] += float(st[k]) * n
                nb += n
        res = {k: v / max(nb, 1) for k, v in tot.items()}
        print(f"{tag:10s} base_acc {res['base_acc']:.3f} sel_acc {res['sel_acc']:.3f}",
              flush=True)
        return res

    base = evaluate("bf16")
    rtn = {k: dequant(*int4_codes(w)) for k, w in orig.items()}
    set_weights(rtn)
    evaluate("rtn int4")
    set_weights(orig)

    # ── activation-aware scales where they fold ─────────────────────────────
    nkv, nh, hd = cfg["num_key_value_heads"], cfg["num_attention_heads"], cfg["head_dim"]
    xin = dict(X)                      # each projection's input after the folds
    with torch.no_grad():
        for i, l in enumerate(model.layers):
            p = f"layers.{i}."
            # post_attention_layernorm -> gate, up (+ the MLP conv's kernels)
            s = scale_search(X[p + "gate_proj"], [l.gate_proj.weight, l.up_proj.weight])
            l.post_attention_layernorm.weight.div_(s)
            l.gate_proj.weight.mul_(s[None, :])
            l.up_proj.weight.mul_(s[None, :])
            l.mlp_conv.kernel_projection.weight.mul_(s[None, :])
            xin[p + "gate_proj"] = X[p + "gate_proj"] / s
            xin[p + "up_proj"] = X[p + "up_proj"] / s
            xin[p + "mlp_conv.kernel_projection"] = X[p + "mlp_conv.kernel_projection"] / s
            # up -> down
            s = scale_search(X[p + "down_proj"], [l.down_proj.weight])
            l.up_proj.weight.div_(s[:, None])
            l.down_proj.weight.mul_(s[None, :])
            xin[p + "down_proj"] = X[p + "down_proj"] / s
            # v -> o, one scale per KV channel for the heads that share it
            x = X[p + "o_proj"]
            xm = x.abs().mean(0).view(nkv, nh // nkv, hd).mean(1)          # [nkv, hd]
            wo = l.o_proj.weight
            ref = x @ wo.t()
            best, best_s = None, None
            for gi in range(21):
                sv = xm.pow(gi / 20).clamp_min(1e-4)
                sv = sv / (sv.max() * sv.min()).sqrt()
                so = sv[:, None, :].expand(nkv, nh // nkv, hd).reshape(-1)
                q, sc, z = int4_codes(wo * so[None, :])
                err = (((x / so[None, :]) @ dequant(q, sc, z).t() - ref) ** 2).mean().item()
                if best is None or err < best:
                    best, best_s = err, (sv.reshape(-1), so)
            sv, so = best_s
            l.v_proj.weight.div_(sv[:, None])
            l.o_proj.weight.mul_(so[None, :])
            xin[p + "o_proj"] = x / so
    scaled = {k: m.weight.detach().clone() for k, m in lin.items()}
    set_weights(scaled)
    evaluate("scaled fp")      # the same function, up to rounding

    # ── clip search and the codes ───────────────────────────────────────────
    codes = {}
    with torch.no_grad():
        for k, w in scaled.items():
            x = xin[k][:1024]
            r = clip_search(w, x)
            codes[k] = int4_codes(w, r)
    set_weights({k: dequant(*c) for k, c in codes.items()})
    res = evaluate("awq int4")

    # ── write it: compressed-tensors pack-quantized, the rest as it was ─────
    os.makedirs(a.out, exist_ok=True)
    out = {}
    if vocab is not None:
        out["draft_vocab"] = vocab.to(torch.int32).contiguous()

    def pub(name):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            name = name.replace(f".{n}", f".self_attn.{n}")
        for n in ("gate_proj", "up_proj", "down_proj"):
            name = name.replace(f".{n}", f".mlp.{n}")
        return name

    for k, (q, s, z) in codes.items():
        N, K = q.shape
        qi = q.to(torch.int64).view(N, K // 8, 8)
        packed = torch.zeros(N, K // 8, dtype=torch.int64, device=q.device)
        for j in range(8):
            packed |= qi[:, :, j] << (4 * j)
        zi = z.to(torch.int64).view(N // 8, 8, K // G)
        zp = torch.zeros(N // 8, K // G, dtype=torch.int64, device=q.device)
        for j in range(8):
            zp |= zi[:, j, :] << (4 * j)
        def i32(t):     # the unsigned words, bit for bit, as int32
            return torch.where(t >= 2 ** 31, t - 2 ** 32, t).to(torch.int32).cpu().contiguous()
        base_name = pub(k)
        out[base_name + ".weight_packed"] = i32(packed)
        out[base_name + ".weight_scale"] = s.half().cpu().contiguous()
        out[base_name + ".weight_zero_point"] = i32(zp)
        out[base_name + ".weight_shape"] = torch.tensor([N, K], dtype=torch.int64)
    for k, v in model.state_dict().items():
        mod_name = k[:-len(".weight")] if k.endswith(".weight") else None
        if mod_name in lin:
            continue
        # The layernorms the scales went into keep FP32: BF16 would round
        # away what the folding put there.
        folded = k.endswith("post_attention_layernorm.weight")
        out[pub(k)] = v.detach().to(torch.float32 if folded else torch.bfloat16).cpu().contiguous()
    save_file(out, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    qcfg = dict(cfg)
    qcfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": False,
                        "strategy": "group", "group_size": G, "dynamic": False,
                        "observer": "minmax"},
            "input_activations": None, "output_activations": None}},
        "ignore": ["candidate_selector.hidden_projection"],
        "kv_cache_scheme": None,
    }
    qcfg["vv_awq"] = {"calibration_traces": len(files), "blocks_per_trace": a.anchors,
                      "folds": ["post_attention_layernorm->gate,up",
                                "up->down", "v->o"],
                      "clip_ratios": RATIOS,
                      "eval": {"bf16": base, "awq_int4": res}}
    json.dump(qcfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
