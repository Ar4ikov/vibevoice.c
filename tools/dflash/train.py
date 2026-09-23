#!/usr/bin/env python3
"""Train a DFlash 2 drafter for a vibevoice.c target from its own traces.

The drafter is the DFlash 2 architecture (https://inco.ai/blog/dflash2/, the
reference inference code is `dflash/model.py` in pip `dflash`, the reference
trainer NeMo Automodel's `dflash2_core.py`): a few Qwen3-style layers whose
keys and values are extended with the target's hidden states at a handful of
layers, run once over a block of [last token, MASK x (B-1)] to propose B-1
tokens; a two-tap dynamic convolution around every sublayer, and a pairwise
selector that walks one path through each position's top-k candidates.

Training data are traces written by `vv_dflash_data trace` (the target's own
greedy transcripts replayed through the runtime, hidden states kept), read
from a directory -- a ring the trace server keeps full (--ring-dir, files are
deleted once read) or a static set (--eval-dir).

    train.py --target models/asr-awq --ring-dir /dev/shm/vvd \
             --eval-dir traces/eval --out drafters/asr-7b \
             --layers 1,7,13,19,25 --steps 2000

What the runtime needs is `model.safetensors` + `config.json` in --out; the
names follow the published DFlash 2 drafters (z-lab/Qwen3.8-27B-DFlash2).

The drafter proposes tokens from a subset of the vocabulary only: the
--draft-vocab ids the target generates most often in --vocab-from (the
transcripts the traces were made from), kept as `draft_vocab` in the
checkpoint. Its head is those rows of the target's LM head -- a fifth of
the bytes a draft pass reads for the full one -- and the target still
checks every token against the whole vocabulary.
"""
import argparse
import glob
import json
import math
import os
import queue
import random
import struct
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

MAGIC = 0x54445656
GEN, LABEL = 1, 2

# Eager flex_attention materializes the whole score matrix; compiled it runs
# the sparse block kernel. Shapes change with every trace, hence dynamic.
flex_attention_c = torch.compile(flex_attention, dynamic=True)


# ─── traces ─────────────────────────────────────────────────────────────────

def read_vvdt(path):
    b = open(path, "rb").read()
    magic, ver, n, nt, hs, idl = struct.unpack_from("<6I", b, 0)
    if magic != MAGIC or ver != 1:
        raise ValueError(f"{path}: not a v1 trace")
    off = 24
    layers = list(struct.unpack_from(f"<{nt}i", b, off))
    off += 4 * nt
    sid = b[off:off + idl].decode("utf-8", "replace")
    off += idl + (4 - idl % 4) % 4
    ids = np.frombuffer(b, np.int32, n, off).copy()
    off += 4 * n
    kind = np.frombuffer(b, np.uint8, n, off).copy()
    off += n + (4 - n % 4) % 4
    feats = np.frombuffer(b, np.float16, n * nt * hs, off).reshape(n, nt * hs)
    return {"id": sid, "layers": layers, "ids": ids, "kind": kind,
            "feats": feats, "hidden": hs}


def ring_reader(d, poll=0.05):
    """Files as the trace server writes them, oldest first, deleted once
    read; ends when the server leaves DONE and the ring is empty."""
    while True:
        files = sorted(f for f in glob.glob(os.path.join(d, "*.vvdt")))
        if not files:
            if os.path.exists(os.path.join(d, "DONE")):
                return
            time.sleep(poll)
            continue
        for f in files:
            try:
                t = read_vvdt(f)
            except (OSError, ValueError, struct.error):
                time.sleep(poll)
                continue
            os.remove(f)
            yield t


def static_reader(d, epochs=1, seed=0):
    files = sorted(glob.glob(os.path.join(d, "*.vvdt")))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(files)
        for f in files:
            yield read_vvdt(f)


def prefetch(gen, depth=4):
    """Run a trace reader in a thread, `depth` traces ahead: a trace is ~90
    MB, and reading one from a disk takes about as long as training on it."""
    q = queue.Queue(maxsize=depth)
    end = object()

    def run():
        try:
            for x in gen:
                q.put(x)
        finally:
            q.put(end)
    threading.Thread(target=run, daemon=True).start()
    while True:
        x = q.get()
        if x is end:
            return
        yield x


# ─── model ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rope(x, cos, sin):
    """NEOX rotary on [..., T, heads, d] with cos/sin [T, d]."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos[:, None, :] + rot * sin[:, None, :]).to(x.dtype)


def rope_tables(pos, d, theta):
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=pos.device,
                                        dtype=torch.float32) / d))
    f = pos.float()[:, None] * inv[None, :]
    emb = torch.cat((f, f), dim=-1)
    return emb.cos(), emb.sin()


def dyn_conv(h, dyn, base, B):
    """Two-tap grouped dynamic conv inside blocks of B rows.

    h [T, H], dyn [T, K, G], base [K, H]; row t reads t - o of its own block,
    zeros before the block start -- dflash2 `_grouped_dynamic_convolve`."""
    T, H = h.shape
    K, G = dyn.shape[1], dyn.shape[2]
    gs = H // G
    blocks = h.view(T // B, B, G, gs)
    dyn = dyn.view(T // B, B, K, G, 1).to(h.dtype)
    out = torch.zeros_like(blocks)
    for o in range(K):
        vals = blocks if o == 0 else F.pad(blocks[:, :-o], (0, 0, 0, 0, o, 0))
        out = out + base[o].view(1, 1, G, gs).to(h.dtype) * vals
        out = torch.addcmul(out, dyn[:, :, o], vals)
    return out.view(T, H)


class DynConv(nn.Module):
    def __init__(self, H, K, gs):
        super().__init__()
        self.K, self.gs = K, gs
        self.base_kernel = nn.Parameter(torch.zeros(2, K, H))
        self.kernel_projection = nn.Linear(H, 2 * K * (H // gs), bias=False)
        with torch.no_grad():
            self.base_kernel[:, 0, :] = 1.0
        nn.init.zeros_(self.kernel_projection.weight)

    def prepare(self, x, B):
        G = x.shape[-1] // self.gs
        dyn = self.kernel_projection(x).view(x.shape[0], 2, self.K, G)
        return dyn_conv(x, dyn[:, 0], self.base_kernel[0], B), dyn[:, 1]

    def finish(self, x, dyn, B):
        return dyn_conv(x, dyn, self.base_kernel[1], B)


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        H, nh, nkv, hd, I = (c["hidden_size"], c["num_attention_heads"],
                             c["num_key_value_heads"], c["head_dim"],
                             c["intermediate_size"])
        eps = c["rms_norm_eps"]
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.q_proj = nn.Linear(H, nh * hd, bias=False)
        self.k_proj = nn.Linear(H, nkv * hd, bias=False)
        self.v_proj = nn.Linear(H, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, H, bias=False)
        self.q_norm = RMSNorm(hd, eps)
        self.k_norm = RMSNorm(hd, eps)
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        dc = c["dflash_config"]
        self.attention_conv = DynConv(H, dc["conv_kernel_size"], dc["conv_group_size"])
        self.mlp_conv = DynConv(H, dc["conv_kernel_size"], dc["conv_group_size"])

    def ctx_kv(self, ctx, cos, sin):
        S = ctx.shape[0]
        k = self.k_norm(self.k_proj(ctx).view(S, self.nkv, self.hd))
        v = self.v_proj(ctx).view(S, self.nkv, self.hd)
        return rope(k, cos, sin), v

    def forward(self, h, ctx, cc, cs, bc, bs, mask, B):
        T = h.shape[0]
        r = h
        x = self.input_layernorm(h)
        x, d = self.attention_conv.prepare(x, B)
        q = rope(self.q_norm(self.q_proj(x).view(T, self.nh, self.hd)), bc, bs)
        kb = rope(self.k_norm(self.k_proj(x).view(T, self.nkv, self.hd)), bc, bs)
        vb = self.v_proj(x).view(T, self.nkv, self.hd)
        kc, vc = self.ctx_kv(ctx, cc, cs)
        k = torch.cat([kc, kb], 0)
        v = torch.cat([vc, vb], 0)
        o = flex_attention_c(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                             v.transpose(0, 1)[None], block_mask=mask,
                             scale=self.hd ** -0.5, enable_gqa=True)
        x = self.o_proj(o[0].transpose(0, 1).reshape(T, self.nh * self.hd))
        h = r + self.attention_conv.finish(x, d, B)
        r = h
        x = self.post_attention_layernorm(h)
        x, d = self.mlp_conv.prepare(x, B)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return r + self.mlp_conv.finish(x, d, B)


class Selector(nn.Module):
    def __init__(self, V, H, rank, top_k):
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.randn(V, rank) * 0.02)
        self.successor_codebook = nn.Parameter(torch.zeros(V, rank))
        self.hidden_projection = nn.Linear(H, rank, bias=False)

    def pair_scores(self, hidden, unary, cand, prev):
        gate = F.embedding(prev, self.predecessor_codebook) * \
            self.hidden_projection(hidden)
        pair = torch.einsum("tr,tkr->tk", gate.float(),
                            F.embedding(cand, self.successor_codebook).float())
        return unary.float() + pair


class Drafter(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        H = c["hidden_size"]
        dc = c["dflash_config"]
        self.B = dc["block_size"]
        self.layers = nn.ModuleList([Layer(c) for _ in range(c["num_hidden_layers"])])
        self.fc = nn.Linear(len(dc["target_layer_ids"]) * H, H, bias=False)
        self.hidden_norm = RMSNorm(H, c["rms_norm_eps"])
        self.norm = RMSNorm(H, c["rms_norm_eps"])
        self.candidate_selector = Selector(c["vocab_size"], H,
                                           dc["selector_rank"], dc["selector_top_k"])
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.abs().sum() != 0:
                nn.init.normal_(m.weight, std=0.02)
        for l in self.layers:
            nn.init.zeros_(l.attention_conv.kernel_projection.weight)
            nn.init.zeros_(l.mlp_conv.kernel_projection.weight)

    def forward(self, feats, noise, anchors, ckpt=True):
        """feats [S, taps*H] bf16, noise [N*B, H], anchors [N] -> [N*B, H]."""
        B = self.B
        S = feats.shape[0]
        N = anchors.shape[0]
        dev = feats.device
        ctx = self.hidden_norm(self.fc(feats))
        hd = self.c["head_dim"]
        theta = self.c["rope_theta"]
        cc, cs = rope_tables(torch.arange(S, device=dev), hd, theta)
        bpos = (anchors[:, None] + torch.arange(B, device=dev)[None, :]).reshape(-1)
        bc, bs = rope_tables(bpos, hd, theta)
        anc = anchors.to(torch.int32)

        def mask_mod(b, h, q, kv):
            qb = q // B
            ctx_ok = (kv < S) & (kv < anc[qb])
            blk_ok = (kv >= S) & ((kv - S) // B == qb)
            return ctx_ok | blk_ok

        mask = create_block_mask(mask_mod, None, None, N * B, S + N * B,
                                 device=dev, _compile=True)
        h = noise
        for l in self.layers:
            if ckpt and self.training:
                h = checkpoint(l, h, ctx, cc, cs, bc, bs, mask, B,
                               use_reentrant=False)
            else:
                h = l(h, ctx, cc, cs, bc, bs, mask, B)
        return self.norm(h)


# ─── target pieces ──────────────────────────────────────────────────────────

def load_target(tdir, device):
    """Embedding table and LM head of the target, frozen, BF16."""
    from safetensors import safe_open
    idx_path = os.path.join(tdir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
    else:
        wm = {k: "model.safetensors" for k in
              safe_open(os.path.join(tdir, "model.safetensors"), "pt").keys()}
    emb_k = next(k for k in wm if k.endswith("embed_tokens.weight"))
    head_k = next((k for k in wm if k == "lm_head.weight"), None)

    def get(k):
        with safe_open(os.path.join(tdir, wm[k]), "pt") as f:
            return f.get_tensor(k)
    emb = get(emb_k).to(device=device, dtype=torch.bfloat16)
    head = get(head_k).to(device=device, dtype=torch.bfloat16) if head_k else emb
    return emb, head


def target_config(tdir):
    c = json.load(open(os.path.join(tdir, "config.json")))
    return c.get("decoder_config") or c.get("text_config") or c


def build_target_layer_ids(n_target, n_draft):
    if n_draft == 1:
        return [n_target // 2]
    start, end = 1, n_target - 3
    return [int(round(start + i * (end - start) / (n_draft - 1)))
            for i in range(n_draft)]


# ─── batches and loss ───────────────────────────────────────────────────────

def make_blocks(t, B, max_anchors, rng):
    ids, kind = t["ids"], t["kind"]
    n = len(ids)
    elig = np.nonzero((kind == GEN) & (np.arange(n) + 1 < n))[0]
    if len(elig) == 0:
        return None
    if len(elig) > max_anchors:
        elig = np.sort(rng.choice(elig, max_anchors, replace=False))
    offs = np.arange(B)
    lab_idx = elig[:, None] + offs[None, :]                 # [N, B]
    inb = lab_idx < n
    li = np.minimum(lab_idx, n - 1)
    labels = ids[li]
    ok = inb & ((kind[li] == GEN) | (kind[li] == LABEL))
    ok[:, 0] = True
    ok = np.cumprod(ok, axis=1).astype(bool)                 # nothing past a gap
    ok[:, 0] = False
    return elig, labels, ok


def draft_vocab(path, n, V):
    """The n ids generated most often in a gen.jsonl, ascending, or None.

    The ids that stopped a chunk or a transcript (`stops`) are always in: a
    transcript's `tokens` leave them out, but a streaming chunk ends on one
    every ~10 tokens, and a drafter that cannot propose it loses every
    block that reaches a chunk's end (and is never trained on those labels,
    which fall outside its head)."""
    if n <= 0:
        return None
    counts = np.zeros(V, np.int64)
    stops = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)
        ids = np.asarray(j["tokens"], np.int64)
        ids = ids[(ids >= 0) & (ids < V)]
        np.add.at(counts, ids, 1)
        for sid in j.get("stops", []):
            if 0 <= sid < V:
                counts[sid] += 1
                stops.add(int(sid))
    order = np.argsort(-counts, kind="stable")
    keep = [int(i) for i in order[:n] if counts[i] > 0]
    missing = [sid for sid in sorted(stops) if sid not in set(keep)]
    if missing:
        keep = keep[:n - len(missing)] + missing
    return np.sort(np.asarray(keep, np.int64))


def step_loss(model, emb, head, t, dev, B, gamma, max_anchors, rng,
              chunk=768, mask_id=151662, vocab=None):
    """vocab: (sub ids [Vd] long tensor, full->sub map [V] long tensor) when
    the head is the subset `head`; labels outside it cannot be drafted."""
    blk = make_blocks(t, B, max_anchors, rng)
    if blk is None:
        return None
    anchors, labels, ok = blk
    S = int(anchors.max())                 # context a block can see: < anchor
    feats = torch.from_numpy(t["feats"][:S]).to(dev, non_blocking=True)
    feats = feats.to(torch.bfloat16)
    N = len(anchors)
    noise_ids = np.full((N, B), mask_id, np.int64)
    noise_ids[:, 0] = t["ids"][anchors]
    noise = F.embedding(torch.from_numpy(noise_ids).to(dev).view(-1), emb)
    anc = torch.from_numpy(anchors).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(feats, noise, anc)                        # [N*B, H]
    hid = out.view(N, B, -1)[:, 1:].reshape(N * (B - 1), -1)
    lab = torch.from_numpy(labels[:, 1:].reshape(-1)).to(dev).long()
    okm = torch.from_numpy(ok[:, 1:].reshape(-1)).to(dev)
    if vocab is not None:
        sub_ids, full2sub = vocab
        lab_h = full2sub[lab]                  # head row of the label, -1: none
    else:
        sub_ids, lab_h = None, lab
    depth = torch.exp(-torch.arange(B - 1, device=dev, dtype=torch.float32)
                      / gamma).repeat(N)
    w = okm.float() * depth * (lab_h >= 0).float()
    prev = torch.from_numpy(labels[:, :-1].reshape(-1)).to(dev).long()
    sel = model.candidate_selector

    def head_chunk(h, y):
        logits = F.linear(h.to(torch.bfloat16), head).float()
        nll = torch.logsumexp(logits, -1) - \
            logits.gather(1, y.clamp_min(0)[:, None])[:, 0]
        unary, cand = logits.topk(sel.top_k, dim=-1)
        best = logits.argmax(-1)
        if sub_ids is not None:
            cand, best = sub_ids[cand], sub_ids[best]
        return nll, unary, cand, best

    nlls, unaries, cands, base_ids = [], [], [], []
    for a in range(0, hid.shape[0], chunk):
        r = checkpoint(head_chunk, hid[a:a + chunk], lab_h[a:a + chunk],
                       use_reentrant=False)
        nlls.append(r[0]); unaries.append(r[1]); cands.append(r[2])
        base_ids.append(r[3])
    nll = torch.cat(nlls)
    unary = torch.cat(unaries)
    cand = torch.cat(cands)
    base_ids = torch.cat(base_ids)
    wsum = w.sum().clamp_min(1e-6)
    base_loss = (nll * w).sum() / wsum

    scores = sel.pair_scores(hid, unary, cand, prev)
    in_c = cand == lab[:, None]
    has = in_c.any(-1)
    tgt = in_c.to(torch.int64).argmax(-1)
    sw = w * has.float()
    sel_nll = F.cross_entropy(scores, tgt, reduction="none")
    sel_loss = (sel_nll * sw).sum() / sw.sum().clamp_min(1e-6)

    with torch.no_grad():
        picked = cand.gather(1, scores.argmax(-1, keepdim=True))[:, 0]
        m = okm.view(N, B - 1)

        def acc_len(pred):
            good = ((pred.view(N, B - 1) == lab.view(N, B - 1)) | ~m).long()
            return ((good.cumprod(1) * m.long()).sum(1).float() + 1.0).mean()

        # Per depth: the pick right, given every one before it was.
        good = (picked.view(N, B - 1) == lab.view(N, B - 1)) & m
        prior = torch.ones(N, dtype=torch.bool, device=dev)
        hit, seen = [], []
        for k in range(B - 1):
            live = prior & m[:, k]
            seen.append(live.float().sum())
            hit.append((live & good[:, k]).float().sum())
            prior = prior & good[:, k]
        stats = {"n_blocks": N, "base_acc": acc_len(base_ids),
                 "sel_acc": acc_len(picked),
                 "recall": (has & okm).float().sum() / okm.float().sum().clamp_min(1),
                 "depth_hit": torch.stack(hit), "depth_seen": torch.stack(seen)}
    return base_loss + sel_loss, base_loss.detach(), sel_loss.detach(), stats


# ─── export ─────────────────────────────────────────────────────────────────

def export(model, cfg, out_dir, vocab_ids=None):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    if vocab_ids is not None:
        sd["draft_vocab"] = torch.as_tensor(vocab_ids, dtype=torch.int32).contiguous()
    for k, v in model.state_dict().items():
        name = k
        for a in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            name = name.replace(f".{a}.", f".self_attn.{a}.")
        for a in ("gate_proj", "up_proj", "down_proj"):
            name = name.replace(f".{a}.", f".mlp.{a}.")
        sd[name] = v.detach().to(torch.bfloat16).contiguous().cpu()
    save_file(sd, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--ring-dir")
    ap.add_argument("--train-dir", help="static trace set instead of a ring")
    ap.add_argument("--eval-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--draft-layers", type=int, default=5)
    ap.add_argument("--intermediate", type=int, default=0)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--mask-id", type=int, default=151662)
    ap.add_argument("--draft-vocab", type=int, default=32768,
                    help="ids the drafter can propose, the most frequent in "
                         "--vocab-from; 0: the whole vocabulary")
    ap.add_argument("--vocab-from", default="",
                    help="gen.jsonl whose tokens rank the vocabulary")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--accum-blocks", type=int, default=4096,
                    help="draft blocks per optimizer step")
    ap.add_argument("--max-anchors", type=int, default=1024,
                    help="blocks drawn from one trace per visit")
    ap.add_argument("--warmup", type=float, default=0.04)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1, help="--train-dir only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    tc = target_config(a.target)
    H = tc["hidden_size"]
    layers = ([int(x) for x in a.layers.split(",")] if a.layers else
              build_target_layer_ids(tc["num_hidden_layers"], a.draft_layers))
    cfg = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "hidden_act": "silu",
        "hidden_size": H,
        "intermediate_size": a.intermediate or tc["intermediate_size"] // 2,
        "num_hidden_layers": a.draft_layers,
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": H // tc["num_attention_heads"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "rope_theta": tc["rope_theta"],
        "max_position_embeddings": tc.get("max_position_embeddings", 131072),
        "vocab_size": tc["vocab_size"],
        "num_target_layers": tc["num_hidden_layers"],
        "is_causal": False,
        "layer_types": ["full_attention"] * a.draft_layers,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "dflash_config": {
            "block_size": a.block,
            "mask_token_id": a.mask_id,
            "target_layer_ids": layers,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": a.rank,
            "selector_top_k": a.top_k,
            "loss_decay_gamma": a.gamma,
        },
    }
    vocab_ids = None
    sd = None
    if a.resume:
        from safetensors.torch import load_file
        sd = load_file(a.resume)
    if sd is not None and "draft_vocab" in sd:
        # A resumed drafter keeps the vocabulary its head was trained on.
        vocab_ids = sd.pop("draft_vocab").numpy().astype(np.int64)
        cfg["dflash_config"]["draft_vocab_size"] = int(len(vocab_ids))
    elif a.draft_vocab > 0:
        if not a.vocab_from:
            raise SystemExit("--draft-vocab needs --vocab-from (a gen.jsonl)")
        vocab_ids = draft_vocab(a.vocab_from, a.draft_vocab, tc["vocab_size"])
        cfg["dflash_config"]["draft_vocab_size"] = int(len(vocab_ids))
    model = Drafter(cfg).to(dev)
    if sd is not None:
        own = {}
        for k, v in sd.items():
            k = k.replace(".self_attn.", ".").replace(".mlp.", ".")
            own[k] = v.float()
        model.load_state_dict(own)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"drafter: {n_par / 1e6:.0f}M params, taps {layers}, "
          f"hidden {H}, inter {cfg['intermediate_size']}", flush=True)
    emb, head = load_target(a.target, dev)
    vocab = None
    if vocab_ids is not None:
        sub = torch.from_numpy(vocab_ids).to(dev)
        full2sub = torch.full((tc["vocab_size"],), -1, dtype=torch.long, device=dev)
        full2sub[sub] = torch.arange(len(vocab_ids), device=dev)
        head = head[sub].contiguous()
        vocab = (sub, full2sub)
        print(f"draft vocabulary: {len(vocab_ids)} ids", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)
    warm = max(1, int(a.warmup * a.steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, a.steps - warm)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    if a.ring_dir:
        stream = prefetch(ring_reader(a.ring_dir))
    else:
        stream = prefetch(static_reader(a.train_dir, a.epochs, a.seed))
    evals = None
    if a.eval_dir:
        evals = [read_vvdt(f) for f in sorted(glob.glob(os.path.join(a.eval_dir, "*.vvdt")))]
        print(f"eval: {len(evals)} traces", flush=True)

    def evaluate():
        model.eval()
        tot = {"base_acc": 0.0, "sel_acc": 0.0, "recall": 0.0}
        nb, bl = 0, 0.0
        dh = torch.zeros(a.block - 1, device=dev)
        ds = torch.zeros(a.block - 1, device=dev)
        erng = np.random.default_rng(1234)
        with torch.no_grad():
            for t in evals:
                r = step_loss(model, emb, head, t, dev, a.block, a.gamma,
                              256, erng, mask_id=a.mask_id, vocab=vocab)
                if r is None:
                    continue
                _, b, s, st = r
                n = st["n_blocks"]
                for k in tot:
                    tot[k] += float(st[k]) * n
                bl += float(b) * n
                nb += n
                dh += st["depth_hit"]
                ds += st["depth_seen"]
        model.train()
        depth = " ".join(f"{float(h / max(float(c), 1.0)):.2f}" for h, c in zip(dh, ds))
        print(f"EVAL depth acc (given all before right): {depth}", flush=True)
        return {k: v / max(nb, 1) for k, v in tot.items()} | {"base_loss": bl / max(nb, 1)}

    model.train()
    best = -1.0
    step, blocks, t0 = 0, 0, time.time()
    run = {"loss": 0.0, "base": 0.0, "sel": 0.0, "base_acc": 0.0,
           "sel_acc": 0.0, "recall": 0.0, "n": 0}
    for g in opt.param_groups:
        g["lr"] = lr_at(0)
    while step < a.steps:
        try:
            t = next(stream)
        except StopIteration:
            print("data ended", flush=True)
            break
        r = step_loss(model, emb, head, t, dev, a.block, a.gamma,
                      a.max_anchors, rng, mask_id=a.mask_id, vocab=vocab)
        if r is None:
            continue
        loss, bl, sl, st = r
        n = st["n_blocks"]
        (loss * (n / a.accum_blocks)).backward()
        blocks += n
        run["loss"] += float(loss) * n
        run["base"] += float(bl) * n
        run["sel"] += float(sl) * n
        for k in ("base_acc", "sel_acc", "recall"):
            run[k] += float(st[k]) * n
        run["n"] += n
        if blocks < a.accum_blocks:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        blocks = 0
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if step % 10 == 0:
            k = max(run["n"], 1)
            el = time.time() - t0
            print(f"step {step} loss {run['loss']/k:.3f} base {run['base']/k:.3f} "
                  f"sel {run['sel']/k:.3f} acc_base {run['base_acc']/k:.2f} "
                  f"acc_sel {run['sel_acc']/k:.2f} recall {run['recall']/k:.3f} "
                  f"lr {lr_at(step):.2e} {el/step:.1f}s/step", flush=True)
            run = {k2: 0.0 for k2 in run}
            run["n"] = 0
        if evals and step % a.eval_every == 0:
            e = evaluate()
            print(f"EVAL step {step} " + " ".join(f"{k} {v:.3f}" for k, v in e.items()),
                  flush=True)
            # --out keeps the best drafter the held-out traces have seen; on
            # a small corpus the last one is often past it.
            if e["sel_acc"] > best:
                best = e["sel_acc"]
                export(model, cfg, a.out, vocab_ids)
                print(f"BEST step {step} sel_acc {best:.3f}", flush=True)
        if not evals and (step % a.save_every == 0 or step == a.steps):
            export(model, cfg, a.out, vocab_ids)
    if evals:
        e = evaluate()
        print("EVAL final " + " ".join(f"{k} {v:.3f}" for k, v in e.items()), flush=True)
        if e["sel_acc"] > best:
            best = e["sel_acc"]
            export(model, cfg, a.out, vocab_ids)
            print(f"BEST final sel_acc {best:.3f}", flush=True)
        export(model, cfg, a.out + "-last", vocab_ids)
    else:
        export(model, cfg, a.out, vocab_ids)


if __name__ == "__main__":
    main()
