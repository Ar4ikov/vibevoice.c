#!/usr/bin/env python3
"""Check the runtime's drafts against the trainer's model on the same context.

    VV_SPEC_LOG=drafts.log vv_cli --model M --audio clip.wav --draft D ...
    check_drafter.py --target M --draft D --trace clip.vvdt --log drafts.log

The runtime logs, per cycle, the cache length p, the anchor and the drafted
block. For each logged cycle this runs the drafter as trained (train.py's
model, BF16) over the trace's first p positions and walks the selector the
way the runtime does, then reports how often the two agree per block
position. Agreement well below ~95% means the runtime computes something
other than what was trained.
"""
import argparse
import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F


def load_train_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("train", os.path.join(here, "train.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-cycles", type=int, default=200)
    a = ap.parse_args()

    T = load_train_module()
    dev = torch.device("cuda")
    import json
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(a.draft, "config.json")))
    model = T.Drafter(cfg).to(dev)
    sd = load_file(os.path.join(a.draft, "model.safetensors"))
    vocab = sd.pop("draft_vocab", None)
    own = {k.replace(".self_attn.", ".").replace(".mlp.", "."): v.float()
           for k, v in sd.items()}
    model.load_state_dict(own)
    model.eval()
    emb, head = T.load_target(a.target, dev)
    sub = None
    if vocab is not None:
        sub = vocab.long().to(dev)
        head = head[sub].contiguous()
    t = T.read_vvdt(a.trace)
    ids = t["ids"]
    dc = cfg["dflash_config"]
    B, mask_id = dc["block_size"], dc["mask_token_id"]
    sel = model.candidate_selector

    rows = [list(map(int, l.split())) for l in open(a.log) if l.strip()]
    rows = rows[: a.max_cycles]
    agree = np.zeros(B - 1)
    bad_anchor = 0
    with torch.no_grad():
        for r in rows:
            p, anchor, drafts = r[0], r[1], r[2:]
            if p >= len(ids) or ids[p] != anchor:
                bad_anchor += 1
                continue
            feats = torch.from_numpy(t["feats"][:p]).to(dev).to(torch.bfloat16)
            noise_ids = torch.full((B,), mask_id, dtype=torch.long, device=dev)
            noise_ids[0] = anchor
            noise = F.embedding(noise_ids, emb)
            anc = torch.tensor([p], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(feats, noise, anc, ckpt=False)
            hid = out[1:].float()
            logits = F.linear(hid.to(torch.bfloat16), head).float()
            unary, cand = logits.topk(sel.top_k, dim=-1)
            if sub is not None:
                cand = sub[cand]
            hp = sel.hidden_projection(hid.to(sel.hidden_projection.weight.dtype)).float()
            prev = anchor
            walk = []
            for i in range(B - 1):
                gate = sel.predecessor_codebook[prev].float() * hp[i]
                pair = (sel.successor_codebook[cand[i]].float() * gate[None, :]).sum(-1)
                k = int((unary[i] + pair).argmax())
                prev = int(cand[i, k])
                walk.append(prev)
            for i in range(B - 1):
                agree[i] += walk[i] == drafts[i]
    n = len(rows) - bad_anchor
    print(f"cycles {n} (anchor mismatches {bad_anchor})")
    print("agreement per position:", " ".join(f"{x / max(n, 1):.3f}" for x in agree))


if __name__ == "__main__":
    main()
