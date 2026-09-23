#!/usr/bin/env python3
"""Where a drafter is right: depth-1 accuracy by kind of target token.

    eval_types.py --target M --draft D --eval-dir traces/eval [--anchors 256]

Kinds: JSON structure (punctuation and the key names), digits, whitespace,
words. A drafter that misses structure is undertrained or broken; one that
only misses digits and words is short of data or capacity.
"""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--anchors", type=int, default=256)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("train", os.path.join(here, "train.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(a.draft, "config.json")))
    model = T.Drafter(cfg).to(dev)
    sd = load_file(os.path.join(a.draft, "model.safetensors"))
    vocab = sd.pop("draft_vocab", None)
    model.load_state_dict({k.replace(".self_attn.", ".").replace(".mlp.", "."): v.float()
                           for k, v in sd.items()})
    model.eval()
    emb, head = T.load_target(a.target, dev)
    sub = vocab.long().to(dev) if vocab is not None else None
    if sub is not None:
        head = head[sub].contiguous()
    tok = Tokenizer.from_file(os.path.join(a.target, "tokenizer.json"))
    keys = {"Start", "End", "Speaker", "Content"}

    def kind(i):
        s = tok.decode([int(i)])
        st = s.strip()
        if not st:
            return "space"
        if st in keys or all(c in '[]{}":,' for c in st):
            return "struct"
        if all(c.isdigit() or c == "." for c in st):
            return "digit"
        return "word"

    dc = cfg["dflash_config"]
    B, mask_id = dc["block_size"], dc["mask_token_id"]
    rng = np.random.default_rng(0)
    hit, seen = {}, {}
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(a.eval_dir, "*.vvdt"))):
            t = T.read_vvdt(f)
            blk = T.make_blocks(t, B, a.anchors, rng)
            if blk is None:
                continue
            anchors, labels, ok = blk
            S = int(anchors.max())
            feats = torch.from_numpy(t["feats"][:S]).to(dev).to(torch.bfloat16)
            N = len(anchors)
            noise_ids = np.full((N, B), mask_id, np.int64)
            noise_ids[:, 0] = t["ids"][anchors]
            noise = F.embedding(torch.from_numpy(noise_ids).to(dev).view(-1), emb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(feats, noise, torch.from_numpy(anchors).to(dev), ckpt=False)
            h1 = out.view(N, B, -1)[:, 1].float()
            logits = F.linear(h1.to(torch.bfloat16), head).float()
            pred = logits.argmax(-1)
            if sub is not None:
                pred = sub[pred]
            pred = pred.cpu().numpy()
            for n in range(N):
                if not ok[n, 1]:
                    continue
                k = kind(labels[n, 1])
                seen[k] = seen.get(k, 0) + 1
                hit[k] = hit.get(k, 0) + int(pred[n] == labels[n, 1])
    tot = sum(seen.values())
    for k in sorted(seen):
        print(f"{k:7s} {seen[k] / tot:6.1%} of labels, depth-1 accuracy {hit[k] / seen[k]:.3f}")
    print(f"all     depth-1 accuracy {sum(hit.values()) / tot:.3f}")


if __name__ == "__main__":
    main()
