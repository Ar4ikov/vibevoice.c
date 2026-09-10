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
"""

import argparse
import os
import sys

import numpy as np

SEGMENT_SAMPLES = 60 * 24000

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
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa").eval()

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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
