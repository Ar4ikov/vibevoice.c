#!/usr/bin/env python3
"""Re-quantize an NF4 checkpoint into the AWQ container format.

The runtime can load any AutoAWQ or GPTQ checkpoint (qweight / qzeros /
scales). This tool produces one from the bitsandbytes NF4 weights so the INT4
path can be exercised without a second download, and so a machine with no
AWQ build of the model can still use the faster kernels.

What it does NOT do is AWQ's activation-aware scale search, which needs
calibration data and a forward pass per layer. The output is therefore plain
group-wise INT4 in the AWQ layout — slightly worse than a real AutoAWQ
checkpoint of the same model, and strictly better than nothing. If you have a
proper AWQ checkpoint, use it; the runtime treats both identically.

Usage:
    python tools/convert_awq.py --src ./model_hf --dst ./model_awq [--group 128]
"""

import argparse
import json
import os
import shutil
import struct

import numpy as np

# bitsandbytes NF4 code book.
NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float32)

# bitsandbytes' default code book for the second-level (block scale)
# quantization. Checkpoints only carry nested_quant_map when it differs.
BNB_CODE = np.array([
    -9.9296875e-01, -9.7890625e-01, -9.6484375e-01, -9.5078125e-01, -9.3671875e-01, -9.2265625e-01,
    -9.0859375e-01, -8.9453125e-01, -8.8046875e-01, -8.6640625e-01, -8.5234375e-01, -8.3828125e-01,
    -8.2421875e-01, -8.1015625e-01, -7.9609375e-01, -7.8203125e-01, -7.6796875e-01, -7.5390625e-01,
    -7.3984375e-01, -7.2578125e-01, -7.1171875e-01, -6.9765625e-01, -6.8359375e-01, -6.6953125e-01,
    -6.5546875e-01, -6.4140625e-01, -6.2734375e-01, -6.1328125e-01, -5.9921875e-01, -5.8515625e-01,
    -5.7109375e-01, -5.5703125e-01, -5.4296875e-01, -5.2890625e-01, -5.1484375e-01, -5.0078125e-01,
    -4.8671875e-01, -4.7265625e-01, -4.5859375e-01, -4.4453125e-01, -4.3046875e-01, -4.1640625e-01,
    -4.0234375e-01, -3.8828125e-01, -3.7421875e-01, -3.6015625e-01, -3.4609375e-01, -3.3203125e-01,
    -3.1796875e-01, -3.0390625e-01, -2.8984375e-01, -2.7578125e-01, -2.6171875e-01, -2.4765625e-01,
    -2.3359375e-01, -2.1953125e-01, -2.0546875e-01, -1.9140625e-01, -1.7734375e-01, -1.6328125e-01,
    -1.4921875e-01, -1.3515625e-01, -1.2109375e-01, -1.0703125e-01, -9.8593750e-02, -9.5781250e-02,
    -9.2968750e-02, -9.0156250e-02, -8.7343750e-02, -8.4531250e-02, -8.1718750e-02, -7.8906250e-02,
    -7.6093750e-02, -7.3281250e-02, -7.0468750e-02, -6.7656250e-02, -6.4843750e-02, -6.2031250e-02,
    -5.9218750e-02, -5.6406250e-02, -5.3593750e-02, -5.0781250e-02, -4.7968750e-02, -4.5156250e-02,
    -4.2343750e-02, -3.9531250e-02, -3.6718750e-02, -3.3906250e-02, -3.1093750e-02, -2.8281250e-02,
    -2.5468750e-02, -2.2656250e-02, -1.9843750e-02, -1.7031250e-02, -1.4218750e-02, -1.1406250e-02,
    -9.7187500e-03, -9.1562500e-03, -8.5937500e-03, -8.0312500e-03, -7.4687500e-03, -6.9062500e-03,
    -6.3437500e-03, -5.7812500e-03, -5.2187500e-03, -4.6562500e-03, -4.0937500e-03, -3.5312500e-03,
    -2.9687500e-03, -2.4062500e-03, -1.8437500e-03, -1.2812500e-03, -9.4375000e-04, -8.3125000e-04,
    -7.1875000e-04, -6.0625000e-04, -4.9375000e-04, -3.8125000e-04, -2.6875000e-04, -1.5625000e-04,
    -8.8750000e-05, -6.6250000e-05, -4.3750000e-05, -2.1250000e-05, -7.7500000e-06, -3.2500000e-06,
    -5.5000000e-07, 0.0000000e+00, 5.5000000e-07, 3.2500000e-06, 7.7500000e-06, 2.1250000e-05,
    4.3750000e-05, 6.6250000e-05, 8.8750000e-05, 1.5625000e-04, 2.6875000e-04, 3.8125000e-04,
    4.9375000e-04, 6.0625000e-04, 7.1875000e-04, 8.3125000e-04, 9.4375000e-04, 1.2812500e-03,
    1.8437500e-03, 2.4062500e-03, 2.9687500e-03, 3.5312500e-03, 4.0937500e-03, 4.6562500e-03,
    5.2187500e-03, 5.7812500e-03, 6.3437500e-03, 6.9062500e-03, 7.4687500e-03, 8.0312500e-03,
    8.5937500e-03, 9.1562500e-03, 9.7187500e-03, 1.1406250e-02, 1.4218750e-02, 1.7031250e-02,
    1.9843750e-02, 2.2656250e-02, 2.5468750e-02, 2.8281250e-02, 3.1093750e-02, 3.3906250e-02,
    3.6718750e-02, 3.9531250e-02, 4.2343750e-02, 4.5156250e-02, 4.7968750e-02, 5.0781250e-02,
    5.3593750e-02, 5.6406250e-02, 5.9218750e-02, 6.2031250e-02, 6.4843750e-02, 6.7656250e-02,
    7.0468750e-02, 7.3281250e-02, 7.6093750e-02, 7.8906250e-02, 8.1718750e-02, 8.4531250e-02,
    8.7343750e-02, 9.0156250e-02, 9.2968750e-02, 9.5781250e-02, 9.8593750e-02, 1.0703125e-01,
    1.2109375e-01, 1.3515625e-01, 1.4921875e-01, 1.6328125e-01, 1.7734375e-01, 1.9140625e-01,
    2.0546875e-01, 2.1953125e-01, 2.3359375e-01, 2.4765625e-01, 2.6171875e-01, 2.7578125e-01,
    2.8984375e-01, 3.0390625e-01, 3.1796875e-01, 3.3203125e-01, 3.4609375e-01, 3.6015625e-01,
    3.7421875e-01, 3.8828125e-01, 4.0234375e-01, 4.1640625e-01, 4.3046875e-01, 4.4453125e-01,
    4.5859375e-01, 4.7265625e-01, 4.8671875e-01, 5.0078125e-01, 5.1484375e-01, 5.2890625e-01,
    5.4296875e-01, 5.5703125e-01, 5.7109375e-01, 5.8515625e-01, 5.9921875e-01, 6.1328125e-01,
    6.2734375e-01, 6.4140625e-01, 6.5546875e-01, 6.6953125e-01, 6.8359375e-01, 6.9765625e-01,
    7.1171875e-01, 7.2578125e-01, 7.3984375e-01, 7.5390625e-01, 7.6796875e-01, 7.8203125e-01,
    7.9609375e-01, 8.1015625e-01, 8.2421875e-01, 8.3828125e-01, 8.5234375e-01, 8.6640625e-01,
    8.8046875e-01, 8.9453125e-01, 9.0859375e-01, 9.2265625e-01, 9.3671875e-01, 9.5078125e-01,
    9.6484375e-01, 9.7890625e-01, 9.9296875e-01, 1.0000000e+00,
], dtype=np.float32)

AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

DTYPE_NP = {
    "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "U8": np.uint8, "I8": np.int8, "I32": np.int32, "I64": np.int64,
    "BOOL": np.bool_,
}


class SafeTensors:
    """Minimal read-only safetensors reader (mmap, no torch dependency)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.offset = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def info(self, name):
        return self.header[name]

    def get(self, name):
        meta = self.header[name]
        a, b = meta["data_offsets"]
        raw = self.mm[self.offset + a: self.offset + b]
        arr = raw.view(DTYPE_NP[meta["dtype"]])
        return arr.reshape(meta["shape"]) if meta["shape"] else arr


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def write_safetensors(path, tensors):
    """tensors: {name: (dtype_str, np.ndarray)} — written in insertion order."""
    header, offset = {}, 0
    for name, (dt, arr) in tensors.items():
        nbytes = arr.nbytes
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, (_, arr) in tensors.items():
            fh.write(np.ascontiguousarray(arr).tobytes())


def dequant_nf4(st_map, name):
    """Undo bitsandbytes NF4 double quantization into a flat float32 array."""
    packed = st_map[name + ""].get(name).reshape(-1)
    absmax = st_map[name + ".absmax"].get(name + ".absmax").reshape(-1)

    if absmax.dtype != np.uint8:
        scales = absmax.astype(np.float32)
    else:
        nested = st_map[name + ".nested_absmax"].get(
            name + ".nested_absmax").reshape(-1).astype(np.float32)
        offset, nbs, code = 0.0, 256, None
        qs_name = None
        for suffix in (".quant_state.bitsandbytes__nf4",
                       ".quant_state.bitsandbytes__fp4"):
            if name + suffix in st_map:
                qs_name = name + suffix
                break
        if qs_name:
            raw = st_map[qs_name].get(qs_name).tobytes()
            try:
                js = json.loads(raw.decode("utf-8").rstrip("\x00"))
                offset = float(js.get("nested_offset", 0.0))
                nbs = int(js.get("nested_blocksize", 256))
                if "nested_quant_map" in js:
                    code = np.asarray(js["nested_quant_map"], dtype=np.float32)
            except Exception:
                pass
        if code is None:
            code = BNB_CODE
        si = np.minimum(np.arange(absmax.size) // nbs, nested.size - 1)
        scales = code[absmax] * nested[si] + offset

    hi = NF4[packed >> 4]
    lo = NF4[packed & 0xF]
    vals = np.empty(packed.size * 2, dtype=np.float32)
    vals[0::2] = hi
    vals[1::2] = lo
    return vals * np.repeat(scales, 64)[: vals.size]


def quantize_awq(w, group):
    """w: [N, K] float32 -> (qweight[K,N/8], qzeros[K/G,N/8], scales[K/G,N])."""
    n, k = w.shape
    assert k % group == 0, (n, k, group)
    g = k // group

    wg = w.reshape(n, g, group)
    lo = wg.min(-1)
    hi = wg.max(-1)
    scale = (hi - lo) / 15.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    zero = np.clip(np.rint(-lo / scale), 0, 15).astype(np.int32)

    q = np.rint(wg / scale[..., None]) + zero[..., None]
    q = np.clip(q, 0, 15).astype(np.uint32).reshape(n, k)

    # AWQ stores [K, N/8]; column j of a word sits at nibble slot ORDER[j].
    qT = q.T.reshape(k, n // 8, 8)
    qweight = np.zeros((k, n // 8), dtype=np.uint32)
    qzeros = np.zeros((g, n // 8), dtype=np.uint32)
    zT = zero.T.reshape(g, n // 8, 8).astype(np.uint32)
    for j in range(8):
        qweight |= qT[:, :, j] << (4 * AWQ_ORDER[j])
        qzeros |= zT[:, :, j] << (4 * AWQ_ORDER[j])

    return (qweight.view(np.int32), qzeros.view(np.int32),
            scale.T.astype(np.float16))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for extra in ("config.json", "generation_config.json",
                  "preprocessor_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt"):
        p = os.path.join(args.src, extra)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, extra))

    files = sorted(f for f in os.listdir(args.src) if f.endswith(".safetensors"))
    readers = [SafeTensors(os.path.join(args.src, f)) for f in files]
    st_map = {}
    for r in readers:
        for nm in r.names():
            st_map[nm] = r

    def is_quant(nm):
        return (".self_attn." in nm or ".mlp." in nm) and nm.endswith("_proj.weight")

    def is_meta(nm):
        return any(t in nm for t in (".absmax", ".quant_state", ".quant_map",
                                     ".nested_", "SCB"))

    out, weight_map, n_out = {}, {}, 0
    part = 1
    part_bytes = 0
    LIMIT = 4 << 30

    def flush():
        nonlocal out, part, part_bytes
        if not out:
            return
        fn = f"model-{part:05d}.safetensors"
        write_safetensors(os.path.join(args.dst, fn), out)
        for nm in out:
            weight_map[nm] = fn
        print(f"  wrote {fn}  ({part_bytes / 1e9:.2f} GB, {len(out)} tensors)")
        out = {}
        part += 1
        part_bytes = 0

    for r in readers:
        for nm in r.names():
            if is_meta(nm):
                continue
            if is_quant(nm):
                shape = None
                for suf in (".quant_state.bitsandbytes__nf4",):
                    qs = nm + suf
                    if qs in st_map:
                        try:
                            js = json.loads(st_map[qs].get(qs).tobytes()
                                            .decode("utf-8").rstrip("\x00"))
                            shape = tuple(js["shape"])
                        except Exception:
                            pass
                if shape is None:
                    raise SystemExit(f"{nm}: no shape in quant_state")
                flat = dequant_nf4(st_map, nm)
                w = flat[: shape[0] * shape[1]].reshape(shape)
                qw, qz, sc = quantize_awq(w, args.group)
                base = nm[: -len(".weight")]
                out[base + ".qweight"] = ("I32", qw)
                out[base + ".qzeros"] = ("I32", qz)
                out[base + ".scales"] = ("F16", sc)
                part_bytes += qw.nbytes + qz.nbytes + sc.nbytes
                n_out += 1
                if n_out % 20 == 0:
                    print(f"  {n_out} projections quantized ({nm})")
            else:
                meta = r.info(nm)
                out[nm] = (meta["dtype"], r.get(nm))
                part_bytes += out[nm][1].nbytes
            if part_bytes > LIMIT:
                flush()
    flush()

    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {}, "weight_map": weight_map}, fh, indent=1)

    cfg_path = os.path.join(args.dst, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        cfg["quantization_config"] = {
            "quant_method": "awq", "bits": 4, "group_size": args.group,
            "version": "gemm", "zero_point": True,
        }
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)

    print(f"done: {n_out} projections -> {args.dst}")


if __name__ == "__main__":
    main()
