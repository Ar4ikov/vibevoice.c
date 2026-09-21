"""The acoustic encoder's real graph, shapes read from the checkpoint."""
import json, struct, sys, numpy as np, coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

import os
MODEL = os.environ.get('VV_MODEL', './model_hf').rstrip('/') + '/'
def shapes(prefix):
    """Tensor shapes from every safetensors shard in the checkpoint."""
    import glob
    h={}
    for f in sorted(glob.glob(MODEL + '*.safetensors')):
        with open(f,'rb') as fh:
            n=struct.unpack('<Q',fh.read(8))[0]; h.update(json.loads(fh.read(n)))
    return {k[len(prefix):]: v['shape'] for k,v in h.items() if k.startswith(prefix)}

S = shapes('model.acoustic_tokenizer.encoder.')
F = np.float16
def rnd(shape, s=0.05): return (np.random.randn(*shape)*s).astype(F)

SECONDS = float(sys.argv[1]) if len(sys.argv)>1 else 30.0
T = int(SECONDS*24000)
STRIDES = [1,2,2,4,5,5,8]          # downsample_layers 0..6, product 3200

def n_blocks(st):
    return len({k.split('.')[1] for k in S if k.startswith(f'stages.{st}.')})

def build():
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,1,T), dtype=types.fp32)])
    def prog(x):
        x = mb.cast(x=x, dtype="fp16")
        for st in range(7):
            w = S[f'downsample_layers.{st}.0.conv.conv.weight']
            k, s = w[2], STRIDES[st]
            pad = max(k - s, 0)
            x = mb.pad(x=mb.expand_dims(x=x, axes=[2]), pad=[0,0,0,0,0,0,pad,0], mode="constant")
            x = mb.squeeze(x=x, axes=[2])
            x = mb.conv(x=x, weight=rnd(tuple(w)), bias=rnd((w[0],)),
                        strides=[s], pad_type="valid", name=f"ds{st}")
            C = w[0]
            for b in range(n_blocks(st)):
                p = f'stages.{st}.{b}.'
                for tag, wk in (("mix", 'mixer.conv.conv.conv.weight'), ("ffn", None)):
                    if tag == "mix":
                        mw = S[p+wk]; kk = mw[2]
                        y = rms(x, C)
                        y = mb.pad(x=mb.expand_dims(x=y, axes=[2]),
                                   pad=[0,0,0,0,0,0,kk-1,0], mode="constant")
                        y = mb.squeeze(x=y, axes=[2])
                        y = mb.conv(x=y, weight=rnd(tuple(mw)), bias=rnd((mw[0],)),
                                    groups=C, pad_type="valid", name=f"mx{st}_{b}")
                    else:
                        l1 = S[p+'ffn.linear1.weight']; l2 = S[p+'ffn.linear2.weight']
                        y = rms(x, C)
                        y = mb.conv(x=y, weight=rnd((l1[0],l1[1],1)), bias=rnd((l1[0],)),
                                    pad_type="valid", name=f"f1_{st}_{b}")
                        y = mb.gelu(x=y, mode="EXACT")
                        y = mb.conv(x=y, weight=rnd((l2[0],l2[1],1)), bias=rnd((l2[0],)),
                                    pad_type="valid", name=f"f2_{st}_{b}")
                    y = mb.mul(x=y, y=rnd((1,C,1), 1e-6))
                    x = mb.add(x=x, y=y)
        hw = S['head.conv.conv.weight']
        x = mb.conv(x=x, weight=rnd(tuple(hw)), bias=rnd((hw[0],)),
                    pad_type="valid", name="head")
        return mb.cast(x=x, dtype="fp32")
    return prog

def rms(x, C, eps=1e-5):
    v = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[1], keep_dims=True)
    return mb.mul(x=mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=v, y=np.float16(eps)))),
                  y=rnd((1,C,1), 1.0))

np.random.seed(0)
m = ct.convert(build(), minimum_deployment_target=ct.target.macOS15,
               compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
name = f"enc_{int(SECONDS)}s"
m.save(name+".mlpackage"); print("saved", name, "T =", T)
