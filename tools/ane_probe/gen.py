"""Emit Core ML models for the shapes vibevoice.c actually runs."""
import numpy as np, coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

FP16 = np.float16

def save(prog, name, fp16=True):
    m = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
                   compute_units=ct.ComputeUnit.ALL)
    m.save(f"{name}.mlpackage"); print("saved", name)

def matmul(M, K, N, name):
    W = (np.random.randn(K, N) * 0.02).astype(FP16)
    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp16)])
    def prog(x):
        return mb.matmul(x=x, y=W, name="mm")
    save(prog, name)

def conv_stack(name, C=512, T=1024, layers=8, k=7):
    """The encoder's inner shape: depthwise mixer + 4x FFN, repeated."""
    ws = [( (np.random.randn(C, 1, k)*0.05).astype(FP16),
            (np.random.randn(4*C, C, 1)*0.05).astype(FP16),
            (np.random.randn(C, 4*C, 1)*0.05).astype(FP16) ) for _ in range(layers)]
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, C, T), dtype=types.fp16)])
    def prog(x):
        for i, (dw, up, dn) in enumerate(ws):
            y = mb.conv(x=x, weight=dw, groups=C, pad_type="same", name=f"dw{i}")
            x = mb.add(x=x, y=y)
            y = mb.conv(x=x, weight=up, pad_type="valid", name=f"up{i}")
            y = mb.gelu(x=y, mode="EXACT")
            y = mb.conv(x=y, weight=dn, pad_type="valid", name=f"dn{i}")
            x = mb.add(x=x, y=y)
        return x
    save(prog, name)

if __name__ == "__main__":
    np.random.seed(0)
    matmul(1,   3584, 18944, "mm_decode")     # one token through gate/up
    matmul(256, 3584, 18944, "mm_prefill")    # a prefill chunk
    conv_stack("vae_stack")                   # the encoder's shape
