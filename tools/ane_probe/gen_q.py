import numpy as np, coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import coremltools.optimize as cto

M,K,N = 256, 3584, 18944
W = (np.random.randn(K,N)*0.02).astype(np.float16)
@mb.program(input_specs=[mb.TensorSpec(shape=(M,K), dtype=types.fp32)])
def prog(x):
    return mb.matmul(x=mb.cast(x=x, dtype="fp16"), y=W, name="mm")
m = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
               compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.ALL)
cfg = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_block",
        block_size=64))
q = cto.coreml.linear_quantize_weights(m, config=cfg)
q.save("mm_prefill_int4.mlpackage"); print("saved int4")
