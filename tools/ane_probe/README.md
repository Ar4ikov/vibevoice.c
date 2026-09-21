# What runs on the Neural Engine, and how fast

The probe behind `docs/ANE.md`. Nothing here is part of the runtime: it
builds Core ML models for the shapes vibevoice.c runs, asks Core ML where it
would put each operation, and times the result on the Neural Engine, the GPU
and the CPU.

```bash
pip install coremltools
python3 gen.py            # a decode-shaped and a prefill-shaped matmul, a conv stack
python3 gen_q.py          # the prefill matmul again, int4 blockwise weights
python3 gen_enc.py 4      # the real acoustic encoder, 4 seconds of audio

clang -fobjc-arc -O2 -o probe probe.m -framework Foundation -framework CoreML
./probe mm_prefill.mlpackage 30 $(python3 -c "print(2*256*3584*18944)")
./probe enc_4s.mlpackage 10
```

`probe` prints the `MLComputePlan` placement (how many operations Core ML
puts on each device) and then the mean time under `MLComputeUnitsAll`,
`CPUAndNeuralEngine`, `CPUAndGPU` and `CPUOnly`. The third argument is the
FLOP count of one run, and is optional.

`gen_enc.py` reads the encoder's shapes straight from a checkpoint
(`VV_MODEL=/path/to/model_hf`, `./model_hf` by default) and fills them with
random weights: the question it answers is how long that graph takes, not
what it outputs.
