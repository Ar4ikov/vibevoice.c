# Skill: TensorRT Engine (Conv-VAE Encoders)

## Purpose
Build TRT engines for the Conv-VAE tokenizer encoders (acoustic + semantic).
These use standard ops (Conv1d, RMSNorm) and benefit greatly from TRT FP16.
LLM decoder stays as custom CUDA (NF4 not natively supported in TRT).

## Strategy
- Export Conv-VAE encoders to ONNX via tools/export_onnx.py
- Build TRT .plan via C API or tools/build_trt_engine.py
- Runtime: load .plan and execute via TRT C API

## Key API
```c
vv_status_t vv_trt_engine_load(const char* plan_path, vv_trt_engine_t** out);
vv_status_t vv_trt_engine_infer(vv_trt_engine_t* engine,
                                 const void** inputs, void** outputs,
                                 cudaStream_t stream);
vv_status_t vv_trt_engine_free(vv_trt_engine_t* engine);
```

## Dynamic Shapes
- input: [1, 1, num_samples] where num_samples varies
- output: [1, num_frames, vae_dim] where num_frames = num_samples / 3200
