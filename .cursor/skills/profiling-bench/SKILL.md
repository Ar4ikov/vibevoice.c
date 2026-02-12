# Skill: Profiling & Benchmarking

## Metrics
| Metric | Target (RTX 3080) |
|--------|-------------------|
| Speech encoding (10s audio) | < 200ms |
| Prefill (1K tokens) | < 200ms |
| Decode per token | < 15ms |
| Throughput | > 70 tok/s |
| RTF (10 min audio) | < 0.5 |
| GPU Memory | < 10 GB |

## Tools
- CUDA Events for GPU timing
- NVTX markers (VV_NVTX_PUSH/POP macros)
- Nsight Systems / Nsight Compute
- vv_bench.exe --model ... --audio ... --warmup 3 --runs 10
