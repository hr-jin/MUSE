# Scientific coding benchmark setup

The memory package operates on historical source records and current interactions. Scientific coding task execution and scoring use the benchmark-specific environment and official verifier. Obtain benchmark data and tooling from the upstream repositories linked below.

## Memory extraction and testing

Approximately 30% of the problems in each benchmark provide the historical trajectories used for memory extraction. The remaining tasks form the test set. All compared memory methods use the same source trajectories and freeze their memory before testing.

| Benchmark | Memory extraction tasks | Test tasks | Metrics |
| --- | ---: | ---: | --- |
| [PDEAgent-Bench](https://github.com/YusanX/pde-agent-bench) | 194 | 451 | Verifier pass rate |
| [KernelBenchX](https://github.com/BonnieW05/KernelBenchX) | 53 | 122 | Correctness and acceleration rates |
| [KernelGenBench-MS](https://github.com/flagos-ai/KernelGenBench) | 63 | 140 | Correctness and acceleration rates |
| [SWE Science](https://github.com/OpenMOSS/SWE-bench-Science) | 35 | 84 | Verifier pass rate |

## Configuration used in the study

The scientific coding agent, memory construction agents, and gate use ZCode with GLM-5.3-flash at high reasoning effort. Task-agent and gate requests have a per-response `max_tokens` limit of 128,000. The task wall-clock limits are two hours for PDE, four hours for each kernel benchmark, and three hours for SWE Science.

Retrieval uses Qwen3-Embedding-8B with 4,096-dimensional float32 unit vectors and cosine similarity. The retrieval client adds no instruction prefix and does not chunk or truncate runtime queries. Initialization retrieves Top5 references, and the four runtime routes each retrieve Top1 when their query is available.

| Environment | Python | Main scientific coding dependencies |
| --- | --- | --- |
| PDE | 3.11.15 | DOLFINx 0.10.0; PETSc 3.25.4 |
| KernelBenchX | 3.12.13 | PyTorch 2.9.1+cu128; Triton 3.5.1 |
| KernelGenBench-MS | 3.12.13 | PyTorch 2.9.0+cu128; Triton 3.5.0 |
| SWE Science | Repository-specific | Task and verifier containers supplied by the benchmark |

These are separate evaluation environments. Install the appropriate environment for the scientific coding benchmark being evaluated rather than combining all dependencies into one environment.

Kernel reference and candidate code are measured on the same GPU under a shared GPU lock. The reported server has two AMD EPYC 7742 processors and eight NVIDIA A100-SXM4 40 GB GPUs. Hosted APIs serve the scientific coding agents and gate.

## Scoring

PDE and SWE Science report the fraction of test tasks passing their official verifiers. Kernel correctness is the fraction of test tasks passing correctness checks. Kernel acceleration is the fraction passing correctness with valid timing and a speedup of at least 1.01×. Candidate failures and timeouts count as failures. Reference and candidate timings use matched conditions.

## Integration

A benchmark adapter should export each historical scientific coding trajectory with its source problem, final submission, and verifier feedback. During a target run, provide the task summary and public tool interactions to MUSE, and append any returned memory context to the scientific coding agent's context. The benchmark continues to control tool permissions and official evaluation.

This repository provides the portable memory components. The original distributed experiment orchestration, GPU scheduling, and complete benchmark installation are separate from this package.
