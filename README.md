# MUSE

**Multi-Level Memory with Selective Runtime Injection for Scientific Coding Agents**

MUSE is a memory management framework for scientific coding agents. It combines global guidance, problem-matched approaches, and implementation experience retrieved as a solution develops.

This release provides the memory construction and retrieval components, LLM gate prompts, and their schemas. A scientific coding agent supplies its task and public tool interactions to these components. Benchmark environments and official verifiers are installed separately.

## Install

From the repository directory, install the Python package and inspect its command-line interface.

```bash
python -m pip install -e .
muse --help
```

Memory extraction, consolidation, global review, and gating require an OpenAI-compatible chat endpoint. Embedding-based retrieval requires an embedding endpoint.

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `MUSE_API_KEY` | Chat endpoint credential | Required |
| `MUSE_BASE_URL` | Chat endpoint URL | Required |
| `MUSE_MODEL` | Chat model | `glm-5.3-flash` |
| `MUSE_REASONING_EFFORT` | Chat reasoning effort | `high` |
| `MUSE_EMBEDDING_BASE_URL` | Embedding endpoint URL | `http://127.0.0.1:18081/v1` |
| `MUSE_EMBEDDING_MODEL` | Embedding model | `qwen3-embedding-8b` |
| `MUSE_EMBEDDING_API_KEY` | Embedding endpoint credential | `local` |

Set these variables in your shell. Keep credentials out of input files and source code.

## End-to-end use

Prepare your own historical evidence, target prompts, and scientific coding interactions as described in [input formats](docs/inputs.md). The default Top5 setup requires at least five historical sources. Run the following commands from the repository directory with a new `work/artifacts` output path.

```bash
muse extract --input data/evidence.json --output work/memories.json
muse consolidate --memories work/memories.json --evidence data/evidence.json --output work/global-draft.json
muse review --memories work/memories.json --evidence data/evidence.json --bank work/global-draft.json --output work/global.json
muse build-bank --memories work/memories.json --targets data/targets.json --global-bank work/global.json --output work/artifacts
muse initial --bank work/artifacts/bank --task target-001 --output work/initial-memory.txt
muse runtime --bank work/artifacts/bank --routes work/artifacts/routes --task target-001 --task-tldr "Solve the target scientific coding task." --interactions data/interactions.json --output work/runtime
```

The `data/` paths refer to user-supplied inputs, not bundled benchmark trajectories. Supply `initial-memory.txt` to the scientific coding agent at initialization. The runtime command processes supplied interactions and saves memory messages in `work/runtime/delivered.json`; a live scientific coding agent can instead use the Python runtime callback to receive messages after its tool returns. The `--no-gate` runtime option selects the similarity-only injection control.

## Workflow

| Stage | Command | Role |
| --- | --- | --- |
| Per-source extraction | `muse extract` | Extract approach-level and implementation-level lessons from a historical scientific coding trajectory, its final submission, and verifier feedback. |
| Global consolidation | `muse consolidate` | Add, replace, or retire shared guidance while retaining source references. |
| Global review | `muse review` | Review global guidance against its historical source material. |
| Retrieval bank | `muse build-bank` | Build the frozen memory bank and semantic indexes. |
| Initialization | `muse initial` | Supply complete global memory and the Top5 approach-level references matched to a target problem. |
| Runtime retrieval | `muse retrieve` | Retrieve implementation experience using the current interaction, merge matches by source, and apply source exclusions. |
| Selection and review | `muse gate` | Select at most one source and generate specific review advice, or skip injection. |
| Runtime integration | `muse runtime` | Process supplied scientific coding interactions and write selected memory messages. |

Use `muse <command> --help` for each stage. Memory construction precedes evaluation. The scientific coding agent continues its own execution after receiving initialization or runtime context.

## Method

- **L0 — Global memory.** Shared lessons supplied in full at initialization.
- **L1 — Approach-level memory.** Five references retrieved by problem similarity at initialization.
- **L2 — Implementation-level memory.** Experience retrieved from the current scientific coding interaction.

Runtime retrieval has four routes, each returning Top1 when its query is available.

| Current-interaction query | Historical index |
| --- | --- |
| Assistant message body | Implementation-level memory |
| Tool arguments | Final submissions |
| Tool arguments | Implementation-level memory |
| Tool return | Historical errors |

Implementation-level keys combine the implementation description with the recorded choice and rationale. The initial five sources are excluded from runtime search. Retrieved hits are merged by source problem; previously injected sources are then removed before the gate, without backfilling. The gate evaluates the task summary, preceding public trajectory, current interaction, and candidate records. Selected historical experience and specific review advice enter the scientific coding agent's context together.

See [method details](docs/method.md), [benchmark setup](docs/benchmarks.md), and [prompt documentation](prompts/README.md).

## Tests

The tests use synthetic inputs and fake model responses. They do not call model APIs or run benchmark tasks.

```bash
python -m unittest discover -s tests -v
```

## External assets

Benchmark datasets, verifier implementations, model weights, and the scientific coding agent runtime are obtained from their upstream projects. Their references and licenses are documented in [third-party assets](docs/THIRD_PARTY.md).
