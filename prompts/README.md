# Prompt templates

The packaged templates in [`muse/prompts/`](../muse/prompts/) implement memory construction and selective runtime injection for scientific coding agents.

| Files | Purpose |
| --- | --- |
| `extraction.txt`, `extraction_schema.json` | Per-source approach-level and implementation-level memory extraction. |
| `extraction-kbx.txt`, `extraction-kbx-variant.txt` | KernelBenchX extraction instructions and the corresponding variant text. |
| `consolidation.txt`, `operations_schema.json` | Incremental global memory updates. |
| `consolidation-pde.txt`, `consolidation-pde-variant.txt` | PDE global memory consolidation instructions and the corresponding variant text. |
| `global-review.txt` | Bank-wide review against historical scientific coding material. |
| `gate-system.txt`, `gate-user.txt` | Runtime experience selection and specific review advice. |
| `runtime-injection.txt` | Context containing the review advice and selected historical experience. |

Braced placeholders represent source- or task-specific inputs. The gate schema's eligible-candidate placeholder represents the candidate IDs supplied in the current invocation; the runtime constructs the corresponding ID enumeration.

Historical extraction templates call L1/L2 a "two-level structure" because they describe per-source extraction; L0 is constructed by subsequent global consolidation. Their use of "training" refers to the memory-extraction sources, not model-weight training. The stored `decision` field describes an implementation choice and corresponds to implementation-level memory in the manuscript.

Extraction and consolidation use the supplied historical trajectory, final submission, and verifier feedback. They do not execute the historical scientific coding solution to create new outcomes. The gate receives public current work and the retrieved historical records, and can select at most one source or skip injection.
