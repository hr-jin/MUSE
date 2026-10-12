# Memory construction and runtime injection

## Historical sources

A source record contains a scientific coding problem, the public assistant/tool trajectory, the final submission, and the benchmark verifier outcome and feedback. The extraction prompts distinguish the approach actually implemented from proposed changes. They retain successful and failed historical trajectories.

Per-source extraction produces an approach-level account and implementation-level lessons. Each implementation-level lesson records the concrete implementation choice, its rationale, and its relation to the observed outcome. The serialized prompt schemas use the historical field names `final_route`, `decisions`, `decision`, `choice_and_reason`, and `failure_relation`.

Global consolidation reads the per-source lessons with their historical material, then adds, replaces, or retires shared entries. A bank-wide review checks these entries against their sources. This process constructs L0 without changing the accepted L1/L2 source lessons. Freeze the resulting bank and indexes before evaluating held-out scientific coding tasks.

## Initialization

L0 is supplied in full. L1 retrieves the Top5 source problems by cosine similarity using their complete first user prompts before memory injection. The retrieval client does not trim these prompts or add an embedding instruction prefix. Each reference contains its approach-level lesson and source material. Their source IDs form the initial exclusion set for runtime retrieval.

## Runtime retrieval

Invoke runtime retrieval after a designated tool return, such as a `Write` or `Bash` interaction. Queries use the current interaction rather than the whole execution history.

1. Assistant message body queries implementation-level memory.
2. Tool arguments query historical final submissions.
3. Tool arguments query implementation-level memory.
4. Tool return queries historical errors.

Each route retrieves one nearest eligible match. Implementation-level keys consist of the implementation description and `choice_and_reason`. Initial L1 source IDs are excluded from all runtime indexes before retrieval. Merge route hits sharing a source problem, then remove sources already injected at runtime. Do not retrieve replacement hits after this second exclusion. At most four source records reach the gate.

## Gate and context update

The tool-free LLM gate receives the task summary, preceding public scientific coding trajectory, current interaction, and candidate records. It returns `inject_ids` and `review_note`, selecting at most one source record. The review note identifies a concrete operation, assumption, or result to inspect and explains what to check using the selected experience. An implementation that already exists may still warrant review.

Inject the selected historical record and its review note together. Track its source ID for subsequent exclusion. An empty selection leaves the context unchanged, and a rejected source can be retrieved again later.

## Ablations

The cumulative memory-level configurations are L0 alone, L0+L1, and the full L0+L1+L2 system. The no-gate configuration retains the three levels, four-route retrieval, and source exclusions. At each trigger it injects the highest-similarity eligible source record directly, without LLM selection or generated review advice.
