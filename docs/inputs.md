# Input formats

The command-line interface reads JSON files. Supply your own historical scientific coding material; the repository does not contain benchmark trajectories or evaluation outputs.

## Historical evidence

`muse extract --input` accepts one evidence object or a list of evidence objects. Each object has the following fields.

| Field | Type | Content |
| --- | --- | --- |
| `source_ref` | String | Unique ID of the historical source problem. |
| `scientific_problem` | String | The complete first user prompt before memory injection. |
| `submission` | String | The historical final submission text. |
| `official_evaluation` | Object | The historical benchmark verifier outcome and feedback. |
| `trajectory` | String, list, or object | The historical public assistant and tool trajectory. |

Extraction returns a map from `source_ref` to a memory object containing the original `scientific_problem`, `official_evaluation`, `L1.final_route`, `L1.submissioned_solution`, and `L2` lessons. Serialized implementation-level lessons retain the historical fields `decision`, `choice_and_reason`, and `failure_relation`.

Consolidation and global review read the extracted memory map and the corresponding evidence. Source insertion order determines the incremental consolidation order. Global entries retain stable IDs, supporting source IDs, and the status `observed`, `derived_from_contract`, or `proposed_unverified`. The operations schema and source-order checks reject unknown sources, future sources, and multiple operations on the same existing entry.

## Target prompts

`muse build-bank --targets` reads a map from target task IDs to complete first user prompt strings before memory injection. Target IDs must be disjoint from the historical source IDs. Target trajectories, submissions, and verifier outcomes are not bank-construction inputs. Use a new output directory to preserve existing frozen artifacts.

## Current interactions

`muse runtime --interactions` accepts one interaction object or a list in chronological order. Each interaction contains an `assistant` object, a `hook` object, and an optional `turn_id` string.

- `assistant.content` is the current public assistant message body.
- `hook.tool_name` identifies the completed tool, such as `Write` or `Bash`.
- `hook.tool_use_id` identifies that tool call.
- `hook.tool_input` contains the tool arguments, such as `command` for Bash or `content` for Write.
- `hook.tool_response` contains the tool's returned text.

Private reasoning is not a retrieval input. Historical error retrieval uses Bash returns; Write acknowledgment messages are not error queries. The runtime maintains source exclusions after acknowledged delivery and writes the delivered memory messages to its output directory.

The separate `retrieve` and `gate` commands expose the same stages for integration. Use their `--help` output for file arguments. A scientific coding agent remains responsible for executing its tools under the benchmark's permissions and passing the submitted solution to the official verifier.
