# CIMA coverage matrix

This matrix separates three different statuses that must not be conflated:

- **Specified**: the behavior is part of the CIMA specification or conformance model.
- **Implemented**: the public demonstrator contains an implementation path for the behavior.
- **Demonstrated in the published profile**: the archived evidence package directly exercises the behavior in the published evaluation profile.

The published profile referenced here is the Zenodo-archived CIMA package associated with DOI `10.5281/zenodo.20139206`.

| CIMA capability | Specified | Implemented in demonstrator | Demonstrated in published profile | Evidence boundary |
|---|---:|---:|---:|---|
| Budget-bounded ContextView construction | Yes | Yes | Yes | Demonstrated for the published 230-run profile with a 6k token budget. |
| Prompt-visible evidence markers | Yes | Yes | Yes | Demonstrated structurally; does not imply semantic entailment. |
| Marker resolution to stored source spans | Yes | Yes | Yes | Demonstrated for active selected markers in the published profile. |
| Publication gate for cited factual output | Yes | Yes | Yes | Demonstrated as structural citation enforcement. |
| Traceable insufficient-evidence abstention | Yes | Yes | Yes | Demonstrated for abstention cases in the published profile. |
| Zoom from marker to source evidence | Yes | Yes | Yes | Demonstrated for selected markers in the published profile. |
| L1 Zoom-out with witness lineage | Yes | Yes | Yes | Demonstrated at L1; deeper summary pyramids are outside the current evidence boundary. |
| Run-level cleanup | Yes | Yes | Yes | Demonstrated as local run cleanup; not equivalent to full lifecycle governance. |
| Episodic memory representation | Yes | Yes | Partial | Exercised by the open-scenario harness, but not a broad multi-turn memory evaluation. |
| TaskMemory for goals, constraints, and decisions | Yes | Partial | No | Specified, but not demonstrated as a central multi-turn operational authority in the published profile. |
| Global memory promotion/demotion | Yes | Partial | No | Specified, but not empirically established by the published profile. |
| Full lifecycle governance: archive, thinning, purge, retention | Yes | Partial | No | Cleanup is demonstrated; the full lifecycle model remains outside the current evidence boundary. |
| Validate / Resolve for conflicting or stale memory | Yes | Partial | No | Specified, but not covered by a published adversarial conflict-resolution suite. |
| Semantic claim-evidence entailment | No claim | No | No | CIMA currently enforces structural citation and lineage, not semantic entailment. |
| Factual-correctness improvement over RAG baselines | No claim | No | No | No baseline comparison is claimed by the published package. |
| Persistent-memory poisoning resistance | No claim | No | No | Relevant future work; not demonstrated by the current public profile. |
| Autonomous agent runtime or planner | No claim | No | No | CIMA is not an agent runtime or planning framework. |
| Legal or regulatory compliance | No claim | No | No | CIMA provides traceability primitives, not compliance certification. |

## Publication rule

A paper, README, release note, or external description should only describe a capability as **demonstrated** when the archived evidence package exercises it directly. Capabilities marked as specified or partially implemented may be discussed as design scope, future work, or conformance requirements, but not as empirical results.
