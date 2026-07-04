# CIMA claim ledger

This ledger defines publication-safe wording for CIMA claims. It is intended to prevent over-reading of the specification or demonstrator.

## Claim status categories

| Status | Meaning |
|---|---|
| Demonstrated | Directly exercised by the published evidence profile. |
| Partially demonstrated | Exercised only under a restricted form or as a supporting behavior. |
| Specified only | Present in the CIMA specification or conformance model, but not demonstrated by the published profile. |
| Non-claim | Explicitly outside the current CIMA claim boundary. |

## Ledger

| Claim ID | Claim | Status | Supported wording | Forbidden or unsafe wording | Critical note |
|---|---|---:|---|---|---|
| CIMA-C1 | CIMA specifies governed memory and context construction for LLM-based systems. | Demonstrated / specified | "CIMA is a specification and demonstrator for governed memory and verifiable context construction." | "CIMA is a complete autonomous agent framework." | CIMA should be evaluated as a memory/context governance layer, not as an agent runtime. |
| CIMA-C2 | ContextView construction is bounded by an explicit budget. | Demonstrated | "The demonstrator builds budget-bounded ContextViews in the published profile." | "CIMA provides optimal context selection." | Budget compliance is structural; optimality is not claimed. |
| CIMA-C3 | Active evidence markers are prompt-visible and structurally admissible. | Demonstrated | "Published factual outputs must cite prompt-visible evidence markers." | "Cited evidence necessarily entails each generated claim." | Marker visibility and admissibility are not semantic entailment. |
| CIMA-C4 | Selected markers resolve to stored source spans. | Demonstrated | "Markers selected into the ContextView resolve to stored source spans." | "All possible memory references are globally resolved under every deployment." | Demonstrated for the public profile and selected markers. |
| CIMA-C5 | The publication gate rejects uncited factual output or accepts traceable abstention. | Demonstrated | "The publication gate enforces cited factual output or traceable abstention in the demonstrator." | "CIMA guarantees factual correctness." | Structural publication integrity is not truth. |
| CIMA-C6 | Zoom resolves selected markers to literal evidence. | Demonstrated | "Zoom allows selected markers to be resolved back to evidence." | "Zoom proves the generated answer is true." | Zoom is a reconstruction primitive. |
| CIMA-C7 | L1 Zoom-out preserves witness lineage. | Demonstrated / partial | "The public profile demonstrates L1 Zoom-out with witness lineage." | "CIMA demonstrates full multi-level lossless summarization." | Deeper summary pyramids are not covered by the current evidence boundary. |
| CIMA-C8 | Run-level cleanup is exercised. | Demonstrated / partial | "The demonstrator exercises run-level cleanup." | "CIMA demonstrates full lifecycle governance." | Archive, thinning, retention, and hard-delete are not fully demonstrated. |
| CIMA-C9 | TaskMemory models active goals, constraints, and decisions. | Specified only | "CIMA specifies TaskMemory as part of its memory/context model." | "The public profile empirically validates multi-turn TaskMemory." | Requires separate multi-turn evaluation. |
| CIMA-C10 | Global memory promotion/demotion preserves governance boundaries. | Specified only | "CIMA specifies global memory promotion/demotion requirements." | "The demonstrator validates global memory at scale." | Not established by the published profile. |
| CIMA-C11 | Validate and Resolve govern stale, missing, or contradictory memory. | Specified only | "CIMA specifies validation and resolution mechanisms for governed memory." | "CIMA has demonstrated robust contradiction resolution." | Requires adversarial conflict-resolution tests. |
| CIMA-N1 | Semantic entailment between every claim and cited evidence. | Non-claim | "CIMA does not currently claim semantic entailment guarantees." | "CIMA guarantees that every generated claim is entailed by its citation." | Entailment would require claim decomposition and semantic verification. |
| CIMA-N2 | Factuality improvement over RAG or long-context baselines. | Non-claim | "The current evidence evaluates structural guarantees, not factuality improvement over baselines." | "CIMA outperforms RAG on factual accuracy." | No such baseline comparison is part of the current package. |
| CIMA-N3 | Complete persistent-memory security. | Non-claim | "CIMA provides governance primitives relevant to memory safety." | "CIMA solves memory poisoning or persistent-memory security." | Security evaluation remains future work. |
| CIMA-N4 | Legal, regulatory, or compliance certification. | Non-claim | "CIMA supports auditability and traceability primitives." | "CIMA provides legal compliance." | Compliance is outside the current claim boundary. |

## Review checklist

Before publication, release notes, papers, READMEs, and external summaries should answer:

1. Is the claim structural, empirical, architectural, or aspirational?
2. Is the claim directly supported by the Zenodo-archived evidence package?
3. Does the wording accidentally imply semantic entailment or factual correctness?
4. Does the wording treat CIMA as an agent runtime, RAG system, planner, or security solution?
5. If the claim is specified but not demonstrated, is it clearly marked as such?

If any answer is unclear, use the more conservative wording from this ledger.
