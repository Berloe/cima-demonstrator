# CIMA v1.0-rc2 publication status

This note records the current public publication state for CIMA v1.0-rc2 after Zenodo metadata review and GitHub citation alignment.

## Public records

- **Zenodo version:** `1.0-rc2`
- **Version DOI:** `10.5281/zenodo.20465483`
- **Version DOI URL:** <https://doi.org/10.5281/zenodo.20465483>
- **Concept DOI / all versions DOI:** `10.5281/zenodo.20139205`
- **Concept DOI URL:** <https://doi.org/10.5281/zenodo.20139205>
- **GitHub repository:** <https://github.com/Berloe/cima-demonstrator>
- **GitHub release tag:** <https://github.com/Berloe/cima-demonstrator/releases/tag/v1.0-rc2>

## Title policy

The Zenodo title is preserved:

> CIMA: Bounded, Traceable, and Navigable Memory for Language Model Systems

This title remains within the demonstrated and specified scope because the public package supports bounded context construction, traceable source-span lineage, and navigable memory/context behavior through markers, marker resolution, Zoom, and evidence recovery.

## Metadata clarifications

The Zenodo description was clarified to avoid reading evidence recoverability as semantic losslessness. The intended claim is that source text is retained and can be navigated back to from summaries or selected context items. This does not imply that compressed abstractions are semantically lossless, nor that cited evidence entails every generated claim.

The Zenodo subjects were aligned with the publication scope, including memory architecture, governed memory, LLM memory, context engineering, context construction, traceability, provenance, auditability, auditable AI, and citation contract.

The related-work metadata was set to point to the GitHub repository or release URL as the associated software artifact.

## GitHub citation alignment

GitHub-facing metadata now follows this DOI policy:

- The README badge points to the Zenodo concept DOI / all-versions DOI: `10.5281/zenodo.20139205`.
- Version-specific citations for v1.0-rc2 point to the Zenodo version DOI: `10.5281/zenodo.20465483`.
- `CITATION.cff` uses the v1.0-rc2 title, version, and version DOI.
- The coverage matrix references the v1.0-rc2 package using the version DOI and records the concept DOI separately.

The previously used DOI `10.5281/zenodo.20139206` is not the confirmed v1.0-rc2 version DOI or concept DOI for the current public record and should not be used in current publication metadata.

## Scope of the update

The post-publication alignment was documentation and metadata only. It did not change:

- runtime code;
- tests;
- generated evidence;
- evaluation harness behavior;
- the published 230-run evidence profile;
- the demonstrated claim boundary.

## Current claim boundary

The public CIMA v1.0-rc2 package supports claims about structural memory/context governance under the published demonstrator profile:

- budget-bounded ContextView construction;
- prompt-visible evidence markers;
- marker resolution to stored source spans;
- publication gating for cited factual output or traceable abstention;
- Zoom from marker to source evidence;
- L1 Zoom-out with witness lineage;
- run-level cleanup under the published profile.

The public package should not be described as demonstrating:

- semantic claim-evidence entailment;
- factual-correctness improvement over RAG baselines;
- complete autonomous-agent runtime functionality;
- full multi-turn TaskMemory authority;
- full global memory promotion/demotion;
- complete lifecycle governance;
- persistent-memory poisoning resistance;
- legal or regulatory compliance.

## Versioning decision

No new Zenodo version is required solely for the current metadata and GitHub citation alignment. A new Zenodo version should be created only if the archived package itself needs to include additional files, materially revised papers, new evidence, changed runtime behavior, or a revised demonstrated-claim boundary.
