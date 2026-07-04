# CIMA — Contextual Item Memory Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20139206.svg)](https://doi.org/10.5281/zenodo.20139206)

CIMA is a specification and demonstrator for governed memory and verifiable context construction in LLM-based systems. It separates stored memory from active context, represents selected context as an auditable ContextView, preserves lineage from context markers to source spans, and constrains publication through a gate that requires prompt-visible evidence markers or traceable abstention.

This repository contains the CIMA Demonstrator: a runnable Python implementation of the CIMA-core architecture with a full evaluation harness. The published evidence package is archived at [doi.org/10.5281/zenodo.20139206](https://doi.org/10.5281/zenodo.20139206).

---

## What CIMA is

CIMA is intended to specify and test structural governance properties for LLM memory/context pipelines:

- governed memory items with lineage to evidence;
- budget-bounded ContextView construction;
- prompt-visible evidence markers;
- marker resolution to stored source spans;
- publication gating for cited factual output or traceable abstention;
- Zoom and L1 Zoom-out operations with recoverable evidence;
- run-level cleanup in the published demonstrator profile.

The literal source text is retained so that abstractions remain evidence-recoverable: a summary or selected context item can be traced back to its underlying evidence. This does not mean that compressed abstractions are semantically lossless, nor that cited evidence entails every generated claim.

## What CIMA is not

CIMA is not an autonomous agent runtime, a planning framework, a general RAG system, a vector database, a tool-use platform, a benchmark of reasoning, or a factuality verifier. The demonstrator evaluates structural publication and lineage guarantees; it does not claim to improve intrinsic model reasoning, outperform RAG baselines on factual accuracy, guarantee semantic entailment, or solve persistent-memory security in full.

---

## Demonstrated profile and non-claims

The published evaluation profile covers 230 open-scenario runs using GPT-4o and a 6k token ContextView budget. Within that profile, the evidence package reports passing structural checks for bounded context construction, prompt-visible citation markers, marker resolution, publication integrity, Zoom, L1 Zoom-out, and run-level cleanup.

The current public demonstrator should not be read as empirical evidence for full multi-turn TaskMemory, global memory promotion/demotion, complete lifecycle governance, semantic claim-evidence entailment, adversarial memory-poisoning resistance, or legal/regulatory compliance.

For a precise boundary between specified, implemented, and demonstrated behavior, see:

- [`docs/publication/cima_coverage_matrix.md`](docs/publication/cima_coverage_matrix.md)
- [`docs/publication/cima_claim_ledger.md`](docs/publication/cima_claim_ledger.md)

---

## Quick start

```bash
# 1. Install dependencies
pip install poetry
poetry install

# 2. Start the server (standalone CI profile — no external services or API key required)
./scripts/run_demo.sh
# Health check: curl http://localhost:8000/health
```

The default standalone server uses in-memory stores and a deterministic rule backend. No Postgres, Qdrant, TEI, llama.cpp, or hosted LLM API is required for a smoke test.

To run the same standalone server with a hosted OpenAI-compatible model:

```bash
cp .env.example .env
# Edit .env and set CIMA_DEMO_STANDALONE_LLM_BACKEND=openai
# Set CIMA_DEMO_OPENAI_API_KEY=sk-...
./scripts/run_demo.sh
```

---

## Reproduce the published evaluation

The published evidence package (230 cases, GPT-4o) is at the Zenodo DOI above. To reproduce from scratch:

```bash
# 1. Download and normalize datasets
python -m cima_demo.demo.open_scenarios.download
python -m cima_demo.demo.open_scenarios.normalize

# 2. Start the server in a separate terminal using the public-eval profile
CIMA_DEMO_STANDALONE_LLM_BACKEND=openai \
CIMA_DEMO_LLM_PROVIDER=openai \
CIMA_DEMO_LLM_MODEL=gpt-4o \
./scripts/run_demo.sh

# 3. Run the evaluation (4 datasets, GPT-4o, 6k token budget)
./scripts/broad_launch.sh \
  --model gpt-4o \
  --max-context-tokens 6000 \
  --out artifacts/open_scenarios/runs_broad_v1

# Evidence report is generated automatically:
# artifacts/open_scenarios/runs_broad_v1_evidence/publication_evidence_report.md
```

Expected runtime: 45–90 minutes depending on OpenAI API latency.

---

## What the evaluation produces

Each case generates a deterministic set of artifacts:

| Artifact | What it records |
|---|---|
| `context.json` | ContextView: token counts, marker resolution, lineage |
| `citation_contract.json` | Publication gate decision: passed / blocked / abstention |
| `zoom.json` | Evidence block resolved from a selected marker |
| `zoom_out.json` | L1 abstraction with summary witness lineage |
| `cleanup.json` | Run-level purge confirmation |
| `prompt_trace.json` | Prompt construction trace with marker anchoring |
| `llm_calls.jsonl` | Raw LLM calls and responses |

Generate the claim matrix from existing artifacts:

```bash
python -m cima_demo.demo.publication.audit \
  --runs artifacts/open_scenarios/runs_broad_v1 \
  --out  artifacts/open_scenarios/runs_broad_v1_evidence
```

---

## Architecture overview

```
Sources (documents, conversations, structured data)
    │ Ingest
    ▼
Total memory  (C-Items · source spans · L1 summaries · lineage links)
    │ Select — budget-bounded, policy-driven
    ▼
ContextView  [S1] [S2] [S3] …  (each marker → stored source span)
    │ Prompt
    ▼
LLM inference  (any OpenAI-compatible backend)
    │ Raw output
    ▼
Publication gate  (citation contract + citation sanitizer)
    │
    ▼
Published answer  [S1][S4] …  or  [INSUFFICIENT_EVIDENCE]

◁── Zoom: any marker resolves to its literal source span ──▷
```

Full architecture diagram and paper: [doi.org/10.5281/zenodo.20139206](https://doi.org/10.5281/zenodo.20139206)

---

## Repository structure

```
cima_demo/
├── api/                FastAPI application and settings
├── cognitive/          Cognitive phase kernel and policy
├── demo/
│   ├── context/        ContextView construction and marker registry
│   ├── open_scenarios/ Dataset download, normalize, execute, audit
│   ├── publication/    Evidence report generation
│   └── runtime/        Citation sanitizer, prompt trace, journal
├── domain/             Core domain entities and ports
├── memory/             Ingestion, summarization, lifecycle
├── retrieval/          Context builder, multi-hop, query planner
└── witness_backend/    Ephemeral in-memory store (standalone mode)

scripts/
├── run_demo.sh         Start server in standalone mode
└── broad_launch.sh     Run full evaluation across all datasets
```

---

## Configuration

All public settings use the `CIMA_DEMO_` prefix and can be set via environment variables or a `.env` file. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `CIMA_DEMO_RUNTIME_MODE` | `standalone` | `standalone` (in-memory) or `full` (Postgres+Qdrant) |
| `CIMA_DEMO_STANDALONE_LLM_BACKEND` | `rule` | Standalone backend: `rule`, `openai`, or `llamacpp` |
| `CIMA_DEMO_OPENAI_API_KEY` | — | Required only when using the `openai` backend |
| `CIMA_DEMO_LLM_PROVIDER` | `llamacpp` | Full-runtime LLM provider: `openai` or `llamacpp` |
| `CIMA_DEMO_LLM_MODEL` | `mistral` | Model ID used by the selected real LLM backend |
| `CIMA_DEMO_PORT` | `8000` | Server port |

For a local llama.cpp backend:

```bash
export CIMA_DEMO_STANDALONE_LLM_BACKEND=llamacpp
export CIMA_DEMO_LLM_PROVIDER=llamacpp
export CIMA_DEMO_LLM_URL=http://localhost:8080
export CIMA_DEMO_LLM_MODEL=mistral
export CIMA_DEMO_LLM_TIMEOUT=3600
./scripts/run_demo.sh
```

---

## Citation

```bibtex
@misc{fuentes2026cima,
  title     = {{CIMA}: A Specification for Governed Memory and Verifiable Context Construction in LLM Systems},
  author    = {Fuentes, Alberto},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20139206},
  url       = {https://doi.org/10.5281/zenodo.20139206}
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## License

Code: MIT — see `LICENSE`.  
Paper and specification documents: CC BY 4.0.
