# CIMA — Contextual Item Memory Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20139206.svg)](https://doi.org/10.5281/zenodo.20139206)

CIMA is a memory-and-context architecture for systems that use language models as bounded-context inference engines. Its central property is that every published answer either cites markers that resolve to specific source spans in stored memory, or declares a traceable abstention when evidence is insufficient. The literal source text is always retained, so any abstraction is lossless in the sense that the underlying evidence is always recoverable.

This repository contains the CIMA Demonstrator: a runnable Python implementation of the CIMA-core architecture with a full evaluation harness.

---

## Quick start

```bash
# 1. Install dependencies
pip install poetry
poetry install

# 2. Configure your OpenAI key
cp .env.example .env
# Edit .env and set CIMA_DEMO_OPENAI_API_KEY=sk-...

# 3. Start the server (standalone mode — no external services required)
./scripts/run_demo.sh
# Health check: curl http://localhost:8000/health
```

The server runs in **standalone mode** using in-memory stores. No Postgres, Qdrant, or TEI required.

---

## Reproduce the published evaluation

The published evidence package (230 cases, GPT-4o) is at the Zenodo DOI above. To reproduce from scratch:

```bash
# 1. Download and normalize datasets
python -m cima_demo.demo.open_scenarios.download
python -m cima_demo.demo.open_scenarios.normalize

# 2. Start the server in a separate terminal
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

All settings use the `CIMA_DEMO_` prefix and can be set via environment variables or a `.env` file. See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `CIMA_DEMO_OPENAI_API_KEY` | — | OpenAI API key |
| `CIMA_DEMO_RUNTIME_MODE` | `standalone` | `standalone` (in-memory) or `full` (Postgres+Qdrant) |
| `CIMA_DEMO_LLM_PROVIDER` | `llamacpp` | `openai` or `llamacpp` |
| `CIMA_DEMO_LLM_MODEL` | `mistral` | Model ID |
| `CIMA_DEMO_PORT` | `8000` | Server port |

For a local llama.cpp backend:

```bash
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
  title  = {{CIMA}: Bounded, Traceable, and Navigable Memory for Language Model Systems},
  author = {Fuentes, Alberto},
  year   = {2026},
  doi    = {10.5281/zenodo.20139206},
  url    = {https://doi.org/10.5281/zenodo.20139206}
}
```

---

## License

Code: MIT — see `LICENSE`.  
Paper and specification documents: CC BY 4.0.
