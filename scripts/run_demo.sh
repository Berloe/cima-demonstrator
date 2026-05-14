#!/usr/bin/env bash
# Start the CIMA Demonstrator API server in standalone mode.
#
# Standalone mode uses in-memory stores — no Postgres, Qdrant, or TEI required.
# The server is ready to accept open-scenario evaluation runs immediately.
#
# Prerequisites:
#   1. poetry install (or pip install -e .)
#   2. OPENAI_API_KEY set in environment or .env file
#
# Usage:
#   ./scripts/run_demo.sh            # start on default port 8000
#   CIMA_DEMO_PORT=8090 ./scripts/run_demo.sh

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

# ── LLM backend ──────────────────────────────────────────────────────────────
export CIMA_DEMO_LLM_PROVIDER="${CIMA_DEMO_LLM_PROVIDER:-openai}"
export CIMA_DEMO_LLM_MODEL="${CIMA_DEMO_LLM_MODEL:-gpt-4o}"
export CIMA_DEMO_LLM_TEMPERATURE="${CIMA_DEMO_LLM_TEMPERATURE:-0}"
export CIMA_DEMO_LLM_TOP_P="${CIMA_DEMO_LLM_TOP_P:-1}"
export CIMA_DEMO_LLM_MAX_TOKENS="${CIMA_DEMO_LLM_MAX_TOKENS:-512}"
export CIMA_DEMO_LLM_TIMEOUT="${CIMA_DEMO_LLM_TIMEOUT:-300}"
export CIMA_DEMO_LLM_MAX_RETRIES="${CIMA_DEMO_LLM_MAX_RETRIES:-2}"

# ── Runtime: standalone (in-memory, no external services) ────────────────────
export CIMA_DEMO_RUNTIME_MODE="standalone"
export CIMA_DEMO_STANDALONE_LLM_BACKEND="openai"
export CIMA_DEMO_DEMO_MODE="true"
export CIMA_DEMO_API_KEY_REQUIRED="false"

# ── Server ────────────────────────────────────────────────────────────────────
export CIMA_DEMO_PORT="${CIMA_DEMO_PORT:-8000}"

echo "[cima] Starting CIMA Demonstrator (standalone, ${CIMA_DEMO_LLM_PROVIDER}/${CIMA_DEMO_LLM_MODEL})"
echo "[cima] Listening on http://localhost:${CIMA_DEMO_PORT}"
echo "[cima] Health check: http://localhost:${CIMA_DEMO_PORT}/health"

python main.py
