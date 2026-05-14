#!/usr/bin/env bash
# Run a broad open-scenario execution across all available normalized datasets
# and generate the publication evidence report.
#
# Prerequisites:
#   1. OKD port-forwards active:  ./scripts/port_forward_okd.sh &
#   2. CIMA API running:          PYTHONPATH=. ./scripts/run_demo_local.sh &
#   3. llama-server on :8080      (started separately)
#
# Usage:
#   ./scripts/broad_launch.sh [OPTIONS]
#
# Options:
#   --base-url URL         CIMA API base URL        [default: http://localhost:8000]
#   --out DIR              Output root dir           [default: artifacts/open_scenarios/runs_broad_v1]
#   --limit N              Max cases per dataset     [default: all]
#   --mode MODE            context | chat | both     [default: both]
#   --max-context-tokens N Token budget              [default: 4096]
#   --settle-seconds N     Post-registration wait    [default: 2.0]
#   --model NAME           LLM model id              [default: mistral]
#   --skip-audit           Skip evidence report      [default: false]
#   --dry-run              Print commands only        [default: false]
#
# Dataset execution order for the main CIMA traceability evaluation.
# FEVER is intentionally excluded: the current bundle lacks the FEVER wiki-pages
# corpus and FEVER-standard claim-to-evidence retrieval, so claim-only FEVER runs
# are not meaningful evidence for the demonstrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── defaults ──────────────────────────────────────────────────────────────────
BASE_URL="http://localhost:8090"
OUT_ROOT="artifacts/open_scenarios/runs_broad_v1"
LIMIT=""
MODE="both"
MAX_CONTEXT_TOKENS="4096"
SETTLE_SECONDS="2.0"
MODEL="gpt-4o"
SKIP_AUDIT="false"
DRY_RUN="false"

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url)           BASE_URL="$2";           shift 2 ;;
        --out)                OUT_ROOT="$2";            shift 2 ;;
        --limit)              LIMIT="$2";               shift 2 ;;
        --mode)               MODE="$2";                shift 2 ;;
        --max-context-tokens) MAX_CONTEXT_TOKENS="$2";  shift 2 ;;
        --settle-seconds)     SETTLE_SECONDS="$2";      shift 2 ;;
        --model)              MODEL="$2";               shift 2 ;;
        --skip-audit)         SKIP_AUDIT="true";        shift ;;
        --dry-run)            DRY_RUN="true";           shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

CASES_ROOT="artifacts/open_scenarios/cases"
EVIDENCE_OUT="${OUT_ROOT}_evidence"

# Dataset order: short diverse first, long last
DATASETS=(hotpotqa qasper explainmeetsum qmsum)

# ── helpers ────────────────────────────────────────────────────────────────────
log()  { echo "[broad-launch] $*"; }
die()  { echo "[broad-launch] ERROR: $*" >&2; exit 1; }

wait_for_api() {
    local url="$BASE_URL/health"
    local retries=40
    log "Waiting for CIMA API at $url ..."
    for ((i=1; i<=retries; i++)); do
        # Require a valid JSON response with status ok/degraded.
        # Also reject HTML responses (e.g. snap-store-proxy on :8000).
        local body
        body=$(curl -sf --max-time 5 "$url" 2>/dev/null) || true
        if echo "$body" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    assert d.get('status') in ('ok', 'degraded'), 'unexpected status'
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
            log "API ready (status=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null))."
            return 0
        fi
        echo -n "."
        sleep 3
    done
    echo ""
    die "CIMA API did not become healthy after $((retries*3))s. Is it running on $BASE_URL?"
}

run_dataset() {
    local ds="$1"
    local cases_file="$CASES_ROOT/$ds/test.jsonl"
    local out_dir="$OUT_ROOT/$ds"

    if [[ ! -f "$cases_file" ]]; then
        log "SKIP $ds — $cases_file not found"
        return 0
    fi

    local case_count
    case_count=$(wc -l < "$cases_file")
    log "── $ds: $case_count cases → $out_dir"

    local cmd=(
        python -m cima_demo.demo.open_scenarios.execute
        --cases "$cases_file"
        --out   "$out_dir"
        --base-url "$BASE_URL"
        --mode "$MODE"
        --model "$MODEL"
        --max-context-tokens "$MAX_CONTEXT_TOKENS"
        --settle-seconds "$SETTLE_SECONDS"
        --cleanup
    )
    [[ -n "$LIMIT" ]] && cmd+=(--limit "$LIMIT")

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  DRY-RUN: ${cmd[*]}"
        return 0
    fi

    PYTHONPATH="${PYTHONPATH:-.}" "${cmd[@]}"
    log "── $ds done."
}

# ── main ───────────────────────────────────────────────────────────────────────
log "Starting broad launch"
log "  base-url            : $BASE_URL"
log "  output root         : $OUT_ROOT"
log "  limit per dataset   : ${LIMIT:-all}"
log "  mode                : $MODE"
log "  max context tokens  : $MAX_CONTEXT_TOKENS"
log "  settle seconds      : $SETTLE_SECONDS"
log "  model               : $MODEL"

if [[ "$DRY_RUN" != "true" ]]; then
    wait_for_api
fi

START_TS=$(date +%s)

for ds in "${DATASETS[@]}"; do
    run_dataset "$ds"
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
log "All datasets done in ${ELAPSED}s."

# ── publication evidence report ────────────────────────────────────────────────
if [[ "$SKIP_AUDIT" != "true" && "$DRY_RUN" != "true" ]]; then
    log "Generating publication evidence report → $EVIDENCE_OUT"

    # Count actual runs produced
    total_runs=0
    for ds in "${DATASETS[@]}"; do
        ds_dir="$OUT_ROOT/$ds"
        if [[ -d "$ds_dir" ]]; then
            n=$(find "$ds_dir" -name "run_manifest.json" | wc -l)
            total_runs=$((total_runs + n))
            log "  $ds: $n runs"
        fi
    done
    log "Total runs: $total_runs"

    PYTHONPATH="${PYTHONPATH:-.}" python -m cima_demo.demo.publication.audit \
        --runs "$OUT_ROOT" \
        --out  "$EVIDENCE_OUT"

    log "Evidence report: $EVIDENCE_OUT/publication_evidence_report.md"
    log ""
    # Print key rates from JSON
    python3 -c "
import json, pathlib
p = pathlib.Path('$EVIDENCE_OUT/publication_evidence_report.json')
if p.exists():
    d = json.loads(p.read_text())
    print('[broad-launch] Key rates:')
    for k, v in d.get('rates', {}).items():
        print(f'  {k}: {v}')
    eur = d.get('evidence_utilization', {}).get('global', {})
    if eur:
        print(f'  EUR global: mean={eur[\"mean\"]}, min={eur[\"min\"]}, max={eur[\"max\"]}')
"
fi

log "Broad launch complete."
