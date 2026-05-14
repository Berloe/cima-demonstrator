from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cima_demo.demo.runtime import c3_sanitize


@dataclass(slots=True)
class RunnerConfig:
    base_url: str
    api_key: str | None
    model: str
    mode: str
    max_context_tokens: int
    reserve_output_tokens: int
    settle_seconds: float
    cleanup: bool
    request_timeout_seconds: float = 3600.0
    chat_timeout_seconds: float = 3600.0
    cleanup_timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 30.0


class OpenScenarioExecutor:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        headers: dict[str, str] = {}
        if config.api_key:
            headers["x-api-key"] = config.api_key
        self._base_url = config.base_url.rstrip("/")
        self._headers = headers
        self._client = self._new_client()

    def _timeout(self, *, read_seconds: float | None = None) -> httpx.Timeout:
        read = float(read_seconds if read_seconds is not None else self.config.request_timeout_seconds)
        connect = min(float(self.config.connect_timeout_seconds), read)
        # Open-scenario runs may use a local llama.cpp server on slow hardware.
        # Non-streaming /v1/chat/completions does not send bytes until the whole
        # generation is ready, so httpx's read timeout must be sized for model
        # inference, not for ordinary API latency.
        return httpx.Timeout(read, connect=connect, read=read, write=read, pool=connect)

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout(),
            headers=self._headers,
        )

    async def _reset_client(self) -> None:
        try:
            await self._client.aclose()
        finally:
            self._client = self._new_client()

    async def close(self) -> None:
        await self._client.aclose()

    async def _post_first_json(self, candidates: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        errors: list[str] = []
        for path, payload in candidates:
            resp = await self._client.post(path, json=payload)
            if 200 <= resp.status_code < 300:
                return resp.json()
            errors.append(f"{path}: HTTP {resp.status_code} {resp.text[:240]}")
            if resp.status_code not in {404, 405, 422}:
                resp.raise_for_status()
        raise RuntimeError("No compatible CIMA endpoint accepted the request: " + " | ".join(errors))

    async def _create_conversation(self, *, metadata: dict[str, Any]) -> dict[str, Any]:
        external_conversation_id = f"open-scenario-{uuid.uuid4()}"
        payload = await self._post_first_json([
            (
                "/cima/v1/conversations/upsert",
                {
                    "external_system": "open_scenarios",
                    "external_conversation_id": external_conversation_id,
                    "metadata": metadata,
                },
            ),
            ("/cima_demo/conversations", {"metadata": metadata}),
        ])
        conversation_id = payload.get("conversation_id") or payload.get("id")
        if not conversation_id:
            raise RuntimeError(f"Conversation creation response lacks conversation_id: {payload}")
        return {**payload, "conversation_id": str(conversation_id)}

    async def _register_document(self, *, conversation_id: str, document: dict[str, Any], request_id: str) -> dict[str, Any]:
        resp = await self._client.post(
            "/cima/v1/sources/register_text",
            json={
                "conversation_id": conversation_id,
                "text": document["text"],
                "source_kind": "file_text",
                "displayable": False,
                "processable": True,
                "request_id": request_id,
                "external_provider": "open_scenarios",
                "external_message_id": document["doc_id"],
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def _context_get(self, *, conversation_id: str, prompt: str, request_id: str) -> dict[str, Any]:
        payload = {
            "conversation_id": conversation_id,
            "request_id": request_id,
            "user_text": prompt,
            "query": prompt,
            "mode": "chat",
            "selected_artifact_ids": [],
            "selected_scope": "conversation",
            "max_context_tokens": self.config.max_context_tokens,
            "reserve_output_tokens": self.config.reserve_output_tokens,
            "tokenizer_id": "llama.cpp",
            "model_id": self.config.model,
        }
        return await self._post_first_json([
            ("/cima/v1/context/get", payload),
            ("/cima_demo/context/get", payload),
        ])

    async def _zoom(self, *, context_id: str, markers: list[str]) -> dict[str, Any]:
        payload = {
            "context_id": context_id,
            "zoom_targets": markers[:1],
            "max_evidence_tokens": 1200,
        }
        return await self._post_first_json([
            ("/cima/v1/context/zoom", payload),
            ("/cima_demo/context/zoom", payload),
        ])

    async def _zoom_out(self, *, context_id: str, markers: list[str]) -> dict[str, Any]:
        payload = {
            "context_id": context_id,
            "targets": markers[:1] or ["MASTER"],
            "max_perspective_tokens": 1200,
        }
        return await self._post_first_json([
            ("/cima/v1/context/zoom_out", payload),
            ("/cima_demo/context/zoom_out", payload),
        ])

    async def _chat(self, *, conversation_id: str, prompt: str) -> dict[str, Any]:
        resp = await self._client.post(
            "/v1/chat/completions",
            headers={"X-Conversation-Id": conversation_id},
            timeout=self._timeout(read_seconds=self.config.chat_timeout_seconds),
            json={
                "model": self.config.model,
                "stream": False,
                "conversation_id": conversation_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_context_tokens": self.config.max_context_tokens,
                "reserve_output_tokens": self.config.reserve_output_tokens,
                "max_tokens": self.config.reserve_output_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()


    async def _load_prompt_trace(self, conversation_id: str) -> dict[str, Any]:
        path = f"/cima_demo/runs/conversations/{conversation_id}/prompt-trace"
        try:
            resp = await self._client.get(path, timeout=self._timeout(read_seconds=min(self.config.request_timeout_seconds, 30.0)))
        except Exception as exc:
            return {
                "prompt_trace_available": False,
                "error_class": type(exc).__name__,
                "error": str(exc),
            }
        if 200 <= resp.status_code < 300:
            payload = resp.json()
            if isinstance(payload, dict):
                return payload
        return {
            "prompt_trace_available": False,
            "status_code": int(resp.status_code),
            "error": resp.text[:500],
        }

    async def _load_gc_audits(self, conversation_id: str) -> list[dict[str, Any]]:
        candidates = [
            f"/cima_demo/runs/conversations/{conversation_id}/gc-audits",
        ]
        for path in candidates:
            try:
                resp = await self._client.get(path, timeout=self._timeout(read_seconds=min(self.config.cleanup_timeout_seconds, 30.0)))
            except Exception:
                continue
            if 200 <= resp.status_code < 300:
                payload = resp.json()
                audits = payload.get("gc_audits") if isinstance(payload, dict) else None
                if isinstance(audits, list):
                    return [dict(v) for v in audits if isinstance(v, dict)]
        return []

    @staticmethod
    def _cleanup_audit_ok(audit: dict[str, Any] | None) -> bool:
        if not isinstance(audit, dict):
            return False
        consistency = audit.get("consistency") if isinstance(audit.get("consistency"), dict) else {}
        return bool(
            audit.get("status") == "ok"
            and consistency.get("cleanup_ok") is True
            and consistency.get("conversation_deleted") is True
            and consistency.get("qdrant_zeroed") is True
        )

    async def _wait_for_cleanup_audit(self, conversation_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not hasattr(self._client, "get"):
            return None, []
        deadline = asyncio.get_running_loop().time() + float(self.config.cleanup_timeout_seconds)
        audits: list[dict[str, Any]] = []
        while True:
            audits = await self._load_gc_audits(conversation_id)
            for audit in reversed(audits):
                if self._cleanup_audit_ok(audit):
                    return audit, audits
            if asyncio.get_running_loop().time() >= deadline:
                return (audits[-1] if audits else None), audits
            await asyncio.sleep(0.25)

    async def _delete_conversation_once(self, conversation_id: str) -> tuple[bool, list[dict[str, Any]], str | None]:
        attempts: list[dict[str, Any]] = []
        accepted_path: str | None = None
        for path in (f"/cima/v1/conversations/{conversation_id}?purge=true", f"/cima_demo/conversations/{conversation_id}"):
            try:
                resp = await self._client.delete(path, timeout=self._timeout(read_seconds=self.config.cleanup_timeout_seconds))
            except httpx.HTTPError as exc:
                attempts.append({"path": path, "ok": False, "error_class": type(exc).__name__, "error": str(exc)})
                await self._reset_client()
                continue
            except Exception as exc:  # pragma: no cover - defensive cleanup guard
                attempts.append({"path": path, "ok": False, "error_class": type(exc).__name__, "error": str(exc)})
                continue
            status_code = int(resp.status_code)
            # Publication cleanup uses Option A: synchronous audited delete.
            # HTTP 202 only proves async acceptance, so it is recorded but never
            # accepted as cleanup evidence.  A verified first delete may be
            # followed by 404 on the idempotence probe.
            ok = status_code in {200, 204, 404}
            attempts.append({"path": path, "ok": ok, "status_code": status_code})
            if status_code in {200, 204, 404}:
                accepted_path = path
                return True, attempts, accepted_path
            if status_code == 202:
                continue
            if status_code not in {405, 422}:
                continue
        return False, attempts, accepted_path

    async def _delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Cleanup with final-state verification, not just HTTP acceptance.

        A cleanup claim is valid only if the service records a GC audit proving
        that relational state, Qdrant state and conversation rows converged to
        zero for this conversation.  A second delete is recorded as an
        idempotence probe; 404 after verified deletion is acceptable.
        """
        accepted, attempts, accepted_path = await self._delete_conversation_once(conversation_id)
        if not accepted:
            return {
                "requested": True,
                "ok": False,
                "conversation_id": conversation_id,
                "attempts": attempts,
                "accepted_path": accepted_path,
                "final_verified": False,
            }
        final_audit, all_audits = await self._wait_for_cleanup_audit(conversation_id)
        final_verified = self._cleanup_audit_ok(final_audit)
        second_ok = False
        second_attempts: list[dict[str, Any]] = []
        if final_verified:
            second_ok, second_attempts, _ = await self._delete_conversation_once(conversation_id)
        return {
            "requested": True,
            "ok": bool(final_verified and (second_ok or not second_attempts)),
            "conversation_id": conversation_id,
            "attempts": attempts,
            "accepted_path": accepted_path,
            "final_verified": final_verified,
            "final_audit": final_audit or {},
            "gc_audit_count": len(all_audits),
            "idempotence": {
                "checked": bool(final_verified),
                "ok": bool(second_ok) if final_verified else None,
                "attempts": second_attempts,
            },
        }

    async def run_case(self, case: dict[str, Any], *, out_root: Path) -> dict[str, Any]:
        case_id = str(case["case_id"])
        case_root = out_root / case_id.replace("::", "__")
        case_root.mkdir(parents=True, exist_ok=True)
        request_id = str(uuid.uuid4())
        conversation = await self._create_conversation(metadata={"case_id": case_id, "dataset_id": case.get("dataset_id")})
        conversation_id = str(conversation["conversation_id"])
        register_results = []
        try:
            for document in case.get("documents", []):
                register_results.append(
                    await self._register_document(conversation_id=conversation_id, document=document, request_id=request_id)
                )
            if self.config.settle_seconds > 0:
                await asyncio.sleep(self.config.settle_seconds)
            context_payload = None
            zoom_payload = None
            zoom_out_payload = None
            # Keep the explicit /context/get + /zoom + /zoom_out navigation
            # artifacts separate from the ContextView used inside the chat
            # runtime.  The chat runtime is authoritative for the citation
            # contract, but the preflight navigation artifacts are still the
            # evidence for C4/C5 navigation tests and publication metrics.
            preflight_context_payload = None
            preflight_zoom_payload = None
            preflight_zoom_out_payload = None
            citation_context_payload = None
            citation_zoom_payload = None
            citation_zoom_out_payload = None
            navigation_contract = {"checked": False, "passed": None}
            chat_payload = None
            if self.config.mode in {"context", "both"}:
                preflight_context_payload = await self._context_get(
                    conversation_id=conversation_id,
                    prompt=str(case["prompt"]),
                    request_id=request_id,
                )
                context_payload = preflight_context_payload
                citation_context_payload = preflight_context_payload
                (case_root / "context.json").write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                (case_root / "preflight_context.json").write_text(json.dumps(preflight_context_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                context_id = str(preflight_context_payload.get("context_id") or "")
                markers = [str(v) for v in preflight_context_payload.get("markers", []) if str(v)]
                if context_id and markers:
                    preflight_zoom_payload = await self._zoom(context_id=context_id, markers=markers)
                    preflight_zoom_out_payload = await self._zoom_out(context_id=context_id, markers=markers)
                    zoom_payload = preflight_zoom_payload
                    zoom_out_payload = preflight_zoom_out_payload
                    (case_root / "zoom.json").write_text(json.dumps(preflight_zoom_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    (case_root / "zoom_out.json").write_text(json.dumps(preflight_zoom_out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    (case_root / "preflight_zoom.json").write_text(json.dumps(preflight_zoom_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    (case_root / "preflight_zoom_out.json").write_text(json.dumps(preflight_zoom_out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    navigation_contract = _navigation_contract_from_artifacts(zoom_payload=zoom_payload, zoom_out_payload=zoom_out_payload)
                    navigation_contract["source"] = "preflight_navigation"
                    (case_root / "navigation_contract.json").write_text(json.dumps(navigation_contract, ensure_ascii=False, indent=2), encoding="utf-8")
            chat_error: dict[str, Any] | None = None
            if self.config.mode in {"chat", "both"}:
                try:
                    chat_payload = await self._chat(conversation_id=conversation_id, prompt=str(case["prompt"]))
                except Exception as exc:
                    chat_error = _serialize_chat_error(exc)
                    (case_root / "chat_error.json").write_text(json.dumps(chat_error, ensure_ascii=False, indent=2), encoding="utf-8")
            prompt_trace_payload: dict[str, Any] = {"prompt_trace_available": False}
            if self.config.mode in {"chat", "both"}:
                prompt_trace_payload = await self._load_prompt_trace(conversation_id)
                (case_root / "prompt_trace.json").write_text(json.dumps(prompt_trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                llm_calls = prompt_trace_payload.get("llm_calls") if isinstance(prompt_trace_payload, dict) else None
                if isinstance(llm_calls, list) and llm_calls:
                    with (case_root / "llm_calls.jsonl").open("w", encoding="utf-8") as fh:
                        for row in llm_calls:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                prompt_lint = prompt_trace_payload.get("prompt_lint") if isinstance(prompt_trace_payload, dict) else None
                if isinstance(prompt_lint, dict) and prompt_lint:
                    (case_root / "prompt_lint.json").write_text(json.dumps(prompt_lint, ensure_ascii=False, indent=2), encoding="utf-8")
            runtime_artifacts = _runtime_artifacts_from_prompt_trace(prompt_trace_payload)
            if runtime_artifacts:
                runtime_context_payload = _context_payload_from_runtime_snapshot(runtime_artifacts.get("context"))
                if runtime_context_payload is not None:
                    citation_context_payload = runtime_context_payload
                    context_payload = runtime_context_payload
                    (case_root / "context.json").write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    (case_root / "runtime_context.json").write_text(json.dumps(runtime_context_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                runtime_zoom = runtime_artifacts.get("zoom")
                if isinstance(runtime_zoom, dict):
                    citation_zoom_payload = runtime_zoom
                    (case_root / "runtime_zoom.json").write_text(json.dumps(runtime_zoom, ensure_ascii=False, indent=2), encoding="utf-8")
                    # Only use runtime zoom as the public zoom artifact when no
                    # explicit preflight zoom was executed.  Otherwise preserve
                    # zoom.json as the independent navigation proof.
                    if preflight_zoom_payload is None:
                        zoom_payload = runtime_zoom
                        (case_root / "zoom.json").write_text(json.dumps(runtime_zoom, ensure_ascii=False, indent=2), encoding="utf-8")

                runtime_zoom_out = runtime_artifacts.get("zoom_out")
                if isinstance(runtime_zoom_out, dict):
                    citation_zoom_out_payload = runtime_zoom_out
                    (case_root / "runtime_zoom_out.json").write_text(json.dumps(runtime_zoom_out, ensure_ascii=False, indent=2), encoding="utf-8")
                    if preflight_zoom_out_payload is None:
                        zoom_out_payload = runtime_zoom_out
                        (case_root / "zoom_out.json").write_text(json.dumps(runtime_zoom_out, ensure_ascii=False, indent=2), encoding="utf-8")

                # Navigation is an explicit CIMA capability and should remain
                # backed by explicit preflight /zoom and /zoom_out calls when
                # they were executed.  Citation consistency is checked below
                # against the chat runtime registry, not against these
                # navigation artifacts.
                navigation_contract = _navigation_contract_from_artifacts(zoom_payload=zoom_payload, zoom_out_payload=zoom_out_payload)
                navigation_contract["source"] = (
                    "preflight_navigation"
                    if preflight_zoom_payload is not None or preflight_zoom_out_payload is not None
                    else "runtime_chat_navigation"
                )
                (case_root / "navigation_contract.json").write_text(json.dumps(navigation_contract, ensure_ascii=False, indent=2), encoding="utf-8")

            raw_chat_payload = _clone_jsonable(chat_payload) if chat_payload is not None else None
            runtime_citation_contract = _runtime_citation_contract_from_prompt_trace(prompt_trace_payload)
            citation_contract = _citation_contract_from_artifacts(
                chat_payload=chat_payload,
                context_payload=citation_context_payload or context_payload,
                zoom_payload=citation_zoom_payload,
                zoom_out_payload=citation_zoom_out_payload,
                runtime_citation_contract=runtime_citation_contract,
                prompt_trace_payload=prompt_trace_payload,
            )
            if chat_payload is not None and citation_contract.get("published_answer_text") is not None:
                published_answer = str(citation_contract.get("published_answer_text") or "")
                raw_answer = _extract_answer_text(chat_payload)
                if published_answer != raw_answer:
                    (case_root / "chat_raw.json").write_text(
                        json.dumps(raw_chat_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    chat_payload = _chat_with_answer(chat_payload, published_answer)
                (case_root / "chat.json").write_text(json.dumps(chat_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            generation_contract = _generation_contract_from_artifacts(chat_payload=chat_payload, chat_error=chat_error)
            (case_root / "generation_contract.json").write_text(
                json.dumps(generation_contract, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if not generation_contract.get("passed"):
                failure_kind = str(generation_contract.get("failure_kind") or "generation_failed")
                publication_gate = c3_sanitize.build_publication_gate(
                    raw_answer=_extract_answer_text(raw_chat_payload or chat_payload),
                    published_answer=_extract_answer_text(chat_payload),
                    published_integrity_passed=False,
                    c3_published_report=(citation_contract.get("c3_published") or {}) if isinstance(citation_contract, dict) else {},
                    c3a_abstention_report=(citation_contract.get("c3a_traceable_abstention") or {}) if isinstance(citation_contract, dict) else {},
                    generation_passed=False,
                    generation_failure_kind=failure_kind,
                    factual_citations_required=False,
                    sanitization_applied=bool((citation_contract or {}).get("deterministic_sanitization_applied")),
                )
                citation_contract = {
                    **citation_contract,
                    "requires_citations": False,
                    "passed": False,
                    "published_integrity_passed": False,
                    "publication_gate": publication_gate,
                    "publication_status": publication_gate["publication_status"],
                    "publishable": publication_gate["publishable"],
                    "blocked_by_cima": publication_gate["blocked_by_cima"],
                    "blocked_reason": publication_gate["blocked_reason"],
                    "invalid_published_as_valid": publication_gate["invalid_published_as_valid"],
                    "not_applicable_reason": failure_kind,
                }
            (case_root / "citation_contract.json").write_text(
                json.dumps(citation_contract, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cleanup_result: dict[str, Any] = {"requested": False, "ok": None}
            if self.config.cleanup:
                cleanup_result = await self._delete_conversation(conversation_id)
                (case_root / "cleanup.json").write_text(json.dumps(cleanup_result, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest = {
                "case_id": case_id,
                "conversation_id": conversation_id,
                "dataset_id": case.get("dataset_id"),
                "task_family": case.get("task_family"),
                "prompt": case.get("prompt"),
                "gold_answers": case.get("gold_answers", []),
                "gold_label": case.get("gold_label"),
                "gold_evidence_count": len(case.get("gold_evidence", [])),
                "register_results": register_results,
                "generation_contract": generation_contract,
                "citation_contract": citation_contract,
                "navigation_contract": navigation_contract,
                "prompt_trace": {
                    "available": bool(prompt_trace_payload.get("prompt_trace_available")) if isinstance(prompt_trace_payload, dict) else False,
                    "run_id": prompt_trace_payload.get("run_id") if isinstance(prompt_trace_payload, dict) else None,
                    "llm_call_count": len(prompt_trace_payload.get("llm_calls") or []) if isinstance(prompt_trace_payload, dict) else 0,
                    "prompt_lint_passed": ((prompt_trace_payload.get("prompt_lint") or {}).get("passed") if isinstance(prompt_trace_payload, dict) and isinstance(prompt_trace_payload.get("prompt_lint"), dict) else None),
                },
                "cleanup": cleanup_result,
                "artifacts": {
                    "context": "context.json" if context_payload is not None else None,
                    "preflight_context": "preflight_context.json" if (case_root / "preflight_context.json").exists() else None,
                    "runtime_context": "runtime_context.json" if (case_root / "runtime_context.json").exists() else None,
                    "zoom": "zoom.json" if zoom_payload is not None else None,
                    "zoom_out": "zoom_out.json" if zoom_out_payload is not None else None,
                    "runtime_zoom": "runtime_zoom.json" if (case_root / "runtime_zoom.json").exists() else None,
                    "runtime_zoom_out": "runtime_zoom_out.json" if (case_root / "runtime_zoom_out.json").exists() else None,
                    "artifact_semantics": {
                        "context": "runtime_context_view_used_by_prompt" if (case_root / "runtime_context.json").exists() else "preflight_context_projection",
                        "zoom": "preflight_navigation_evidence_projection" if preflight_zoom_payload is not None else ("runtime_zoom_evidence_used_by_prompt" if (case_root / "runtime_zoom.json").exists() else None),
                        "zoom_out": "preflight_navigation_perspective_projection" if preflight_zoom_out_payload is not None else ("runtime_zoom_out_used_by_prompt" if (case_root / "runtime_zoom_out.json").exists() else None),
                        "runtime_context": "runtime_context_view_used_by_prompt" if (case_root / "runtime_context.json").exists() else None,
                        "runtime_zoom": "runtime_zoom_evidence_used_by_prompt" if (case_root / "runtime_zoom.json").exists() else None,
                        "runtime_zoom_out": "runtime_zoom_out_used_by_prompt" if (case_root / "runtime_zoom_out.json").exists() else None,
                    },
                    "navigation_contract": "navigation_contract.json" if navigation_contract.get("checked") else None,
                    "chat": "chat.json" if chat_payload is not None else None,
                    "chat_raw": "chat_raw.json" if (case_root / "chat_raw.json").exists() else None,
                    "chat_error": "chat_error.json" if chat_error is not None else None,
                    "generation_contract": "generation_contract.json",
                    "citation_contract": "citation_contract.json",
                    "prompt_trace": "prompt_trace.json" if (case_root / "prompt_trace.json").exists() else None,
                    "llm_calls": "llm_calls.jsonl" if (case_root / "llm_calls.jsonl").exists() else None,
                    "prompt_lint": "prompt_lint.json" if (case_root / "prompt_lint.json").exists() else None,
                    "cleanup": "cleanup.json" if self.config.cleanup else None,
                },
            }
            (case_root / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
            (case_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return manifest
        except Exception:
            if self.config.cleanup:
                cleanup_result = await self._delete_conversation(conversation_id)
                (case_root / "cleanup.json").write_text(json.dumps(cleanup_result, ensure_ascii=False, indent=2), encoding="utf-8")
            raise


def _iter_case_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def _load_cases(path: Path) -> list[dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _extract_answer_text(chat_payload: dict[str, Any] | None) -> str:
    if not isinstance(chat_payload, dict):
        return ""
    parts: list[str] = []
    for choice in chat_payload.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            parts.append(delta["content"])
    if parts:
        return "".join(parts).strip()
    return str(chat_payload.get("content") or "").strip()


_MARKER_GROUP_RE = re.compile(r"\[((?:[A-Za-z]\d+)(?:\s*,\s*[A-Za-z]\d+)*)\]")


def _extract_markers(text: str) -> list[str]:
    markers: list[str] = []
    for match in _MARKER_GROUP_RE.finditer(text or ""):
        for part in match.group(1).split(","):
            marker = part.strip()
            if marker:
                markers.append(marker)
    return list(dict.fromkeys(markers))


def _collect_context_markers(value: Any) -> set[str]:
    markers: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            markers.update(_extract_markers(node))
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            marker = node.get("marker")
            if marker:
                markers.add(str(marker))
            for child in node.values():
                walk(child)

    walk(value)
    return markers


def _marker_resolution_by_marker(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key in ("marker_resolution", "zoom_out_marker_resolution"):
        for row in payload.get(key) or []:
            if isinstance(row, dict) and str(row.get("marker") or ""):
                out[str(row["marker"])] = dict(row)
    return out


def _zoom_out_lineage_ok(zoom_out_payload: dict[str, Any] | None) -> bool:
    if not isinstance(zoom_out_payload, dict):
        return False
    markers = [str(v) for v in zoom_out_payload.get("markers_added") or [] if str(v)]
    if not markers or not str(zoom_out_payload.get("perspective_block") or "").strip():
        return False
    if zoom_out_payload.get("summary_lineage_valid") is True:
        return True
    by_marker = _marker_resolution_by_marker(zoom_out_payload)
    for marker in markers:
        row = by_marker.get(marker)
        if not isinstance(row, dict):
            return False
        if str(row.get("ref_kind") or "") not in {"summary", "local_summary", "global_summary"}:
            return False
        if list(row.get("unresolved_ref_ids") or []):
            return False
        if int(row.get("resolved_source_count") or 0) <= 0 or int(row.get("resolved_span_count") or 0) <= 0:
            return False
        if not list(row.get("citem_ids") or []):
            return False
    return True



def _normalize_answer_line(line: str) -> str:
    normalized = re.sub(r"[*_`#]", "", line or "").strip()
    normalized = re.sub(r"^\((note\s*:.*)\)\.?$", r"\1", normalized, flags=re.IGNORECASE).strip()
    return normalized


def _is_visible_scaffolding_or_meta(normalized: str) -> bool:
    if not normalized:
        return False
    if re.fullmatch(r"[-—–_\s]{3,}", normalized):
        return True
    if re.match(
        r"^here\s+is\s+(the\s+)?(corrected|revised|updated)?\s*(summary|answer)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"^corrected\s+(summary|answer)\s*(with\s+valid\s+citations)?\s*:?$",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(r"^note\s*:", normalized, flags=re.IGNORECASE):
        return True
    if re.search(
        r"\b(claims?\s+about|lacked\s+direct\s+contextual\s+support|were\s+omitted|has\s+been\s+omitted|removed\s+because)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    return False

def _is_non_claim_block(block: str, *, position: int) -> bool:
    normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", block or "").strip()
    normalized = _normalize_answer_line(normalized)
    if not normalized:
        return True
    if _is_visible_scaffolding_or_meta(normalized):
        return True
    if _extract_markers(normalized):
        return False
    if len(normalized) < 24:
        return True
    if normalized.endswith(":") and len(normalized) < 160:
        return True
    if len(normalized) < 140 and not re.search(r"[.!?]", normalized):
        return True
    if position <= 2 and len(normalized) < 420 and re.match(
        r"^(this|the|overall,?\s+the)\s+(meeting|document|discussion|source|text)\s+",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if len(normalized) < 420 and re.match(
        r"^(this|the|overall,?\s+the)\s+(meeting|document|discussion|source|text)\s+"
        r"(addressed|explored|discussed|highlighted|covered|focused|identified|summarized)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if position <= 2 and len(normalized) < 160 and re.match(
        r"^here\s+is\s+(the\s+)?(corrected\s+)?(summary|answer)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(no direct evidence cited|evidence was insufficient|insufficient evidence|not enough evidence|no concrete examples|limited data|available evidence is limited|evidence is limited|context is limited|source material is limited|lacked direct contextual support|were omitted|broader\s+[^.!?]{0,80}\s+remain(?:s)?\s+(?:unspecified|unclear|unknown|not covered))\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _answer_blocks(answer_text: str) -> list[str]:
    text = (answer_text or "").strip()
    if not text:
        return []
    raw_blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) <= 1:
            raw_blocks.extend(lines)
        else:
            raw_blocks.extend(lines)
    blocks: list[str] = []
    for index, block in enumerate(raw_blocks):
        normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", block).strip()
        if _is_non_claim_block(normalized, position=index):
            continue
        blocks.append(normalized)
    return blocks


def _serialize_chat_error(exc: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_class": type(exc).__name__,
        "error": str(exc),
    }
    if isinstance(exc, httpx.HTTPStatusError):
        payload["status_code"] = int(exc.response.status_code)
        payload["response_text"] = exc.response.text[:4000]
        payload["url"] = str(exc.request.url)
    elif isinstance(exc, httpx.TimeoutException):
        payload["failure_kind"] = "client_timeout"
    return payload


def _generation_contract_from_artifacts(*, chat_payload: dict[str, Any] | None, chat_error: dict[str, Any] | None) -> dict[str, Any]:
    if chat_error is not None:
        text = (chat_error.get("response_text") or chat_error.get("error") or "").lower()
        if "oai_stream_timeout" in text or "stream_timeout" in text:
            kind = "oai_stream_timeout"
        elif "timeout" in text:
            kind = "llm_or_client_timeout"
        else:
            kind = str(chat_error.get("failure_kind") or "chat_error")
        return {
            "checked": True,
            "passed": False,
            "failure_kind": kind,
            "error": chat_error,
        }
    answer = _extract_answer_text(chat_payload)
    normalized = answer.strip()
    if not normalized:
        return {"checked": True, "passed": False, "failure_kind": "empty_answer", "answer_char_count": 0}
    if normalized in {"[TIMEOUT]", "TIMEOUT"} or normalized.endswith("[TIMEOUT]"):
        return {"checked": True, "passed": False, "failure_kind": "synthetic_timeout_answer", "answer_char_count": len(answer)}
    if normalized.startswith("[ERROR:"):
        return {"checked": True, "passed": False, "failure_kind": "backend_error_answer", "answer_char_count": len(answer)}
    return {"checked": True, "passed": True, "answer_char_count": len(answer)}


def _clone_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False)) if value is not None else None


def _chat_with_answer(chat_payload: dict[str, Any] | None, answer: str) -> dict[str, Any] | None:
    if chat_payload is None:
        return None
    cloned = _clone_jsonable(chat_payload)
    try:
        cloned["choices"][0]["message"]["content"] = answer
    except Exception:
        pass
    return cloned


def _structural_marker_list(payload: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    values = payload.get(key) or []
    if not isinstance(values, list):
        return []
    markers: list[str] = []
    for value in values:
        if isinstance(value, dict):
            marker = value.get("marker")
        else:
            marker = value
        marker_s = str(marker or "")
        if marker_s:
            markers.append(marker_s)
    return list(dict.fromkeys(markers))


def _context_payload_from_runtime_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    budget = snapshot.get("budget") if isinstance(snapshot.get("budget"), dict) else {}
    return {
        "context_id": snapshot.get("context_id"),
        "context_pack": snapshot.get("context_text") or "",
        "markers": [str(m) for m in (snapshot.get("markers") or []) if str(m)],
        "token_usage": {
            "context": budget.get("tokens_used"),
            "available_for_content": budget.get("available_for_content"),
            "overhead": budget.get("overhead_tokens"),
        },
        "resolved_source_ids": snapshot.get("resolved_source_ids") or [],
        "resolved_span_ids": snapshot.get("resolved_span_ids") or [],
        "resolved_source_count": snapshot.get("resolved_source_count") or 0,
        "resolved_span_count": snapshot.get("resolved_span_count") or 0,
        "unresolved_ref_ids": snapshot.get("unresolved_ref_ids") or [],
        "resolution_mode": snapshot.get("resolution_mode"),
        "marker_resolution": snapshot.get("marker_resolution") or [],
        "evidence_marker_registry": snapshot.get("evidence_marker_registry") or [],
        "visible_marker_support": snapshot.get("visible_marker_support") or [],
        "visible_support_metrics": snapshot.get("visible_support_metrics") or {},
        "auxiliary_items": snapshot.get("auxiliary_items") or [],
        "dropped_uncitable_items": snapshot.get("dropped_uncitable_items") or [],
        "context_drop_metrics": snapshot.get("context_drop_metrics") or {},
        "run_id": snapshot.get("run_id"),
        "turn_id": snapshot.get("turn_id"),
        "source": "runtime_chat_context_snapshot",
    }


def _runtime_artifacts_from_prompt_trace(prompt_trace_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(prompt_trace_payload, dict):
        return {}
    artifacts = prompt_trace_payload.get("runtime_artifacts")
    return artifacts if isinstance(artifacts, dict) else {}


def _answer_generation_allowed_markers(prompt_trace_payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(prompt_trace_payload, dict):
        return []
    calls = prompt_trace_payload.get("llm_calls")
    if not isinstance(calls, list):
        return []
    for call in calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("call_kind") or "") != "answer_generation":
            continue
        markers = call.get("allowed_markers")
        if isinstance(markers, list):
            return [str(m) for m in markers if str(m)]
    return []


def _marker_registry_consistency(
    *,
    prompt_trace_payload: dict[str, Any] | None,
    citation_contract: dict[str, Any],
    context_payload: dict[str, Any] | None,
    zoom_payload: dict[str, Any] | None,
    zoom_out_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt_allowed = set(_answer_generation_allowed_markers(prompt_trace_payload))
    contract_allowed = set(str(m) for m in (citation_contract.get("allowed_markers") or []) if str(m))
    reconstructed, reconstructed_source = _allowed_markers_from_artifacts(
        context_payload=context_payload,
        zoom_payload=zoom_payload,
        zoom_out_payload=zoom_out_payload,
    )
    return {
        "schema_version": "cima_demo.marker_registry_consistency.v1",
        "passed": bool(prompt_allowed == contract_allowed == reconstructed),
        "prompt_equals_citation_contract": prompt_allowed == contract_allowed,
        "citation_contract_equals_reconstructed_artifacts": contract_allowed == reconstructed,
        "prompt_equals_reconstructed_artifacts": prompt_allowed == reconstructed,
        "prompt_allowed_markers": sorted(prompt_allowed),
        "citation_contract_allowed_markers": sorted(contract_allowed),
        "reconstructed_allowed_markers": sorted(reconstructed),
        "reconstructed_allowed_markers_source": reconstructed_source,
    }


def _runtime_citation_contract_from_prompt_trace(prompt_trace_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    artifacts = _runtime_artifacts_from_prompt_trace(prompt_trace_payload)
    contract = artifacts.get("citation_contract")
    return contract if isinstance(contract, dict) else None


def _marker_is_citable(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    if list(row.get("unresolved_ref_ids") or []):
        return False
    if int(row.get("resolved_source_count") or 0) <= 0 or int(row.get("resolved_span_count") or 0) <= 0:
        return False
    kind = str(row.get("ref_kind") or row.get("summary_ref_kind") or "").lower()
    if kind in {"summary", "local_summary", "global_summary", "summary_chunk"}:
        citem_ids = [str(v) for v in list(row.get("citem_ids") or []) if str(v)]
        if not citem_ids or list(row.get("unresolved_citem_ids") or []):
            return False
        witnesses = [dict(v) for v in list(row.get("citem_witnesses") or []) if isinstance(v, dict)]
        if not witnesses:
            return False
        by_citem = {str(w.get("citem_id") or ""): w for w in witnesses if str(w.get("citem_id") or "")}
        return all(list((by_citem.get(cid) or {}).get("source_ids") or []) and list((by_citem.get(cid) or {}).get("span_ids") or []) for cid in citem_ids)
    return True


def _citable_markers_from_payload(payload: dict[str, Any] | None, field: str) -> list[str]:
    markers = _structural_marker_list(payload, field)
    if not markers:
        return []
    rows = {
        str(row.get("marker") or ""): row
        for row in ((payload or {}).get("marker_resolution") or [])
        if isinstance(row, dict) and str(row.get("marker") or "")
    }
    if not rows:
        # Backward compatibility for very old artifacts that did not carry
        # marker_resolution. New runtime artifacts must include it.
        return markers
    return [marker for marker in markers if _marker_is_citable(rows.get(marker))]


def _allowed_markers_from_artifacts(
    *,
    context_payload: dict[str, Any] | None,
    zoom_payload: dict[str, Any] | None = None,
    zoom_out_payload: dict[str, Any] | None = None,
) -> tuple[set[str], dict[str, list[str]]]:
    # Closed marker set must be reconstructed from structural marker fields only.
    # Do not recursively scan arbitrary text: evidence/source text can contain
    # bracketed tokens that are not CIMA citation markers.
    context_markers = _citable_markers_from_payload(context_payload, "markers")
    zoom_markers = _citable_markers_from_payload(zoom_payload, "markers_added")
    zoom_out_markers: list[str] = []
    if isinstance(zoom_out_payload, dict) and zoom_out_payload.get("summary_lineage_valid") is True:
        zoom_out_markers = _citable_markers_from_payload(zoom_out_payload, "markers_added")
    allowed = set(context_markers) | set(zoom_markers) | set(zoom_out_markers)
    return allowed, {
        "context_markers": list(dict.fromkeys(context_markers)),
        "zoom_markers": list(dict.fromkeys(zoom_markers)),
        "zoom_out_markers": list(dict.fromkeys(zoom_out_markers)),
        "extra_prompt_markers": [],
    }


def _citation_contract_from_artifacts(
    *,
    chat_payload: dict[str, Any] | None,
    context_payload: dict[str, Any] | None,
    zoom_payload: dict[str, Any] | None = None,
    zoom_out_payload: dict[str, Any] | None = None,
    runtime_citation_contract: dict[str, Any] | None = None,
    prompt_trace_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(runtime_citation_contract, dict) and runtime_citation_contract:
        contract = json.loads(json.dumps(runtime_citation_contract))
        contract["source"] = "runtime_chat_citation_contract"
        contract["marker_registry_consistency"] = _marker_registry_consistency(
            prompt_trace_payload=prompt_trace_payload,
            citation_contract=contract,
            context_payload=context_payload,
            zoom_payload=zoom_payload,
            zoom_out_payload=zoom_out_payload,
        )
        return contract

    raw_answer = _extract_answer_text(chat_payload)
    allowed, allowed_source = _allowed_markers_from_artifacts(
        context_payload=context_payload,
        zoom_payload=zoom_payload,
        zoom_out_payload=zoom_out_payload,
    )
    raw_result = c3_sanitize.sanitize(raw_answer, allowed)
    published_answer = raw_result.sanitized_answer
    published_result = c3_sanitize.sanitize(published_answer, allowed)

    c3a_abstention = c3_sanitize.build_c3a_abstention_report(
        answer=published_answer,
        allowed_markers=allowed,
        context_view_id=str((context_payload or {}).get("context_id") or "") or None,
        inspected_markers=list(dict.fromkeys(allowed_source.get("context_markers", []) + allowed_source.get("zoom_markers", []) + allowed_source.get("zoom_out_markers", []))),
        zoom_attempted=bool(zoom_payload),
        zoom_out_attempted=bool(zoom_out_payload),
        extra_trace={
            "context_marker_count": len(allowed_source.get("context_markers", [])),
            "zoom_marker_count": len(allowed_source.get("zoom_markers", [])),
            "zoom_out_marker_count": len(allowed_source.get("zoom_out_markers", [])),
        },
    )
    answer_type = str(c3a_abstention.get("answer_type") or "factual_answer")
    factual_citations_required = answer_type != "insufficient_evidence"

    cited = _extract_markers(published_answer)
    valid = [marker for marker in cited if marker in allowed]
    invalid = [marker for marker in cited if marker not in allowed]
    blocks = _answer_blocks(published_answer)
    uncited_blocks = [block for block in blocks if not any(marker in allowed for marker in _extract_markers(block))]
    requires = bool(allowed) and bool(published_answer.strip()) and factual_citations_required
    c3_normal_passed = (not requires) or (bool(valid) and not invalid and not uncited_blocks and bool(published_result.passed))
    published_integrity_passed = bool(c3_normal_passed or c3a_abstention.get("passed") is True)
    sanitization_applied = published_answer != raw_answer
    sanitization_reports = []
    if sanitization_applied or raw_result.report.get("deterministic_sanitization_applied"):
        sanitization_reports.append(raw_result.report)
    publication_gate = c3_sanitize.build_publication_gate(
        raw_answer=raw_answer,
        published_answer=published_answer,
        published_integrity_passed=published_integrity_passed,
        c3_published_report=published_result.report,
        c3a_abstention_report=c3a_abstention,
        generation_passed=bool(raw_answer.strip()),
        generation_failure_kind="empty_generation" if not raw_answer.strip() else None,
        factual_citations_required=factual_citations_required,
        sanitization_applied=bool(sanitization_reports),
    )
    return {
        "schema_version": "cima_demo.citation_contract.v2",
        "requires_citations": requires,
        "factual_citations_required": factual_citations_required,
        "answer_type": answer_type,
        "passed": publication_gate["publishable"],
        "legacy_marker_citation_passed": c3_normal_passed,
        "published_integrity_passed": published_integrity_passed,
        "publication_gate": publication_gate,
        "publication_status": publication_gate["publication_status"],
        "publishable": publication_gate["publishable"],
        "blocked_by_cima": publication_gate["blocked_by_cima"],
        "blocked_reason": publication_gate["blocked_reason"],
        "invalid_published_as_valid": publication_gate["invalid_published_as_valid"],
        "answer_char_count": len(published_answer),
        "available_marker_count": len(allowed),
        "allowed_markers": sorted(allowed),
        "allowed_markers_source": allowed_source,
        "raw_model_answer_text": raw_answer,
        "published_answer_text": published_answer,
        "cited_markers": cited,
        "valid_cited_markers": list(dict.fromkeys(valid)),
        "invalid_cited_markers": list(dict.fromkeys(invalid)),
        "raw_cited_markers": raw_result.report.get("raw_cited_markers", []),
        "valid_cited_markers_raw": raw_result.report.get("valid_cited_markers_raw", []),
        "invalid_cited_markers_raw": raw_result.report.get("invalid_cited_markers_raw", []),
        "deterministic_sanitization_applied": bool(sanitization_reports),
        "citation_sanitization_reports": sanitization_reports,
        "c3_raw_model": raw_result.report,
        "c3_published": published_result.report,
        "c3a_traceable_abstention": c3a_abstention,
        "source": "reconstructed_from_exported_artifacts",
        "marker_registry_consistency": _marker_registry_consistency(
            prompt_trace_payload=prompt_trace_payload,
            citation_contract={"allowed_markers": sorted(allowed)},
            context_payload=context_payload,
            zoom_payload=zoom_payload,
            zoom_out_payload=zoom_out_payload,
        ),
        "answer_block_count": len(blocks),
        "uncited_answer_block_count": len(uncited_blocks) if factual_citations_required else 0,
        "uncited_answer_block_preview": [block[:180] for block in uncited_blocks[:5]] if factual_citations_required else [],
    }


def _navigation_contract_from_artifacts(*, zoom_payload: dict[str, Any] | None, zoom_out_payload: dict[str, Any] | None) -> dict[str, Any]:
    zoom_ok = bool(
        zoom_payload
        and str(zoom_payload.get("resolution_mode") or "empty") != "empty"
        and int(zoom_payload.get("resolved_source_count") or 0) > 0
        and int(zoom_payload.get("resolved_span_count") or 0) > 0
        and not list(zoom_payload.get("unresolved_ref_ids") or [])
    )
    zoom_out_lineage_ok = _zoom_out_lineage_ok(zoom_out_payload)
    zoom_out_ok = bool(
        zoom_out_payload
        and str(zoom_out_payload.get("resolution_mode") or "empty") != "empty"
        and str(zoom_out_payload.get("perspective_block") or "").strip()
        and zoom_out_lineage_ok
    )
    return {
        "checked": bool(zoom_payload or zoom_out_payload),
        "passed": bool(zoom_ok and zoom_out_ok),
        "zoom_passed": zoom_ok,
        "zoom_out_passed": zoom_out_ok,
        "zoom_resolution_mode": str((zoom_payload or {}).get("resolution_mode") or "empty"),
        "zoom_out_resolution_mode": str((zoom_out_payload or {}).get("resolution_mode") or "empty"),
        "zoom_resolved_source_count": int((zoom_payload or {}).get("resolved_source_count") or 0),
        "zoom_resolved_span_count": int((zoom_payload or {}).get("resolved_span_count") or 0),
        "zoom_out_marker_count": len(list((zoom_out_payload or {}).get("markers_added") or [])),
        "zoom_out_lineage_passed": zoom_out_lineage_ok,
    }


def _case_file_inventory(root: Path) -> dict[str, Any]:
    files = _iter_case_files(root)
    per_file: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}
    empty_files: list[str] = []
    for path in files:
        loaded = _load_cases(path)
        if not loaded:
            empty_files.append(str(path))
        counts: dict[str, int] = {}
        for case in loaded:
            ds = str(case.get("dataset_id") or "unknown")
            counts[ds] = counts.get(ds, 0) + 1
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        per_file.append({"path": str(path), "case_count": len(loaded), "dataset_counts": counts})
    return {"files": per_file, "dataset_case_counts": dataset_counts, "empty_files": empty_files}


def _normalization_expectations(root: Path) -> dict[str, Any]:
    if root.is_file():
        base = root.parent
    else:
        base = root
    index_path = base / "normalization_index.json"
    if not index_path.exists():
        return {"index_path": None, "datasets": [], "skipped": [], "zero_case": []}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"index_path": str(index_path), "error": f"{type(exc).__name__}: {exc}", "datasets": [], "skipped": [], "zero_case": []}
    datasets: list[str] = []
    skipped: list[dict[str, Any]] = []
    zero_case: list[dict[str, Any]] = []
    for row in index.get("results", []):
        dataset_id = str(row.get("dataset_id") or "")
        if not dataset_id:
            continue
        datasets.append(dataset_id)
        case_count = int(row.get("case_count") or 0)
        reason = row.get("skipped_reason")
        if reason:
            skipped.append({"dataset_id": dataset_id, "reason": str(reason)})
        if case_count == 0:
            zero_case.append({"dataset_id": dataset_id, "reason": str(reason or "no cases generated")})
    return {"index_path": str(index_path), "datasets": datasets, "skipped": skipped, "zero_case": zero_case}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute normalized open-scenario cases against a running CIMA API.")
    parser.add_argument("--cases", type=Path, required=True, help="Normalized JSONL file or directory produced by normalize.py")
    parser.add_argument("--out", type=Path, required=True, help="Directory for execution artifacts")
    parser.add_argument("--base-url", required=True, help="Base URL of the running CIMA API")
    parser.add_argument("--api-key", default=None, help="Optional API key for the running CIMA API")
    parser.add_argument("--model", default="cima_demo", help="Model id to send to /v1/chat/completions")
    parser.add_argument("--mode", choices=["context", "chat", "both"], default="both")
    parser.add_argument("--limit", "--max-cases", dest="limit", type=int, default=None, help="Optional max number of cases to execute")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--reserve-output-tokens", type=int, default=768)
    parser.add_argument("--settle-seconds", type=float, default=0.0, help="Optional wait after source registration")
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=3600.0,
        help="Default HTTP timeout for non-generation calls. Large default supports local llama.cpp runs.",
    )
    parser.add_argument(
        "--chat-timeout-seconds",
        type=float,
        default=3600.0,
        help="HTTP read timeout for /v1/chat/completions. Non-streaming local LLM calls can take many minutes.",
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=60.0,
        help="Best-effort cleanup timeout per delete endpoint.",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP connect/pool timeout.",
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete each conversation after execution")
    return parser


async def _run(args: argparse.Namespace) -> int:
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    inventory = _case_file_inventory(args.cases)
    expectations = _normalization_expectations(args.cases)
    files = _iter_case_files(args.cases)
    cases: list[dict[str, Any]] = []
    for path in files:
        cases.extend(_load_cases(path))
    if args.limit is not None:
        cases = cases[: args.limit]
    executor = OpenScenarioExecutor(
        RunnerConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            mode=args.mode,
            max_context_tokens=args.max_context_tokens,
            reserve_output_tokens=args.reserve_output_tokens,
            settle_seconds=args.settle_seconds,
            cleanup=bool(args.cleanup),
            request_timeout_seconds=args.request_timeout_seconds,
            chat_timeout_seconds=args.chat_timeout_seconds,
            cleanup_timeout_seconds=args.cleanup_timeout_seconds,
            connect_timeout_seconds=args.connect_timeout_seconds,
        )
    )
    try:
        manifests = [await executor.run_case(case, out_root=out_root) for case in cases]
    finally:
        await executor.close()
    executed_dataset_counts: dict[str, int] = {}
    for manifest in manifests:
        ds = str(manifest.get("dataset_id") or "unknown")
        executed_dataset_counts[ds] = executed_dataset_counts.get(ds, 0) + 1
    expected_datasets = set(expectations.get("datasets") or [])
    case_datasets = set(inventory.get("dataset_case_counts") or {})
    executed_datasets = set(executed_dataset_counts)
    generation_contract_summary = {
        "checked_runs": sum(1 for manifest in manifests if isinstance(manifest.get("generation_contract"), dict)),
        "passed_runs": sum(1 for manifest in manifests if (manifest.get("generation_contract") or {}).get("passed") is True),
        "failed_runs": sum(1 for manifest in manifests if (manifest.get("generation_contract") or {}).get("passed") is False),
    }
    citation_contract_summary = {
        "checked_runs": sum(1 for manifest in manifests if isinstance(manifest.get("citation_contract"), dict)),
        "published_integrity_passed_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("published_integrity_passed") is True or (manifest.get("citation_contract") or {}).get("passed") is True),
        "published_integrity_failed_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("published_integrity_passed") is False or (manifest.get("citation_contract") or {}).get("passed") is False),
        "publishable_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("publishable") is True),
        "blocked_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("publication_status") == "blocked"),
        "invalid_published_as_valid_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("invalid_published_as_valid") is True),
        "raw_model_passed_runs": sum(1 for manifest in manifests if ((manifest.get("citation_contract") or {}).get("c3_raw_model") or {}).get("raw_model_passed") is True),
        "sanitized_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("deterministic_sanitization_applied") is True),
        "abstention_runs": sum(1 for manifest in manifests if ((manifest.get("citation_contract") or {}).get("c3a_traceable_abstention") or {}).get("applicable") is True),
        "not_applicable_runs": sum(1 for manifest in manifests if (manifest.get("citation_contract") or {}).get("passed") is None),
    }
    navigation_contract_summary = {
        "checked_runs": sum(1 for manifest in manifests if (manifest.get("navigation_contract") or {}).get("checked") is True),
        "passed_runs": sum(1 for manifest in manifests if (manifest.get("navigation_contract") or {}).get("passed") is True),
        "failed_runs": sum(1 for manifest in manifests if (manifest.get("navigation_contract") or {}).get("passed") is False),
    }
    prompt_trace_summary = {
        "available_runs": sum(1 for manifest in manifests if (manifest.get("prompt_trace") or {}).get("available") is True),
        "missing_runs": sum(1 for manifest in manifests if (manifest.get("prompt_trace") or {}).get("available") is not True),
        "prompt_lint_passed_runs": sum(1 for manifest in manifests if (manifest.get("prompt_trace") or {}).get("prompt_lint_passed") is True),
        "prompt_lint_failed_runs": sum(1 for manifest in manifests if (manifest.get("prompt_trace") or {}).get("prompt_lint_passed") is False),
        "llm_call_count": sum(int((manifest.get("prompt_trace") or {}).get("llm_call_count") or 0) for manifest in manifests),
    }
    index = {
        "runs": manifests,
        "case_count": len(manifests),
        "generation_contract_summary": generation_contract_summary,
        "citation_contract_summary": citation_contract_summary,
        "navigation_contract_summary": navigation_contract_summary,
        "prompt_trace_summary": prompt_trace_summary,
        "case_inventory": inventory,
        "normalization_expectations": expectations,
        "executed_dataset_counts": executed_dataset_counts,
        "datasets_with_cases_not_executed": sorted(case_datasets - executed_datasets),
        "expected_datasets_without_cases": sorted(expected_datasets - case_datasets),
        "expected_datasets_not_executed": sorted(expected_datasets - executed_datasets),
        "runner_config": {
            "base_url": args.base_url,
            "model": args.model,
            "mode": args.mode,
            "max_context_tokens": args.max_context_tokens,
            "reserve_output_tokens": args.reserve_output_tokens,
            "settle_seconds": args.settle_seconds,
            "request_timeout_seconds": args.request_timeout_seconds,
            "chat_timeout_seconds": args.chat_timeout_seconds,
            "cleanup_timeout_seconds": args.cleanup_timeout_seconds,
            "connect_timeout_seconds": args.connect_timeout_seconds,
            "cleanup": bool(args.cleanup),
        },
    }
    index_path = out_root / "open_scenario_run_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "index": str(index_path),
        "case_count": len(manifests),
        "executed_dataset_counts": executed_dataset_counts,
        "expected_datasets_not_executed": index["expected_datasets_not_executed"],
        "generation_contract_summary": generation_contract_summary,
        "citation_contract_summary": citation_contract_summary,
        "navigation_contract_summary": navigation_contract_summary,
        "prompt_trace_summary": prompt_trace_summary,
    }, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
