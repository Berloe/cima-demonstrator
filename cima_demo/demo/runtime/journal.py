"""Durable run journal for the CIMA Demonstrator."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from cima_demo.demo.contracts import RunBundle, RunCheckpoint, RunManifest, RunPhaseRecord
from cima_demo.domain.ports import RelDBPort


class DemoRunJournal:
    """Persists demonstrator runs to PostgreSQL and filesystem artifacts."""

    def __init__(self, rel_db: RelDBPort, artifacts_root: Path) -> None:
        self._db = rel_db
        self._root = artifacts_root

    def _run_dir(self, conversation_id: str, run_id: str) -> Path:
        return self._root / conversation_id / run_id

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def open_skeleton_run(
        self,
        *,
        run_id: str,
        conversation_id: str,
        turn_id: str,
        user_message: str,
        attached_files: list[dict[str, Any]] | None = None,
    ) -> RunManifest:
        manifest = RunManifest(
            run_id=run_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message=user_message,
            attached_files=attached_files or [],
        )
        await self._db.create_demo_run(
            run_id=run_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            status=manifest.status,
            user_message=user_message,
            manifest_json=manifest.to_dict(),
        )
        run_dir = self._run_dir(conversation_id, run_id)
        self._write_json(run_dir / "run_manifest.json", manifest.to_dict())
        seq = await self.append_phase(
            run_id=run_id,
            conversation_id=conversation_id,
            phase_name="OPEN",
            payload={
                "turn_id": turn_id,
                "attached_files": attached_files or [],
            },
        )
        manifest.phase_count = seq
        self._write_json(run_dir / "run_manifest.json", manifest.to_dict())
        return manifest

    async def append_phase(
        self,
        *,
        run_id: str,
        conversation_id: str,
        phase_name: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        payload = payload or {}
        seq = await self._db.append_demo_run_phase(
            run_id=run_id,
            phase_name=phase_name,
            payload_json=payload,
        )
        phase = RunPhaseRecord(
            run_id=run_id,
            sequence=seq,
            phase_name=phase_name,
            payload=payload,
        )
        self._append_jsonl(
            self._run_dir(conversation_id, run_id) / "run_phases.jsonl",
            phase.to_dict(),
        )
        return seq

    async def checkpoint(
        self,
        *,
        run_id: str,
        conversation_id: str,
        checkpoint_kind: str,
        state: dict[str, Any],
    ) -> RunCheckpoint:
        checkpoint_id = str(uuid.uuid4())
        seq = await self._db.save_demo_checkpoint(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            checkpoint_kind=checkpoint_kind,
            state_json=state,
        )
        cp = RunCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            sequence=seq,
            checkpoint_kind=checkpoint_kind,
            state=state,
        )
        self._write_json(
            self._run_dir(conversation_id, run_id) / "checkpoints" / f"checkpoint_{seq:04d}_{checkpoint_kind.lower()}.json",
            cp.to_dict(),
        )
        await self._db.touch_demo_run_counters(run_id=run_id, checkpoint_count=seq)
        return cp

    async def update_manifest(self, manifest: RunManifest) -> None:
        await self._db.update_demo_run_manifest(
            run_id=manifest.run_id,
            status=manifest.status,
            cognitive_phase=manifest.cognitive_phase,
            execution_mode=manifest.execution_mode,
            active_plan_id=manifest.active_plan_id,
            assistant_reply=manifest.assistant_reply,
            error_class=manifest.error_class,
            manifest_json=manifest.to_dict(),
            finished_at=manifest.finished_at.isoformat() if manifest.finished_at else None,
        )
        self._write_json(
            self._run_dir(manifest.conversation_id, manifest.run_id) / "run_manifest.json",
            manifest.to_dict(),
        )

    async def finalize_run(self, manifest: RunManifest) -> None:
        await self.append_phase(
            run_id=manifest.run_id,
            conversation_id=manifest.conversation_id,
            phase_name="CLOSED",
            payload={
                "status": manifest.status,
                "error_class": manifest.error_class,
            },
        )
        phases = await self._db.load_demo_run_phases(manifest.run_id)
        manifest.phase_count = len(phases)
        await self.update_manifest(manifest)


    async def write_json_artifact(
        self,
        *,
        conversation_id: str,
        run_id: str,
        relative_path: str,
        payload: dict[str, Any],
    ) -> None:
        self._write_json(self._run_dir(conversation_id, run_id) / relative_path, payload)

    async def append_jsonl_artifact(
        self,
        *,
        conversation_id: str,
        run_id: str,
        relative_path: str,
        payload: dict[str, Any],
    ) -> None:
        self._append_jsonl(self._run_dir(conversation_id, run_id) / relative_path, payload)

    def _latest_run_dir_for_conversation(self, conversation_id: str) -> Path | None:
        base = self._root / conversation_id
        if not base.exists():
            return None
        run_dirs = [p for p in base.iterdir() if p.is_dir()]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda p: p.stat().st_mtime)

    async def load_latest_prompt_trace(self, conversation_id: str) -> dict[str, Any] | None:
        run_dir = self._latest_run_dir_for_conversation(conversation_id)
        if run_dir is None:
            return None
        llm_calls_path = run_dir / "llm_calls.jsonl"
        prompt_lint_path = run_dir / "prompt_lint.json"
        phases_path = run_dir / "run_phases.jsonl"
        llm_calls: list[dict[str, Any]] = []
        if llm_calls_path.exists():
            for line in llm_calls_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    llm_calls.append(parsed)
        prompt_lint: dict[str, Any] = {}
        if prompt_lint_path.exists():
            try:
                parsed = json.loads(prompt_lint_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    prompt_lint = parsed
            except Exception:
                prompt_lint = {}

        phases: list[dict[str, Any]] = []
        if phases_path.exists():
            for line in phases_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    phases.append(parsed)

        latest_by_phase: dict[str, dict[str, Any]] = {}
        for phase in phases:
            name = str(phase.get("phase_name") or "")
            if name:
                latest_by_phase[name] = phase

        context_snapshot: dict[str, Any] | None = None
        context_phase = latest_by_phase.get("CONTEXT_0")
        context_id = None
        if isinstance(context_phase, dict):
            payload = context_phase.get("payload")
            if isinstance(payload, dict):
                context_id = payload.get("context_id")
        if context_id:
            context_path = run_dir / f"context_snapshot_{context_id}.json"
            if context_path.exists():
                try:
                    parsed = json.loads(context_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        context_snapshot = parsed
                except Exception:
                    context_snapshot = None
        if context_snapshot is None:
            snapshots = sorted(run_dir.glob("context_snapshot_*.json"), key=lambda p: p.stat().st_mtime)
            if snapshots:
                try:
                    parsed = json.loads(snapshots[-1].read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        context_snapshot = parsed
                except Exception:
                    context_snapshot = None

        def _phase_payload(name: str) -> dict[str, Any] | None:
            phase = latest_by_phase.get(name)
            if not isinstance(phase, dict):
                return None
            payload = phase.get("payload")
            return payload if isinstance(payload, dict) else None

        runtime_artifacts = {
            "context": context_snapshot,
            "zoom": _phase_payload("ENRICH_ZOOM"),
            "zoom_out": _phase_payload("ENRICH_ZOOM_OUT"),
            "citation_contract": _phase_payload("CITATION_CONTRACT"),
        }
        return {
            "conversation_id": conversation_id,
            "run_id": run_dir.name,
            "prompt_trace_available": bool(llm_calls),
            "llm_calls": llm_calls,
            "prompt_lint": prompt_lint,
            "runtime_artifacts": runtime_artifacts,
            "artifacts": {
                "llm_calls": "llm_calls.jsonl" if llm_calls_path.exists() else None,
                "prompt_lint": "prompt_lint.json" if prompt_lint_path.exists() else None,
                "run_phases": "run_phases.jsonl" if phases_path.exists() else None,
                "context_snapshot": f"context_snapshot_{context_snapshot.get('context_id')}.json" if isinstance(context_snapshot, dict) and context_snapshot.get("context_id") else None,
            },
        }

    async def write_text_artifact(
        self,
        *,
        conversation_id: str,
        run_id: str,
        relative_path: str,
        text: str,
    ) -> None:
        path = self._run_dir(conversation_id, run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    async def load_bundle(self, run_id: str) -> RunBundle | None:
        manifest = await self._db.load_demo_run(run_id)
        if manifest is None:
            return None
        phases = await self._db.load_demo_run_phases(run_id)
        checkpoints = await self._db.load_demo_checkpoints(run_id)
        return RunBundle(manifest=manifest, phases=phases, checkpoints=checkpoints)
