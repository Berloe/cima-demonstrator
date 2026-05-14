from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cima_demo.demo.runtime import DemoRunJournal
from cima_demo.demo.contracts import RunManifest
from cima_demo.application.orchestrator import AgentOrchestrator
from cima_demo.domain.entities import TaskMemory
from cima_demo.domain.value_objects import ContextBudget


class _FakeRunDB:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.phases: dict[str, list[dict]] = {}
        self.checkpoints: dict[str, list[dict]] = {}
        self.turns: list[tuple[str, str, str]] = []
        self.saved_task_memory: list[TaskMemory] = []
        self.turn_metadata: dict[str, dict] = {}

    # --- run journal methods ---
    async def create_demo_run(self, **kwargs):
        self.runs[kwargs["run_id"]] = {
            "run_id": kwargs["run_id"],
            "conversation_id": kwargs["conversation_id"],
            "turn_id": kwargs["turn_id"],
            "status": kwargs["status"],
            "user_message": kwargs["user_message"],
            **kwargs["manifest_json"],
            "checkpoint_count": 0,
            "phase_count": 0,
        }

    async def append_demo_run_phase(self, *, run_id: str, phase_name: str, payload_json: dict):
        seq = len(self.phases.setdefault(run_id, [])) + 1
        self.phases[run_id].append({
            "run_id": run_id,
            "sequence": seq,
            "phase_name": phase_name,
            "payload": payload_json,
            "created_at": datetime.now(UTC).isoformat(),
        })
        self.runs[run_id]["phase_count"] = seq
        return seq

    async def save_demo_checkpoint(self, *, run_id: str, checkpoint_id: str, checkpoint_kind: str, state_json: dict):
        seq = len(self.checkpoints.setdefault(run_id, [])) + 1
        self.checkpoints[run_id].append({
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "sequence": seq,
            "checkpoint_kind": checkpoint_kind,
            "state": state_json,
            "created_at": datetime.now(UTC).isoformat(),
        })
        self.runs[run_id]["checkpoint_count"] = seq
        return seq

    async def touch_demo_run_counters(self, *, run_id: str, checkpoint_count: int | None = None, phase_count: int | None = None):
        if checkpoint_count is not None:
            self.runs[run_id]["checkpoint_count"] = checkpoint_count
        if phase_count is not None:
            self.runs[run_id]["phase_count"] = phase_count

    async def update_demo_run_manifest(self, *, run_id: str, status: str, cognitive_phase: str | None, execution_mode: str | None, active_plan_id: str | None, assistant_reply: str, error_class: str | None, manifest_json: dict, finished_at: str | None = None):
        self.runs[run_id].update(manifest_json)
        self.runs[run_id].update({
            "status": status,
            "cognitive_phase": cognitive_phase,
            "execution_mode": execution_mode,
            "active_plan_id": active_plan_id,
            "assistant_reply": assistant_reply,
            "error_class": error_class,
            "finished_at": finished_at,
        })

    async def load_demo_run(self, run_id: str):
        return dict(self.runs.get(run_id)) if run_id in self.runs else None

    async def load_demo_run_phases(self, run_id: str):
        return list(self.phases.get(run_id, []))

    async def load_demo_checkpoints(self, run_id: str):
        return list(self.checkpoints.get(run_id, []))

    # --- orchestrator methods used by PR2 path ---
    async def load_task_memory(self, conversation_id: str):
        return TaskMemory(conversation_id=conversation_id)

    async def load_turn_metadata(self, conversation_id: str):
        return None

    async def save_task_memory(self, task_memory: TaskMemory):
        self.saved_task_memory.append(task_memory)

    async def load_plan(self, plan_id: str):
        return None

    async def save_turn_metadata(self, conversation_id: str, json_data: str):
        self.turn_metadata[conversation_id] = {"json": json_data}

    async def append_turn(self, conversation_id: str, user_message: str, assistant_message: str, created_at=None):
        self.turns.append((conversation_id, user_message, assistant_message))

    async def release_turn_in_progress(self, conversation_id: str):
        return None

    async def save_chm_refs(self, conversation_id: str, citem_ids: list[str]):
        return None

    async def load_chm_refs(self, conversation_id: str):
        return {}


class _DummyLLM:
    def abort(self) -> None:
        return None


class _DummyMemory:
    def __init__(self) -> None:
        self.ingested: list[str] = []
        self.promotions: list[str] = []

    async def ingest_citem(self, req):
        self.ingested.append(req.content)

    async def fetch_by_conversation(self, conversation_id: str, scope_status: str = "active"):
        return []

    async def check_promotions(self, conversation_id: str, chm_counts: dict[str, int]):
        self.promotions.append(conversation_id)


class _DummyStream:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, delta):
        self.published.append(delta)


class _DummyRetrieval:
    def set_rag_config(self, value):
        self.value = value


class _DummyContext:
    def __init__(self) -> None:
        self._retrieval = _DummyRetrieval()




class _DummyDemoController:
    async def run_turn(self, rt, task_memory, plan):
        rt.assistant_reply_buffer = "La respuesta final del demostrador"
        rt.iteration_count = 1
        return None

class _DummyEngine:
    def __init__(self) -> None:
        self.flush_called = False
        self.events = []
        self.errors = []

    async def run_turn(self, rt, task_memory, plan):
        rt.assistant_reply_buffer = "La respuesta final del demostrador"
        rt.iteration_count = 1

    async def _flush_tool_results(self, plan):
        self.flush_called = True

    async def _emit_error(self, code, message, recoverable):
        self.errors.append((code, message, recoverable))

    def _emit_domain(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_demo_run_journal_persists_manifest_phases_and_checkpoints(tmp_path: Path):
    db = _FakeRunDB()
    journal = DemoRunJournal(rel_db=db, artifacts_root=tmp_path)

    manifest = await journal.open_skeleton_run(
        run_id="11111111-1111-4111-8111-111111111111",
        conversation_id="22222222-2222-4222-8222-222222222222",
        turn_id="33333333-3333-4333-8333-333333333333",
        user_message="hola",
        attached_files=[{"filename": "a.txt", "mime_type": "text/plain", "size_bytes": 12}],
    )
    cp = await journal.checkpoint(
        run_id=manifest.run_id,
        conversation_id=manifest.conversation_id,
        checkpoint_kind="BOOTSTRAP",
        state={"stage": "BOOTSTRAP"},
    )
    manifest.checkpoint_count = cp.sequence
    manifest.status = "completed"
    manifest.finished_at = datetime.now(UTC)
    manifest.assistant_reply = "ok"
    await journal.finalize_run(manifest)

    bundle = await journal.load_bundle(manifest.run_id)
    assert bundle is not None
    assert bundle.manifest["status"] == "completed"
    assert bundle.manifest["phase_count"] >= 2
    assert bundle.manifest["checkpoint_count"] == 1
    assert [p["phase_name"] for p in bundle.phases] == ["OPEN", "CLOSED"]
    assert bundle.checkpoints[0]["checkpoint_kind"] == "BOOTSTRAP"
    assert (tmp_path / manifest.conversation_id / manifest.run_id / "run_manifest.json").exists()
    assert (tmp_path / manifest.conversation_id / manifest.run_id / "checkpoints" / "checkpoint_0001_bootstrap.json").exists()


@pytest.mark.asyncio
async def test_orchestrator_generates_demo_run_artifacts(tmp_path: Path):
    db = _FakeRunDB()
    journal = DemoRunJournal(rel_db=db, artifacts_root=tmp_path)
    orch = AgentOrchestrator(
        llm_port=_DummyLLM(),
        rel_db=db,
        memory_service=_DummyMemory(),
        context_builder=_DummyContext(),
        stream_manager=_DummyStream(),
        plan_executor=object(),
        context_budget=ContextBudget(max_tokens=2048, overhead_tokens=256),
        system_prompt_factory=lambda **_: "prompt",
        demo_run_journal=journal,
        demo_turn_controller=_DummyDemoController(),
    )
    orch._run_promotions = lambda conversation_id: _noop()  # type: ignore[method-assign]

    await orch.handle_turn(
        conversation_id="44444444-4444-4444-8444-444444444444",
        user_message="What is 2 + 2?",
        attached_files=None,
    )

    assert len(db.runs) == 1
    run_id = next(iter(db.runs))
    manifest = db.runs[run_id]
    assert manifest["status"] == "completed"
    assert manifest["assistant_reply"] == "La respuesta final del demostrador"
    assert manifest["execution_mode"] is not None
    assert manifest["checkpoint_count"] >= 2
    assert manifest["phase_count"] >= 4
    assert [p["phase_name"] for p in db.phases[run_id]] == [
        "OPEN", "BOOTSTRAPPED", "DEMO_RUNTIME_PREPARED", "DEMO_CONTROLLER_RUNNING", "FINALIZING", "CLOSED"
    ]
    assert [c["checkpoint_kind"] for c in db.checkpoints[run_id]] == [
        "BOOTSTRAP", "FINAL_STATE"
    ]
    assert db.turns == [(
        "44444444-4444-4444-8444-444444444444",
        "What is 2 + 2?",
        "La respuesta final del demostrador",
    )]
    assert (tmp_path / "44444444-4444-4444-8444-444444444444" / run_id / "run_manifest.json").exists()


async def _noop() -> None:
    return None
