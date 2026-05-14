"""Reproducible scenario runner for the CIMA Demonstrator."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cima_demo.cognitive.kernel.state import TurnRuntime
from cima_demo.demo.context.service import DemoContextService
from cima_demo.demo.handoff.service import DemoHandoffService
from cima_demo.demo.lineage.service import DemoLineageService
from cima_demo.demo.runtime.controller import DemoTurnController
from cima_demo.demo.runtime.journal import DemoRunJournal
from cima_demo.domain.entities import CItem, Plan, PlanStep, TaskMemory, SummaryNode
from cima_demo.domain.value_objects import ContextBudget, StepStatus

from .acceptance import ScenarioObserved, evaluate_scenario
from .acceptance_package import write_acceptance_package
from .datasets import HarnessScenario, dataset_dir, load_all_scenarios, load_scenario
from .evidence_book import write_evidence_book
from .conformance_matrix import write_conformance_matrix
from .final_handoff import write_final_handoff
from .release_candidate import write_release_candidate
from .fakes import HarnessContextBuilder, HarnessLLM, HarnessMemoryService, HarnessStreamManager, InMemoryCItemStore, InMemoryDemoDB
from .report_builder import ScenarioExecutionResult, build_acceptance_report, write_acceptance_report, write_demo_report


class DemoScenarioRunner:
    def __init__(self, *, artifacts_root: Path, scenarios: list[HarnessScenario] | None = None) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.scenarios = scenarios or load_all_scenarios()

    async def run_all(self, scenario_ids: list[str] | None = None) -> list[ScenarioExecutionResult]:
        selected = self.scenarios
        if scenario_ids:
            wanted = set(scenario_ids)
            selected = [scenario for scenario in self.scenarios if scenario.scenario_id in wanted]
        results: list[ScenarioExecutionResult] = []
        for scenario in selected:
            results.append(await self.run_scenario(scenario))
        acceptance_report = build_acceptance_report(results)
        acceptance_report_path = write_acceptance_report(self.artifacts_root, acceptance_report)
        demo_report_path = write_demo_report(self.artifacts_root, results, acceptance_report)
        evidence_paths = write_evidence_book(root=self.artifacts_root, results=results, dataset_root=dataset_dir())
        conformance_paths = write_conformance_matrix(root=self.artifacts_root)
        evidence_index_payload = json.loads(evidence_paths["index"].read_text(encoding="utf-8"))
        scenario_manifest_paths = [self.artifacts_root / entry["path"] for entry in evidence_index_payload.get("manifests", [])]
        acceptance_paths = write_acceptance_package(
            root=self.artifacts_root,
            results=results,
            scenario_manifest_paths=scenario_manifest_paths,
            acceptance_report_path=acceptance_report_path,
            demo_report_path=demo_report_path,
            evidence_book_index_path=evidence_paths["index"],
            evidence_book_md_path=evidence_paths["markdown"],
            conformance_matrix_json_path=conformance_paths["json"],
            conformance_matrix_md_path=conformance_paths["markdown"],
        )
        release_paths = write_release_candidate(
            root=self.artifacts_root,
            repo_root=Path(__file__).resolve().parents[3],
            acceptance_report_path=acceptance_report_path,
            acceptance_package_bundle_path=acceptance_paths["bundle"],
            conformance_matrix_json_path=conformance_paths["json"],
        )
        write_final_handoff(
            root=self.artifacts_root,
            release_candidate_index_path=release_paths["index"],
            acceptance_package_index_path=acceptance_paths["index"],
            conformance_matrix_json_path=conformance_paths["json"],
        )
        return results

    async def run_scenario(self, scenario: HarnessScenario) -> ScenarioExecutionResult:
        env = await self._bootstrap_environment(scenario)
        run_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        conversation_id = f"{scenario.scenario_id.lower()}-conv-1"
        journal = env["journal"]
        task_memory = env["task_memory"]
        plan = env.get("plan")
        context_service: DemoContextService = env["context_service"]
        controller: DemoTurnController = env["controller"]
        db: InMemoryDemoDB = env["db"]
        lineage: DemoLineageService = env["lineage"]
        llm: HarnessLLM = env["llm"]

        manifest = await journal.open_skeleton_run(
            run_id=run_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message=scenario.query,
            attached_files=[{"filename": doc.title, "mime_type": "text/plain", "size_bytes": len(doc.text.encode("utf-8"))} for doc in scenario.documents],
        )
        await journal.checkpoint(
            run_id=run_id,
            conversation_id=conversation_id,
            checkpoint_kind="BOOTSTRAP",
            state={
                "task_memory": self._task_memory_dict(task_memory),
                "plan": self._plan_dict(plan),
                "query": scenario.query,
            },
        )
        token = context_service.bind_run(run_id=run_id, conversation_id=conversation_id, turn_id=turn_id, query_text=scenario.query)
        try:
            rt = self._make_turn_runtime(conversation_id=conversation_id, turn_id=turn_id, run_id=run_id, query=scenario.query)
            await controller.run_turn(rt, task_memory, plan)
        finally:
            context_service.reset_run(token)

        await lineage.record_answer_lineage(
            conversation_id=conversation_id,
            run_id=run_id,
            response_turn_id=turn_id,
            context_id=context_service.last_snapshot_id() or await self._latest_context_id(db, run_id),
            answer_text=rt.assistant_reply_buffer,
            cited_markers=list(rt.cited_markers),
            selected_items=list((await db.load_latest_demo_context_snapshot_for_run(run_id) or {}).get("items", [])),
        )
        visible_transcript = [
            {"role": "user", "content": scenario.query},
            {"role": "assistant", "content": rt.assistant_reply_buffer},
        ]
        await journal.write_text_artifact(
            conversation_id=conversation_id,
            run_id=run_id,
            relative_path="visible_transcript.jsonl",
            text="\n".join(json.dumps(row, ensure_ascii=False) for row in visible_transcript),
        )
        snapshot = await db.load_latest_demo_context_snapshot_for_run(run_id)
        await self._export_trace_artifacts(db=db, journal=journal, conversation_id=conversation_id, run_id=run_id, snapshot=snapshot)
        manifest.task_memory = self._task_memory_dict(task_memory)
        manifest.task_state = {"task_memory": self._task_memory_dict(task_memory)}
        manifest.active_plan_id = plan.plan_id if plan is not None else None
        manifest.cognitive_phase = task_memory.phase
        manifest.assistant_reply = rt.assistant_reply_buffer
        manifest.status = "completed"
        await journal.checkpoint(
            run_id=run_id,
            conversation_id=conversation_id,
            checkpoint_kind="FINAL_STATE",
            state={
                "task_memory": self._task_memory_dict(task_memory),
                "plan": self._plan_dict(plan),
                "assistant_reply": rt.assistant_reply_buffer,
            },
        )
        await journal.finalize_run(manifest)

        zoom_result = None
        zoom_out_result = None
        if snapshot is not None and scenario.expectations.requires_zoom:
            targets = [item.get("marker") for item in snapshot.get("items", [])[:2] if item.get("marker")]
            zoom_result = await context_service.zoom(context_id=snapshot["context_id"], zoom_targets=targets)
            await journal.write_json_artifact(
                conversation_id=conversation_id,
                run_id=run_id,
                relative_path=f"zoom_trace_{snapshot['context_id']}.json",
                payload=zoom_result,
            )
        if snapshot is not None and scenario.expectations.requires_zoom_out:
            zoom_out_result = await context_service.zoom_out(context_id=snapshot["context_id"], targets=["MASTER"])
            await journal.write_json_artifact(
                conversation_id=conversation_id,
                run_id=run_id,
                relative_path=f"zoom_out_trace_{snapshot['context_id']}.json",
                payload=zoom_out_result,
            )

        handoff_manifest = None
        handoff_validation = None
        handoff_restore = None
        resumed_answer = None
        if scenario.expectations.requires_handoff:
            handoff_manifest, handoff_validation, handoff_restore, resumed_answer = await self._run_handoff_continuation(
                scenario=scenario,
                env=env,
                source_conversation_id=conversation_id,
                source_run_id=run_id,
            )

        observed = ScenarioObserved(
            answer_text=rt.assistant_reply_buffer,
            answer_lineage=self._latest_answer_lineage(db, conversation_id, run_id),
            context_snapshot=snapshot,
            budget_trace=(snapshot or {}).get("budget") if snapshot else None,
            corpus_tokens=scenario.corpus_tokens,
            visible_transcript=visible_transcript,
            zoom_result=zoom_result,
            zoom_out_result=zoom_out_result,
            handoff_manifest=handoff_manifest,
            handoff_validation=handoff_validation,
            handoff_restore=handoff_restore,
            resumed_answer_text=resumed_answer,
        )
        if observed.answer_lineage is not None:
            resolved = await self._resolve_lineage_counts(db, conversation_id, observed.answer_lineage)
            observed.answer_lineage.update(resolved)
        acceptance = evaluate_scenario(scenario, observed)
        result = ScenarioExecutionResult(
            scenario=scenario,
            conversation_id=conversation_id,
            run_id=run_id,
            answer_text=rt.assistant_reply_buffer,
            resumed_answer_text=resumed_answer,
            artifacts_dir=self.artifacts_root / conversation_id / run_id,
            corpus_tokens=scenario.corpus_tokens,
            context_snapshot=snapshot,
            budget_trace=(snapshot or {}).get("budget") if snapshot else None,
            answer_lineage=observed.answer_lineage,
            zoom_result=zoom_result,
            zoom_out_result=zoom_out_result,
            handoff_manifest=handoff_manifest,
            handoff_validation=handoff_validation,
            handoff_restore=handoff_restore,
            visible_transcript=visible_transcript,
            acceptance=acceptance,
        )
        return result

    async def _bootstrap_environment(self, scenario: HarnessScenario) -> dict[str, Any]:
        db = InMemoryDemoDB()
        store = InMemoryCItemStore()
        memory = HarnessMemoryService(store)
        stream = HarnessStreamManager()
        conversation_id = f"{scenario.scenario_id.lower()}-conv-1"
        await db.create_conversation(conversation_id)
        task_memory = TaskMemory(conversation_id=conversation_id, phase="recall")
        await db.save_task_memory(task_memory)
        run_root = self.artifacts_root
        journal = DemoRunJournal(rel_db=db, artifacts_root=run_root)
        lineage = DemoLineageService(rel_db=db)
        source_map: dict[str, tuple[str, str | None]] = {}
        for doc in scenario.documents:
            source, full_span = await lineage.register_text_source(
                conversation_id=conversation_id,
                source_kind="dataset_document",
                role="evidence",
                display_text=doc.title,
                process_text=doc.text,
                origin_ref=doc.doc_id,
                metadata={"dataset": scenario.scenario_id},
            )
            source_map[doc.doc_id] = (source.source_id, full_span.span_id if full_span is not None else None)
        for item in scenario.citems:
            citem = CItem(
                citem_id=item.citem_id,
                conversation_id=conversation_id,
                content=item.content,
                item_type=item.item_type,
                token_count=max(1, len(item.content.split())),
                dependency_ids=list(item.dependencies),
            )
            await store.save(citem)
            source_id, span_id = source_map[item.source_doc_id]
            await lineage.record_citem_lineage(
                conversation_id=conversation_id,
                citem_id=citem.citem_id,
                source_id=source_id,
                source_span_ids=[span_id] if span_id is not None else [],
                dependency_ids=list(item.dependencies),
                metadata={"dataset": scenario.scenario_id, "doc_id": item.source_doc_id},
            )
        for summary in scenario.summaries:
            node = SummaryNode(
                node_id=summary.summary_id,
                conversation_id=conversation_id,
                level=summary.level,
                content=summary.content,
                token_count=max(1, len(summary.content.split())),
                origin_citem_ids=list(summary.origin_citem_ids),
            )
            await db.save_summary(node)
            await lineage.record_summary_resolution(
                conversation_id=conversation_id,
                summary_id=node.node_id,
                summary_text=node.content,
                origin_citem_ids=list(summary.origin_citem_ids),
                metadata={"dataset": scenario.scenario_id},
            )
        plan = None
        if scenario.expectations.requires_handoff:
            plan = Plan(
                conversation_id=conversation_id,
                goal="Complete the migration safely",
                steps=[
                    PlanStep(plan_id="", description="Rename the public surface", status=StepStatus.COMPLETED),
                    PlanStep(plan_id="", description="Preserve compatibility during migration", status=StepStatus.ACTIVE),
                ],
                auto_continue=False,
            )
            for step in plan.steps:
                step.plan_id = plan.plan_id
            task_memory.active_plan_id = plan.plan_id
            await db.save_plan_with_task_memory(plan, task_memory)
        builder = HarnessContextBuilder(store=store, db=db)
        llm = HarnessLLM(
            need_proposal=scenario.llm.need_proposal,
            memory_proposal=scenario.llm.memory_proposal,
            answer=scenario.llm.answer,
        )
        handoff = DemoHandoffService(rel_db=db, citem_store=store, run_journal=journal, artifacts_root=run_root)
        context_service = DemoContextService(
            base_builder=builder,
            memory_service=memory,
            rel_db=db,
            run_journal=journal,
            geometry_service=None,
            handoff_service=handoff,
        )
        controller = DemoTurnController(
            llm_port=llm,
            stream_manager=stream,
            context_service=context_service,
            memory_service=memory,
            context_budget=ContextBudget(max_tokens=scenario.window_tokens * 2, overhead_tokens=scenario.window_tokens),
            run_journal=journal,
            llm_max_tokens=scenario.window_tokens,
        )
        return {
            "db": db,
            "store": store,
            "memory": memory,
            "stream": stream,
            "journal": journal,
            "lineage": lineage,
            "llm": llm,
            "task_memory": task_memory,
            "plan": plan,
            "context_service": context_service,
            "controller": controller,
            "handoff": handoff,
        }

    async def _run_handoff_continuation(self, *, scenario: HarnessScenario, env: dict[str, Any], source_conversation_id: str, source_run_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
        context_service: DemoContextService = env["context_service"]
        handoff_manifest = await context_service.create_handoff(conversation_id=source_conversation_id, source_run_id=source_run_id, rationale="pause for portable continuation")
        handoff_validation = await context_service.validate_handoff(handoff_id=handoff_manifest["handoff_id"])
        target_conversation_id = f"{scenario.scenario_id.lower()}-conv-2"
        db: InMemoryDemoDB = env["db"]
        await db.create_conversation(target_conversation_id)
        target_run_id = str(uuid.uuid4())
        journal: DemoRunJournal = env["journal"]
        restore_manifest = await journal.open_skeleton_run(
            run_id=target_run_id,
            conversation_id=target_conversation_id,
            turn_id=str(uuid.uuid4()),
            user_message=scenario.resume_query or "Continue after restore",
            attached_files=[],
        )
        await journal.checkpoint(
            run_id=target_run_id,
            conversation_id=target_conversation_id,
            checkpoint_kind="BOOTSTRAP",
            state={"handoff_id": handoff_manifest["handoff_id"], "resume_query": scenario.resume_query},
        )
        handoff_restore = await context_service.restore_handoff(
            handoff_id=handoff_manifest["handoff_id"],
            target_conversation_id=target_conversation_id,
            target_run_id=target_run_id,
        )
        resumed_answer = None
        if handoff_restore.get("valid"):
            task_memory = await db.load_task_memory(target_conversation_id)
            plan = await db.load_plan(task_memory.active_plan_id) if task_memory and task_memory.active_plan_id else None
            llm = HarnessLLM(
                need_proposal=scenario.llm.need_proposal,
                memory_proposal=scenario.llm.memory_proposal,
                answer=scenario.llm.resume_answer or scenario.llm.answer,
            )
            controller = DemoTurnController(
                llm_port=llm,
                stream_manager=env["stream"],
                context_service=context_service,
                memory_service=env["memory"],
                context_budget=ContextBudget(max_tokens=scenario.window_tokens * 2, overhead_tokens=scenario.window_tokens),
                run_journal=journal,
                llm_max_tokens=scenario.window_tokens,
            )
            resume_query = scenario.resume_query or "Continue after restore"
            token = context_service.bind_run(run_id=target_run_id, conversation_id=target_conversation_id, turn_id=str(uuid.uuid4()), query_text=resume_query)
            try:
                rt = self._make_turn_runtime(conversation_id=target_conversation_id, turn_id=str(uuid.uuid4()), run_id=target_run_id, query=resume_query)
                await controller.run_turn(rt, task_memory or TaskMemory(conversation_id=target_conversation_id, phase="recall"), plan)
                resumed_answer = rt.assistant_reply_buffer
                await journal.write_text_artifact(
                    conversation_id=target_conversation_id,
                    run_id=target_run_id,
                    relative_path="visible_transcript.jsonl",
                    text="\n".join([
                        json.dumps({"role": "user", "content": resume_query}, ensure_ascii=False),
                        json.dumps({"role": "assistant", "content": resumed_answer}, ensure_ascii=False),
                    ]),
                )
            finally:
                context_service.reset_run(token)
            restore_manifest.task_memory = self._task_memory_dict(task_memory or TaskMemory(conversation_id=target_conversation_id))
            restore_manifest.assistant_reply = resumed_answer or ""
            restore_manifest.active_plan_id = plan.plan_id if plan is not None else None
            restore_manifest.status = "completed"
            await journal.checkpoint(
                run_id=target_run_id,
                conversation_id=target_conversation_id,
                checkpoint_kind="FINAL_STATE",
                state={"task_memory": self._task_memory_dict(task_memory or TaskMemory(conversation_id=target_conversation_id)), "plan": self._plan_dict(plan), "assistant_reply": resumed_answer},
            )
            await journal.finalize_run(restore_manifest)
        return handoff_manifest, handoff_validation, handoff_restore, resumed_answer

    async def _export_trace_artifacts(self, *, db: InMemoryDemoDB, journal: DemoRunJournal, conversation_id: str, run_id: str, snapshot: dict[str, Any] | None) -> None:
        lineages = await db.load_demo_lineage_edges(conversation_id)
        if lineages:
            await journal.write_text_artifact(
                conversation_id=conversation_id,
                run_id=run_id,
                relative_path="lineage_edges.jsonl",
                text="\n".join(json.dumps(row, ensure_ascii=False) for row in lineages),
            )
        for resolution in await db.load_demo_summary_resolutions(conversation_id):
            await journal.write_json_artifact(
                conversation_id=conversation_id,
                run_id=run_id,
                relative_path=f"summary_resolution_{resolution['summary_id']}.json",
                payload=resolution,
            )
        answer_lineage = self._latest_answer_lineage(db, conversation_id, run_id)
        if answer_lineage is not None:
            answer_lineage.update(await self._resolve_lineage_counts(db, conversation_id, answer_lineage))
            await journal.write_json_artifact(
                conversation_id=conversation_id,
                run_id=run_id,
                relative_path=f"answer_lineage_{answer_lineage['answer_lineage_id']}.json",
                payload=answer_lineage,
            )

    def _latest_answer_lineage(self, db: InMemoryDemoDB, conversation_id: str, run_id: str) -> dict[str, Any] | None:
        rows = [row for row in db.demo_answer_lineage if row.get("conversation_id") == conversation_id and row.get("run_id") == run_id]
        if not rows:
            return None
        return dict(rows[-1])

    async def _resolve_lineage_counts(self, db: InMemoryDemoDB, conversation_id: str, answer_lineage: dict[str, Any]) -> dict[str, Any]:
        stored_sources = answer_lineage.get("resolved_source_ids") or []
        stored_spans = answer_lineage.get("resolved_span_ids") or []
        stored_marker_resolution = list(answer_lineage.get("marker_resolution") or [])
        stored_unresolved = list(answer_lineage.get("unresolved_ref_ids") or [])
        if stored_sources or stored_spans or stored_marker_resolution or stored_unresolved:
            return {
                "resolved_source_count": int(answer_lineage.get("resolved_source_count", len(stored_sources)) or 0),
                "resolved_span_count": int(answer_lineage.get("resolved_span_count", len(stored_spans)) or 0),
                "resolved_source_ids": [str(v) for v in stored_sources if str(v)],
                "resolved_span_ids": [str(v) for v in stored_spans if str(v)],
                "unresolved_ref_ids": [str(v) for v in stored_unresolved if str(v)],
                "marker_resolution": [dict(row) for row in stored_marker_resolution],
                "resolution_mode": str(answer_lineage.get("resolution_mode") or "empty"),
            }
        detailed = await WitnessLineageResolver(db).resolve_selected_items_detailed(
            conversation_id=conversation_id,
            selected_items=list(answer_lineage.get("lineage") or []),
        )
        resolved_source_ids = sorted({str(v) for row in detailed for v in (row.get("resolved_source_ids") or []) if str(v)})
        resolved_span_ids = sorted({str(v) for row in detailed for v in (row.get("resolved_span_ids") or []) if str(v)})
        unresolved_ref_ids = sorted({str(v) for row in detailed for v in (row.get("unresolved_ref_ids") or []) if str(v)})
        modes = [str(row.get("resolution_mode") or "") for row in detailed if str(row.get("resolution_mode") or "")]
        modes.extend(str(row.get("support_resolution_mode") or "") for row in detailed if str(row.get("support_resolution_mode") or ""))
        witness = any(mode in {"witness_first", "mixed"} for mode in modes)
        legacy = any(mode in {"legacy_fallback", "mixed"} for mode in modes)
        if witness and legacy:
            resolution_mode = "mixed"
        elif witness:
            resolution_mode = "witness_first"
        elif legacy:
            resolution_mode = "legacy_fallback"
        else:
            resolution_mode = "empty"
        return {
            "resolved_source_count": len(resolved_source_ids),
            "resolved_span_count": len(resolved_span_ids),
            "resolved_source_ids": resolved_source_ids,
            "resolved_span_ids": resolved_span_ids,
            "unresolved_ref_ids": unresolved_ref_ids,
            "marker_resolution": [dict(row) for row in detailed],
            "resolution_mode": resolution_mode,
        }

    async def _latest_context_id(self, db: InMemoryDemoDB, run_id: str) -> str | None:
        snapshot = await db.load_latest_demo_context_snapshot_for_run(run_id)
        return None if snapshot is None else str(snapshot.get("context_id") or "") or None

    def _make_turn_runtime(self, *, conversation_id: str, turn_id: str, run_id: str, query: str) -> TurnRuntime:
        rt = TurnRuntime(
            conversation_id=conversation_id,
            turn_id=turn_id,
            run_id=run_id,
            user_message=query,
            phase="recall",
        )
        rt.output_contract = SimpleNamespace(
            format="text",
            representation="plain_text",
            base_unit=None,
            display_scale=None,
            rounding_rule=None,
            precision=None,
            required_evidence=True,
        )
        return rt

    def _task_memory_dict(self, task_memory: TaskMemory) -> dict[str, Any]:
        return {
            "conversation_id": task_memory.conversation_id,
            "turn_count": task_memory.turn_count,
            "phase": task_memory.phase,
            "active_plan_id": task_memory.active_plan_id,
            "awaiting_user_input": task_memory.awaiting_user_input,
            "turn_in_progress": task_memory.turn_in_progress,
            "stall_count": task_memory.stall_count,
            "last_turn_at": task_memory.last_turn_at.isoformat() if task_memory.last_turn_at else None,
            "created_at": task_memory.created_at.isoformat() if task_memory.created_at else None,
        }

    def _plan_dict(self, plan: Plan | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "status": plan.status,
            "auto_continue": plan.auto_continue,
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "status": step.status,
                    "tool_name": step.tool_name,
                    "result_summary": step.result_summary,
                }
                for step in plan.steps
            ],
        }


async def run_harness(*, artifacts_root: Path, scenario_ids: list[str] | None = None) -> list[ScenarioExecutionResult]:
    runner = DemoScenarioRunner(artifacts_root=artifacts_root)
    return await runner.run_all(scenario_ids=scenario_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen demonstrator scenarios.")
    parser.add_argument("--artifacts-root", default="./demo_artifacts", help="Directory where harness artifacts will be written")
    parser.add_argument("--scenario", action="append", dest="scenario_ids", help="Scenario ID to run (repeatable)")
    args = parser.parse_args()
    import asyncio
    asyncio.run(run_harness(artifacts_root=Path(args.artifacts_root), scenario_ids=args.scenario_ids))


__all__ = ["DemoScenarioRunner", "ScenarioExecutionResult", "run_harness", "main"]

if __name__ == "__main__":
    main()
