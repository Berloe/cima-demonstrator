"""Portable handoff service for the CIMA Demonstrator."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cima_demo.demo.contracts import HandoffManifest, HandoffRestore, HandoffValidation
from cima_demo.demo.runtime.journal import DemoRunJournal
from cima_demo.domain.entities import CItem, Plan, PlanStep, SummaryNode, TaskMemory
from cima_demo.domain.ports import CItemStorePort, RelDBPort


class DemoHandoffService:
    """Create, validate and restore portable handoff manifests.

    The service is deliberately independent from the visible transcript. A
    handoff is constructed from durable run artifacts plus memory/context
    references and can be restored into another conversation without replaying
    the full chat history.
    """

    def __init__(
        self,
        *,
        rel_db: RelDBPort,
        citem_store: CItemStorePort,
        run_journal: DemoRunJournal,
        artifacts_root: Path,
    ) -> None:
        self._db = rel_db
        self._store = citem_store
        self._runs = run_journal
        self._root = Path(artifacts_root)

    async def create_handoff(
        self,
        *,
        conversation_id: str,
        source_run_id: str,
        rationale: str | None = None,
    ) -> HandoffManifest:
        bundle = await self._runs.load_bundle(source_run_id)
        if bundle is None:
            raise ValueError(f"run not found: {source_run_id}")
        snapshot = await self._db.load_latest_demo_context_snapshot_for_run(source_run_id)
        citem_refs: list[str] = []
        pyramid_refs: list[str] = []
        context_id: str | None = None
        if snapshot is not None:
            context_id = snapshot.get("context_id")
            for item in snapshot.get("items", []) or []:
                ref_id = str(item.get("ref_id", "") or "")
                if not ref_id:
                    continue
                ref_kind = str(item.get("ref_kind", "citem"))
                if ref_kind == "summary":
                    pyramid_refs.append(ref_id)
                else:
                    citem_refs.append(ref_id)
        citem_refs = list(dict.fromkeys(citem_refs))
        pyramid_refs = list(dict.fromkeys(pyramid_refs))

        bundled_citems, citem_lineage = await self._load_citem_bundle(
            conversation_id=conversation_id,
            citem_refs=citem_refs,
        )
        bundled_summaries, summary_lineage = await self._load_summary_bundle(
            conversation_id=conversation_id,
            summary_refs=pyramid_refs,
        )
        lineage_edges = self._dedupe_lineage_edges(citem_lineage + summary_lineage)
        source_ids = sorted({str(edge["dst_id"]) for edge in lineage_edges if edge.get("dst_kind") == "source"})
        span_ids = sorted({str(edge["dst_id"]) for edge in lineage_edges if edge.get("dst_kind") == "source_span"})
        sources = await self._db.load_demo_sources(conversation_id, source_ids)
        spans = await self._db.load_demo_source_spans(conversation_id, span_ids)

        task_state = {
            "context_id": context_id,
            "task_memory": bundle.manifest.get("task_memory", {}),
            "task_spec": bundle.manifest.get("task_spec", {}),
            "task_state": bundle.manifest.get("task_state"),
            "output_contract": bundle.manifest.get("output_contract"),
            "active_plan_id": bundle.manifest.get("active_plan_id"),
            "plan": self._extract_plan_snapshot(bundle),
            "phase": bundle.manifest.get("cognitive_phase"),
            "assistant_reply": bundle.manifest.get("assistant_reply", ""),
        }

        checksum = self._checksum(
            citem_refs=citem_refs,
            pyramid_refs=pyramid_refs,
            task_state=task_state,
            bundled_citems=bundled_citems,
            bundled_summaries=bundled_summaries,
            bundled_sources=sources,
            bundled_spans=spans,
            bundled_lineage=lineage_edges,
        )
        manifest = HandoffManifest(
            handoff_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            source_run_id=source_run_id,
            context_id=context_id,
            citem_refs=citem_refs,
            pyramid_refs=pyramid_refs,
            task_state=task_state,
            rationale=rationale,
            bundled_citems=bundled_citems,
            bundled_summaries=bundled_summaries,
            bundled_sources=sources,
            bundled_spans=spans,
            bundled_lineage=lineage_edges,
            checksum=checksum,
        )
        await self._db.save_demo_handoff_manifest(manifest.to_dict())
        await self._runs.write_json_artifact(
            conversation_id=conversation_id,
            run_id=source_run_id,
            relative_path=f"handoff_manifest_{manifest.handoff_id}.json",
            payload=manifest.to_dict(),
        )
        return manifest

    async def _load_citem_bundle(
        self,
        *,
        conversation_id: str,
        citem_refs: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not citem_refs:
            return [], []
        bundled: list[dict[str, Any]] = []
        lineage_edges: list[dict[str, Any]] = []

        local_rows = await self._db.list_local_citem_records(conversation_id, citem_ids=citem_refs)
        local_by_id = {str(row["local_citem_id"]): row for row in local_rows}
        global_rows = await self._db.list_global_citem_records(global_citem_ids=citem_refs)
        global_by_id = {str(row["global_citem_id"]): row for row in global_rows}

        missing_ids: list[str] = []
        for ref_id in citem_refs:
            if ref_id in local_by_id:
                row = local_by_id[ref_id]
                bundled.append(self._local_citem_record_to_dict(row))
                for evidence in await self._db.list_local_citem_evidence(ref_id):
                    lineage_edges.extend(self._local_evidence_to_lineage(conversation_id=conversation_id, local_citem_id=ref_id, evidence_row=evidence))
                continue
            if ref_id in global_by_id:
                row = global_by_id[ref_id]
                bundled.append(self._global_citem_record_to_dict(row))
                for evidence in await self._db.list_global_citem_evidence(ref_id):
                    lineage_edges.extend(self._global_evidence_to_lineage(conversation_id=conversation_id, global_citem_id=ref_id, evidence_row=evidence))
                continue
            missing_ids.append(ref_id)

        if missing_ids:
            found_citems = await self._store.fetch_batch(missing_ids)
            citems_by_id = {item.citem_id: item for item in found_citems}
            bundled.extend(self._citem_to_dict(citems_by_id[ref_id]) for ref_id in missing_ids if ref_id in citems_by_id)
            legacy_edges = await self._db.load_demo_lineage_edges(
                conversation_id,
                src_kind="citem",
                src_ids=list(citems_by_id.keys()),
            )
            lineage_edges.extend(dict(edge) for edge in legacy_edges)

        bundled_by_id = {str(row.get("citem_id")): row for row in bundled}
        ordered = [bundled_by_id[ref_id] for ref_id in citem_refs if ref_id in bundled_by_id]
        return ordered, self._dedupe_lineage_edges(lineage_edges)

    async def _load_summary_bundle(
        self,
        *,
        conversation_id: str,
        summary_refs: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not summary_refs:
            return [], []
        bundled: list[dict[str, Any]] = []
        lineage_edges: list[dict[str, Any]] = []

        local_rows = await self._db.list_local_summary_records(conversation_id, summary_ids=summary_refs)
        local_by_id = {str(row["local_summary_id"]): row for row in local_rows}
        global_rows = await self._db.list_global_summary_records(summary_ids=summary_refs)
        global_by_id = {str(row["global_summary_id"]): row for row in global_rows}

        legacy_ids: list[str] = []
        for ref_id in summary_refs:
            if ref_id in global_by_id:
                row = global_by_id[ref_id]
                origins = await self._db.list_global_summary_origins(ref_id)
                bundled.append(self._global_summary_record_to_dict(row, origins))
                lineage_edges.extend(self._summary_origins_to_lineage(summary_id=ref_id, origin_rows=origins))
                continue
            if ref_id in local_by_id:
                row = local_by_id[ref_id]
                origins = await self._db.list_local_summary_origins(ref_id)
                bundled.append(self._local_summary_record_to_dict(row, origins))
                lineage_edges.extend(self._summary_origins_to_lineage(summary_id=ref_id, origin_rows=origins))
                continue
            legacy_ids.append(ref_id)

        if legacy_ids:
            summaries = [node for node in await self._db.load_summaries(conversation_id) if node.node_id in set(legacy_ids)]
            summary_by_id = {node.node_id: node for node in summaries}
            summary_resolutions = await self._db.load_demo_summary_resolutions(conversation_id, legacy_ids)
            summary_res_map = {row["summary_id"]: row for row in summary_resolutions}
            bundled.extend(
                self._summary_to_dict(summary_by_id[ref_id], summary_res_map.get(ref_id))
                for ref_id in legacy_ids if ref_id in summary_by_id
            )
            summary_edges = await self._db.load_demo_lineage_edges(
                conversation_id,
                src_kind="summary",
                src_ids=list(summary_by_id.keys()),
            )
            lineage_edges.extend(dict(edge) for edge in summary_edges)

        bundled_by_id = {str(row.get("summary_id")): row for row in bundled}
        ordered = [bundled_by_id[ref_id] for ref_id in summary_refs if ref_id in bundled_by_id]
        return ordered, self._dedupe_lineage_edges(lineage_edges)

    def _local_citem_record_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "citem_id": str(row.get("local_citem_id")),
            "conversation_id": str(row.get("conversation_id")),
            "content": str(row.get("text", "")),
            "item_type": str(row.get("type", "FACT")),
            "scope": "episodic",
            "scope_status": "active",
            "importance": float(row.get("salience", 0.5) or 0.5),
            "confidence": 1.0,
            "validation_label": row.get("validity"),
            "conflict_status": "none",
            "phase_ingested": "IDLE",
            "actor": "agent",
            "motivation": "witness_handoff_bundle",
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "dependency_ids": [],
            "token_count": max(1, len(str(row.get("text", "")).split())),
            "content_hash": None,
            "chunk_kind": None,
            "semantic_identity_id": row.get("semantic_identity_id"),
            "meta_json": dict(row.get("meta_json") or {}),
            "provenance_json": dict(row.get("provenance_json") or {}),
            "validity": row.get("validity"),
            "salience": float(row.get("salience", 0.0) or 0.0),
            "vector_state": row.get("vector_state"),
            "expires_at": row.get("expires_at"),
            "is_pinned": bool(row.get("is_pinned", False)),
            "was_cited": bool(row.get("was_cited", False)),
            "last_used_at": row.get("last_used_at"),
            "normalizer_version": row.get("normalizer_version"),
            "citem_builder_version": row.get("citem_builder_version"),
        }

    def _global_citem_record_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "citem_id": str(row.get("global_citem_id")),
            "conversation_id": str(row.get("origin_conversation_id", "")),
            "content": str(row.get("text", "")),
            "item_type": str(row.get("type", "FACT")),
            "scope": "global",
            "scope_status": "active",
            "importance": float(row.get("salience", 0.5) or 0.5),
            "confidence": 1.0,
            "validation_label": row.get("validity"),
            "conflict_status": "none",
            "phase_ingested": "IDLE",
            "actor": "agent",
            "motivation": "witness_handoff_bundle",
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "dependency_ids": [],
            "token_count": max(1, len(str(row.get("text", "")).split())),
            "content_hash": None,
            "chunk_kind": None,
            "semantic_identity_id": row.get("semantic_identity_id"),
            "origin_conversation_id": row.get("origin_conversation_id"),
            "promotion_origin_local_citem_id": row.get("promotion_origin_local_citem_id"),
            "meta_json": dict(row.get("meta_json") or {}),
            "provenance_json": dict(row.get("provenance_json") or {}),
            "validity": row.get("validity"),
            "salience": float(row.get("salience", 0.0) or 0.0),
            "vector_state": row.get("vector_state"),
            "expires_at": row.get("expires_at"),
            "is_pinned": bool(row.get("is_pinned", False)),
            "was_cited": bool(row.get("was_cited", False)),
            "last_used_at": row.get("last_used_at"),
        }

    def _local_summary_record_to_dict(self, row: dict[str, Any], origin_rows: list[dict[str, Any]]) -> dict[str, Any]:
        origin_ids = [str(origin.get("origin_id")) for origin in origin_rows if origin.get("origin_id")]
        return {
            "summary_id": str(row.get("local_summary_id")),
            "conversation_id": str(row.get("conversation_id")),
            "level": self._summary_level_to_int(row.get("level")),
            "content": str(row.get("text", "")),
            "token_count": max(1, len(str(row.get("text", "")).split())),
            "parent_id": None,
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "origin_citem_ids": origin_ids,
            "metadata": {
                "summary_scope": "local",
                "covers_json": dict(row.get("covers_json") or {}),
                "cluster_id": row.get("cluster_id"),
                "epoch_no": row.get("epoch_no"),
                "vector_state": row.get("vector_state"),
                "is_pinned": bool(row.get("is_pinned", False)),
                "was_cited": bool(row.get("was_cited", False)),
                "last_used_at": row.get("last_used_at"),
            },
        }

    def _global_summary_record_to_dict(self, row: dict[str, Any], origin_rows: list[dict[str, Any]]) -> dict[str, Any]:
        origin_ids = [str(origin.get("origin_id")) for origin in origin_rows if origin.get("origin_id")]
        return {
            "summary_id": str(row.get("global_summary_id")),
            "conversation_id": str(row.get("origin_conversation_id") or ""),
            "level": self._summary_level_to_int(row.get("level")),
            "content": str(row.get("text", "")),
            "token_count": max(1, len(str(row.get("text", "")).split())),
            "parent_id": None,
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "origin_citem_ids": origin_ids,
            "metadata": {
                "summary_scope": "global",
                "covers_json": dict(row.get("covers_json") or {}),
                "cluster_id": row.get("cluster_id"),
                "vector_state": row.get("vector_state"),
            },
        }

    def _summary_level_to_int(self, value: Any) -> int:
        mapping = {"EPOCH": 1, "CLUSTER": 2, "MASTER": 3}
        if isinstance(value, int):
            return value
        if value is None:
            return 1
        return mapping.get(str(value).upper(), 1)

    def _local_evidence_to_lineage(
        self,
        *,
        conversation_id: str,
        local_citem_id: str,
        evidence_row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        source_id = evidence_row.get("source_id")
        if source_id:
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "src_kind": "citem",
                "src_id": local_citem_id,
                "dst_kind": "source",
                "dst_id": str(source_id),
                "relation": "DERIVED_FROM_SOURCE",
                "metadata": {"ordinal": int(evidence_row.get("ordinal", 0))},
            })
        locator = dict(evidence_row.get("locator_json") or {})
        source_span_id = locator.get("source_span_id")
        if source_span_id:
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "src_kind": "citem",
                "src_id": local_citem_id,
                "dst_kind": "source_span",
                "dst_id": str(source_span_id),
                "relation": "DERIVED_FROM_SPAN",
                "metadata": {"ordinal": int(evidence_row.get("ordinal", 0))},
            })
        return edges

    def _global_evidence_to_lineage(
        self,
        *,
        conversation_id: str,
        global_citem_id: str,
        evidence_row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        locator = dict(evidence_row.get("locator_json") or {})
        source_id = locator.get("source_id")
        if source_id:
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "src_kind": "citem",
                "src_id": global_citem_id,
                "dst_kind": "source",
                "dst_id": str(source_id),
                "relation": "DERIVED_FROM_SOURCE",
                "metadata": {"ordinal": int(evidence_row.get("ordinal", 0)), "evidence_kind": evidence_row.get("evidence_kind")},
            })
        source_span_id = locator.get("source_span_id")
        if source_span_id:
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "src_kind": "citem",
                "src_id": global_citem_id,
                "dst_kind": "source_span",
                "dst_id": str(source_span_id),
                "relation": "DERIVED_FROM_SPAN",
                "metadata": {"ordinal": int(evidence_row.get("ordinal", 0)), "evidence_kind": evidence_row.get("evidence_kind")},
            })
        return edges

    def _summary_origins_to_lineage(self, *, summary_id: str, origin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for row in origin_rows:
            origin_id = row.get("origin_id")
            if not origin_id:
                continue
            origin_kind = str(row.get("origin_kind") or "citem")
            dst_kind = "citem" if origin_kind.endswith("citem") else "summary"
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "conversation_id": row.get("conversation_id"),
                "src_kind": "summary",
                "src_id": summary_id,
                "dst_kind": dst_kind,
                "dst_id": str(origin_id),
                "relation": "SUMMARIZES",
                "metadata": {"origin_kind": origin_kind, "ordinal": int(row.get("ordinal", 0))},
            })
        return edges

    def _dedupe_lineage_edges(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for edge in edges:
            key = (
                str(edge.get("src_kind", "")),
                str(edge.get("src_id", "")),
                str(edge.get("dst_kind", "")),
                str(edge.get("dst_id", "")),
                str(edge.get("relation", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(edge))
        return out

    async def validate_handoff(self, *, handoff_id: str) -> HandoffValidation:
        manifest_json = await self._db.load_demo_handoff_manifest(handoff_id)
        if manifest_json is None:
            validation = HandoffValidation(handoff_id=handoff_id, valid=False, issues=["manifest_not_found"], evidence_coverage=0.0)
            await self._db.save_demo_handoff_validation(validation.to_dict())
            return validation
        manifest = HandoffManifest(**manifest_json)
        issues: list[str] = []
        expected = self._checksum(
            citem_refs=manifest.citem_refs,
            pyramid_refs=manifest.pyramid_refs,
            task_state=manifest.task_state,
            bundled_citems=manifest.bundled_citems,
            bundled_summaries=manifest.bundled_summaries,
            bundled_sources=manifest.bundled_sources,
            bundled_spans=manifest.bundled_spans,
            bundled_lineage=manifest.bundled_lineage,
        )
        if manifest.checksum != expected:
            issues.append("checksum_mismatch")
        bundled_citem_ids = {str(row.get("citem_id")) for row in manifest.bundled_citems}
        missing_citems = [ref_id for ref_id in manifest.citem_refs if ref_id not in bundled_citem_ids]
        if missing_citems:
            issues.append(f"missing_citem_bundle:{','.join(sorted(missing_citems))}")
        bundled_summary_ids = {str(row.get("summary_id")) for row in manifest.bundled_summaries}
        missing_summaries = [ref_id for ref_id in manifest.pyramid_refs if ref_id not in bundled_summary_ids]
        if missing_summaries:
            issues.append(f"missing_summary_bundle:{','.join(sorted(missing_summaries))}")
        source_lineage_count = 0
        citems_with_evidence: set[str] = set()
        for edge in manifest.bundled_lineage:
            if edge.get("src_kind") == "citem" and edge.get("src_id") in bundled_citem_ids and edge.get("dst_kind") in {"source", "source_span"}:
                source_lineage_count += 1
                citems_with_evidence.add(str(edge.get("src_id")))
        evidence_coverage = 1.0 if not manifest.citem_refs else len(citems_with_evidence) / float(len(manifest.citem_refs))
        if evidence_coverage < 0.80:
            issues.append("evidence_coverage_below_threshold")
        valid = not issues
        validation = HandoffValidation(
            handoff_id=handoff_id,
            valid=valid,
            issues=issues,
            evidence_coverage=evidence_coverage,
        )
        await self._db.save_demo_handoff_validation(validation.to_dict())
        if manifest.source_run_id:
            await self._runs.write_json_artifact(
                conversation_id=manifest.conversation_id,
                run_id=manifest.source_run_id,
                relative_path=f"handoff_validation_{handoff_id}.json",
                payload=validation.to_dict(),
            )
        return validation

    async def restore_handoff(
        self,
        *,
        handoff_id: str,
        target_conversation_id: str,
        target_run_id: str | None = None,
    ) -> HandoffRestore:
        manifest_json = await self._db.load_demo_handoff_manifest(handoff_id)
        if manifest_json is None:
            restore = HandoffRestore(
                restore_id=str(uuid.uuid4()),
                handoff_id=handoff_id,
                target_conversation_id=target_conversation_id,
                target_run_id=target_run_id,
                valid=False,
                diff={"error": "manifest_not_found"},
            )
            await self._db.save_demo_handoff_restore(restore.to_dict())
            return restore
        manifest = HandoffManifest(**manifest_json)
        validation_json = await self._db.load_demo_handoff_validation(handoff_id)
        validation = HandoffValidation(**validation_json) if validation_json is not None else await self.validate_handoff(handoff_id=handoff_id)
        if not validation.valid:
            restore = HandoffRestore(
                restore_id=str(uuid.uuid4()),
                handoff_id=handoff_id,
                target_conversation_id=target_conversation_id,
                target_run_id=target_run_id,
                valid=False,
                diff={"validation_issues": list(validation.issues)},
            )
            await self._db.save_demo_handoff_restore(restore.to_dict())
            if target_run_id:
                await self._runs.write_json_artifact(
                    conversation_id=target_conversation_id,
                    run_id=target_run_id,
                    relative_path=f"reconstruction_diff_{restore.restore_id}.json",
                    payload=restore.to_dict(),
                )
            return restore

        if await self._db.get_conversation(target_conversation_id) is None:
            await self._db.create_conversation(target_conversation_id)

        restore_timestamp = datetime.now(UTC).isoformat()

        source_map: dict[str, str] = {}
        restored_sources_by_id: dict[str, dict[str, Any]] = {}
        for row in manifest.bundled_sources:
            new_id = str(uuid.uuid4())
            source_map[str(row.get("source_id"))] = new_id
            clone = dict(row)
            clone["source_id"] = new_id
            clone["conversation_id"] = target_conversation_id
            meta = dict(clone.get("metadata") or {})
            meta["handoff_origin_source_id"] = row.get("source_id")
            clone["metadata"] = meta
            await self._db.save_demo_source(clone)
            restored_sources_by_id[new_id] = clone

        span_map: dict[str, str] = {}
        restored_spans_by_id: dict[str, dict[str, Any]] = {}
        for row in manifest.bundled_spans:
            old_span_id = str(row.get("span_id"))
            old_source_id = str(row.get("source_id"))
            if old_source_id not in source_map:
                continue
            new_id = str(uuid.uuid4())
            span_map[old_span_id] = new_id
            clone = dict(row)
            clone["span_id"] = new_id
            clone["conversation_id"] = target_conversation_id
            clone["source_id"] = source_map[old_source_id]
            await self._db.save_demo_source_span(clone)
            restored_spans_by_id[new_id] = clone

        citem_map: dict[str, str] = {str(row.get("citem_id")): str(uuid.uuid4()) for row in manifest.bundled_citems}
        for row in manifest.bundled_citems:
            old_id = str(row.get("citem_id"))
            new_id = citem_map[old_id]
            deps = [citem_map.get(str(dep), str(dep)) for dep in row.get("dependency_ids", [])]
            citem = CItem(
                citem_id=new_id,
                conversation_id=target_conversation_id,
                content=str(row.get("content", "")),
                item_type=str(row.get("item_type", "FACT")),
                scope=str(row.get("scope", "episodic")),
                scope_status=str(row.get("scope_status", "active")),
                importance=float(row.get("importance", 0.5)),
                confidence=float(row.get("confidence", 1.0)),
                validation_label=row.get("validation_label"),
                conflict_status=str(row.get("conflict_status", "none")),
                phase_ingested=str(row.get("phase_ingested", "IDLE")),
                actor=str(row.get("actor", "agent")),
                motivation=str(row.get("motivation", "handoff_restore")),
                dependency_ids=deps,
                token_count=int(row.get("token_count", 0)),
                chunk_kind=row.get("chunk_kind"),
            )
            citem.content_hash = row.get("content_hash")
            citem.summarized_by_node_id = None
            await self._store.save(citem)

        witness_restore_stats = await self._restore_witness_citems(
            manifest=manifest,
            handoff_id=handoff_id,
            target_conversation_id=target_conversation_id,
            restore_timestamp=restore_timestamp,
            citem_map=citem_map,
            source_map=source_map,
            span_map=span_map,
            restored_sources_by_id=restored_sources_by_id,
            restored_spans_by_id=restored_spans_by_id,
        )

        for edge in manifest.bundled_lineage:
            if edge.get("src_kind") != "citem":
                continue
            old_src = str(edge.get("src_id"))
            if old_src not in citem_map:
                continue
            dst_kind = str(edge.get("dst_kind"))
            dst_id = str(edge.get("dst_id"))
            mapped_dst = dst_id
            if dst_kind == "source" and dst_id in source_map:
                mapped_dst = source_map[dst_id]
            elif dst_kind == "source_span" and dst_id in span_map:
                mapped_dst = span_map[dst_id]
            elif dst_kind == "citem" and dst_id in citem_map:
                mapped_dst = citem_map[dst_id]
            else:
                continue
            clone_edge = dict(edge)
            clone_edge["edge_id"] = str(uuid.uuid4())
            clone_edge["conversation_id"] = target_conversation_id
            clone_edge["src_id"] = citem_map[old_src]
            clone_edge["dst_id"] = mapped_dst
            await self._db.save_demo_lineage_edge(clone_edge)

        summary_map: dict[str, str] = {str(row.get("summary_id")): str(uuid.uuid4()) for row in manifest.bundled_summaries}
        for row in manifest.bundled_summaries:
            old_summary_id = str(row.get("summary_id"))
            new_summary_id = summary_map[old_summary_id]
            origin_ids = [citem_map.get(str(cid), str(cid)) for cid in row.get("origin_citem_ids", []) if str(cid) in citem_map]
            node = SummaryNode(
                node_id=new_summary_id,
                conversation_id=target_conversation_id,
                level=int(row.get("level", 1)),
                content=str(row.get("content", row.get("summary_text", ""))),
                token_count=int(row.get("token_count", 0)),
                parent_id=None,
                origin_citem_ids=origin_ids,
            )
            await self._db.save_summary(node)
            await self._db.save_demo_summary_resolution({
                "summary_id": new_summary_id,
                "conversation_id": target_conversation_id,
                "summary_text": node.content,
                "origin_citem_ids": origin_ids,
                "metadata": {"handoff_origin_summary_id": old_summary_id},
            })
            for cid in origin_ids:
                await self._db.save_demo_lineage_edge({
                    "edge_id": str(uuid.uuid4()),
                    "conversation_id": target_conversation_id,
                    "src_kind": "summary",
                    "src_id": new_summary_id,
                    "dst_kind": "citem",
                    "dst_id": cid,
                    "relation": "SUMMARIZES",
                    "metadata": {"handoff_origin_summary_id": old_summary_id},
                })

        witness_restore_stats.update(await self._restore_witness_summaries(
            manifest=manifest,
            target_conversation_id=target_conversation_id,
            restore_timestamp=restore_timestamp,
            citem_map=citem_map,
            summary_map=summary_map,
        ))

        restored_task_memory = self._restore_task_memory(manifest.task_state.get("task_memory", {}), target_conversation_id)
        plan_snapshot = manifest.task_state.get("plan")
        restored_plan = self._restore_plan(plan_snapshot, target_conversation_id) if isinstance(plan_snapshot, dict) else None
        if restored_plan is not None:
            restored_task_memory.active_plan_id = restored_plan.plan_id
            await self._db.save_plan_with_task_memory(restored_plan, restored_task_memory)
        else:
            await self._db.save_task_memory(restored_task_memory)

        diff = self._reconstruction_diff(
            manifest=manifest,
            restored_task_memory=restored_task_memory,
            restored_plan=restored_plan,
            citem_map=citem_map,
            summary_map=summary_map,
            source_map=source_map,
            span_map=span_map,
            witness_restore_stats=witness_restore_stats,
        )
        restore = HandoffRestore(
            restore_id=str(uuid.uuid4()),
            handoff_id=handoff_id,
            target_conversation_id=target_conversation_id,
            target_run_id=target_run_id,
            valid=True,
            reconstructed_task_state={
                "task_memory": self._task_memory_to_dict(restored_task_memory),
                "plan": self._plan_to_dict(restored_plan) if restored_plan is not None else None,
            },
            diff=diff,
        )
        await self._db.save_demo_handoff_restore(restore.to_dict())
        if target_run_id:
            await self._runs.write_json_artifact(
                conversation_id=target_conversation_id,
                run_id=target_run_id,
                relative_path=f"reconstruction_diff_{restore.restore_id}.json",
                payload=restore.to_dict(),
            )
        return restore

    def _citem_to_dict(self, item: CItem) -> dict[str, Any]:
        return {
            "citem_id": item.citem_id,
            "conversation_id": item.conversation_id,
            "content": item.content,
            "item_type": item.item_type,
            "scope": item.scope,
            "scope_status": item.scope_status,
            "importance": item.importance,
            "confidence": item.confidence,
            "validation_label": item.validation_label,
            "conflict_status": item.conflict_status,
            "phase_ingested": item.phase_ingested,
            "actor": item.actor,
            "motivation": item.motivation,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "dependency_ids": list(item.dependency_ids),
            "token_count": item.token_count,
            "content_hash": item.content_hash,
            "chunk_kind": item.chunk_kind,
        }

    def _summary_to_dict(self, node: SummaryNode, resolution_row: dict[str, Any] | None) -> dict[str, Any]:
        origin_ids = list(node.origin_citem_ids)
        if resolution_row is not None and resolution_row.get("origin_citem_ids"):
            origin_ids = [str(v) for v in resolution_row.get("origin_citem_ids", [])]
        metadata = dict((resolution_row or {}).get("metadata") or {})
        metadata.setdefault("summary_scope", "local")
        return {
            "summary_id": node.node_id,
            "conversation_id": node.conversation_id,
            "level": node.level,
            "content": node.content,
            "token_count": node.token_count,
            "parent_id": node.parent_id,
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "updated_at": None,
            "origin_citem_ids": origin_ids,
            "metadata": metadata,
        }

    def _summary_scope(self, row: dict[str, Any]) -> str:
        metadata = dict(row.get("metadata") or {})
        scope = str(metadata.get("summary_scope") or "").lower()
        return "global" if scope == "global" else "local"

    def _summary_level_label(self, value: Any, *, summary_scope: str) -> str:
        if isinstance(value, int):
            level_value = value
        else:
            mapping = {"EPOCH": 1, "CLUSTER": 2, "MASTER": 3}
            level_value = mapping.get(str(value).upper(), 1 if summary_scope == "local" else 2)
        if summary_scope == "global":
            return "MASTER" if level_value >= 3 else "CLUSTER"
        return {1: "EPOCH", 2: "CLUSTER", 3: "MASTER"}.get(level_value, "EPOCH")

    def _collect_citem_evidence_specs(self, *, manifest: HandoffManifest, old_citem_id: str) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        for edge in manifest.bundled_lineage:
            if edge.get("src_kind") != "citem" or str(edge.get("src_id")) != old_citem_id:
                continue
            dst_kind = str(edge.get("dst_kind") or "")
            if dst_kind not in {"source", "source_span"}:
                continue
            metadata = dict(edge.get("metadata") or {})
            ordinal = int(metadata.get("ordinal", 0) or 0)
            spec = grouped.setdefault(
                ordinal,
                {
                    "ordinal": ordinal,
                    "source_id": None,
                    "source_span_id": None,
                    "evidence_kind": metadata.get("evidence_kind"),
                },
            )
            if not spec.get("evidence_kind") and metadata.get("evidence_kind"):
                spec["evidence_kind"] = metadata.get("evidence_kind")
            if dst_kind == "source":
                spec["source_id"] = str(edge.get("dst_id"))
            else:
                spec["source_span_id"] = str(edge.get("dst_id"))
        return [grouped[ordinal] for ordinal in sorted(grouped)]

    def _collect_summary_origin_specs(
        self,
        *,
        manifest: HandoffManifest,
        row: dict[str, Any],
        summary_scope: str,
    ) -> list[dict[str, Any]]:
        old_summary_id = str(row.get("summary_id"))
        specs: list[dict[str, Any]] = []
        for edge in manifest.bundled_lineage:
            if edge.get("src_kind") != "summary" or str(edge.get("src_id")) != old_summary_id:
                continue
            dst_kind = str(edge.get("dst_kind") or "")
            if dst_kind not in {"citem", "summary"}:
                continue
            metadata = dict(edge.get("metadata") or {})
            origin_kind = str(metadata.get("origin_kind") or "")
            if not origin_kind:
                if summary_scope == "global":
                    origin_kind = "global_summary" if dst_kind == "summary" else "global_citem"
                else:
                    origin_kind = "local_summary" if dst_kind == "summary" else "local_citem"
            specs.append({
                "origin_kind": origin_kind,
                "origin_id": str(edge.get("dst_id")),
                "ordinal": int(metadata.get("ordinal", 0) or 0),
            })
        default_kind = "global_citem" if summary_scope == "global" else "local_citem"
        specs.extend(
            {
                "origin_kind": default_kind,
                "origin_id": str(origin_id),
                "ordinal": idx,
            }
            for idx, origin_id in enumerate(row.get("origin_citem_ids", []) or [])
            if origin_id
        )
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for spec in sorted(specs, key=lambda item: (int(item.get("ordinal", 0)), str(item.get("origin_id", "")))):
            key = (str(spec.get("origin_kind", "")), str(spec.get("origin_id", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
        return deduped

    async def _restore_witness_citems(
        self,
        *,
        manifest: HandoffManifest,
        handoff_id: str,
        target_conversation_id: str,
        restore_timestamp: str,
        citem_map: dict[str, str],
        source_map: dict[str, str],
        span_map: dict[str, str],
        restored_sources_by_id: dict[str, dict[str, Any]],
        restored_spans_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        stats = {
            "restored_local_citems_witness": 0,
            "restored_global_citems_witness": 0,
            "restored_local_citem_evidence": 0,
            "restored_global_citem_evidence": 0,
            "skipped_global_citems_witness": 0,
        }
        save_local_citem_record = getattr(self._db, "save_local_citem_record", None)
        save_local_citem_evidence = getattr(self._db, "save_local_citem_evidence", None)
        save_global_citem_record = getattr(self._db, "save_global_citem_record", None)
        save_global_citem_evidence = getattr(self._db, "save_global_citem_evidence", None)

        for row in manifest.bundled_citems:
            old_citem_id = str(row.get("citem_id"))
            new_citem_id = citem_map.get(old_citem_id)
            if not new_citem_id:
                continue
            scope = str(row.get("scope", "episodic") or "episodic").lower()
            semantic_identity_id = str(row.get("semantic_identity_id") or uuid.uuid4())
            meta_json = dict(row.get("meta_json") or {})
            meta_json.setdefault("handoff_origin_citem_id", old_citem_id)
            provenance_json = dict(row.get("provenance_json") or {})
            provenance_json["handoff_restore"] = {
                "handoff_id": handoff_id,
                "restored_at": restore_timestamp,
                "source_conversation_id": manifest.conversation_id,
                "source_citem_id": old_citem_id,
            }
            evidence_specs = self._collect_citem_evidence_specs(manifest=manifest, old_citem_id=old_citem_id)

            if scope == "global":
                if not callable(save_global_citem_record) or not callable(save_global_citem_evidence):
                    continue
                promotion_origin_local_citem_id = None
                if row.get("promotion_origin_local_citem_id") is not None:
                    promotion_origin_local_citem_id = citem_map.get(str(row.get("promotion_origin_local_citem_id")))
                if promotion_origin_local_citem_id is None:
                    stats["skipped_global_citems_witness"] += 1
                    continue
                await save_global_citem_record({
                    "global_citem_id": new_citem_id,
                    "semantic_identity_id": semantic_identity_id,
                    "origin_conversation_id": target_conversation_id,
                    "promotion_origin_local_citem_id": promotion_origin_local_citem_id,
                    "type": str(row.get("item_type", "FACT")),
                    "text": str(row.get("content", "")),
                    "embedding_text": str(row.get("content", "")),
                    "meta_json": meta_json,
                    "provenance_json": provenance_json,
                    "validity": row.get("validity") or row.get("validation_label") or "unknown",
                    "salience": float(row.get("salience", row.get("importance", 0.0)) or 0.0),
                    "created_at": row.get("created_at") or restore_timestamp,
                    "updated_at": restore_timestamp,
                    "vector_state": "INDEXED",
                    "embedding_model_id": None,
                    "embedding_schema_version": None,
                    "expires_at": row.get("expires_at"),
                    "is_pinned": bool(row.get("is_pinned", False)),
                    "was_cited": bool(row.get("was_cited", False)),
                    "last_used_at": row.get("last_used_at"),
                })
                stats["restored_global_citems_witness"] += 1
                for spec in evidence_specs:
                    mapped_span_id = span_map.get(str(spec.get("source_span_id"))) if spec.get("source_span_id") else None
                    mapped_source_id = source_map.get(str(spec.get("source_id"))) if spec.get("source_id") else None
                    if mapped_source_id is None and mapped_span_id is not None:
                        mapped_source_id = restored_spans_by_id.get(mapped_span_id, {}).get("source_id")
                    locator_json: dict[str, Any] = {}
                    if mapped_source_id:
                        locator_json["source_id"] = mapped_source_id
                    if mapped_span_id:
                        locator_json["source_span_id"] = mapped_span_id
                    if not locator_json:
                        continue
                    source_text_snapshot = None
                    if mapped_span_id is not None:
                        source_text_snapshot = restored_spans_by_id.get(mapped_span_id, {}).get("preview_text")
                    if not source_text_snapshot and mapped_source_id is not None:
                        source_row = restored_sources_by_id.get(mapped_source_id, {})
                        source_text_snapshot = source_row.get("process_text") or source_row.get("display_text")
                    evidence_kind = str(spec.get("evidence_kind") or ("chunk_snippet" if mapped_span_id else "source_snippet"))
                    if evidence_kind not in {"source_snippet", "chunk_snippet", "external_ref"}:
                        evidence_kind = "chunk_snippet" if mapped_span_id else "source_snippet"
                    await save_global_citem_evidence({
                        "global_citem_id": new_citem_id,
                        "ordinal": int(spec.get("ordinal", 0) or 0),
                        "evidence_kind": evidence_kind,
                        "source_text_snapshot": source_text_snapshot,
                        "locator_json": locator_json,
                    })
                    stats["restored_global_citem_evidence"] += 1
                continue

            if not callable(save_local_citem_record) or not callable(save_local_citem_evidence):
                continue
            await save_local_citem_record({
                "local_citem_id": new_citem_id,
                "semantic_identity_id": semantic_identity_id,
                "conversation_id": target_conversation_id,
                "type": str(row.get("item_type", "FACT")),
                "text": str(row.get("content", "")),
                "embedding_text": str(row.get("content", "")),
                "meta_json": meta_json,
                "provenance_json": provenance_json,
                "validity": row.get("validity") or row.get("validation_label") or "unknown",
                "salience": float(row.get("salience", row.get("importance", 0.0)) or 0.0),
                "created_at": row.get("created_at") or restore_timestamp,
                "updated_at": restore_timestamp,
                "vector_state": "INDEXED",
                "embedding_model_id": None,
                "embedding_schema_version": None,
                "expires_at": row.get("expires_at"),
                "is_pinned": bool(row.get("is_pinned", False)),
                "was_cited": bool(row.get("was_cited", False)),
                "last_used_at": row.get("last_used_at"),
                "normalizer_version": int(row.get("normalizer_version", 1) or 1),
                "citem_builder_version": int(row.get("citem_builder_version", 1) or 1),
            })
            stats["restored_local_citems_witness"] += 1
            for spec in evidence_specs:
                mapped_span_id = span_map.get(str(spec.get("source_span_id"))) if spec.get("source_span_id") else None
                mapped_source_id = source_map.get(str(spec.get("source_id"))) if spec.get("source_id") else None
                if mapped_source_id is None and mapped_span_id is not None:
                    mapped_source_id = restored_spans_by_id.get(mapped_span_id, {}).get("source_id")
                if mapped_source_id is None and mapped_span_id is None:
                    continue
                locator_json: dict[str, Any] = {}
                if mapped_source_id:
                    locator_json["source_id"] = mapped_source_id
                if mapped_span_id:
                    locator_json["source_span_id"] = mapped_span_id
                await save_local_citem_evidence({
                    "local_citem_id": new_citem_id,
                    "source_id": mapped_source_id,
                    "chunk_id": None,
                    "edu_id": None,
                    "ordinal": int(spec.get("ordinal", 0) or 0),
                    "locator_json": locator_json,
                })
                stats["restored_local_citem_evidence"] += 1
        return stats

    async def _restore_witness_summaries(
        self,
        *,
        manifest: HandoffManifest,
        target_conversation_id: str,
        restore_timestamp: str,
        citem_map: dict[str, str],
        summary_map: dict[str, str],
    ) -> dict[str, int]:
        stats = {
            "restored_local_summaries_witness": 0,
            "restored_global_summaries_witness": 0,
            "restored_local_summary_origins": 0,
            "restored_global_summary_origins": 0,
        }
        save_local_summary_record = getattr(self._db, "save_local_summary_record", None)
        save_local_summary_origin = getattr(self._db, "save_local_summary_origin", None)
        save_global_summary_record = getattr(self._db, "save_global_summary_record", None)
        save_global_summary_origin = getattr(self._db, "save_global_summary_origin", None)

        for row in manifest.bundled_summaries:
            old_summary_id = str(row.get("summary_id"))
            new_summary_id = summary_map.get(old_summary_id)
            if not new_summary_id:
                continue
            summary_scope = self._summary_scope(row)
            level_label = self._summary_level_label(row.get("level"), summary_scope=summary_scope)
            metadata = dict(row.get("metadata") or {})
            covers_json = dict(metadata.get("covers_json") or {})
            origin_specs = self._collect_summary_origin_specs(manifest=manifest, row=row, summary_scope=summary_scope)

            if summary_scope == "global":
                if not callable(save_global_summary_record) or not callable(save_global_summary_origin):
                    continue
                mapped_origin_specs: list[dict[str, Any]] = []
                mapped_origin_global_ids: list[str] = []
                for spec in origin_specs:
                    origin_kind = "global_summary" if str(spec.get("origin_kind", "")).endswith("summary") else "global_citem"
                    origin_id = summary_map.get(str(spec.get("origin_id"))) if origin_kind == "global_summary" else citem_map.get(str(spec.get("origin_id")))
                    if origin_id is None:
                        continue
                    mapped_origin_specs.append({
                        "origin_kind": origin_kind,
                        "origin_id": origin_id,
                        "ordinal": int(spec.get("ordinal", 0) or 0),
                    })
                    if origin_kind == "global_citem":
                        mapped_origin_global_ids.append(origin_id)
                local_covers = [str(v) for v in covers_json.pop("origin_citem_ids", []) if v]
                covers_json["origin_global_citem_ids"] = mapped_origin_global_ids or [citem_map[str(v)] for v in local_covers if str(v) in citem_map]
                await save_global_summary_record({
                    "global_summary_id": new_summary_id,
                    "level": level_label,
                    "cluster_id": metadata.get("cluster_id"),
                    "text": str(row.get("content", row.get("summary_text", ""))),
                    "covers_json": covers_json,
                    "created_at": row.get("created_at") or restore_timestamp,
                    "updated_at": restore_timestamp,
                    "vector_state": "NONE",
                    "embedding_model_id": None,
                    "embedding_schema_version": None,
                })
                stats["restored_global_summaries_witness"] += 1
                for spec in mapped_origin_specs:
                    await save_global_summary_origin({
                        "global_summary_id": new_summary_id,
                        "origin_kind": spec["origin_kind"],
                        "origin_id": spec["origin_id"],
                        "ordinal": spec["ordinal"],
                    })
                    stats["restored_global_summary_origins"] += 1
                continue

            if not callable(save_local_summary_record) or not callable(save_local_summary_origin):
                continue
            mapped_origin_specs = []
            mapped_origin_citem_ids: list[str] = []
            for spec in origin_specs:
                origin_kind = "local_summary" if str(spec.get("origin_kind", "")).endswith("summary") else "local_citem"
                origin_id = summary_map.get(str(spec.get("origin_id"))) if origin_kind == "local_summary" else citem_map.get(str(spec.get("origin_id")))
                if origin_id is None:
                    continue
                mapped_origin_specs.append({
                    "origin_kind": origin_kind,
                    "origin_id": origin_id,
                    "ordinal": int(spec.get("ordinal", 0) or 0),
                })
                if origin_kind == "local_citem":
                    mapped_origin_citem_ids.append(origin_id)
            covers_json["origin_citem_ids"] = mapped_origin_citem_ids
            await save_local_summary_record({
                "local_summary_id": new_summary_id,
                "conversation_id": target_conversation_id,
                "level": level_label,
                "cluster_id": metadata.get("cluster_id"),
                "epoch_no": metadata.get("epoch_no"),
                "text": str(row.get("content", row.get("summary_text", ""))),
                "covers_json": covers_json,
                "created_at": row.get("created_at") or restore_timestamp,
                "updated_at": restore_timestamp,
                "vector_state": "NONE",
                "embedding_model_id": None,
                "embedding_schema_version": None,
                "is_pinned": bool(metadata.get("is_pinned", False)),
                "was_cited": bool(metadata.get("was_cited", False)),
                "last_used_at": metadata.get("last_used_at"),
            })
            stats["restored_local_summaries_witness"] += 1
            for spec in mapped_origin_specs:
                await save_local_summary_origin({
                    "local_summary_id": new_summary_id,
                    "origin_kind": spec["origin_kind"],
                    "origin_id": spec["origin_id"],
                    "ordinal": spec["ordinal"],
                    "conversation_id": target_conversation_id,
                })
                stats["restored_local_summary_origins"] += 1
        return stats

    def _extract_plan_snapshot(self, bundle: Any) -> dict[str, Any] | None:
        checkpoints = list(bundle.checkpoints or [])
        for checkpoint in reversed(checkpoints):
            state = checkpoint.get("state") or {}
            plan = state.get("plan")
            if isinstance(plan, dict):
                return plan
        return None

    def _restore_task_memory(self, snapshot: dict[str, Any], conversation_id: str) -> TaskMemory:
        last_turn_at = self._parse_dt(snapshot.get("last_turn_at"))
        created_at = self._parse_dt(snapshot.get("created_at")) or datetime.now(UTC)
        return TaskMemory(
            conversation_id=conversation_id,
            turn_count=int(snapshot.get("turn_count", 0)),
            phase=str(snapshot.get("phase") or "IDLE"),
            active_plan_id=snapshot.get("active_plan_id"),
            awaiting_user_input=bool(snapshot.get("awaiting_user_input", False)),
            turn_in_progress=False,
            stall_count=int(snapshot.get("stall_count", 0)),
            last_turn_at=last_turn_at,
            created_at=created_at,
        )

    def _restore_plan(self, snapshot: dict[str, Any], conversation_id: str) -> Plan | None:
        if not snapshot:
            return None
        plan_id = str(snapshot.get("plan_id") or uuid.uuid4())
        steps: list[PlanStep] = []
        for step in snapshot.get("steps", []) or []:
            steps.append(
                PlanStep(
                    step_id=str(step.get("step_id") or uuid.uuid4()),
                    plan_id=plan_id,
                    description=str(step.get("description", "")),
                    status=str(step.get("status", "PENDING")),
                    tool_name=step.get("tool_name"),
                    result_summary=step.get("result_summary"),
                )
            )
        return Plan(
            plan_id=plan_id,
            conversation_id=conversation_id,
            goal=str(snapshot.get("goal", "")),
            status=str(snapshot.get("status", "RUNNING")),
            steps=steps,
            auto_continue=bool(snapshot.get("auto_continue", False)),
        )

    def _task_memory_to_dict(self, task_memory: TaskMemory) -> dict[str, Any]:
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

    def _plan_to_dict(self, plan: Plan | None) -> dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "plan_id": plan.plan_id,
            "conversation_id": plan.conversation_id,
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

    def _reconstruction_diff(
        self,
        *,
        manifest: HandoffManifest,
        restored_task_memory: TaskMemory,
        restored_plan: Plan | None,
        citem_map: dict[str, str],
        summary_map: dict[str, str],
        source_map: dict[str, str],
        span_map: dict[str, str],
        witness_restore_stats: dict[str, int],
    ) -> dict[str, Any]:
        diff = {
            "source_turn_count": int((manifest.task_state.get("task_memory") or {}).get("turn_count", 0)),
            "restored_turn_count": restored_task_memory.turn_count,
            "source_phase": (manifest.task_state.get("task_memory") or {}).get("phase"),
            "restored_phase": restored_task_memory.phase,
            "source_active_plan_id": (manifest.task_state.get("task_memory") or {}).get("active_plan_id"),
            "restored_active_plan_id": restored_task_memory.active_plan_id,
            "restored_plan_goal": restored_plan.goal if restored_plan is not None else None,
            "restored_plan_steps": len(restored_plan.steps) if restored_plan is not None else 0,
            "restored_citems": len(citem_map),
            "restored_summaries": len(summary_map),
            "restored_sources": len(source_map),
            "restored_spans": len(span_map),
            "citem_ref_map": dict(citem_map),
            "summary_ref_map": dict(summary_map),
        }
        diff.update({str(key): int(value) for key, value in witness_restore_stats.items()})
        return diff

    def _checksum(
        self,
        *,
        citem_refs: list[str],
        pyramid_refs: list[str],
        task_state: dict[str, Any],
        bundled_citems: list[dict[str, Any]],
        bundled_summaries: list[dict[str, Any]],
        bundled_sources: list[dict[str, Any]],
        bundled_spans: list[dict[str, Any]],
        bundled_lineage: list[dict[str, Any]],
    ) -> str:
        canonical = json.dumps(
            {
                "citem_refs": list(citem_refs),
                "pyramid_refs": list(pyramid_refs),
                "task_state": task_state,
                "bundled_citems": bundled_citems,
                "bundled_summaries": bundled_summaries,
                "bundled_sources": bundled_sources,
                "bundled_spans": bundled_spans,
                "bundled_lineage": bundled_lineage,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None
