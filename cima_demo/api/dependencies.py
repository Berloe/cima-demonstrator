"""FastAPI Depends() factories (KIMA_API_Layer_v0.5 §11.2)."""
from __future__ import annotations

from typing import cast

from fastapi import Request

from cima_demo.application.orchestrator import AgentOrchestrator
from cima_demo.application.stream_manager import StreamManager
from cima_demo.domain.ports import CItemStorePort, RelDBPort, RerankerPort


def get_db(request: Request) -> RelDBPort:
    return cast(RelDBPort, request.app.state.db)


def get_citem_store(request: Request) -> CItemStorePort:
    return cast(CItemStorePort, request.app.state.citem_store)


def get_reranker(request: Request) -> RerankerPort:
    return cast(RerankerPort, request.app.state.reranker)


def get_orchestrator(request: Request) -> AgentOrchestrator:
    return cast(AgentOrchestrator, request.app.state.orchestrator)


def get_stream_manager(request: Request) -> StreamManager:
    return cast(StreamManager, request.app.state.stream_manager)


def get_handoff_service(request: Request):
    return request.app.state.handoff_service


def get_geometry_reader(request: Request):
    return request.app.state.geometry_reader


def get_geometry_commands(request: Request):
    return request.app.state.geometry_commands


def get_lifecycle_audit_service(request: Request):
    return request.app.state.lifecycle_audit_service


def get_context_service(request: Request):
    return request.app.state.context_service


def get_run_journal(request: Request):
    return request.app.state.demo_run_journal


def get_lineage_service(request: Request):
    return request.app.state.lineage_service


def get_memory_service(request: Request):
    return request.app.state.memory_service


def get_source_registration_service(request: Request):
    return request.app.state.source_registration_service
