from __future__ import annotations

from cima_demo.witness_backend.sql import load_logical_ddl_v1


def test_logical_ddl_contains_required_schemas_and_core_tables() -> None:
    ddl = load_logical_ddl_v1()
    assert "CREATE SCHEMA IF NOT EXISTS cima;" in ddl
    assert "CREATE SCHEMA IF NOT EXISTS cima_rm;" in ddl
    assert "CREATE SCHEMA IF NOT EXISTS geom;" in ddl
    for required in [
        "CREATE TABLE IF NOT EXISTS cima.local_citem",
        "CREATE TABLE IF NOT EXISTS cima.global_citem",
        "CREATE TABLE IF NOT EXISTS cima.handoff_manifest",
        "CREATE TABLE IF NOT EXISTS cima.ephemeral_vector",
        "CREATE TABLE IF NOT EXISTS cima.consumer_effect",
        "CREATE TABLE IF NOT EXISTS cima.outbox",
        "CREATE TABLE IF NOT EXISTS geom.run",
        "CREATE TABLE IF NOT EXISTS geom.item_state",
    ]:
        assert required in ddl


def test_logical_ddl_preserves_frontend_fidelity_and_canonical_fts_scope() -> None:
    ddl = load_logical_ddl_v1()
    assert "display_text" in ddl
    assert "content_text" in ddl
    assert "global_citem_search_gin" in ddl
    assert "local_citem_search_gin" in ddl
    assert "source_search_gin" in ddl
    assert "chunk_search_gin" not in ddl
    assert "CREATE TABLE IF NOT EXISTS cima.job" not in ddl
