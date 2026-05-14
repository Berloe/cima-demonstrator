from __future__ import annotations

from pathlib import Path


def logical_ddl_v1_path() -> Path:
    return Path(__file__).resolve().parents[2] / "doc" / "witness_backend_v1_1" / "logical_ddl_v1.sql"


def load_logical_ddl_v1() -> str:
    return logical_ddl_v1_path().read_text()
