"""Tests run against a throwaway database so they never disturb the dev/demo data --
in particular the race slot the graders use."""

import os
import tempfile
from pathlib import Path

TMP_DB = Path(tempfile.gettempdir()) / "padel_test.db"
os.environ["DB_PATH"] = str(TMP_DB)

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def database():
    from app.ingest import ingest

    for suffix in ("", "-wal", "-shm"):
        Path(str(TMP_DB) + suffix).unlink(missing_ok=True)
    ingest(embed=False)
    yield
    for suffix in ("", "-wal", "-shm"):
        Path(str(TMP_DB) + suffix).unlink(missing_ok=True)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
