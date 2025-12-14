import io
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.web import get_ingestion_service
from src.web.routes.upload import upload_router, get_file_type
from src.infra.db.database import get_session


@pytest_asyncio.fixture
async def app_client():
    app = FastAPI()
    app.include_router(upload_router)

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield app, client

    app.dependency_overrides.clear()


def _override_session(app, mock_session):
    async def override():
        yield mock_session
    app.dependency_overrides[get_session] = override


@pytest.mark.asyncio
async def test_upload_document_success(app_client):
    app, client = app_client

    mock_session = SimpleNamespace()
    _override_session(app, mock_session)

    mock_ingestion_result = SimpleNamespace(
        chunks_count=3,
        file_type="document",
        files_processed=1,
        errors=["warning"]
    )

    ingestion_service = SimpleNamespace(
        processFile=AsyncMock(return_value=mock_ingestion_result)
    )
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service

    with patch("src.web.routes.upload.UserRepository") as repo_cls:
        repo_instance = SimpleNamespace()
        repo_instance.get_by_telegram_id = AsyncMock(return_value=SimpleNamespace(role="admin"))
        repo_cls.return_value = repo_instance

        files = {"file": ("contract.pdf", io.BytesIO(b"data"), "application/pdf")}
        response = await client.post("/upload", data={"user_id": "1"}, files=files)

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks_added"] == 3
    assert payload["file_type"] == "document"
    assert payload["errors"] == ["warning"]
    ingestion_service.processFile.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_document_unsupported(app_client):
    app, client = app_client

    mock_session = SimpleNamespace()
    _override_session(app, mock_session)
    app.dependency_overrides[get_ingestion_service] = lambda: SimpleNamespace()

    with patch("src.web.routes.upload.UserRepository") as repo_cls:
        repo_instance = SimpleNamespace()
        repo_instance.get_by_telegram_id = AsyncMock(return_value=SimpleNamespace(role="admin"))
        repo_cls.return_value = repo_instance

        files = {"file": ("unknown.xyz", io.BytesIO(b"data"), "application/octet-stream")}
        response = await client.post("/upload", data={"user_id": "1"}, files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Неподдерживаемый формат"


@pytest.mark.asyncio
async def test_upload_document_forbidden(app_client):
    app, client = app_client

    mock_session = SimpleNamespace()
    _override_session(app, mock_session)
    app.dependency_overrides[get_ingestion_service] = lambda: SimpleNamespace()

    with patch("src.web.routes.upload.UserRepository") as repo_cls:
        repo_instance = SimpleNamespace()
        repo_instance.get_by_telegram_id = AsyncMock(return_value=SimpleNamespace(role="user"))
        repo_cls.return_value = repo_instance

        files = {"file": ("contract.pdf", io.BytesIO(b"data"), "application/pdf")}
        response = await client.post("/upload", data={"user_id": "1"}, files=files)

    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden: admin only"


@pytest.mark.asyncio
async def test_get_file_type_helper():
    assert get_file_type("archive.zip") == (True, "archive")
    assert get_file_type("contract.pdf") == (True, "document")
    assert get_file_type("notes.txt") == (True, "document")
    assert get_file_type("fun.exe") == (False, "unsupported")
