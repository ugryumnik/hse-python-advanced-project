import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.web import get_doc_generation_service
from src.web.routes.generate import generate_router
from src.infra.db.database import get_session


@pytest_asyncio.fixture
async def app_client():
    app = FastAPI()
    app.include_router(generate_router)

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield app, client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_document_unauthorized(app_client):
    app, client = app_client

    mock_service = SimpleNamespace()
    mock_service.generate = AsyncMock()
    app.dependency_overrides[get_doc_generation_service] = lambda: mock_service

    async def override_session():
        yield SimpleNamespace()

    app.dependency_overrides[get_session] = override_session

    with patch("src.web.routes.generate.UserRepository") as repo_cls:
        repo_instance = SimpleNamespace()
        repo_instance.get_by_telegram_id = AsyncMock(return_value=None)
        repo_cls.return_value = repo_instance

        response = await client.post(
            "/generate",
            json={"request": "contract", "context": None, "user_id": 1, "use_rag": True},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_generate_document_success(app_client):
    app, client = app_client

    mock_result = SimpleNamespace(
        title="Test Document",
        markdown_content="# Title",
        document_type="contract",
        pdf_bytes=b"%PDF-1.4"
    )

    mock_service = SimpleNamespace()
    mock_service.generate = AsyncMock(return_value=mock_result)
    app.dependency_overrides[get_doc_generation_service] = lambda: mock_service

    mock_session = SimpleNamespace()

    async def override_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_session

    with patch("src.web.routes.generate.UserRepository") as repo_cls:
        repo_instance = SimpleNamespace()
        repo_instance.get_by_telegram_id = AsyncMock(return_value=SimpleNamespace(id=1))
        repo_cls.return_value = repo_instance

        response = await client.post(
            "/generate",
            json={"request": "contract", "context": "ctx", "user_id": 1, "use_rag": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "title": "Test Document",
        "markdown": "# Title",
        "document_type": "contract"
    }
    mock_service.generate.assert_awaited_once_with(request="contract", context="ctx", use_rag=False)


@pytest.mark.asyncio
async def test_generate_document_pdf(app_client):
    app, client = app_client

    mock_result = SimpleNamespace(
        title="Document 1",
        markdown_content="# Title",
        document_type=None,
        pdf_bytes=b"PDFDATA"
    )

    mock_service = SimpleNamespace()
    mock_service.generate = AsyncMock(return_value=mock_result)
    app.dependency_overrides[get_doc_generation_service] = lambda: mock_service

    mock_session = SimpleNamespace()

    async def override_session():
        yield mock_session

    app.dependency_overrides[get_session] = override_session

    with patch("src.web.routes.generate.UserRepository") as repo_cls:
        repo_instance = SimpleNamespace()
        repo_instance.get_by_telegram_id = AsyncMock(return_value=SimpleNamespace(id=1))
        repo_cls.return_value = repo_instance

        response = await client.post(
            "/generate/pdf",
            json={"request": "contract", "context": None, "user_id": 1, "use_rag": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "Content-Disposition" in response.headers
    assert response.content == b"PDFDATA"


@pytest.mark.asyncio
async def test_get_document_types(app_client):
    app, client = app_client

    mock_service = SimpleNamespace()
    mock_service.get_document_types = lambda: {"contract": "Offer"}
    app.dependency_overrides[get_doc_generation_service] = lambda: mock_service

    response = await client.get("/generate/types")

    assert response.status_code == 200
    assert response.json() == {"types": {"contract": "Оферта"}}
