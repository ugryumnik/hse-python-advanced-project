import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.web import get_rag_service
from src.web.routes.ask import ask_router


@pytest_asyncio.fixture
async def app_client():
    app = FastAPI()
    app.include_router(ask_router)

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield app, client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ask_question_success(app_client):
    app, client = app_client
    mock_service = SimpleNamespace()
    mock_service.query = AsyncMock(return_value=("Answer", [{"title": "doc1"}]))

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = await client.post("/ask", json={"query": "What is this?", "user_id": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"answer": "Answer", "sources": [{"title": "doc1"}]}


@pytest.mark.asyncio
async def test_ask_question_error(app_client):
    app, client = app_client
    mock_service = SimpleNamespace()
    mock_service.query = AsyncMock(side_effect=RuntimeError("boom"))

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = await client.post("/ask", json={"query": "?", "user_id": 1})

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"
