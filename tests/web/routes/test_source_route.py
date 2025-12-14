import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.web import get_rag_service
from src.web.routes.source import source_router


@pytest_asyncio.fixture
async def app_client():
    app = FastAPI()
    app.include_router(source_router)

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield app, client

    app.dependency_overrides.clear()


def _build_rag_service(search_side_effect):
    vector_store = SimpleNamespace(search=AsyncMock(side_effect=search_side_effect))
    agent = SimpleNamespace(vector_store=vector_store)
    return SimpleNamespace(agent=agent)


@pytest.mark.asyncio
async def test_get_source_chunk_primary_match(app_client):
    app, client = app_client

    doc = SimpleNamespace(
        metadata={"filename": "case.pdf", "page": 2},
        page_content="Paragraph"
    )

    rag_service = _build_rag_service([ [doc] ])
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    response = await client.post(
        "/source",
        json={"filename": "case.pdf", "page": 2, "limit": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "chunks": [
            {"filename": "case.pdf", "page": 2, "text": "Paragraph"}
        ]
    }
    rag_service.agent.vector_store.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_source_chunk_fallback(app_client):
    app, client = app_client

    doc = SimpleNamespace(
        metadata={"filename": "case.pdf", "page": None},
        page_content="Fallback"
    )
    rag_service = _build_rag_service([[], [doc]])
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    response = await client.post(
        "/source",
        json={"filename": "case.pdf", "page": 3, "limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["chunks"][0]["text"] == "Fallback"
    assert rag_service.agent.vector_store.search.await_count == 2


@pytest.mark.asyncio
async def test_get_source_chunk_no_results(app_client):
    app, client = app_client

    rag_service = _build_rag_service([[], []])
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    response = await client.post(
        "/source",
        json={"filename": "case.pdf", "page": 1, "limit": 2},
    )

    assert response.status_code == 200
    assert response.json() == {"chunks": []}


@pytest.mark.asyncio
async def test_get_source_chunk_handles_error(app_client):
    app, client = app_client

    rag_service = _build_rag_service(RuntimeError("boom"))
    app.dependency_overrides[get_rag_service] = lambda: rag_service

    response = await client.post(
        "/source",
        json={"filename": "case.pdf", "page": 1, "limit": 2},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"
