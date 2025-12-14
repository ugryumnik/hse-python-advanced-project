import pytest
from fastapi import FastAPI
from types import SimpleNamespace
from unittest.mock import AsyncMock

import src.web as web


@pytest.mark.asyncio
async def test_lifespan_initializes_services(monkeypatch):
    fake_agent = SimpleNamespace()
    fake_rag_service = SimpleNamespace(close=AsyncMock())
    fake_ingestion_service = SimpleNamespace()
    fake_doc_service = SimpleNamespace()

    monkeypatch.setattr(web, "LegalRAGAgent", lambda: fake_agent)
    monkeypatch.setattr(web, "RAGService", lambda agent: fake_rag_service)
    monkeypatch.setattr(web, "IngestionService", lambda agent: fake_ingestion_service)
    monkeypatch.setattr(web, "DocumentGenerationService", lambda agent: fake_doc_service)

    web.rag_service = None
    web.ingestion_service = None
    web.doc_generation_service = None

    async with web.lifespan(FastAPI()):
        assert web.rag_service is fake_rag_service
        assert web.ingestion_service is fake_ingestion_service
        assert web.doc_generation_service is fake_doc_service

    fake_rag_service.close.assert_awaited_once()
    web.rag_service = None
    web.ingestion_service = None
    web.doc_generation_service = None


def test_get_services_raise_when_uninitialized(monkeypatch):
    monkeypatch.setattr(web, "rag_service", None)
    monkeypatch.setattr(web, "ingestion_service", None)
    monkeypatch.setattr(web, "doc_generation_service", None)

    with pytest.raises(RuntimeError):
        web.get_rag_service()
    with pytest.raises(RuntimeError):
        web.get_ingestion_service()
    with pytest.raises(RuntimeError):
        web.get_doc_generation_service()


def test_get_services_return_when_initialized(monkeypatch):
    rag = object()
    ing = object()
    doc = object()

    monkeypatch.setattr(web, "rag_service", rag)
    monkeypatch.setattr(web, "ingestion_service", ing)
    monkeypatch.setattr(web, "doc_generation_service", doc)

    assert web.get_rag_service() is rag
    assert web.get_ingestion_service() is ing
    assert web.get_doc_generation_service() is doc

    web.rag_service = None
    web.ingestion_service = None
    web.doc_generation_service = None
