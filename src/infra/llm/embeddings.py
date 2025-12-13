"""Асинхронные Yandex Embeddings для векторизации текста"""

import logging
import asyncio
from typing import List

import httpx
from langchain_core.embeddings import Embeddings

from .config import YandexGPTConfig

logger = logging.getLogger(__name__)


class YandexEmbeddings(Embeddings):
    """Асинхронные LangChain-совместимые Yandex Embeddings"""

    DOC_MODEL = "text-search-doc"
    QUERY_MODEL = "text-search-query"
    MAX_TEXT_LENGTH = 8000

    def __init__(self, config: YandexGPTConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        logger.info("YandexEmbeddings инициализированы")

    async def _get_client(self) -> httpx.AsyncClient:
        """Ленивая инициализация клиента"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30,
                headers={
                    "Content-Type": "application/json",
                    **self.config.get_auth_header(),
                },
            )
        return self._client

    def _get_model_uri(self, model: str) -> str:
        return f"emb://{self.config.folder_id}/{model}/latest"

    async def _embed_async(self, text: str, model: str) -> List[float]:
        """Асинхронно получить эмбеддинг для текста"""
        text = text[:self.MAX_TEXT_LENGTH] if len(text) > self.MAX_TEXT_LENGTH else text
        
        body = {
            "modelUri": self._get_model_uri(model),
            "text": text,
        }

        client = await self._get_client()

        for attempt in range(3):
            try:
                response = await client.post(self.config.embeddings_url, json=body)
                response.raise_for_status()
                return response.json()["embedding"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embeddings API error: {e.response.status_code}")
            except httpx.RequestError as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                raise RuntimeError(f"Connection error: {e}")

        raise RuntimeError("Превышено количество попыток")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Асинхронные эмбеддинги для документов"""
        embeddings = []
        for i, text in enumerate(texts):
            if not text.strip():
                embeddings.append([0.0] * 256)
                continue
            embedding = await self._embed_async(text, self.DOC_MODEL)
            embeddings.append(embedding)
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)}")
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Асинхронный эмбеддинг для поискового запроса"""
        return await self._embed_async(text, self.QUERY_MODEL)

    # Синхронные методы для совместимости с LangChain (вызываются через asyncio.run если нужно)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Синхронная обёртка для embed_documents"""
        return asyncio.get_event_loop().run_until_complete(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """Синхронная обёртка для embed_query"""
        return asyncio.get_event_loop().run_until_complete(self.aembed_query(text))

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()