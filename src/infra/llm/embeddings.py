"""Yandex Embeddings для векторизации текста"""

import logging
import time
from typing import List

import httpx
from langchain_core.embeddings import Embeddings

from .config import YandexGPTConfig

logger = logging.getLogger(__name__)


class YandexEmbeddings(Embeddings):
    """LangChain-совместимые Yandex Embeddings"""

    DOC_MODEL = "text-search-doc"
    QUERY_MODEL = "text-search-query"
    MAX_TEXT_LENGTH = 8000

    def __init__(self, config: YandexGPTConfig):
        self.config = config
        self._client = httpx.Client(
            timeout=30,
            headers={
                "Content-Type": "application/json",
                **config.get_auth_header(),
            },
        )
        logger.info("YandexEmbeddings инициализированы")

    def _get_model_uri(self, model: str) -> str:
        return f"emb://{self.config.folder_id}/{model}/latest"

    def _embed(self, text: str, model: str) -> List[float]:
        """Получить эмбеддинг для текста"""
        text = text[:self.MAX_TEXT_LENGTH] if len(text) > self.MAX_TEXT_LENGTH else text
        
        body = {
            "modelUri": self._get_model_uri(model),
            "text": text,
        }

        for attempt in range(3):
            try:
                response = self._client.post(self.config.embeddings_url, json=body)
                response.raise_for_status()
                return response.json()["embedding"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embeddings API error: {e.response.status_code}")
            except httpx.RequestError as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"Connection error: {e}")

        raise RuntimeError("Превышено количество попыток")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинги для документов"""
        embeddings = []
        for i, text in enumerate(texts):
            if not text.strip():
                embeddings.append([0.0] * 256)
                continue
            embeddings.append(self._embed(text, self.DOC_MODEL))
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг для поискового запроса"""
        return self._embed(text, self.QUERY_MODEL)

    def close(self):
        self._client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass