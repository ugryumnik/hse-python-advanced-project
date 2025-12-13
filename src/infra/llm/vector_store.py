"""Асинхронное векторное хранилище на базе Qdrant"""

import logging
import uuid
from typing import List, Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import QdrantConfig
from .embeddings import YandexEmbeddings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Асинхронное векторное хранилище Qdrant"""

    def __init__(self, config: QdrantConfig, embeddings: YandexEmbeddings):
        self.config = config
        self.embeddings = embeddings
        self._client: AsyncQdrantClient | None = None
        self._initialized = False
        logger.info(f"Qdrant config: {config.host}:{config.port}")

    async def _get_client(self) -> AsyncQdrantClient:
        """Ленивая инициализация клиента"""
        if self._client is None:
            self._client = AsyncQdrantClient(host=self.config.host, port=self.config.port)
            logger.info(f"Qdrant connected: {self.config.host}:{self.config.port}")
        
        if not self._initialized:
            await self._ensure_collection()
            self._initialized = True
        
        return self._client

    async def _ensure_collection(self) -> None:
        """Создать коллекцию если не существует"""
        client = self._client
        collections_response = await client.get_collections()
        collections = [c.name for c in collections_response.collections]
        
        if self.config.collection_name not in collections:
            await client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Создана коллекция: {self.config.collection_name}")

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Асинхронно добавить документы"""
        if not documents:
            return []
        
        client = await self._get_client()
        
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embeddings.aembed_documents(texts)
        
        points = []
        ids = []
        
        for doc, embedding in zip(documents, embeddings):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "filename": doc.metadata.get("filename", ""),
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page"),
                    "file_hash": doc.metadata.get("file_hash", ""),
                },
            ))
        
        # Загрузка батчами
        for i in range(0, len(points), 100):
            await client.upsert(
                collection_name=self.config.collection_name,
                points=points[i:i + 100],
            )
        
        logger.info(f"Добавлено {len(points)} документов")
        return ids

    def _points_to_documents(self, points) -> List[Document]:
        """Преобразовать точки Qdrant в документы"""
        documents = []
        for point in points:
            payload = point.payload or {}
            documents.append(Document(
                page_content=payload.get("text", ""),
                metadata={
                    "filename": payload.get("filename", ""),
                    "source": payload.get("source", ""),
                    "page": payload.get("page"),
                    "file_hash": payload.get("file_hash", ""),
                    "score": getattr(point, "score", None),
                },
            ))
        return documents

    async def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_dict: dict | None = None,
    ) -> List[Document]:
        """Асинхронный поиск по сходству"""
        k = k or self.config.search_k
        client = await self._get_client()
        
        query_vector = await self.embeddings.aembed_query(query)
        
        qdrant_filter = None
        if filter_dict:
            qdrant_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_dict.items()
            ])

        results = await client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return self._points_to_documents(results.points)

    async def mmr_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float | None = None,
    ) -> List[Document]:
        """Асинхронный MMR поиск для разнообразия результатов"""
        k = k or self.config.search_k
        lambda_mult = lambda_mult or self.config.mmr_lambda
        
        client = await self._get_client()
        
        query_vector = np.array(await self.embeddings.aembed_query(query))
        
        results = await client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector.tolist(),
            limit=fetch_k,
            with_vectors=True,
            with_payload=True,
        )
        
        if not results.points:
            return []
        
        # MMR алгоритм
        candidates = []
        for point in results.points:
            if point.vector:
                candidates.append({
                    "vector": np.array(point.vector),
                    "payload": point.payload,
                    "score": getattr(point, "score", 0),
                })
        
        if not candidates:
            return self._points_to_documents(results.points[:k])
        
        selected = []
        remaining = list(range(len(candidates)))
        
        for _ in range(min(k, len(candidates))):
            if not remaining:
                break
                
            if not selected:
                best_idx = max(remaining, key=lambda i: candidates[i]["score"] or 0)
            else:
                mmr_scores = []
                for idx in remaining:
                    relevance = float(np.dot(query_vector, candidates[idx]["vector"]))
                    diversity = max(
                        float(np.dot(candidates[idx]["vector"], candidates[sel]["vector"]))
                        for sel in selected
                    )
                    mmr = lambda_mult * relevance - (1 - lambda_mult) * diversity
                    mmr_scores.append((idx, mmr))
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        documents = []
        for idx in selected:
            payload = candidates[idx]["payload"] or {}
            documents.append(Document(
                page_content=payload.get("text", ""),
                metadata={
                    "filename": payload.get("filename", ""),
                    "source": payload.get("source", ""),
                    "page": payload.get("page"),
                    "score": candidates[idx]["score"],
                },
            ))
        
        return documents

    async def search(self, query: str, k: int | None = None) -> List[Document]:
        """Универсальный асинхронный поиск"""
        if self.config.use_mmr:
            return await self.mmr_search(query, k)
        return await self.similarity_search(query, k)

    async def clear_collection(self) -> None:
        """Очистить коллекцию"""
        client = await self._get_client()
        await client.delete_collection(self.config.collection_name)
        self._initialized = False
        await self._ensure_collection()
        logger.info(f"Коллекция очищена: {self.config.collection_name}")

    async def count(self) -> int:
        """Количество документов"""
        client = await self._get_client()
        info = await client.get_collection(self.config.collection_name)
        return info.points_count

    async def get_info(self) -> dict[str, Any]:
        """Информация о коллекции"""
        client = await self._get_client()
        info = await client.get_collection(self.config.collection_name)
        return {
            "name": self.config.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Qdrant закрыт")