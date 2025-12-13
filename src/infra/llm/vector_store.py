"""Векторное хранилище на базе Qdrant"""

import logging
import uuid
from typing import List, Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
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
    """Векторное хранилище Qdrant"""

    def __init__(self, config: QdrantConfig, embeddings: YandexEmbeddings):
        self.config = config
        self.embeddings = embeddings

        self._client = QdrantClient(host=config.host, port=config.port)
        logger.info(f"Qdrant: {config.host}:{config.port}")
        
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Создать коллекцию если не существует"""
        collections = [c.name for c in self._client.get_collections().collections]
        
        if self.config.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Создана коллекция: {self.config.collection_name}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Добавить документы"""
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
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
            self._client.upsert(
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

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_dict: dict | None = None,
    ) -> List[Document]:
        """Поиск по сходству"""
        k = k or self.config.search_k
        query_vector = self.embeddings.embed_query(query)
        
        qdrant_filter = None
        if filter_dict:
            qdrant_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_dict.items()
            ])

        results = self._client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return self._points_to_documents(results.points)

    def mmr_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float | None = None,
    ) -> List[Document]:
        """MMR поиск для разнообразия результатов"""
        k = k or self.config.search_k
        lambda_mult = lambda_mult or self.config.mmr_lambda
        
        query_vector = np.array(self.embeddings.embed_query(query))
        
        results = self._client.query_points(
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
                # Первый - самый релевантный
                best_idx = max(remaining, key=lambda i: candidates[i]["score"] or 0)
            else:
                # MMR: баланс релевантности и разнообразия
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
        
        # Собираем результаты
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

    def search(self, query: str, k: int | None = None) -> List[Document]:
        """Универсальный поиск (выбирает метод по конфигу)"""
        if self.config.use_mmr:
            return self.mmr_search(query, k)
        return self.similarity_search(query, k)

    def clear_collection(self) -> None:
        """Очистить коллекцию"""
        self._client.delete_collection(self.config.collection_name)
        self._ensure_collection()
        logger.info(f"Коллекция очищена: {self.config.collection_name}")

    def count(self) -> int:
        """Количество документов"""
        return self._client.get_collection(self.config.collection_name).points_count

    def get_info(self) -> dict[str, Any]:
        """Информация о коллекции"""
        info = self._client.get_collection(self.config.collection_name)
        return {
            "name": self.config.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }

    def close(self) -> None:
        self._client.close()
        logger.info("Qdrant закрыт")