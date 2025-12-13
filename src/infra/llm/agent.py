"""RAG-агент для юридических документов"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig, YandexGPTConfig, QdrantConfig
from .document_loader import LegalDocumentLoader
from .embeddings import YandexEmbeddings
from .prompts import LEGAL_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from .vector_store import QdrantVectorStore
from .yandex_gpt import YandexGPTClient, YandexGPTMessage, YandexGPTError

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Ответ RAG системы"""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    tokens_used: int = 0


class LegalRAGAgent:
    """RAG-агент для юридических документов"""

    def __init__(
        self,
        config: RAGConfig | None = None,
        yandex_config: YandexGPTConfig | None = None,
        qdrant_config: QdrantConfig | None = None,
    ):
        self.config = config or RAGConfig()
        self.yandex_config = yandex_config or YandexGPTConfig()
        self.qdrant_config = qdrant_config or QdrantConfig()
        
        # Ленивая инициализация
        self._gpt_client: YandexGPTClient | None = None
        self._embeddings: YandexEmbeddings | None = None
        self._vector_store: QdrantVectorStore | None = None
        
        self.document_loader = LegalDocumentLoader(self.config.documents_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators,
        )

        logger.info(f"LegalRAGAgent: {self.yandex_config.model_uri}")

    @property
    def gpt_client(self) -> YandexGPTClient:
        if self._gpt_client is None:
            self._gpt_client = YandexGPTClient(self.yandex_config)
        return self._gpt_client

    @property
    def embeddings(self) -> YandexEmbeddings:
        if self._embeddings is None:
            self._embeddings = YandexEmbeddings(self.yandex_config)
        return self._embeddings

    @property
    def vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(self.qdrant_config, self.embeddings)
        return self._vector_store

    def index_documents(self, force_reindex: bool = False) -> int:
        """Индексировать документы из директории"""
        documents = list(self.document_loader.load_directory())
        if not documents:
            logger.warning("Документы не найдены")
            return 0

        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Создано {len(chunks)} чанков из {len(documents)} документов")

        if force_reindex:
            self.vector_store.clear_collection()

        # Индексация батчами
        for i in range(0, len(chunks), 50):
            self.vector_store.add_documents(chunks[i:i + 50])
            logger.info(f"Проиндексировано {min(i + 50, len(chunks))}/{len(chunks)}")

        return len(chunks)

    def add_document(self, file_path: str | Path) -> int:
        """Добавить один документ"""
        documents = self.document_loader.load_file(Path(file_path))
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        logger.info(f"Добавлен {file_path}: {len(chunks)} чанков")
        return len(chunks)

    def _format_context(self, docs: list[Document]) -> str:
        """Форматировать документы для контекста"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "?")
            page = doc.metadata.get("page", "?")
            parts.append(f"[Источник {i}: {source}, стр. {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """Извлечь уникальные источники"""
        seen = set()
        sources = []
        
        for doc in docs:
            key = (doc.metadata.get("filename", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": doc.metadata.get("filename", "Неизвестно"),
                    "page": doc.metadata.get("page"),
                    "score": doc.metadata.get("score"),
                })
        
        return sources

    def query(self, question: str, k: int | None = None) -> RAGResponse:
        """Выполнить RAG-запрос"""
        logger.info(f"Запрос: {question[:50]}...")

        # Поиск релевантных документов
        docs = self.vector_store.search(question, k)
        
        if not docs:
            return RAGResponse(
                answer="В базе не найдено релевантной информации.",
                query=question,
            )

        # Генерация ответа
        context = self._format_context(docs)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        
        messages = [
            YandexGPTMessage(role="system", text=LEGAL_SYSTEM_PROMPT),
            YandexGPTMessage(role="user", text=prompt),
        ]

        try:
            response = self.gpt_client.complete(messages)
            return RAGResponse(
                answer=response.text,
                sources=self._extract_sources(docs),
                query=question,
                tokens_used=response.total_tokens,
            )
        except YandexGPTError as e:
            logger.error(f"Ошибка GPT: {e}")
            return RAGResponse(
                answer=f"Ошибка генерации: {e}",
                sources=self._extract_sources(docs),
                query=question,
            )

    def get_stats(self) -> dict[str, Any]:
        """Статистика системы"""
        info = self.vector_store.get_info()
        return {
            "total_chunks": info["points_count"],
            "collection": info["name"],
            "status": info["status"],
            "model": self.yandex_config.model_uri,
            "documents_dir": str(self.config.documents_dir),
        }

    def health_check(self) -> bool:
        """Проверка работоспособности"""
        try:
            # Qdrant
            self.vector_store.count()
            logger.info("✓ Qdrant OK")
            
            # Embeddings
            self.embeddings.embed_query("тест")
            logger.info("✓ Embeddings OK")
            
            # GPT
            response = self.gpt_client.complete([
                YandexGPTMessage(role="user", text="Ответь: OK")
            ])
            logger.info("✓ GPT OK")
            
            return len(response.text) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """Закрыть ресурсы"""
        if self._gpt_client:
            self._gpt_client.close()
        if self._embeddings:
            self._embeddings.close()
        if self._vector_store:
            self._vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()