"""Асинхронный RAG-агент для юридических документов"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig, YandexGPTConfig, QdrantConfig
from .document_loader import LegalDocumentLoader
from .embeddings import YandexEmbeddings
from .prompts import LEGAL_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, CONVERSATIONAL_SYSTEM_PROMPT
from .vector_store import QdrantVectorStore
from .yandex_gpt import YandexGPTClient, YandexGPTMessage, YandexGPTError

logger = logging.getLogger(__name__)


# Паттерны для определения разговорных вопросов
CONVERSATIONAL_PATTERNS = [
    r'^привет',
    r'^здравствуй',
    r'^добрый\s+(день|вечер|утро)',
    r'^как\s+(дела|поживаешь|жизнь)',
    r'^что\s+(ты\s+)?умеешь',
    r'^кто\s+ты',
    r'^ты\s+кто',
    r'^спасибо',
    r'^благодар',
    r'^пока$',
    r'^до\s+свидания',
    r'^хорошо$',
    r'^ок$',
    r'^понятно',
    r'^ясно',
    r'^помощь$',
    r'^help$',
    r'^start$',
]

# Минимальный score для считания источника релевантным
MIN_RELEVANCE_SCORE = 0.25

# Минимальная длина вопроса для RAG
MIN_QUESTION_LENGTH = 10


@dataclass
class RAGResponse:
    """Ответ RAG системы"""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    tokens_used: int = 0
    used_rag: bool = True  # Флаг: использовался ли RAG


class LegalRAGAgent:
    """Асинхронный RAG-агент для юридических документов"""

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

    def _is_conversational(self, question: str) -> bool:
        """
        Определить, является ли вопрос разговорным (не требует поиска в документах).

        Returns:
            True если вопрос разговорный, False если требуется RAG
        """
        question_lower = question.lower().strip()

        # Проверяем паттерны разговорных фраз
        for pattern in CONVERSATIONAL_PATTERNS:
            if re.match(pattern, question_lower):
                logger.debug(f"Conversational pattern matched: {pattern}")
                return True

        # Очень короткие вопросы без вопросительных слов часто разговорные
        words = question_lower.split()
        if len(words) <= 2 and '?' not in question:
            # Но проверяем, не содержит ли юридических терминов
            legal_keywords = ['закон', 'право', 'статья', 'договор', 'суд', 'иск',
                            'ответственность', 'штраф', 'увольнение', 'трудов']
            if not any(kw in question_lower for kw in legal_keywords):
                return True

        return False

    def _is_legal_question(self, question: str) -> bool:
        """
        Определить, является ли вопрос юридическим/правовым.
        """
        question_lower = question.lower()

        legal_keywords = [
            'закон', 'право', 'статья', 'пункт', 'договор', 'контракт',
            'суд', 'иск', 'истец', 'ответчик', 'заявление', 'жалоба',
            'ответственность', 'штраф', 'наказание', 'санкц',
            'увольнение', 'трудов', 'работодатель', 'работник',
            'налог', 'ндфл', 'ндс', 'взнос',
            'собственность', 'имущество', 'наследств',
            'развод', 'алимент', 'опека', 'брак',
            'аренд', 'найм', 'лицензи', 'разрешени',
            'регистрац', 'документ', 'справк',
            'обязан', 'должен', 'можно ли', 'имею ли право',
            'как оформить', 'как составить', 'как подать',
            'какие документы', 'какой срок', 'какой порядок',
            'что делать если', 'что грозит', 'что будет',
            'гк рф', 'тк рф', 'ук рф', 'коап', 'конституц',
        ]

        return any(kw in question_lower for kw in legal_keywords)

    def _filter_relevant_sources(
        self,
        docs: list[Document],
        min_score: float = MIN_RELEVANCE_SCORE
    ) -> list[Document]:
        """
        Фильтрация документов по релевантности.
        Возвращает только документы с достаточно высоким score.
        """
        relevant = []
        for doc in docs:
            score = doc.metadata.get("score")
            if score is not None and score >= min_score:
                relevant.append(doc)
            elif score is None:
                # Если score нет, включаем документ (на всякий случай)
                relevant.append(doc)

        logger.debug(f"Filtered {len(docs)} docs to {len(relevant)} relevant")
        return relevant

    async def index_documents(self, force_reindex: bool = False) -> int:
        """Асинхронно индексировать документы из директории"""
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(
            None,
            lambda: list(self.document_loader.load_directory())
        )

        if not documents:
            logger.warning("Документы не найдены")
            return 0

        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Создано {len(chunks)} чанков из {len(documents)} документов")

        if force_reindex:
            await self.vector_store.clear_collection()

        for i in range(0, len(chunks), 50):
            await self.vector_store.add_documents(chunks[i:i + 50])
            logger.info(f"Проиндексировано {min(i + 50, len(chunks))}/{len(chunks)}")

        return len(chunks)

    async def add_document(self, file_path: str | Path) -> int:
        """Асинхронно добавить один документ"""
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(
            None,
            lambda: self.document_loader.load_file(Path(file_path))
        )
        chunks = self.text_splitter.split_documents(documents)
        await self.vector_store.add_documents(chunks)
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
            filename = doc.metadata.get("filename", "")
            page = doc.metadata.get("page", 0)
            archive = doc.metadata.get("archive_source")

            key = (filename, page, archive)
            if key not in seen:
                seen.add(key)
                source_info = {
                    "filename": filename or "Неизвестно",
                    "page": page,
                    "score": doc.metadata.get("score"),
                }
                if archive:
                    source_info["archive"] = archive

                sources.append(source_info)

        return sources

    async def _answer_conversational(self, question: str) -> RAGResponse:
        """
        Ответить на разговорный вопрос без RAG.
        """
        messages = [
            YandexGPTMessage(role="system", text=CONVERSATIONAL_SYSTEM_PROMPT),
            YandexGPTMessage(role="user", text=question),
        ]

        try:
            response = await self.gpt_client.complete(messages)
            return RAGResponse(
                answer=response.text,
                sources=[],  # Без источников
                query=question,
                tokens_used=response.total_tokens,
                used_rag=False,
            )
        except YandexGPTError as e:
            logger.error(f"Ошибка GPT: {e}")
            return RAGResponse(
                answer="Извините, произошла ошибка. Попробуйте ещё раз.",
                query=question,
                used_rag=False,
            )

    async def query(self, question: str, k: int | None = None) -> RAGResponse:
        """
        Асинхронно выполнить RAG-запрос.

        Автоматически определяет тип вопроса:
        - Разговорные вопросы обрабатываются без RAG
        - Юридические вопросы обрабатываются с поиском по документам
        """
        logger.info(f"Запрос: {question[:50]}...")

        # Проверяем, разговорный ли вопрос
        if self._is_conversational(question):
            logger.info("Detected conversational question, skipping RAG")
            return await self._answer_conversational(question)

        # Проверяем минимальную длину для RAG
        if len(question.strip()) < MIN_QUESTION_LENGTH and not self._is_legal_question(question):
            logger.info("Question too short for RAG, using conversational mode")
            return await self._answer_conversational(question)

        # Поиск релевантных документов
        docs = await self.vector_store.search(question, k)

        # Фильтруем по релевантности
        relevant_docs = self._filter_relevant_sources(docs)

        # Если нет релевантных документов
        if not relevant_docs:
            # Для юридических вопросов сообщаем об отсутствии информации
            if self._is_legal_question(question):
                return RAGResponse(
                    answer="К сожалению, в моей базе знаний нет информации по этому вопросу. "
                           "Рекомендую обратиться к профильному юристу.",
                    query=question,
                    sources=[],
                    used_rag=True,
                )
            else:
                # Для общих вопросов отвечаем без источников
                return await self._answer_conversational(question)

        # Генерация ответа с контекстом
        context = self._format_context(relevant_docs)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [
            YandexGPTMessage(role="system", text=LEGAL_SYSTEM_PROMPT),
            YandexGPTMessage(role="user", text=prompt),
        ]

        try:
            response = await self.gpt_client.complete(messages)

            # Проверяем, использовал ли ответ контекст
            sources = self._extract_sources(relevant_docs)

            # Если в ответе явно указано, что информации нет - не показываем источники
            no_info_phrases = [
                "в предоставленном контексте нет",
                "в документах не найдено",
                "информация отсутствует",
                "не содержит информации",
                "нет данных по этому вопросу",
            ]

            answer_lower = response.text.lower()
            if any(phrase in answer_lower for phrase in no_info_phrases):
                sources = []

            return RAGResponse(
                answer=response.text,
                sources=sources,
                query=question,
                tokens_used=response.total_tokens,
                used_rag=True,
            )
        except YandexGPTError as e:
            logger.error(f"Ошибка GPT: {e}")
            return RAGResponse(
                answer=f"Ошибка генерации: {e}",
                sources=self._extract_sources(relevant_docs),
                query=question,
                used_rag=True,
            )

    async def get_stats(self) -> dict[str, Any]:
        """Статистика системы"""
        info = await self.vector_store.get_info()
        return {
            "total_chunks": info["points_count"],
            "collection": info["name"],
            "status": info["status"],
            "model": self.yandex_config.model_uri,
            "documents_dir": str(self.config.documents_dir),
        }

    async def health_check(self) -> bool:
        """Асинхронная проверка работоспособности"""
        try:
            await self.vector_store.count()
            logger.info("✓ Qdrant OK")

            await self.embeddings.aembed_query("тест")
            logger.info("✓ Embeddings OK")

            response = await self.gpt_client.complete([
                YandexGPTMessage(role="user", text="Ответь: OK")
            ])
            logger.info("✓ GPT OK")
            
            return len(response.text) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Закрыть ресурсы"""
        if self._gpt_client:
            await self._gpt_client.close()
        if self._embeddings:
            await self._embeddings.close()
        if self._vector_store:
            await self._vector_store.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()