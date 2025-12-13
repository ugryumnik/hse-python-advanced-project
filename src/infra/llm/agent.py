from pathlib import Path
from typing import Iterator, Any
from dataclasses import dataclass, field
import logging
import atexit

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from .config import RAGConfig
from .embeddings import EmbeddingsManager
from .document_loader import LegalDocumentLoader
from .prompts import LEGAL_SYSTEM_PROMPT, LEGAL_RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Структура ответа RAG системы"""
    answer: str
    sources: list[dict] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)
    query: str = ""


class LegalRAGAgent:
    """
    RAG-агент для работы с юридическими документами.
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()

        self._llm = None
        self._vector_store = None

        self.embeddings_manager = EmbeddingsManager(self.config.embeddings)
        self.document_loader = LegalDocumentLoader(self.config.documents_dir)
        self.text_splitter = self._create_text_splitter()

        logger.info("LegalRAGAgent инициализирован")

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Создать сплиттер текста"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators,
            length_function=len,
        )

    def _get_model_path(self) -> Path:
        """Получить путь к модели"""
        llm_config = self.config.llm

        if llm_config.local_model_path is not None:
            model_path = Path(llm_config.local_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Модель не найдена: {model_path}")
            logger.info(f"Используем локальную модель: {model_path}")
            return model_path

        if llm_config.model_id and llm_config.model_file:
            cache_dir = Path(llm_config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_path = cache_dir / llm_config.model_file
            if model_path.exists():
                logger.info(f"Модель в кеше: {model_path}")
                return model_path

            logger.info(f"Скачиваем: {llm_config.model_id}/{llm_config.model_file}")
            downloaded_path = hf_hub_download(
                repo_id=llm_config.model_id,
                filename=llm_config.model_file,
                local_dir=cache_dir,
            )
            return Path(downloaded_path)

        raise ValueError("Укажите local_model_path или model_id + model_file")

    @property
    def llm(self) -> LlamaCpp:
        """Инициализация LLM"""
        if self._llm is None:
            model_path = self._get_model_path()
            llm_config = self.config.llm

            logger.info(f"Инициализируем LLM из {model_path}...")

            self._llm = LlamaCpp(
                model_path=str(model_path),
                n_ctx=llm_config.n_ctx,
                n_gpu_layers=llm_config.n_gpu_layers,
                n_batch=llm_config.n_batch,

                # Параметры генерации
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                top_p=llm_config.top_p,
                top_k=llm_config.top_k,

                # Против повторений
                repeat_penalty=llm_config.repeat_penalty,

                # Стоп-токены
                stop=llm_config.stop,

                verbose=False,
                n_threads=8,
                f16_kv=True,
            )
            logger.info("LLM инициализирована успешно")

        return self._llm

    @property
    def vector_store(self) -> Chroma:
        """Инициализация векторного хранилища"""
        if self._vector_store is None:
            persist_dir = Path(self.config.vector_store.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._vector_store = Chroma(
                collection_name=self.config.vector_store.collection_name,
                embedding_function=self.embeddings_manager.get_embeddings(),
                persist_directory=str(persist_dir),
            )

            count = self._vector_store._collection.count()
            logger.info(f"Векторное хранилище: {count} документов")

        return self._vector_store

    def index_documents(self, force_reindex: bool = False) -> int:
        """Индексировать документы из директории"""
        logger.info("Начинаем индексацию...")

        documents = list(self.document_loader.load_directory())

        if not documents:
            logger.warning("Документы не найдены")
            return 0

        logger.info(f"Загружено {len(documents)} документов")

        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Создано {len(chunks)} чанков")

        if force_reindex:
            self.vector_store._collection.delete(where={})
            logger.info("Коллекция очищена")

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.vector_store.add_documents(batch)

        logger.info(f"Индексация завершена: {len(chunks)} чанков")
        return len(chunks)

    def add_document(self, file_path: str | Path) -> int:
        """Добавить один документ"""
        documents = self.document_loader.load_file(Path(file_path))
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        logger.info(f"Добавлен {file_path}: {len(chunks)} чанков")
        return len(chunks)

    def _format_docs(self, docs: list[Document]) -> str:
        """Форматировать документы для контекста"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "Неизвестный источник")
            page = doc.metadata.get("page", "?")
            formatted.append(
                f"[Источник {i}: {source}, стр. {page}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """Извлечь информацию об источниках"""
        sources = []
        seen = set()

        for doc in docs:
            source_key = (
                doc.metadata.get("filename", ""),
                doc.metadata.get("page", 0)
            )

            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "filename": doc.metadata.get("filename", "Неизвестно"),
                    "page": doc.metadata.get("page"),
                    "file_path": doc.metadata.get("source", ""),
                })

        return sources

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Найти релевантные документы"""
        k = k or self.config.vector_store.search_k

        if self.config.vector_store.search_type == "mmr":
            docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                lambda_mult=self.config.vector_store.mmr_lambda,
            )
        else:
            docs = self.vector_store.similarity_search(query, k=k)

        return docs

    def _clean_response(self, text: str) -> str:
        """Очистить ответ от артефактов"""
        # Убираем повторяющиеся блоки
        lines = text.split('\n')
        seen_blocks = set()
        clean_lines = []
        current_block = []

        for line in lines:
            if line.strip() == '---':
                block = '\n'.join(current_block)
                if block.strip() and block not in seen_blocks:
                    seen_blocks.add(block)
                    clean_lines.extend(current_block)
                    clean_lines.append(line)
                current_block = []
            else:
                current_block.append(line)

        # Добавляем последний блок
        block = '\n'.join(current_block)
        if block.strip() and block not in seen_blocks:
            clean_lines.extend(current_block)

        result = '\n'.join(clean_lines).strip()

        # Если очистка слишком агрессивная, вернём первую часть оригинала
        if len(result) < 50 and len(text) > 100:
            # Берём текст до первого повторения разделителя ---
            parts = text.split('---\n\n---')
            result = parts[0].strip()

        return result

    def query(self, question: str, k: int | None = None) -> RAGResponse:
        """Выполнить RAG-запрос"""
        logger.info(f"Запрос: {question[:50]}...")

        relevant_docs = self.retrieve(question, k)

        if not relevant_docs:
            return RAGResponse(
                answer="В базе документов не найдено релевантной информации.",
                sources=[],
                context_chunks=[],
                query=question,
            )

        context = self._format_docs(relevant_docs)
        sources = self._extract_sources(relevant_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", LEGAL_SYSTEM_PROMPT),
            ("human", LEGAL_RAG_PROMPT_TEMPLATE),
        ])

        chain = prompt | self.llm | StrOutputParser()

        raw_answer = chain.invoke({
            "context": context,
            "question": question,
        })

        # Очищаем от повторений
        answer = self._clean_response(raw_answer)

        logger.info("Ответ сгенерирован")

        return RAGResponse(
            answer=answer,
            sources=sources,
            context_chunks=[doc.page_content for doc in relevant_docs],
            query=question,
        )

    def query_stream(self, question: str, k: int | None = None) -> Iterator[str]:
        """Стриминговый RAG-запрос"""
        relevant_docs = self.retrieve(question, k)

        if not relevant_docs:
            yield "В базе документов не найдено релевантной информации."
            return

        context = self._format_docs(relevant_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", LEGAL_SYSTEM_PROMPT),
            ("human", LEGAL_RAG_PROMPT_TEMPLATE),
        ])

        chain = prompt | self.llm

        for chunk in chain.stream({
            "context": context,
            "question": question,
        }):
            yield chunk

    def get_stats(self) -> dict[str, Any]:
        """Статистика"""
        return {
            "total_chunks": self.vector_store._collection.count(),
            "documents_dir": str(self.config.documents_dir),
        }

    def health_check(self) -> bool:
        """Проверка работоспособности"""
        try:
            _ = self.embeddings_manager.embed_query("тест")
            _ = self.llm.invoke("Скажи OK")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """Явное закрытие ресурсов (опционально)"""
        if self._llm is not None:
            try:
                del self._llm
                self._llm = None
            except:
                pass