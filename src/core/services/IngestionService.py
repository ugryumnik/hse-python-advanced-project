from pathlib import Path
import os
import tempfile
import logging
from dataclasses import dataclass, asdict

import aiofiles
from fastapi import UploadFile

from infra.llm import LegalRAGAgent
from infra.llm.document_loader import (
    ArchiveHandler,
    SUPPORTED_EXTENSIONS,
    ARCHIVE_EXTENSIONS,
    COMPOUND_ARCHIVE_EXTENSIONS,
    ArchiveProcessingStats,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Результат обработки файла/архива"""
    chunks_count: int
    files_processed: int
    file_type: str  # "document" или "archive"
    processed_files: list[dict]  # Список обработанных файлов
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class IngestionService:
    """Сервис для загрузки и индексации документов"""

    def __init__(self, agent: LegalRAGAgent | None = None):
        self.agent = agent

    def _get_file_type(self, filename: str) -> str:
        """Определить тип файла"""
        filename_lower = filename.lower()

        for ext in COMPOUND_ARCHIVE_EXTENSIONS:
            if filename_lower.endswith(ext):
                return "archive"

        if any(filename_lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS):
            return "archive"

        if any(filename_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            return "document"

        return "unknown"

    async def processFile(self, file: UploadFile) -> IngestionResult:
        """
        Сохранить загруженный файл, индексировать и вернуть результат.

        Returns:
            IngestionResult с информацией о всех обработанных файлах
        """
        if not self.agent:
            raise RuntimeError("LegalRAGAgent not initialized")

        filename = file.filename or "unknown"
        file_type = self._get_file_type(filename)

        if file_type == "unknown":
            raise ValueError(f"Неподдерживаемый формат файла: {filename}")

        temp_dir = tempfile.mkdtemp(prefix="legal_upload_")
        temp_path = os.path.join(temp_dir, filename)

        try:
            # Сохраняем файл
            logger.info(f"Сохранение файла: {filename} ({file_type})")

            async with aiofiles.open(temp_path, 'wb') as out_file:
                while True:
                    chunk = await file.read(8192)
                    if not chunk:
                        break
                    await out_file.write(chunk)

            file_size = os.path.getsize(temp_path)
            logger.info(f"Файл сохранён: {filename}, размер: {file_size / 1024 / 1024:.2f} MB")

            # Обработка в зависимости от типа
            if file_type == "archive":
                return await self._process_archive(Path(temp_path))
            else:
                return await self._process_document(Path(temp_path), filename)

        except Exception as e:
            logger.error(f"Ошибка обработки файла {filename}: {e}")
            raise

        finally:
            # Очищаем временные файлы
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Ошибка очистки временных файлов: {e}")

    async def _process_document(self, file_path: Path, filename: str) -> IngestionResult:
        """Обработка одного документа"""
        import asyncio

        loop = asyncio.get_running_loop()
        documents = await loop.run_in_executor(
            None,
            lambda: self.agent.document_loader.load_file(file_path)
        )

        if documents:
            chunks = self.agent.text_splitter.split_documents(documents)
            await self.agent.vector_store.add_documents(chunks)
            chunks_count = len(chunks)
        else:
            chunks_count = 0

        return IngestionResult(
            chunks_count=chunks_count,
            files_processed=1,
            file_type="document",
            processed_files=[{
                "filename": filename,
                "chunks": chunks_count,
            }]
        )

    async def _process_archive(self, archive_path: Path) -> IngestionResult:
        """Обработка архива с детальной статистикой"""
        import asyncio

        loop = asyncio.get_running_loop()

        # Загружаем архив с статистикой
        documents, stats = await loop.run_in_executor(
            None,
            lambda: self.agent.document_loader.load_archive(archive_path)
        )

        chunks_count = 0
        if documents:
            chunks = self.agent.text_splitter.split_documents(documents)

            # Батчевая индексация
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                await self.agent.vector_store.add_documents(chunks[i:i + batch_size])

            chunks_count = len(chunks)

        # Формируем список обработанных файлов
        processed_files = []
        for file_info in stats.processed_files:
            entry = {
                "filename": file_info.filename,
                "chunks": file_info.chunks_count,
            }
            if file_info.archive_path:
                entry["archive_path"] = file_info.archive_path
            processed_files.append(entry)

        return IngestionResult(
            chunks_count=chunks_count,
            files_processed=stats.files_processed,
            file_type="archive",
            processed_files=processed_files,
            errors=stats.errors[:10] if stats.errors else []
        )