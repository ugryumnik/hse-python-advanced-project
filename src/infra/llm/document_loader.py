from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
import hashlib
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Метаданные документа"""
    source: str
    filename: str
    file_hash: str
    page: int | None = None
    total_pages: int | None = None


class DocumentLoaderFactory:
    """Фабрика загрузчиков документов по расширению"""

    LOADERS = {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": UnstructuredFileLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path):
        """Получить загрузчик для файла"""
        suffix = file_path.suffix.lower()
        loader_class = cls.LOADERS.get(suffix)

        if loader_class is None:
            logger.warning(f"Неподдерживаемый формат: {suffix}, попробуем UnstructuredFileLoader")
            loader_class = UnstructuredFileLoader

        return loader_class(str(file_path))


class LegalDocumentLoader:
    """Загрузчик юридических документов из папки"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        if not self.documents_dir.exists():
            self.documents_dir.mkdir(parents=True)
            logger.info(f"Создана директория для документов: {self.documents_dir}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Вычислить хеш файла для отслеживания изменений"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _enrich_metadata(self, doc: Document, file_path: Path, file_hash: str) -> Document:
        """Обогатить документ метаданными"""
        doc.metadata.update({
            "source": str(file_path),
            "filename": file_path.name,
            "file_hash": file_hash,
            "file_type": file_path.suffix.lower(),
        })
        return doc

    def load_file(self, file_path: Path) -> list[Document]:
        """Загрузить один файл"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        file_hash = self._compute_file_hash(file_path)
        loader = DocumentLoaderFactory.get_loader(file_path)

        try:
            documents = loader.load()
            return [self._enrich_metadata(doc, file_path, file_hash) for doc in documents]
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            raise

    def load_directory(self) -> Iterator[Document]:
        """Загрузить все документы из директории"""
        files = list(self.documents_dir.rglob("*"))
        supported_files = [
            f for f in files
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        logger.info(f"Найдено {len(supported_files)} документов для загрузки")

        for file_path in supported_files:
            try:
                docs = self.load_file(file_path)
                for doc in docs:
                    yield doc
                logger.debug(f"Загружен: {file_path.name}")
            except Exception as e:
                logger.error(f"Пропущен {file_path.name}: {e}")
                continue

    def get_file_list(self) -> list[dict]:
        """Получить список файлов с метаданными"""
        files = []
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "hash": self._compute_file_hash(file_path),
                })
        return files