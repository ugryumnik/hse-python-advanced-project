"""Загрузка юридических документов"""

import hashlib
import logging
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)

logger = logging.getLogger(__name__)


LOADERS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": UnstructuredFileLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

SUPPORTED_EXTENSIONS = set(LOADERS.keys())


def compute_file_hash(file_path: Path) -> str:
    """Вычислить MD5 хеш файла"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class LegalDocumentLoader:
    """Загрузчик юридических документов"""

    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: Path) -> list[Document]:
        """Загрузить один файл"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        suffix = file_path.suffix.lower()
        loader_class = LOADERS.get(suffix, UnstructuredFileLoader)
        
        file_hash = compute_file_hash(file_path)
        
        try:
            documents = loader_class(str(file_path)).load()
            
            # Обогащаем метаданными
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_hash": file_hash,
                    "file_type": suffix,
                })
            
            return documents
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            raise

    def load_directory(self) -> Iterator[Document]:
        """Загрузить все документы из директории"""
        files = [
            f for f in self.documents_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        logger.info(f"Найдено {len(files)} документов")

        for file_path in files:
            try:
                yield from self.load_file(file_path)
                logger.debug(f"Загружен: {file_path.name}")
            except Exception as e:
                logger.error(f"Пропущен {file_path.name}: {e}")

    def list_files(self) -> list[dict]:
        """Список файлов с метаданными"""
        return [
            {
                "filename": f.name,
                "path": str(f),
                "size": f.stat().st_size,
            }
            for f in self.documents_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]