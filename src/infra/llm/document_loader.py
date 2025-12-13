"""
Загрузка юридических документов с поддержкой Docling + LangChain
"""

import hashlib
import logging
from pathlib import Path
from typing import Iterator, List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

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
    """Вычислить MD5-хеш файла"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class LegalDocumentLoader:
    """
    Загрузчик юридических документов.
    """

    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def load_directory(self) -> Iterator[Document]:
        files = [
            f
            for f in self.documents_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        logger.info(f"Найдено {len(files)} документов")

        for file_path in files:
            try:
                yield from self.load_file(file_path)
                logger.debug(f"Загружен: {file_path.name}")
            except Exception as e:
                logger.error(f"Пропущен {file_path.name}: {e}")

    def load_file(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        file_hash = compute_file_hash(file_path)

        # Docling
        if suffix in {".pdf", ".docx", ".doc"}:
            try:
                logger.info(f"Docling: loading {file_path.name}")
                documents = self._load_with_docling(file_path)
                if documents:
                    self._enrich_metadata(documents, file_path, file_hash, suffix)
                    return documents
            except Exception as e:
                logger.warning(f"Docling failed with {file_path.name}: {e}")

        # fallback
        loader_class = LOADERS.get(suffix, UnstructuredFileLoader)
        documents = loader_class(str(file_path)).load()
        self._enrich_metadata(documents, file_path, file_hash, suffix)
        return documents

    def _load_with_docling(self, file_path: Path) -> List[Document]:
        loader = DoclingLoader(
            file_path=[str(file_path)],
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer="intfloat/multilingual-e5-large"
            ),
        )

        docs = loader.load()

        # Приводим metadata к единому виду
        for doc in docs:
            dl_meta = doc.metadata.get("dl_meta", {})

            # page number
            page_no = None
            try:
                page_no = (
                    dl_meta.get("doc_items", [{}])[0]
                    .get("prov", [{}])[0]
                    .get("page_no")
                )
            except Exception:
                pass

            doc.metadata["page"] = page_no

        return docs

    def _enrich_metadata(
        self,
        documents: List[Document],
        file_path: Path,
        file_hash: str,
        suffix: str,
    ) -> None:
        for doc in documents:
            doc.metadata.update(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_hash": file_hash,
                    "file_type": suffix,
                }
            )
            doc.metadata.setdefault("page", None)

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
