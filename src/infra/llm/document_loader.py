"""
Загрузка юридических документов с поддержкой Docling + LangChain + архивов
"""

import hashlib
import logging
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

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

SUPPORTED_EXTENSIONS = frozenset(LOADERS.keys())

ARCHIVE_EXTENSIONS = frozenset({".zip", ".tar", ".tgz", ".tbz2", ".txz"})
COMPOUND_ARCHIVE_EXTENSIONS = frozenset({".tar.gz", ".tar.bz2", ".tar.xz"})

# Лимиты безопасности
MAX_ARCHIVE_SIZE_MB = 500
MAX_ARCHIVE_SIZE = MAX_ARCHIVE_SIZE_MB * 1024 * 1024
MAX_FILES_IN_ARCHIVE = 1000
MAX_EXTRACTION_RATIO = 100
MAX_NESTED_DEPTH = 1


@dataclass
class ArchiveProcessingStats:
    """Статистика обработки архива"""
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    nested_archives: int = 0
    errors: list[str] = field(default_factory=list)
    processed_files: list[dict] = field(default_factory=list)

    def add_processed_file(
        self,
        filename: str,
        chunks_count: int,
        archive_chain: list[str] | None = None,
    ) -> None:
        """Добавить информацию об обработанном файле"""
        entry = {"filename": filename, "chunks": chunks_count}
        if archive_chain:
            entry["archive_path"] = " → ".join(archive_chain)
        self.processed_files.append(entry)


class ArchiveError(Exception):
    """Ошибка обработки архива"""
    pass


def compute_file_hash(file_path: Path) -> str:
    """Вычислить MD5-хеш файла"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArchiveHandler:
    """Безопасный обработчик архивов"""

    SKIP_NAMES = frozenset({
        "__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini", ".git", ".svn"
    })
    SKIP_PREFIXES = (".", "__", "~")

    @classmethod
    def is_archive(cls, path: Path) -> bool:
        """Проверить, является ли файл поддерживаемым архивом"""
        name_lower = path.name.lower()
        for ext in COMPOUND_ARCHIVE_EXTENSIONS:
            if name_lower.endswith(ext):
                return True
        return path.suffix.lower() in ARCHIVE_EXTENSIONS

    @classmethod
    def get_archive_type(cls, path: Path) -> str | None:
        """Определить тип архива"""
        name_lower = path.name.lower()
        if name_lower.endswith((".tar.gz", ".tgz")):
            return "tar.gz"
        elif name_lower.endswith((".tar.bz2", ".tbz2")):
            return "tar.bz2"
        elif name_lower.endswith((".tar.xz", ".txz")):
            return "tar.xz"
        elif name_lower.endswith(".tar"):
            return "tar"
        elif name_lower.endswith(".zip"):
            return "zip"
        return None

    @classmethod
    def _validate_path_safety(cls, member_path: str) -> None:
        """Проверка безопасности пути"""
        if member_path.startswith("/") or ".." in member_path:
            raise ArchiveError(f"Небезопасный путь: {member_path}")
        if len(member_path) > 1 and member_path[1] == ":":
            raise ArchiveError(f"Абсолютный Windows путь: {member_path}")

    @classmethod
    def _validate_zip(cls, zip_ref: zipfile.ZipFile, archive_path: Path) -> None:
        """Валидация ZIP архива"""
        members = zip_ref.infolist()
        if len(members) > MAX_FILES_IN_ARCHIVE:
            raise ArchiveError(f"Слишком много файлов: {len(members)}")

        archive_size = archive_path.stat().st_size
        if archive_size > 0:
            total_uncompressed = sum(m.file_size for m in members)
            ratio = total_uncompressed / archive_size
            if ratio > MAX_EXTRACTION_RATIO:
                raise ArchiveError(f"Подозрение на zip-бомбу: сжатие {ratio:.1f}x")

        for member in members:
            cls._validate_path_safety(member.filename)

    @classmethod
    def _validate_tar(cls, tar_ref: tarfile.TarFile) -> None:
        """Валидация TAR архива"""
        members = tar_ref.getmembers()
        if len(members) > MAX_FILES_IN_ARCHIVE:
            raise ArchiveError(f"Слишком много файлов: {len(members)}")

        for member in members:
            cls._validate_path_safety(member.name)
            if member.issym() or member.islnk():
                raise ArchiveError(f"Ссылки запрещены: {member.name}")
            if member.isdev():
                raise ArchiveError(f"Device-файлы запрещены: {member.name}")

    @classmethod
    def extract(cls, archive_path: Path) -> Path:
        """Извлечь архив во временную директорию"""
        size = archive_path.stat().st_size
        if size > MAX_ARCHIVE_SIZE:
            raise ArchiveError(f"Архив слишком большой: {size / 1024 / 1024:.1f} MB")

        archive_type = cls.get_archive_type(archive_path)
        if not archive_type:
            raise ArchiveError(f"Неподдерживаемый формат: {archive_path.suffix}")

        temp_dir = Path(tempfile.mkdtemp(prefix="legal_docs_"))

        try:
            if archive_type == "zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    cls._validate_zip(zf, archive_path)
                    zf.extractall(temp_dir)
            else:
                mode_map = {"tar": "r", "tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}
                with tarfile.open(archive_path, mode_map.get(archive_type, "r")) as tf:
                    cls._validate_tar(tf)
                    tf.extractall(temp_dir, filter="data")

            logger.info(f"Архив распакован: {archive_path.name}")
            return temp_dir

        except (zipfile.BadZipFile, tarfile.TarError) as e:
            cls.cleanup(temp_dir)
            raise ArchiveError(f"Повреждённый архив: {e}") from e
        except Exception:
            cls.cleanup(temp_dir)
            raise

    @classmethod
    def cleanup(cls, path: Path) -> None:
        """Удалить временную директорию"""
        try:
            if path and path.exists():
                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Ошибка очистки {path}: {e}")

    @classmethod
    def should_skip_file(cls, path: Path) -> bool:
        """Проверить, нужно ли пропустить файл"""
        if path.name in cls.SKIP_NAMES:
            return True
        if any(path.name.startswith(p) for p in cls.SKIP_PREFIXES):
            return True
        for part in path.parts:
            if part in cls.SKIP_NAMES or any(part.startswith(p) for p in cls.SKIP_PREFIXES):
                return True
        return False

    @classmethod
    def iter_files(cls, directory: Path) -> Iterator[Path]:
        """Итерация по файлам директории с фильтрацией системных"""
        for path in directory.rglob("*"):
            if path.is_file() and not cls.should_skip_file(path):
                yield path


class LegalDocumentLoader:
    """Загрузчик юридических документов с поддержкой архивов"""

    def __init__(self, documents_dir: Path | str):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def load_directory(self) -> Iterator[Document]:
        """Загрузить все документы из директории"""
        regular_files: list[Path] = []
        archive_files: list[Path] = []

        for f in self.documents_dir.rglob("*"):
            if not f.is_file() or ArchiveHandler.should_skip_file(f):
                continue
            if ArchiveHandler.is_archive(f):
                archive_files.append(f)
            elif f.suffix.lower() in SUPPORTED_EXTENSIONS:
                regular_files.append(f)

        logger.info(f"Найдено: {len(regular_files)} документов, {len(archive_files)} архивов")

        for file_path in regular_files:
            try:
                yield from self._load_single_file(file_path)
            except Exception as e:
                logger.error(f"Пропущен {file_path.name}: {e}")

        for archive_path in archive_files:
            try:
                yield from self._load_archive(archive_path)
            except Exception as e:
                logger.error(f"Ошибка архива {archive_path.name}: {e}")

    def load_file(self, file_path: Path | str) -> list[Document]:
        """Загрузить один файл или архив"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if ArchiveHandler.is_archive(file_path):
            return list(self._load_archive(file_path))
        return self._load_single_file(file_path)

    def load_archive(self, archive_path: Path | str) -> tuple[list[Document], ArchiveProcessingStats]:
        """Загрузить архив с детальной статистикой"""
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Архив не найден: {archive_path}")
        if not ArchiveHandler.is_archive(archive_path):
            raise ValueError(f"Неподдерживаемый формат: {archive_path.suffix}")

        stats = ArchiveProcessingStats()
        documents: list[Document] = []

        for doc in self._process_archive_recursive(archive_path, stats=stats):
            documents.append(doc)

        logger.info(
            f"Архив {archive_path.name}: "
            f"обработано={stats.files_processed}, пропущено={stats.files_skipped}, ошибок={stats.files_failed}"
        )
        return documents, stats

    def _load_single_file(self, file_path: Path, archive_chain: list[str] | None = None) -> list[Document]:
        """Загрузить один документ"""
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            return []

        file_hash = compute_file_hash(file_path)

        # Пробуем Docling для сложных форматов
        if suffix in {".pdf", ".docx", ".doc"}:
            try:
                documents = self._load_with_docling(file_path)
                if documents:
                    self._enrich_metadata(documents, file_path, file_hash, suffix, archive_chain)
                    return documents
            except Exception as e:
                logger.warning(f"Docling fallback для {file_path.name}: {e}")

        # Fallback на стандартные loaders
        loader_class = LOADERS.get(suffix, UnstructuredFileLoader)
        try:
            documents = loader_class(str(file_path)).load()
            self._enrich_metadata(documents, file_path, file_hash, suffix, archive_chain)
            return documents
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path.name}: {e}")
            raise

    def _load_with_docling(self, file_path: Path) -> list[Document]:
        """Загрузка через Docling с чанкингом"""
        loader = DoclingLoader(
            file_path=[str(file_path)],
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer="intfloat/multilingual-e5-large"),
        )
        docs = loader.load()

        for doc in docs:
            dl_meta = doc.metadata.get("dl_meta", {})
            page_no = None
            try:
                doc_items = dl_meta.get("doc_items", [])
                if doc_items:
                    prov = doc_items[0].get("prov", [])
                    if prov:
                        page_no = prov[0].get("page_no")
            except (IndexError, KeyError, TypeError):
                pass
            doc.metadata["page"] = page_no

        return docs

    def _enrich_metadata(
        self,
        documents: list[Document],
        file_path: Path,
        file_hash: str,
        suffix: str,
        archive_chain: list[str] | None = None,
    ) -> None:
        """Обогатить метаданные документов"""
        for doc in documents:
            doc.metadata.update({
                "source": str(file_path),
                "filename": file_path.name,
                "file_hash": file_hash,
                "file_type": suffix,
            })
            doc.metadata.setdefault("page", None)
            if archive_chain:
                doc.metadata["archive_source"] = " → ".join(archive_chain)

    def _load_archive(self, archive_path: Path) -> Iterator[Document]:
        """Загрузка архива"""
        stats = ArchiveProcessingStats()
        yield from self._process_archive_recursive(archive_path, stats=stats)
        if stats.errors:
            logger.warning(f"Архив {archive_path.name}: {len(stats.errors)} ошибок")

    def _process_archive_recursive(
        self,
        archive_path: Path,
        archive_chain: list[str] | None = None,
        depth: int = 0,
        stats: ArchiveProcessingStats | None = None,
    ) -> Iterator[Document]:
        """Обработка архива (с ограниченной вложенностью)"""
        if stats is None:
            stats = ArchiveProcessingStats()

        current_chain = (archive_chain or []) + [archive_path.name]

        if depth >= MAX_NESTED_DEPTH:
            stats.errors.append(f"Превышена глубина вложенности: {' → '.join(current_chain)}")
            return

        temp_dir: Path | None = None
        try:
            temp_dir = ArchiveHandler.extract(archive_path)

            for file_path in ArchiveHandler.iter_files(temp_dir):
                # Вложенные архивы обрабатываем только на первом уровне
                if ArchiveHandler.is_archive(file_path):
                    stats.nested_archives += 1
                    if depth < MAX_NESTED_DEPTH - 1:
                        yield from self._process_archive_recursive(
                            file_path, current_chain, depth + 1, stats
                        )
                    continue

                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    stats.files_skipped += 1
                    continue

                try:
                    documents = self._load_single_file(file_path, current_chain)
                    if documents:
                        stats.files_processed += 1
                        stats.add_processed_file(file_path.name, len(documents), current_chain)
                        yield from documents
                    else:
                        stats.files_skipped += 1
                except Exception as e:
                    stats.files_failed += 1
                    stats.errors.append(f"Ошибка {file_path.name}: {e}")

        except ArchiveError as e:
            stats.errors.append(f"Ошибка архива {archive_path.name}: {e}")
        finally:
            if temp_dir:
                ArchiveHandler.cleanup(temp_dir)