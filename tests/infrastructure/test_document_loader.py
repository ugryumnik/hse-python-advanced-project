import tarfile
import zipfile
from pathlib import Path

import pytest

from src.infra.llm.document_loader import (
    ARCHIVE_EXTENSIONS,
    ArchiveError,
    ArchiveHandler,
    LegalDocumentLoader,
    compute_file_hash,
)


def _make_zip(zip_path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _make_tar(tar_path: Path, members: list[tarfile.TarInfo], data_by_name: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w") as tf:
        for ti in members:
            data = data_by_name.get(ti.name)
            if data is None:
                tf.addfile(ti)
            else:
                import io

                bio = io.BytesIO(data)
                ti.size = len(data)
                tf.addfile(ti, fileobj=bio)


def test_compute_file_hash_is_stable(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello")

    h1 = compute_file_hash(p)
    h2 = compute_file_hash(p)

    assert h1 == h2
    assert len(h1) == 32  # md5 hex


def test_archive_handler_is_archive_and_type(tmp_path: Path):
    assert ArchiveHandler.is_archive(tmp_path / "a.zip")
    assert ArchiveHandler.is_archive(tmp_path / "a.tar")
    assert ArchiveHandler.is_archive(tmp_path / "a.tgz")
    assert ArchiveHandler.is_archive(tmp_path / "a.tar.gz")

    assert ArchiveHandler.get_archive_type(tmp_path / "a.zip") == "zip"
    assert ArchiveHandler.get_archive_type(tmp_path / "a.tar") == "tar"
    assert ArchiveHandler.get_archive_type(tmp_path / "a.tgz") == "tar.gz"
    assert ArchiveHandler.get_archive_type(tmp_path / "a.tar.gz") == "tar.gz"


def test_archive_handler_iter_files_skips_system_entries(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("x", encoding="utf-8")

    macosx = tmp_path / "__MACOSX"
    macosx.mkdir()
    (macosx / "inner.txt").write_text("skip", encoding="utf-8")

    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "a.txt").write_text("skip", encoding="utf-8")

    found = sorted(p.name for p in ArchiveHandler.iter_files(tmp_path))
    assert found == ["ok.txt"]


def test_archive_extract_rejects_zip_slip(tmp_path: Path):
    z = tmp_path / "bad.zip"
    _make_zip(z, {"../evil.txt": b"x"})

    with pytest.raises(ArchiveError, match="Небезопасный путь"):
        ArchiveHandler.extract(z)


def test_archive_extract_rejects_zip_bomb_ratio(tmp_path: Path):
    # Highly compressible payload => very high uncompressed/archive ratio
    z = tmp_path / "bomb.zip"
    _make_zip(z, {"big.txt": b"0" * (2 * 1024 * 1024)})

    with pytest.raises(ArchiveError, match="zip-бомбу"):
        ArchiveHandler.extract(z)


def test_archive_extract_rejects_tar_symlink(tmp_path: Path):
    t = tmp_path / "bad.tar"

    ti = tarfile.TarInfo("link")
    ti.type = tarfile.SYMTYPE
    ti.linkname = "target"

    _make_tar(t, [ti], data_by_name={})

    with pytest.raises(ArchiveError, match="Ссылки запрещены"):
        ArchiveHandler.extract(t)


def test_legal_document_loader_load_file_txt_enriches_metadata(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    p = docs_dir / "doc.txt"
    p.write_text("Hello", encoding="utf-8")

    loader = LegalDocumentLoader(docs_dir)
    docs = loader.load_file(p)

    assert len(docs) >= 1
    d0 = docs[0]
    assert d0.metadata["filename"] == "doc.txt"
    assert d0.metadata["file_type"] == ".txt"
    assert d0.metadata["source"] == str(p)
    assert d0.metadata["file_hash"] == compute_file_hash(p)
    assert "page" in d0.metadata


def test_legal_document_loader_load_archive_zip_stats_and_archive_source(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    archive = docs_dir / "bundle.zip"

    # include a nested archive + one supported file
    nested = b"PK\x03\x04"  # not a valid zip, but will still be detected by suffix only when extracted
    _make_zip(
        archive,
        {
            "inner.zip": nested,
            "a.txt": b"content",
            "ignored.bin": b"x",
        },
    )

    loader = LegalDocumentLoader(docs_dir)
    documents, stats = loader.load_archive(archive)

    assert stats.nested_archives == 1
    assert stats.files_processed == 1
    assert stats.files_skipped >= 1
    assert len(stats.processed_files) == 1
    assert stats.processed_files[0]["filename"] == "a.txt"

    assert len(documents) >= 1
    assert documents[0].metadata.get("archive_source") == "bundle.zip"


def test_legal_document_loader_load_archive_rejects_non_archive(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    p = docs_dir / "x.txt"
    p.write_text("x", encoding="utf-8")

    loader = LegalDocumentLoader(docs_dir)

    with pytest.raises(ValueError):
        loader.load_archive(p)


def test_archive_extensions_are_lowercase_and_start_with_dot():
    # quick sanity: prevents regressions from accidental values like 'zip' or '.ZIP'
    assert all(ext.startswith(".") for ext in ARCHIVE_EXTENSIONS)
    assert all(ext == ext.lower() for ext in ARCHIVE_EXTENSIONS)
