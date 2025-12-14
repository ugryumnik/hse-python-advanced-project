import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from src.core.services.IngestionService import IngestionService, IngestionResult
from src.infra.llm.document_loader import ArchiveProcessingStats


class TestIngestionResult:
    def test_ingestion_result_creation(self):
        """Test IngestionResult dataclass creation."""
        result = IngestionResult(
            chunks_count=10,
            files_processed=2,
            file_type="document",
            processed_files=[{"filename": "test.pdf", "chunks": 10}]
        )
        assert result.chunks_count == 10
        assert result.files_processed == 2
        assert result.file_type == "document"
        assert result.processed_files == [{"filename": "test.pdf", "chunks": 10}]
        assert result.errors == []

    def test_ingestion_result_with_errors(self):
        """Test IngestionResult with errors."""
        result = IngestionResult(
            chunks_count=5,
            files_processed=1,
            file_type="archive",
            processed_files=[],
            errors=["Error 1", "Error 2"]
        )
        assert result.errors == ["Error 1", "Error 2"]


class DummyAsyncFile:
    """Helper async context manager mimicking aiofiles open."""

    def __init__(self):
        self.write = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class TestIngestionService:
    @pytest.fixture
    def mock_agent(self):
        """Create a mock LegalRAGAgent."""
        agent = MagicMock()
        agent.document_loader = MagicMock()
        agent.text_splitter = MagicMock()
        agent.vector_store = MagicMock()
        agent.vector_store.add_documents = AsyncMock()
        return agent

    @pytest.fixture
    def service(self, mock_agent):
        """Create IngestionService with mocked agent."""
        return IngestionService(agent=mock_agent)

    @pytest.fixture
    def mock_upload_file(self):
        """Create a mock UploadFile."""
        file = MagicMock()
        file.filename = "test.pdf"
        file.read = AsyncMock(return_value=b"test content")
        return file

    def test_init_with_agent(self, mock_agent):
        """Test initialization with agent."""
        service = IngestionService(agent=mock_agent)
        assert service.agent == mock_agent

    def test_init_without_agent(self):
        """Test initialization without agent."""
        service = IngestionService()
        assert service.agent is None

    def test_get_file_type_document(self, service):
        """Test file type detection for documents."""
        assert service._get_file_type("document.pdf") == "document"
        assert service._get_file_type("document.DOCX") == "document"
        assert service._get_file_type("document.txt") == "document"

    def test_get_file_type_archive(self, service):
        """Test file type detection for archives."""
        assert service._get_file_type("archive.zip") == "archive"
        assert service._get_file_type("archive.tar.gz") == "archive"
        assert service._get_file_type("archive.ZIP") == "archive"

    def test_get_file_type_unknown(self, service):
        """Test file type detection for unknown files."""
        assert service._get_file_type("file.unknown") == "unknown"
        assert service._get_file_type("file") == "unknown"

    @pytest.mark.asyncio
    async def test_process_file_without_agent(self, mock_upload_file):
        """Test processFile raises error when agent is not initialized."""
        service = IngestionService()

        with pytest.raises(RuntimeError, match="LegalRAGAgent not initialized"):
            await service.processFile(mock_upload_file)

    @pytest.mark.asyncio
    async def test_process_file_unknown_type(self, service, mock_upload_file):
        """Test processFile raises error for unknown file type."""
        mock_upload_file.filename = "test.unknown"

        with pytest.raises(ValueError, match="Неподдерживаемый формат файла"):
            await service.processFile(mock_upload_file)

    @pytest.mark.asyncio
    @patch('tempfile.mkdtemp')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('os.rmdir')
    async def test_process_file_document(self, mock_rmdir, mock_remove, mock_exists,
                                       mock_mkdtemp, service, mock_agent, mock_upload_file):
        """Test processing a document file."""
        mock_mkdtemp.return_value = "/tmp/test"
        mock_exists.return_value = True

        mock_documents = [MagicMock()]
        service.agent.document_loader.load_file.return_value = mock_documents

        mock_chunks = [MagicMock(), MagicMock()]
        service.agent.text_splitter.split_documents.return_value = mock_chunks

        mock_upload_file.read.side_effect = [b"test content", b""]

        def fake_open(*_args, **_kwargs):
            return DummyAsyncFile()

        with patch('aiofiles.open', side_effect=fake_open), \
            patch('os.path.getsize', return_value=1024):

            result = await service.processFile(mock_upload_file)

        assert isinstance(result, IngestionResult)
        assert result.chunks_count == 2
        assert result.files_processed == 1
        assert result.file_type == "document"
        assert len(result.processed_files) == 1

        service.agent.document_loader.load_file.assert_called_once()
        service.agent.text_splitter.split_documents.assert_called_once_with(mock_documents)
        service.agent.vector_store.add_documents.assert_called_once_with(mock_chunks)

    @pytest.mark.asyncio
    @patch('tempfile.mkdtemp')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('os.rmdir')
    async def test_process_file_archive(self, mock_rmdir, mock_remove, mock_exists,
                                      mock_mkdtemp, service, mock_agent):
        """Test processing an archive file."""
        mock_mkdtemp.return_value = "/tmp/test"
        mock_exists.return_value = True

        mock_upload_file = MagicMock()
        mock_upload_file.filename = "test.zip"
        mock_upload_file.read = AsyncMock(side_effect=[b"archive content", b""])

        mock_documents = [MagicMock(), MagicMock()]
        mock_stats = ArchiveProcessingStats()
        mock_stats.files_processed = 2
        mock_stats.processed_files = [
            MagicMock(filename="doc1.pdf", chunks_count=5, archive_path="test.zip"),
            MagicMock(filename="doc2.pdf", chunks_count=3, archive_path="test.zip")
        ]
        service.agent.document_loader.load_archive.return_value = (mock_documents, mock_stats)

        mock_chunks = [MagicMock()] * 8
        service.agent.text_splitter.split_documents.return_value = mock_chunks

        def fake_open(*_args, **_kwargs):
            return DummyAsyncFile()

        with patch('aiofiles.open', side_effect=fake_open), \
            patch('os.path.getsize', return_value=1024):

            result = await service.processFile(mock_upload_file)

        assert isinstance(result, IngestionResult)
        assert result.chunks_count == 8
        assert result.files_processed == 2
        assert result.file_type == "archive"
        assert len(result.processed_files) == 2

        service.agent.document_loader.load_archive.assert_called_once()
        service.agent.text_splitter.split_documents.assert_called_once_with(mock_documents)
        assert service.agent.vector_store.add_documents.call_count == 1

    @pytest.mark.asyncio
    async def test_process_document_no_documents(self, service, mock_agent):
        """Test processing document with no documents loaded."""
        service.agent.document_loader.load_file.return_value = []
        service.agent.text_splitter.split_documents.return_value = []

        result = await service._process_document(Path("test.pdf"), "test.pdf")

        assert result.chunks_count == 0
        assert result.files_processed == 1
        assert result.file_type == "document"
        service.agent.vector_store.add_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_archive_with_errors(self, service, mock_agent):
        """Test processing archive with errors."""
        mock_stats = ArchiveProcessingStats()
        mock_stats.files_processed = 1
        mock_stats.processed_files = [MagicMock(filename="doc.pdf", chunks_count=5)]
        mock_stats.errors = ["Error 1", "Error 2", "Error 3"]

        service.agent.document_loader.load_archive.return_value = ([], mock_stats)
        service.agent.text_splitter.split_documents.return_value = []

        result = await service._process_archive(Path("test.zip"))

        assert result.errors == ["Error 1", "Error 2", "Error 3"]
        assert len(result.errors) == 3