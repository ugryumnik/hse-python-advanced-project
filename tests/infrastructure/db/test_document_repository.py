import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.infra.db.document_repository import DocumentRepository
from src.infra.db.models import Document


class TestDocumentRepository:
    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        session.delete = AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create DocumentRepository with mocked session."""
        return DocumentRepository(mock_session)

    @pytest.fixture
    def sample_document(self):
        """Create a sample Document instance."""
        return Document(
            id=1,
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_hash="abc123",
            uploaded_by=1,
            status="processed",
            uploaded_at=datetime(2023, 1, 1, 12, 0, 0)
        )

    @pytest.mark.asyncio
    async def test_create_document(self, repository, mock_session):
        """Test creating a new document."""
        result = await repository.create(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_hash="abc123",
            uploaded_by=1,
            status="processing"
        )

        assert result.filename == "test.pdf"
        assert result.file_path == "/path/to/test.pdf"
        assert result.file_hash == "abc123"
        assert result.uploaded_by == 1
        assert result.status == "processing"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_create_document_defaults(self, repository, mock_session):
        """Test creating a document with default values."""
        result = await repository.create(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_hash="abc123"
        )

        assert result.uploaded_by is None
        assert result.status == "processing"

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, repository, mock_session, sample_document):
        """Test getting document by ID when document exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1)

        assert result == sample_document
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting document by ID when document doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_hash_found(self, repository, mock_session, sample_document):
        """Test getting document by hash when document exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_hash("abc123")

        assert result == sample_document

    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self, repository, mock_session):
        """Test getting document by hash when document doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_hash("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_user(self, repository, mock_session, sample_document):
        """Test getting documents by user ID."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_document]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_user(1)

        assert result == [sample_document]

    @pytest.mark.asyncio
    async def test_get_by_user_empty(self, repository, mock_session):
        """Test getting documents by user ID when no documents exist."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_user(999)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_default_params(self, repository, mock_session, sample_document):
        """Test getting all documents with default parameters."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_document]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_all()

        assert result == [sample_document]
        call_args = mock_session.execute.call_args[0][0]
        assert str(call_args).find("LIMIT") != -1
        assert str(call_args).find("OFFSET") != -1

    @pytest.mark.asyncio
    async def test_get_all_custom_params(self, repository, mock_session, sample_document):
        """Test getting all documents with custom parameters."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_document]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.get_all(limit=50, offset=10)

        assert result == [sample_document]

    @pytest.mark.asyncio
    async def test_update_status_document_exists(self, repository, mock_session, sample_document):
        """Test updating document status when document exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        result = await repository.update_status(1, "completed")

        assert result == sample_document
        assert result.status == "completed"
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_update_status_document_not_exists(self, repository, mock_session):
        """Test updating document status when document doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.update_status(999, "completed")

        assert result is None
        mock_session.commit.assert_not_called()
        mock_session.refresh.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_document_exists(self, repository, mock_session, sample_document):
        """Test deleting document when document exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_document
        mock_session.execute.return_value = mock_result

        result = await repository.delete(1)

        assert result is True
        mock_session.delete.assert_called_once_with(sample_document)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_not_exists(self, repository, mock_session):
        """Test deleting document when document doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.delete(999)

        assert result is False
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()