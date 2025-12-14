import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.infra.db.chat_history_repository import ChatHistoryRepository
from src.infra.db.models import ChatHistory


class TestChatHistoryRepository:
    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture
    def repository(self, mock_session):
        """Create ChatHistoryRepository with mocked session."""
        return ChatHistoryRepository(mock_session)

    @pytest.fixture
    def sample_chat_history(self):
        """Create a sample ChatHistory instance."""
        return ChatHistory(
            id=1,
            user_id=1,
            query="Test query",
            answer="Test answer",
            used_sources={"sources": ["doc1.pdf"]},
            created_at=datetime(2023, 1, 1, 12, 0, 0)
        )

    @pytest.mark.asyncio
    async def test_create_chat_history(self, repository, mock_session):
        """Test creating a new chat history entry."""
        # Test
        result = await repository.create(
            user_id=1,
            query="Test query",
            answer="Test answer",
            used_sources={"sources": ["doc1.pdf"]}
        )

        # Verify
        assert result.user_id == 1
        assert result.query == "Test query"
        assert result.answer == "Test answer"
        assert result.used_sources == {"sources": ["doc1.pdf"]}
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_create_chat_history_minimal(self, repository, mock_session):
        """Test creating a chat history entry with minimal parameters."""
        result = await repository.create(
            user_id=1,
            query="Query",
            answer="Answer"
        )

        assert result.used_sources is None

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, repository, mock_session, sample_chat_history):
        """Test getting chat history by ID when entry exists."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_chat_history
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_by_id(1)

        # Verify
        assert result == sample_chat_history

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting chat history by ID when entry doesn't exist."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_by_id(999)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_user_default_params(self, repository, mock_session, sample_chat_history):
        """Test getting chat history by user ID with default parameters."""
        # Setup mock result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_chat_history]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_by_user(1)

        # Verify
        assert result == [sample_chat_history]

    @pytest.mark.asyncio
    async def test_get_by_user_custom_params(self, repository, mock_session, sample_chat_history):
        """Test getting chat history by user ID with custom parameters."""
        # Setup mock result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_chat_history]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_by_user(user_id=1, limit=20, offset=5)

        # Verify
        assert result == [sample_chat_history]

    @pytest.mark.asyncio
    async def test_get_by_user_empty(self, repository, mock_session):
        """Test getting chat history by user ID when no entries exist."""
        # Setup mock result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_by_user(999)

        # Verify
        assert result == []

    @pytest.mark.asyncio
    async def test_get_recent_default_limit(self, repository, mock_session, sample_chat_history):
        """Test getting recent chat history with default limit."""
        # Setup mock result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_chat_history]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_recent(1)

        # Verify
        assert result == [sample_chat_history]

    @pytest.mark.asyncio
    async def test_get_recent_custom_limit(self, repository, mock_session, sample_chat_history):
        """Test getting recent chat history with custom limit."""
        # Setup mock result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_chat_history]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.get_recent(1, limit=5)

        # Verify
        assert result == [sample_chat_history]

    @pytest.mark.asyncio
    async def test_count_by_user(self, repository, mock_session):
        """Test counting chat history entries by user ID."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.count_by_user(1)

        # Verify
        assert result == 42

    @pytest.mark.asyncio
    async def test_count_by_user_zero(self, repository, mock_session):
        """Test counting chat history entries when none exist."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.count_by_user(999)

        # Verify
        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_by_user(self, repository, mock_session):
        """Test deleting chat history entries by user ID."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.delete_by_user(1)

        # Verify
        assert result == 5
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_user_no_entries(self, repository, mock_session):
        """Test deleting chat history entries when none exist."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        # Test
        result = await repository.delete_by_user(999)

        # Verify
        assert result == 0
        mock_session.commit.assert_called_once()