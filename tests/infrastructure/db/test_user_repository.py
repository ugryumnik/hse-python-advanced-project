import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.infra.db.user_repository import UserRepository
from src.infra.db.models import User


class TestUserRepository:
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
        """Create UserRepository with mocked session."""
        return UserRepository(mock_session)

    @pytest.fixture
    def sample_user(self):
        """Create a sample User instance."""
        return User(
            id=1,
            telegram_user_id=12345,
            username="testuser",
            role="user",
            created_at=datetime(2023, 1, 1, 12, 0, 0)
        )

    @pytest.mark.asyncio
    async def test_get_by_telegram_id_found(self, repository, mock_session, sample_user):
        """Test getting user by telegram ID when user exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_telegram_id(12345)

        assert result == sample_user
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_telegram_id_not_found(self, repository, mock_session):
        """Test getting user by telegram ID when user doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_telegram_id(99999)

        assert result is None
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, repository, mock_session, sample_user):
        """Test getting user by ID when user exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1)

        assert result == sample_user
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting user by ID when user doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(999)

        assert result is None
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_new_user(self, repository, mock_session):
        """Test upserting a new user."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.upsert(tg_id=12345, role="user", username="newuser")

        assert result.telegram_user_id == 12345
        assert result.role == "user"
        assert result.username == "newuser"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_upsert_existing_user_update_role_only(self, repository, mock_session, sample_user):
        """Test upserting existing user with role update only."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.upsert(tg_id=12345, role="admin")

        assert result == sample_user
        assert result.role == "admin"
        assert result.username == "testuser"
        mock_session.add.assert_not_called()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_upsert_existing_user_update_username(self, repository, mock_session, sample_user):
        """Test upserting existing user with username update."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.upsert(tg_id=12345, role="user", username="updateduser")

        assert result == sample_user
        assert result.role == "user"
        assert result.username == "updateduser"
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_delete_by_telegram_id_user_exists(self, repository, mock_session, sample_user):
        """Test deleting user by telegram ID when user exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result

        result = await repository.delete_by_telegram_id(12345)

        assert result is True
        mock_session.delete.assert_called_once_with(sample_user)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_telegram_id_user_not_exists(self, repository, mock_session):
        """Test deleting user by telegram ID when user doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.delete_by_telegram_id(99999)

        assert result is False
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()