import pytest
from src.config import _build_db_url, Settings


class TestBuildDbUrl:
    def test_build_db_url_basic(self):
        """Test basic database URL construction."""
        url = _build_db_url(
            user="testuser",
            password="testpass",
            host="localhost",
            port=5432,
            db="testdb"
        )
        expected = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
        assert url == expected

    def test_build_db_url_with_special_chars(self):
        """Test database URL construction with special characters."""
        url = _build_db_url(
            user="user@domain",
            password="pass%word",
            host="db.example.com",
            port=9999,
            db="my_db"
        )
        expected = "postgresql+asyncpg://user@domain:pass%word@db.example.com:9999/my_db"
        assert url == expected


class TestSettings:
    def test_settings_with_database_url_provided(self):
        """Test Settings initialization when database_url is provided."""
        settings = Settings(
            bot_token="test_token",
            admin_token="admin_token",
            user_token="user_token",
            database_url="custom_url"
        )
        assert settings.database_url == "custom_url"
        assert settings.bot_token == "test_token"

    def test_settings_without_database_url(self):
        """Test Settings initialization when database_url is not provided."""
        settings = Settings(
            bot_token="test_token",
            admin_token="admin_token",
            user_token="user_token",
            database_url=None,  # Explicitly set to None to trigger building
            db_user="custom_user",
            db_password="custom_pass",
            db_host="custom_host",
            db_port=9999,
            db_name="custom_db"
        )
        expected_url = "postgresql+asyncpg://custom_user:custom_pass@custom_host:9999/custom_db"
        assert settings.database_url == expected_url

    def test_settings_default_values(self):
        """Test Settings with default values."""
        settings = Settings(
            bot_token="test_token",
            admin_token="admin_token",
            user_token="user_token"
        )
        expected_url = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/legal_rag"
        assert settings.database_url == expected_url
        assert settings.api_base_url == "http://localhost:8000"
        assert settings.db_user == "postgres"
        assert settings.db_password == "postgres"
        assert settings.db_host == "127.0.0.1"
        assert settings.db_port == 5432
        assert settings.db_name == "legal_rag"