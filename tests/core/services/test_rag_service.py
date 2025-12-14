import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.core.services.RAGService import RAGService
from src.infra.llm.agent import LegalRAGAgent, RAGResponse


class TestRAGService:
    @pytest.fixture
    def mock_agent(self):
        """Create a mock LegalRAGAgent."""
        agent = MagicMock(spec=LegalRAGAgent)
        agent.query = AsyncMock()
        agent.add_document = AsyncMock()
        agent.index_documents = AsyncMock()
        agent.get_stats = AsyncMock()
        agent.health_check = AsyncMock()
        agent.close = AsyncMock()
        return agent

    @pytest.fixture
    def rag_service_with_mock(self, mock_agent):
        """Create RAGService with mocked agent."""
        return RAGService(agent=mock_agent)

    @pytest.fixture
    def rag_service_default(self):
        """Create RAGService with default initialization."""
        # This will try to create real configs, but we can mock them if needed
        return RAGService()

    def test_init_with_agent(self, mock_agent):
        """Test initialization with provided agent."""
        service = RAGService(agent=mock_agent)
        assert service._agent == mock_agent
        assert service.agent == mock_agent

    @patch('src.core.services.RAGService.LegalRAGAgent', autospec=True)
    def test_init_without_agent(self, mock_legal_agent):
        """Test initialization without agent (creates default)."""
        mock_instance = mock_legal_agent.return_value

        service = RAGService()

        assert service.agent is mock_instance
        mock_legal_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_query(self, rag_service_with_mock, mock_agent):
        """Test query method."""
        # Setup mock response
        mock_response = RAGResponse(
            answer="Test answer",
            sources=[{"title": "doc1", "content": "content1"}]
        )
        mock_agent.query.return_value = mock_response

        # Test
        answer, sources = await rag_service_with_mock.query("Test question", k=5)

        # Verify
        assert answer == "Test answer"
        assert sources == [{"title": "doc1", "content": "content1"}]
        mock_agent.query.assert_called_once_with("Test question", k=5)

    @pytest.mark.asyncio
    async def test_query_default_k(self, rag_service_with_mock, mock_agent):
        """Test query method with default k parameter."""
        mock_response = RAGResponse(answer="Answer", sources=[])
        mock_agent.query.return_value = mock_response

        await rag_service_with_mock.query("Question")

        mock_agent.query.assert_called_once_with("Question", k=None)

    @pytest.mark.asyncio
    async def test_add_document(self, rag_service_with_mock, mock_agent):
        """Test add_document method."""
        mock_agent.add_document.return_value = 42

        result = await rag_service_with_mock.add_document("path/to/doc.pdf")

        assert result == 42
        mock_agent.add_document.assert_called_once_with("path/to/doc.pdf")

    @pytest.mark.asyncio
    async def test_add_document_path_object(self, rag_service_with_mock, mock_agent):
        """Test add_document method with Path object."""
        path = Path("test.pdf")
        mock_agent.add_document.return_value = 1

        result = await rag_service_with_mock.add_document(path)

        assert result == 1
        mock_agent.add_document.assert_called_once_with(path)

    @pytest.mark.asyncio
    async def test_index_all(self, rag_service_with_mock, mock_agent):
        """Test index_all method."""
        mock_agent.index_documents.return_value = 100

        result = await rag_service_with_mock.index_all(force=True)

        assert result == 100
        mock_agent.index_documents.assert_called_once_with(force_reindex=True)

    @pytest.mark.asyncio
    async def test_index_all_default_force(self, rag_service_with_mock, mock_agent):
        """Test index_all method with default force parameter."""
        mock_agent.index_documents.return_value = 50

        result = await rag_service_with_mock.index_all()

        assert result == 50
        mock_agent.index_documents.assert_called_once_with(force_reindex=False)

    @pytest.mark.asyncio
    async def test_get_stats(self, rag_service_with_mock, mock_agent):
        """Test get_stats method."""
        expected_stats = {"documents": 10, "chunks": 100}
        mock_agent.get_stats.return_value = expected_stats

        result = await rag_service_with_mock.get_stats()

        assert result == expected_stats
        mock_agent.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, rag_service_with_mock, mock_agent):
        """Test health_check method."""
        mock_agent.health_check.return_value = True

        result = await rag_service_with_mock.health_check()

        assert result is True
        mock_agent.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, rag_service_with_mock, mock_agent):
        """Test close method."""
        await rag_service_with_mock.close()

        mock_agent.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, rag_service_with_mock, mock_agent):
        """Test async context manager."""
        async with rag_service_with_mock:
            pass

        mock_agent.close.assert_called_once()