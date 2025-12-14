import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.core.services.DocumentGenerationService import (
    DocumentGenerationService,
    GeneratedDocument
)
from src.infra.llm.yandex_gpt import YandexGPTResponse, YandexGPTMessage


def _make_response(text: str) -> YandexGPTResponse:
    """Create YandexGPTResponse with default token counts for tests."""
    return YandexGPTResponse(text=text, tokens_input=0, tokens_output=0)


class TestGeneratedDocument:
    def test_generated_document_creation(self):
        """Test GeneratedDocument dataclass creation."""
        doc = GeneratedDocument(
            title="Test Document",
            markdown_content="# Test\nContent",
            pdf_bytes=b"pdf content",
            generated_at=datetime(2023, 1, 1, 12, 0, 0),
            document_type="contract"
        )

        assert doc.title == "Test Document"
        assert doc.markdown_content == "# Test\nContent"
        assert doc.pdf_bytes == b"pdf content"
        assert doc.generated_at == datetime(2023, 1, 1, 12, 0, 0)
        assert doc.document_type == "contract"


class TestDocumentGenerationService:
    @pytest.fixture
    def mock_agent(self):
        """Create a mock LegalRAGAgent."""
        agent = MagicMock()
        agent.vector_store = MagicMock()
        agent.vector_store.search = AsyncMock()
        agent.gpt_client = MagicMock()
        agent.gpt_client.complete = AsyncMock()
        return agent

    @pytest.fixture
    def service(self, mock_agent):
        """Create DocumentGenerationService with mocked agent."""
        return DocumentGenerationService(agent=mock_agent)

    def test_init_with_agent(self, mock_agent):
        """Test initialization with agent."""
        service = DocumentGenerationService(agent=mock_agent)
        assert service.agent == mock_agent

    def test_init_without_agent(self):
        """Test initialization without agent."""
        service = DocumentGenerationService()
        assert service.agent is None

    def test_detect_document_type_found(self, service):
        """Test document type detection when type is found."""
        request = "I need a договор for services"
        result = service._detect_document_type(request)
        assert result == "договор"

    def test_detect_document_type_not_found(self, service):
        """Test document type detection when type is not found."""
        request = "Some random request without document type"
        result = service._detect_document_type(request)
        assert result is None

    def test_extract_title_with_h1(self, service):
        """Test title extraction with H1 header."""
        content = "# Document Title\n\nSome content"
        title = service._extract_title(content)
        assert title == "Document Title"

    def test_extract_title_without_h1(self, service):
        """Test title extraction without H1 header."""
        content = "Some content without header"
        title = service._extract_title(content)
        assert title == "Документ"

    def test_extract_title_empty_content(self, service):
        """Test title extraction with empty content."""
        content = ""
        title = service._extract_title(content)
        assert title == "Документ"

    def test_add_disclaimer(self, service):
        """Test adding disclaimer to content."""
        content = "# Test Document\n\nContent"
        result = service._add_disclaimer(content)

        assert "ВАЖНО:" in result
        assert "FOR REFERENCE ONLY" in result
        assert content in result

    def test_markdown_to_html(self, service):
        """Test Markdown to HTML conversion."""
        markdown = "# Title\n\nSome **bold** text"
        html = service._markdown_to_html(markdown)

        assert "<!DOCTYPE html>" in html
        assert '<h1 id="title">Title</h1>' in html
        assert "<strong>bold</strong>" in html
        assert "FOR REFERENCE ONLY" in html

    @patch('src.core.services.DocumentGenerationService.HTML')
    @patch('src.core.services.DocumentGenerationService.CSS')
    def test_html_to_pdf(self, mock_css, mock_html, service):
        """Test HTML to PDF conversion."""
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.write_pdf.return_value = b"pdf bytes"
        mock_html.return_value = mock_pdf_doc

        html = "<html><body>Test</body></html>"
        result = service._html_to_pdf(html)

        assert result == b"pdf bytes"
        mock_html.assert_called_once_with(string=html)
        mock_pdf_doc.write_pdf.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_without_agent(self):
        """Test generate raises error when agent is not initialized."""
        service = DocumentGenerationService()

        with pytest.raises(RuntimeError, match="LegalRAGAgent not initialized"):
            await service.generate("test request")

    @patch('src.core.services.DocumentGenerationService.datetime')
    @pytest.mark.asyncio
    async def test_generate_full_flow(self, mock_datetime, service, mock_agent):
        """Test full document generation flow."""
        mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)

        mock_doc = MagicMock()
        mock_doc.metadata = {"filename": "test.pdf"}
        mock_doc.page_content = "Some content"
        mock_agent.vector_store.search.return_value = [mock_doc]

        mock_response = _make_response(text="# Generated Document\n\nContent")
        mock_agent.gpt_client.complete.return_value = mock_response

        with patch.object(service, '_html_to_pdf', return_value=b"pdf bytes"):
            result = await service.generate("Generate a contract")

        assert isinstance(result, GeneratedDocument)
        assert result.title == "Generated Document"
        assert result.markdown_content == "# Generated Document\n\nContent"
        assert result.pdf_bytes == b"pdf bytes"
        assert result.generated_at == datetime(2023, 1, 1, 12, 0, 0)
        assert result.document_type is None

        mock_agent.vector_store.search.assert_called_once_with("Generate a contract", k=3)
        mock_agent.gpt_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_without_rag(self, service, mock_agent):
        """Test generation without RAG."""
        mock_response = _make_response(text="# Document\n\nContent")
        mock_agent.gpt_client.complete.return_value = mock_response

        with patch.object(service, '_html_to_pdf', return_value=b"pdf"):
            result = await service.generate("Request", use_rag=False)

        mock_agent.vector_store.search.assert_not_called()
        assert isinstance(result, GeneratedDocument)

    @pytest.mark.asyncio
    async def test_generate_with_custom_context(self, service, mock_agent):
        """Test generation with custom context."""
        mock_response = _make_response(text="# Document\n\nContent")
        mock_agent.gpt_client.complete.return_value = mock_response

        with patch.object(service, '_html_to_pdf', return_value=b"pdf"):
            result = await service.generate("Request", context="Custom context", use_rag=False)

        call_args = mock_agent.gpt_client.complete.call_args
        messages = call_args[0][0]
        prompt_text = messages[1].text
        assert "Пользовательский контекст:" in prompt_text
        assert "Custom context" in prompt_text

    @pytest.mark.asyncio
    async def test_generate_rag_error_handling(self, service, mock_agent):
        """Test generation handles RAG errors gracefully."""
        mock_agent.vector_store.search.side_effect = Exception("RAG error")

        mock_response = _make_response(text="# Document\n\nContent")
        mock_agent.gpt_client.complete.return_value = mock_response

        with patch.object(service, '_html_to_pdf', return_value=b"pdf"):
            result = await service.generate("Request")

        assert isinstance(result, GeneratedDocument)
        mock_agent.vector_store.search.assert_called_once()

    def test_get_document_types(self, service):
        """Test getting document types."""
        types = service.get_document_types()
        assert isinstance(types, dict)
        assert types is not service.get_document_types()