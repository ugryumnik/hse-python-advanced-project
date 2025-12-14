"""Сервис генерации юридических документов"""

import io
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

from infra.llm import LegalRAGAgent
from infra.llm.prompts import (
    DOCUMENT_GENERATION_SYSTEM_PROMPT,
    DOCUMENT_GENERATION_PROMPT,
    DOCUMENT_TYPES,
)
from infra.llm.yandex_gpt import YandexGPTMessage

logger = logging.getLogger(__name__)

# ============================================================================
# CSS для PDF документа
# ============================================================================

DOCUMENT_CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm 2cm 2.5cm;

    @top-right {
        content: "FOR REFERENCE ONLY";
        font-size: 8pt;
        color: #999;
        font-style: italic;
    }

    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: 'DejaVu Serif', 'Times New Roman', serif;
    font-size: 12pt;
    line-height: 1.5;
    color: #000;
    text-align: justify;
}

h1 {
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 20pt;
    text-transform: uppercase;
}

h2 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 10pt;
}

h3 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 8pt;
}

p {
    margin-bottom: 10pt;
    text-indent: 1.25cm;
}

p:first-of-type {
    text-indent: 0;
}

ul, ol {
    margin-left: 1cm;
    margin-bottom: 10pt;
}

li {
    margin-bottom: 5pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15pt 0;
}

th, td {
    border: 1px solid #000;
    padding: 8pt;
    text-align: left;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
}

.header {
    text-align: right;
    margin-bottom: 20pt;
    font-size: 11pt;
}

.signature-block {
    margin-top: 40pt;
    page-break-inside: avoid;
}

.signature-line {
    border-bottom: 1px solid #000;
    width: 200pt;
    display: inline-block;
    margin-right: 20pt;
}

.watermark {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-45deg);
    font-size: 60pt;
    color: rgba(200, 200, 200, 0.3);
    z-index: -1;
    white-space: nowrap;
}

blockquote {
    margin: 15pt 0;
    padding: 10pt 20pt;
    border-left: 3px solid #ccc;
    background-color: #f9f9f9;
    font-style: italic;
}

code {
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 10pt;
    background-color: #f4f4f4;
    padding: 2pt 4pt;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20pt 0;
}

.disclaimer {
    margin-top: 30pt;
    padding: 10pt;
    border: 1px solid #ccc;
    background-color: #fffef0;
    font-size: 9pt;
    color: #666;
}
"""


# ============================================================================
# Модели данных
# ============================================================================

@dataclass
class GeneratedDocument:
    """Результат генерации документа"""
    title: str
    markdown_content: str
    pdf_bytes: bytes
    generated_at: datetime
    document_type: str | None = None


# ============================================================================
# Сервис генерации
# ============================================================================

class DocumentGenerationService:
    """Сервис для генерации юридических документов"""

    def __init__(self, agent: LegalRAGAgent | None = None):
        self.agent = agent
        self._font_config = FontConfiguration()

    def _detect_document_type(self, request: str) -> str | None:
        """Определить тип документа по запросу"""
        request_lower = request.lower()

        for doc_type, description in DOCUMENT_TYPES.items():
            if doc_type in request_lower:
                return doc_type

        return None

    def _extract_title(self, markdown_content: str) -> str:
        """Извлечь заголовок из Markdown"""
        lines = markdown_content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        return "Документ"

    def _add_disclaimer(self, markdown_content: str) -> str:
        """Добавить дисклеймер в конец документа"""
        disclaimer = """

---

> **ВАЖНО:** Данный документ сгенерирован автоматически с использованием 
> искусственного интеллекта и предоставляется исключительно в справочных целях 
> (FOR REFERENCE ONLY). Перед использованием документа рекомендуется консультация 
> с квалифицированным юристом. Авторы не несут ответственности за последствия 
> использования данного документа.

"""
        return markdown_content + disclaimer

    def _markdown_to_html(self, markdown_content: str) -> str:
        """Конвертировать Markdown в HTML"""
        # Расширения для лучшего форматирования
        extensions = [
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
            'sane_lists',
        ]

        html_body = markdown.markdown(
            markdown_content,
            extensions=extensions,
            output_format='html5'
        )

        # Оборачиваем в полный HTML документ
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Документ</title>
</head>
<body>
    <div class="watermark">FOR REFERENCE ONLY</div>
    {html_body}
</body>
</html>"""

        return html

    def _html_to_pdf(self, html_content: str) -> bytes:
        """Конвертировать HTML в PDF"""
        css = CSS(string=DOCUMENT_CSS, font_config=self._font_config)

        html_doc = HTML(string=html_content)

        pdf_bytes = html_doc.write_pdf(
            stylesheets=[css],
            font_config=self._font_config
        )

        return pdf_bytes

    async def generate(
            self,
            request: str,
            context: str | None = None,
            use_rag: bool = True,
    ) -> GeneratedDocument:
        """
        Сгенерировать юридический документ.

        Args:
            request: Описание документа от пользователя
            context: Дополнительный контекст (необязательно)
            use_rag: Использовать ли RAG для получения контекста

        Returns:
            GeneratedDocument с markdown и PDF
        """
        if not self.agent:
            raise RuntimeError("LegalRAGAgent not initialized")

        logger.info(f"Генерация документа: {request[:100]}...")

        # Определяем тип документа
        doc_type = self._detect_document_type(request)

        # Получаем релевантный контекст из базы знаний
        rag_context = ""
        if use_rag:
            try:
                docs = await self.agent.vector_store.search(request, k=3)
                if docs:
                    rag_context = "\n\n".join([
                        f"[{doc.metadata.get('filename', '?')}]: {doc.page_content[:500]}"
                        for doc in docs
                    ])
            except Exception as e:
                logger.warning(f"RAG context error: {e}")

        # Объединяем контекст
        full_context = ""
        if context:
            full_context += f"Пользовательский контекст:\n{context}\n\n"
        if rag_context:
            full_context += f"Релевантные документы из базы:\n{rag_context}"

        if not full_context:
            full_context = "Не предоставлен"

        # Формируем промпт
        prompt = DOCUMENT_GENERATION_PROMPT.format(
            request=request,
            context=full_context
        )

        messages = [
            YandexGPTMessage(role="system", text=DOCUMENT_GENERATION_SYSTEM_PROMPT),
            YandexGPTMessage(role="user", text=prompt),
        ]

        # Генерируем документ
        response = await self.agent.gpt_client.complete(
            messages,
            temperature=0.3,  # Более детерминированный для документов
            max_tokens=4000,
        )

        markdown_content = response.text

        # Добавляем дисклеймер
        markdown_with_disclaimer = self._add_disclaimer(markdown_content)

        # Извлекаем заголовок
        title = self._extract_title(markdown_content)

        # Конвертируем в PDF
        html_content = self._markdown_to_html(markdown_with_disclaimer)
        pdf_bytes = self._html_to_pdf(html_content)

        logger.info(f"Документ сгенерирован: {title} ({len(pdf_bytes)} bytes)")

        return GeneratedDocument(
            title=title,
            markdown_content=markdown_content,
            pdf_bytes=pdf_bytes,
            generated_at=datetime.utcnow(),
            document_type=doc_type,
        )

    def get_document_types(self) -> dict[str, str]:
        """Получить список поддерживаемых типов документов"""
        return DOCUMENT_TYPES.copy()