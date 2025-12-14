from types import SimpleNamespace

import pytest

from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup

from bot.handlers.handlers import (
    create_sources_keyboard,
    get_supported_formats_text,
    is_supported_file,
)
from bot.keyboards import mode_keyboard
from bot.middlewares.logging import LoggingMiddleware


def test_is_supported_file_by_extension_compound_archive():
    ok, kind = is_supported_file("docs.tar.gz", None)
    assert ok is True
    assert kind == "archive"


def test_is_supported_file_by_extension_document():
    ok, kind = is_supported_file("contract.PDF", None)
    assert ok is True
    assert kind == "document"


def test_is_supported_file_by_mime_type_when_no_filename():
    ok, kind = is_supported_file(None, "application/pdf")
    assert ok is True
    assert kind == "document"

    ok, kind = is_supported_file(None, "application/zip")
    assert ok is True
    assert kind == "archive"


def test_is_supported_file_unsupported():
    ok, kind = is_supported_file("virus.exe", "application/octet-stream")
    assert ok is False
    assert kind == "unsupported"


def test_get_supported_formats_text_contains_expected_extensions():
    text = get_supported_formats_text()
    assert ".pdf" in text
    assert ".zip" in text


def test_create_sources_keyboard_none_on_empty():
    assert create_sources_keyboard([]) is None


def test_create_sources_keyboard_dedup_and_limit():
    sources = [
        {"filename": "a.pdf", "page": 1},
        {"filename": "a.pdf", "page": 1},
        {"filename": "b.pdf", "page": 2},
    ]

    kb = create_sources_keyboard(sources)
    assert isinstance(kb, InlineKeyboardMarkup)
    assert len(kb.inline_keyboard) == 2
    assert kb.inline_keyboard[0][0].callback_data == "src:0"
    assert kb.inline_keyboard[1][0].callback_data == "src:2"

    many = [{"filename": f"f{i}.pdf", "page": i} for i in range(20)]
    kb2 = create_sources_keyboard(many)
    assert isinstance(kb2, InlineKeyboardMarkup)
    assert len(kb2.inline_keyboard) == 8


def test_mode_keyboard_layout_and_texts():
    assert isinstance(mode_keyboard, ReplyKeyboardMarkup)
    rows = mode_keyboard.keyboard
    assert rows[0][0].text == "Задать вопрос"
    assert rows[0][1].text == "Загрузить документ"
    assert rows[1][0].text == "Создать документ"


@pytest.mark.asyncio
async def test_logging_middleware_passes_through(caplog):
    mw = LoggingMiddleware()

    async def handler(event, data):
        return {"ok": True, "event": event, "data": data}

    event = SimpleNamespace(type="test")
    data = {"x": 1}

    with caplog.at_level("INFO"):
        result = await mw(handler, event, data)

    assert result["ok"] is True
    assert result["event"] is event
    assert result["data"] == data
    assert any("Received event" in r.message for r in caplog.records)
