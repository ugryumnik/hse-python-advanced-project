from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiogram.types import InlineKeyboardMarkup

import bot.handlers.handlers as h


def _fake_get_session(session_obj: object):
    def _get_session():
        async def _gen():
            yield session_obj

        return _gen()

    return _get_session


def _fake_message(
    *,
    text: str | None = None,
    user_id: int = 1,
    chat_id: int = 100,
    document: object | None = None,
):
    bot = SimpleNamespace(
        send_chat_action=AsyncMock(),
        download=AsyncMock(),
    )
    message = SimpleNamespace(
        text=text,
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=chat_id),
        bot=bot,
        document=document,
        answer=AsyncMock(),
        answer_document=AsyncMock(),
    )
    return message


class _DummyResponse:
    def __init__(self, *, status: int, json_data=None, text_data: str = ""):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data
        self.headers = {}

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyClientSession:
    def __init__(self, responses: list[_DummyResponse]):
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, **kwargs):
        # Consume responses in order; tests keep it simple (single call or deterministic order)
        if not self._responses:
            raise AssertionError(f"Unexpected POST to {url}")
        return self._responses.pop(0)


class _DummyFormData:
    def __init__(self):
        self.fields: list[tuple[str, object, dict]] = []

    def add_field(self, name: str, value: object, **kwargs):
        self.fields.append((name, value, kwargs))


@pytest.mark.asyncio
async def test_cmd_start_asks_token_for_new_user(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=None))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="/start")

    await h.cmd_start(msg, state)

    state.set_state.assert_awaited_once_with(h.BotStates.auth_token)
    msg.answer.assert_awaited_once()
    assert "секретный токен" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_cmd_start_shows_menu_for_existing_user(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="user")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="/start")

    await h.cmd_start(msg, state)

    state.set_state.assert_not_awaited()
    msg.answer.assert_awaited_once()
    assert "Сол Гудман" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_auth_token_rejects_invalid_token(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(upsert=AsyncMock())

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)
    monkeypatch.setattr(h, "settings", SimpleNamespace(admin_token="admin", user_token="user"))

    state = SimpleNamespace(clear=AsyncMock())
    msg = _fake_message(text="nope", user_id=5)

    await h.handle_auth_token(msg, state)

    repo.upsert.assert_not_awaited()
    state.clear.assert_not_awaited()
    msg.answer.assert_awaited_once()
    assert "чс" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_auth_token_accepts_admin_token(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(upsert=AsyncMock())

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)
    monkeypatch.setattr(h, "settings", SimpleNamespace(admin_token="admin", user_token="user"))

    state = SimpleNamespace(clear=AsyncMock())
    msg = _fake_message(text="admin", user_id=7)

    await h.handle_auth_token(msg, state)

    repo.upsert.assert_awaited_once_with(7, "admin")
    state.clear.assert_awaited_once()
    msg.answer.assert_awaited_once()
    assert "Верный токен" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_select_upload_mode_requires_auth(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=None))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="Загрузить документ")

    await h.select_upload_mode(msg, state)

    state.set_state.assert_awaited_once_with(h.BotStates.auth_token)
    msg.answer.assert_awaited_once()
    assert "Сначала токен" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_select_upload_mode_rejects_non_admin(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="user")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="Загрузить документ")

    await h.select_upload_mode(msg, state)

    state.set_state.assert_not_awaited()
    msg.answer.assert_awaited_once()
    assert "не админ" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_select_upload_mode_sets_state_for_admin(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="admin")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="Загрузить документ")

    await h.select_upload_mode(msg, state)

    state.set_state.assert_awaited_once_with(h.BotStates.upload_mode)
    msg.answer.assert_awaited_once()
    assert "Поддерживаемые форматы" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_ask_skips_menu_commands(monkeypatch):
    # Should return early before any DB or HTTP calls
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock())

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(update_data=AsyncMock())
    msg = _fake_message(text="Задать вопрос")

    await h.handle_ask(msg, state)

    repo.get_by_telegram_id.assert_not_awaited()
    msg.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_ask_requires_auth(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=None))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(update_data=AsyncMock())
    msg = _fake_message(text="What is this?")

    await h.handle_ask(msg, state)

    msg.answer.assert_awaited_once()
    assert "Сначала токен" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_ask_success_with_sources(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="user")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    responses = [
        _DummyResponse(
            status=200,
            json_data={
                "answer": "Answer",
                "sources": [{"filename": "a.pdf", "page": 1}],
            },
        )
    ]

    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _DummyClientSession(responses))

    state = SimpleNamespace(update_data=AsyncMock())
    msg = _fake_message(text="Question")

    await h.handle_ask(msg, state)

    msg.bot.send_chat_action.assert_awaited_once_with(msg.chat.id, "typing")
    state.update_data.assert_awaited_once()

    # Should send with inline keyboard for sources
    kwargs = msg.answer.await_args.kwargs
    assert kwargs.get("parse_mode") == "Markdown"
    assert isinstance(kwargs.get("reply_markup"), InlineKeyboardMarkup)


@pytest.mark.asyncio
async def test_handle_upload_rejects_unsupported(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="admin")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    doc = SimpleNamespace(file_name="bad.exe", mime_type="application/octet-stream", file_size=1024)
    msg = _fake_message(text=None, document=doc)

    await h.handle_upload(msg)

    msg.answer.assert_awaited_once()
    assert "Неподдерживаемый формат" in msg.answer.await_args.args[0]
    msg.bot.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_upload_rejects_too_large(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="admin")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    big_size = 51 * 1024 * 1024
    doc = SimpleNamespace(file_name="doc.pdf", mime_type="application/pdf", file_size=big_size)
    msg = _fake_message(text=None, document=doc)

    await h.handle_upload(msg)

    msg.answer.assert_awaited_once()
    assert "Файл слишком большой" in msg.answer.await_args.args[0]
    msg.bot.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_help_sends_formats_and_keyboard(monkeypatch):
    msg = _fake_message(text="/help")
    await h.cmd_help(msg)

    msg.answer.assert_awaited_once()
    text = msg.answer.await_args.args[0]
    assert "Поддерживаемые форматы" in text
    assert msg.answer.await_args.kwargs.get("reply_markup") is h.mode_keyboard


@pytest.mark.asyncio
async def test_cmd_formats_sends_formats_and_keyboard(monkeypatch):
    msg = _fake_message(text="/formats")
    await h.cmd_formats(msg)

    msg.answer.assert_awaited_once()
    text = msg.answer.await_args.args[0]
    assert "Поддерживаемые форматы" in text
    assert msg.answer.await_args.kwargs.get("reply_markup") is h.mode_keyboard


@pytest.mark.asyncio
async def test_cmd_reauth_deletes_user_and_prompts_token(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(delete_by_telegram_id=AsyncMock())

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="/reauth", user_id=42)

    await h.cmd_reauth(msg, state)

    repo.delete_by_telegram_id.assert_awaited_once_with(42)
    state.set_state.assert_awaited_once_with(h.BotStates.auth_token)
    msg.answer.assert_awaited_once()
    assert "Права сброшены" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_select_ask_mode_sets_state_for_authed_user(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="user")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    state = SimpleNamespace(set_state=AsyncMock())
    msg = _fake_message(text="Задать вопрос")

    await h.select_ask_mode(msg, state)

    state.set_state.assert_awaited_once_with(h.BotStates.ask_mode)
    msg.answer.assert_awaited_once()
    assert "Слушаю" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_ask_non_200_returns_error_message(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="user")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    responses = [_DummyResponse(status=500, text_data="boom")]
    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _DummyClientSession(responses))

    state = SimpleNamespace(update_data=AsyncMock())
    msg = _fake_message(text="Question")

    await h.handle_ask(msg, state)

    msg.answer.assert_awaited_once()
    assert "Ошибка при обработке запроса" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_ask_exception_returns_connection_error(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="user")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    class _ExplodingClientSession:
        async def __aenter__(self):
            raise RuntimeError("network down")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _ExplodingClientSession())

    state = SimpleNamespace(update_data=AsyncMock())
    msg = _fake_message(text="Question")

    await h.handle_ask(msg, state)

    msg.answer.assert_awaited_once()
    assert "ошибка" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_handle_generate_document_guidance(monkeypatch):
    msg = _fake_message(text=None, document=SimpleNamespace())
    await h.handle_generate_document(msg)
    msg.answer.assert_awaited_once()
    assert "В режиме генерации" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_source_callback_bad_index(monkeypatch):
    msg = _fake_message(text=None)
    cb = SimpleNamespace(
        data="src:not_an_int",
        answer=AsyncMock(),
        message=msg,
    )
    state = SimpleNamespace(get_data=AsyncMock(return_value={}))

    await h.handle_source_callback(cb, state)

    cb.answer.assert_awaited_once()
    msg.answer.assert_awaited_once()
    assert "неверный формат" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_source_callback_sources_stale(monkeypatch):
    msg = _fake_message(text=None)
    cb = SimpleNamespace(data="src:0", answer=AsyncMock(), message=msg)
    state = SimpleNamespace(get_data=AsyncMock(return_value={"last_sources": []}))

    await h.handle_source_callback(cb, state)

    msg.answer.assert_awaited_once()
    assert "Источники устарели" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_source_callback_missing_filename(monkeypatch):
    msg = _fake_message(text=None)
    cb = SimpleNamespace(data="src:0", answer=AsyncMock(), message=msg)
    state = SimpleNamespace(get_data=AsyncMock(return_value={"last_sources": [{"page": 1}]}))

    await h.handle_source_callback(cb, state)

    msg.answer.assert_awaited_once()
    assert "имя файла" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_source_callback_success_with_no_chunks(monkeypatch):
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    msg = _fake_message(text=None)
    cb = SimpleNamespace(data="src:0", answer=AsyncMock(), message=msg)
    state = SimpleNamespace(get_data=AsyncMock(return_value={"last_sources": [{"filename": "a.pdf", "page": 2}]}))

    responses = [_DummyResponse(status=200, json_data={"chunks": []})]
    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _DummyClientSession(responses))

    await h.handle_source_callback(cb, state)

    msg.answer.assert_awaited_once()
    assert "не найден" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_source_callback_success_truncates_long_text(monkeypatch):
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    msg = _fake_message(text=None)
    cb = SimpleNamespace(data="src:0", answer=AsyncMock(), message=msg)
    state = SimpleNamespace(get_data=AsyncMock(return_value={"last_sources": [{"filename": "a.pdf", "page": 1}]}))

    long_text = "x" * 4000
    responses = [_DummyResponse(status=200, json_data={"chunks": [{"text": long_text}]})]
    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _DummyClientSession(responses))

    await h.handle_source_callback(cb, state)

    msg.answer.assert_awaited_once()
    sent = msg.answer.await_args.args[0]
    assert "текст сокращён" in sent


@pytest.mark.asyncio
async def test_handle_read_source_unrecognized(monkeypatch):
    state = SimpleNamespace()
    msg = _fake_message(text="nonsense")
    await h.handle_read_source(msg, state)

    msg.answer.assert_awaited_once()
    assert "Не смог распознать" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_read_source_success(monkeypatch):
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    msg = _fake_message(text="doc.pdf (стр. 2)")
    state = SimpleNamespace()

    responses = [_DummyResponse(status=200, json_data={"chunks": [{"text": "Hello"}]})]
    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _DummyClientSession(responses))

    await h.handle_read_source(msg, state)

    msg.bot.send_chat_action.assert_awaited_once_with(msg.chat.id, "typing")
    msg.answer.assert_awaited_once()
    sent = msg.answer.await_args.args[0]
    assert "doc.pdf" in sent
    assert "Hello" in sent


@pytest.mark.asyncio
async def test_handle_upload_download_failure(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="admin")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)

    doc = SimpleNamespace(file_name="doc.pdf", mime_type="application/pdf", file_size=1024)
    msg = _fake_message(text=None, document=doc)
    msg.bot.download = AsyncMock(side_effect=RuntimeError("boom"))

    await h.handle_upload(msg)

    msg.answer.assert_awaited_once()
    assert "ошибка" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_handle_upload_success_document(monkeypatch):
    session_obj = SimpleNamespace()
    repo = SimpleNamespace(get_by_telegram_id=AsyncMock(return_value=SimpleNamespace(role="admin")))

    monkeypatch.setattr(h, "get_session", _fake_get_session(session_obj))
    monkeypatch.setattr(h, "UserRepository", lambda session: repo)
    monkeypatch.setattr(h, "settings", SimpleNamespace(api_base_url="http://api"))

    monkeypatch.setattr(h.aiohttp, "FormData", _DummyFormData)
    monkeypatch.setattr(h.aiohttp, "ClientSession", lambda *a, **k: _DummyClientSession([
        _DummyResponse(status=200, json_data={"chunks_added": 3, "files_processed": 1, "errors": []})
    ]))

    doc = SimpleNamespace(file_name="doc.pdf", mime_type="application/pdf", file_size=1024)
    msg = _fake_message(text=None, document=doc, user_id=9)
    msg.bot.download = AsyncMock(return_value=b"data")

    await h.handle_upload(msg)

    # One "started" message + one final success message
    assert msg.answer.await_count >= 2
    final_text = msg.answer.await_args.args[0]
    assert "успешно" in final_text.lower()
    assert "чанков" in final_text.lower()
