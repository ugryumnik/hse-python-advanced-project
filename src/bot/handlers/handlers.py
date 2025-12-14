import asyncio
import re
from aiogram import Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
import aiohttp

from bot.keyboards import mode_keyboard
from config import settings
from infra.db.database import get_session
from infra.db.user_repository import UserRepository

router = Router()

# ============================================================================
# Поддерживаемые форматы файлов
# ============================================================================

DOCUMENT_MIME_TYPES = frozenset({
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
})

ARCHIVE_MIME_TYPES = frozenset({
    "application/zip",
    "application/x-zip-compressed",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-xz",
    "application/x-compressed-tar",
})

SUPPORTED_MIME_TYPES = DOCUMENT_MIME_TYPES | ARCHIVE_MIME_TYPES

DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".doc", ".txt", ".md"})
ARCHIVE_EXTENSIONS = frozenset({".zip", ".tar", ".tgz", ".tbz2", ".txz", ".tar.gz", ".tar.bz2", ".tar.xz"})
SUPPORTED_EXTENSIONS = DOCUMENT_EXTENSIONS | ARCHIVE_EXTENSIONS


def is_supported_file(filename: str | None, mime_type: str | None) -> tuple[bool, str]:
    """Проверить, поддерживается ли файл."""
    if filename:
        filename_lower = filename.lower()
        for ext in {".tar.gz", ".tar.bz2", ".tar.xz"}:
            if filename_lower.endswith(ext):
                return True, "archive"
        for ext in ARCHIVE_EXTENSIONS:
            if filename_lower.endswith(ext):
                return True, "archive"
        for ext in DOCUMENT_EXTENSIONS:
            if filename_lower.endswith(ext):
                return True, "document"

    if mime_type:
        if mime_type in ARCHIVE_MIME_TYPES:
            return True, "archive"
        if mime_type in DOCUMENT_MIME_TYPES:
            return True, "document"

    return False, "unsupported"


def get_supported_formats_text() -> str:
    """Получить текст с поддерживаемыми форматами"""
    docs = ", ".join(sorted(DOCUMENT_EXTENSIONS))
    archives = ", ".join(sorted(ARCHIVE_EXTENSIONS))
    return f" Документы: {docs}\n Архивы: {archives}"


# ============================================================================
# Создание inline-кнопок для источников
# ============================================================================

def create_sources_keyboard(sources: list[dict]) -> InlineKeyboardMarkup | None:
    """
    Создать inline-клавиатуру с кнопками-источниками.

    Каждая кнопка позволяет получить текст соответствующего фрагмента.
    """
    if not sources:
        return None

    buttons = []
    seen = set()  # Избегаем дублей

    for i, src in enumerate(sources):
        filename = src.get("filename", "?")
        page = src.get("page")
        archive = src.get("archive")

        # Ключ для дедупликации
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)

        # Текст кнопки (сокращаем если длинный)
        if len(filename) > 25:
            short_name = filename[:22] + "..."
        else:
            short_name = filename

        if page:
            button_text = f" {short_name} (стр. {page})"
        else:
            button_text = f" {short_name}"

        # callback_data: src:{index}
        # Индекс используем для получения полных данных из state
        callback_data = f"src:{i}"

        buttons.append([InlineKeyboardButton(
            text=button_text,
            callback_data=callback_data
        )])

    if not buttons:
        return None

    # Ограничиваем количество кнопок (Telegram limit)
    buttons = buttons[:8]

    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ============================================================================
# Состояния FSM
# ============================================================================

class BotStates(StatesGroup):
    ask_mode = State()
    upload_mode = State()
    auth_token = State()
    read_mode = State()


# ============================================================================
# Команды
# ============================================================================

@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    async for session in get_session():
        repo = UserRepository(session)
        existing = await repo.get_by_telegram_id(message.from_user.id)
    if not existing:
        await state.set_state(BotStates.auth_token)
        await message.answer("Твой секретный токен?")
        return
    await message.answer("Сол Гудман у аппарата. Зачем позвонил?", reply_markup=mode_keyboard)


@router.message(BotStates.auth_token, F.text)
async def handle_auth_token(message: Message, state: FSMContext):
    token = message.text.strip()
    role = None
    if token == settings.admin_token:
        role = "admin"
    elif token == settings.user_token:
        role = "user"

    if role is None:
        await message.answer("Ты в чс. Лучше не звони Солу.")
        return

    async for session in get_session():
        repo = UserRepository(session)
        await repo.upsert(message.from_user.id, role)
    await state.clear()
    await message.answer("Верный токен.\nСол Гудман у аппарата. Зачем позвонил?", reply_markup=mode_keyboard)


@router.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "Задай вопрос, и я найду ответ в документах с ссылками на источники.\n\n"
        "После ответа нажми на кнопку источника, чтобы увидеть полный текст фрагмента.\n\n"
        f"Поддерживаемые форматы для загрузки:\n{get_supported_formats_text()}"
    )
    await message.answer(help_text, reply_markup=mode_keyboard)


@router.message(Command("reauth"))
async def cmd_reauth(message: Message, state: FSMContext):
    async for session in get_session():
        repo = UserRepository(session)
        await repo.delete_by_telegram_id(message.from_user.id)
    await state.set_state(BotStates.auth_token)
    await message.answer("Права сброшены, назови токен доступа.")


@router.message(Command("formats"))
async def cmd_formats(message: Message):
    await message.answer(
        f" Поддерживаемые форматы:\n\n{get_supported_formats_text()}\n\n"
        "Архивы могут содержать вложенные архивы (до 3 уровней).",
        reply_markup=mode_keyboard
    )


# ============================================================================
# Выбор режима
# ============================================================================

@router.message(F.text == "Задать вопрос")
async def select_ask_mode(message: Message, state: FSMContext):
    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(message.from_user.id)
    if not user:
        await state.set_state(BotStates.auth_token)
        await message.answer("Сначала токен доступа.")
        return
    await state.set_state(BotStates.ask_mode)
    await message.answer("Слушаю вопросы.",
                         reply_markup=mode_keyboard)


@router.message(F.text == "Прочитать документ")
async def select_read_mode(message: Message, state: FSMContext):
    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(message.from_user.id)
    if not user:
        await state.set_state(BotStates.auth_token)
        await message.answer("Сначала токен доступа.")
        return
    await state.set_state(BotStates.read_mode)
    await message.answer(
        "Пришли строку источника, например:\n"
        "твой_файл.pdf, стр. 1",
        reply_markup=mode_keyboard,
    )


@router.message(F.text == "Загрузить документ")
async def select_upload_mode(message: Message, state: FSMContext):
    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(message.from_user.id)
    if not user:
        await state.set_state(BotStates.auth_token)
        await message.answer("Сначала токен доступа.")
        return
    if user.role != "admin":
        await message.answer("Ты не админ, загрузка запрещена.")
        return
    await state.set_state(BotStates.upload_mode)
    await message.answer(
        f"Отправляй файлы, я сохраню и буду использовать их.\n\n"
        f"Поддерживаемые форматы:\n{get_supported_formats_text()}",
        reply_markup=mode_keyboard
    )


# ============================================================================
# Обработка вопросов
# ============================================================================

@router.message(BotStates.ask_mode, F.text)
async def handle_ask(message: Message, state: FSMContext):
    question = message.text
    user_id = message.from_user.id

    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(user_id)
    if not user:
        await message.answer("Сначала токен доступа.")
        return

    # Отправляем индикатор "печатает..."
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{settings.api_base_url}/ask",
                    json={"query": question, "user_id": user_id}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    # Сохраняем источники в state для callback-обработчика
                    await state.update_data(last_sources=sources)

                    # Формируем текст ответа
                    response_text = answer

                    if sources:
                        response_text += "\n\n *Источники* (нажми для просмотра):"

                    # Создаём inline-кнопки
                    keyboard = create_sources_keyboard(sources)

                    await message.answer(
                        response_text,
                        reply_markup=keyboard,
                        parse_mode="Markdown"
                    )
                else:
                    await message.answer("Ошибка при обработке запроса.", reply_markup=mode_keyboard)
    except Exception as e:
        await message.answer(f"Произошла ошибка при подключении к серверу.", reply_markup=mode_keyboard)


# ============================================================================
# Callback-обработчик для источников
# ============================================================================

@router.callback_query(F.data.startswith("src:"))
async def handle_source_callback(callback: CallbackQuery, state: FSMContext):
    """
    Обработка нажатия на кнопку источника.
    Получает текст фрагмента и отправляет пользователю.
    """
    await callback.answer()  # Убираем "часики" на кнопке

    # Извлекаем индекс источника
    try:
        index = int(callback.data.split(":")[1])
    except (ValueError, IndexError):
        await callback.message.answer(" Ошибка: неверный формат источника.")
        return

    # Получаем сохранённые источники из state
    data = await state.get_data()
    sources = data.get("last_sources", [])

    if not sources or index >= len(sources):
        await callback.message.answer(
            " Источники устарели. Задай вопрос заново.",
            reply_markup=mode_keyboard
        )
        return

    source = sources[index]
    filename = source.get("filename")
    page = source.get("page")
    archive = source.get("archive")

    if not filename:
        await callback.message.answer(" Ошибка: имя файла не найдено.")
        return

    # Показываем индикатор загрузки
    await callback.message.bot.send_chat_action(callback.message.chat.id, "typing")

    # Запрашиваем текст источника
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{settings.api_base_url}/source",
                    json={"filename": filename, "page": page or 1, "limit": 3}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    chunks = data.get("chunks", [])

                    if not chunks:
                        await callback.message.answer(
                            f"Текст для `{filename}` (стр. {page}) не найден в базе.",
                            parse_mode="Markdown"
                        )
                        return

                    # Собираем текст из чанков
                    content = "\n\n---\n\n".join(
                        chunk.get("text", "") for chunk in chunks
                    )

                    # Формируем заголовок
                    if archive:
                        header = f" *{filename}*\nАрхив: `{archive}`"
                    else:
                        header = f" *{filename}*"

                    if page:
                        header += f"\n Страница: {page}"

                    # Ограничиваем длину (Telegram limit 4096 символов)
                    max_content_len = 3500
                    if len(content) > max_content_len:
                        content = content[:max_content_len] + "\n\n... _(текст сокращён)_"

                    full_message = f"{header}\n\n{content}"

                    await callback.message.answer(
                        full_message,
                        parse_mode="Markdown"
                    )
                else:
                    await callback.message.answer(
                        "Ошибка при получении источника.",
                        reply_markup=mode_keyboard
                    )
    except Exception as e:
        await callback.message.answer(
            f"Ошибка подключения: {str(e)[:100]}",
            reply_markup=mode_keyboard
        )


# ============================================================================
# Чтение источников (ручной режим)
# ============================================================================

@router.message(BotStates.read_mode, F.text)
async def handle_read_source(message: Message, state: FSMContext):
    text = message.text.strip()

    filename = None
    page = None

    patterns = [
        r"(?i)(?:источник\s*\d+\s*:\s*)?([^,\s]+\.(pdf|docx|doc|txt|md))[^\d]*(?:стр\.?\s*)(\d+)",
        r"(?i)([^,\s]+\.(pdf|docx|doc|txt|md))[^\d]*(\d+)",
        r"(?i)([^,\s]+\.(pdf|docx|doc|txt|md))",  # Без страницы
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            filename = m.group(1)
            try:
                page = int(m.group(3)) if m.lastindex >= 3 else 1
            except (ValueError, TypeError):
                page = 1
            break

    if not filename:
        await message.answer(
            "Не смог распознать источник. Пришли в формате:\n"
            "`документ.pdf, стр. 1`\n\n"
            "Или просто имя файла: `документ.pdf`",
            reply_markup=mode_keyboard,
            parse_mode="Markdown"
        )
        return

    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                    f"{settings.api_base_url}/source",
                    json={"filename": filename, "page": page, "limit": 5},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    chunks = data.get("chunks", [])

                    if not chunks:
                        await message.answer(
                            f"Текст для `{filename}` (стр. {page}) не найден.",
                            reply_markup=mode_keyboard,
                            parse_mode="Markdown"
                        )
                        return

                    content = "\n\n---\n\n".join(
                        chunk.get("text", "") for chunk in chunks
                    )

                    # Ограничение длины
                    if len(content) > 3500:
                        content = content[:3500] + "\n\n... _(текст сокращён)_"

                    header = f"📄 *{filename}*, стр. {page}\n\n"

                    await message.answer(
                        header + content,
                        reply_markup=mode_keyboard,
                        parse_mode="Markdown"
                    )
                else:
                    await message.answer(
                        "Ошибка при получении источника.",
                        reply_markup=mode_keyboard
                    )
    except Exception:
        await message.answer(
            "Произошла ошибка при подключении к серверу.",
            reply_markup=mode_keyboard
        )


# ============================================================================
# Загрузка файлов
# ============================================================================

@router.message(BotStates.ask_mode, F.document)
async def handle_ask_document(message: Message):
    await message.answer(
        "В режиме вопроса отправьте текст или переключитесь в режим загрузки.",
        reply_markup=mode_keyboard
    )


@router.message(BotStates.upload_mode, F.document)
async def handle_upload(message: Message):
    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(message.from_user.id)
    if not user or user.role != "admin":
        await message.answer("Загрузка запрещена: требуется токен администратора.")
        return

    filename = message.document.file_name
    mime_type = message.document.mime_type

    is_supported, file_type = is_supported_file(filename, mime_type)

    if not is_supported:
        await message.answer(
            f"Неподдерживаемый формат файла.\n\n"
            f"Поддерживаемые форматы:\n{get_supported_formats_text()}",
            reply_markup=mode_keyboard
        )
        return

    file_size_mb = message.document.file_size / (1024 * 1024)
    if file_size_mb > 50:
        await message.answer(
            f"Файл слишком большой ({file_size_mb:.1f} MB). Максимум 50 MB.",
            reply_markup=mode_keyboard
        )
        return

    try:
        file_content = await message.bot.download(message.document)
    except Exception as e:
        await message.answer("Произошла ошибка при загрузке файла.", reply_markup=mode_keyboard)
        return

    try:
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename=filename)
            data.add_field('user_id', str(message.from_user.id))

            if file_type == "archive":
                await message.answer(
                    f"Начал обработку архива `{filename}`.\n"
                    "Это может занять значительное время...",
                    reply_markup=mode_keyboard,
                    parse_mode="Markdown"
                )
            else:
                await message.answer(
                    f"Начал обработку файла `{filename}`...",
                    reply_markup=mode_keyboard,
                    parse_mode="Markdown"
                )

            async with session.post(
                    f"{settings.api_base_url}/upload",
                    data=data
            ) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    chunks_added = response_data.get("chunks_added", 0)
                    files_processed = response_data.get("files_processed", 0)
                    processed_files = response_data.get("processed_files", [])
                    errors = response_data.get("errors", [])

                    if file_type == "archive":
                        success_msg = (
                            f"Архив `{filename}` успешно обработан!\n\n"
                            f"Статистика:\n"
                            f"- Файлов обработано: {files_processed}\n"
                            f"- Всего чанков: {chunks_added}\n"
                        )

                        if processed_files:
                            success_msg += f"\nОбработанные файлы:\n"
                            files_to_show = processed_files[:15]
                            for f in files_to_show:
                                fname = f.get("filename", "?")
                                fchunks = f.get("chunks", 0)
                                success_msg += f"• `{fname}` ({fchunks} чанков)\n"

                            if len(processed_files) > 15:
                                success_msg += f"• ... и ещё {len(processed_files) - 15} файлов\n"

                        if errors:
                            success_msg += f"\nОшибки ({len(errors)}):\n"
                            for err in errors[:3]:
                                success_msg += f"• {err[:50]}...\n" if len(err) > 50 else f"• {err}\n"
                            if len(errors) > 3:
                                success_msg += f"• ... и ещё {len(errors) - 3}\n"
                    else:
                        success_msg = (
                            f"Документ `{filename}` успешно обработан!\n"
                            f"Добавлено чанков: {chunks_added}"
                        )

                    await message.answer(success_msg, reply_markup=mode_keyboard, parse_mode="Markdown")

                elif resp.status == 403:
                    await message.answer("Доступ запрещён.", reply_markup=mode_keyboard)
                else:
                    error_text = await resp.text()
                    await message.answer(
                        f"Ошибка при обработке файла: {error_text[:200]}",
                        reply_markup=mode_keyboard
                    )
    except asyncio.TimeoutError:
        await message.answer("Превышено время ожидания.", reply_markup=mode_keyboard)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)[:100]}", reply_markup=mode_keyboard)


@router.message(BotStates.upload_mode, F.text)
async def handle_upload_text(message: Message):
    await message.answer(
        "В режиме загрузки отправьте файл или переключитесь в режим вопроса.",
        reply_markup=mode_keyboard
    )