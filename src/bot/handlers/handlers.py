from aiogram import Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message
import aiohttp

from bot.keyboards import mode_keyboard
from config import settings
from infra.db.database import get_session
from infra.db.user_repository import UserRepository

router = Router()


class BotStates(StatesGroup):
    ask_mode = State()
    upload_mode = State()
    auth_token = State()


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
    await message.answer("Задай вопрос, и я найду ответ в документах с ссылками на источники.", reply_markup=mode_keyboard)


@router.message(Command("reauth"))
async def cmd_reauth(message: Message, state: FSMContext):
    async for session in get_session():
        repo = UserRepository(session)
        await repo.delete_by_telegram_id(message.from_user.id)
    await state.set_state(BotStates.auth_token)
    await message.answer("Права сброшены, назови токен доступа.")


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
    await message.answer("Слушаю вопросы.", reply_markup=mode_keyboard)


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
    await message.answer("Отправляй PDF файлы, я сохраню и буду использовать их.", reply_markup=mode_keyboard)


@router.message(BotStates.ask_mode, F.text)
async def handle_ask(message: Message):
    question = message.text
    user_id = message.from_user.id
    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(user_id)
    if not user:
        await message.answer("Сначала токен доступа.")
        return

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

                    response_text = answer
                    if sources:
                        response_text += "\n\nРелевантные источники:\n"
                        for i, src in enumerate(sources, 1):
                            response_text += f"{i}. {src['filename']} (стр. {src['page']})\n"

                    await message.answer(response_text, reply_markup=mode_keyboard)
                else:
                    await message.answer("Ошибка при обработке запроса.", reply_markup=mode_keyboard)
    except Exception as e:
        await message.answer("Произошла ошибка при подключении к серверу.", reply_markup=mode_keyboard)


@router.message(BotStates.ask_mode, F.document)
async def handle_ask_document(message: Message):
    await message.answer("В режиме вопроса отправьте текст или переключитесь в режим загрузки.", reply_markup=mode_keyboard)


@router.message(BotStates.upload_mode, F.document)
async def handle_upload(message: Message):
    async for session in get_session():
        repo = UserRepository(session)
        user = await repo.get_by_telegram_id(message.from_user.id)
    if not user or user.role != "admin":
        await message.answer("Загрузка запрещена: требуется токен администратора.")
        return
    if message.document.mime_type != "application/pdf":
        await message.answer("Отправь файл в формате PDF.", reply_markup=mode_keyboard)
        return

    try:
        file_content = await message.bot.download(message.document)
    except Exception as e:
        await message.answer("Произошла ошибка при загрузке файла.", reply_markup=mode_keyboard)
        return

    try:
        timeout = aiohttp.ClientTimeout(total=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            data = aiohttp.FormData()
            data.add_field('file', file_content,
                           filename=message.document.file_name)
            data.add_field('user_id', str(message.from_user.id))

            await message.answer("Начал обработку файла. Это может занять время...")

            async with session.post(
                f"{settings.api_base_url}/upload",
                data=data
            ) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    await message.answer(response_data["message"], reply_markup=mode_keyboard)
                else:
                    await message.answer("Ошибка при обработке файла.", reply_markup=mode_keyboard)
    except Exception as e:
        await message.answer("Произошла ошибка при подключении к серверу.", reply_markup=mode_keyboard)


@router.message(BotStates.upload_mode, F.text)
async def handle_upload_text(message: Message):
    await message.answer("В режиме загрузки отправьте PDF файл или переключитесь в режим вопроса.", reply_markup=mode_keyboard)
