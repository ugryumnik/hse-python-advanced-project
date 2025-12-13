from aiogram import Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message
import aiohttp

from bot.keyboards import mode_keyboard
from config import settings

router = Router()


class BotStates(StatesGroup):
    ask_mode = State()
    upload_mode = State()


@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет! Я Legal RAG Bot. Выберите режим:", reply_markup=mode_keyboard)


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer("Отправьте вопрос, и я найду ответ в документах с ссылками на источники.", reply_markup=mode_keyboard)


@router.message(F.text == "Задать вопрос")
async def select_ask_mode(message: Message, state: FSMContext):
    await state.set_state(BotStates.ask_mode)
    await message.answer("Режим вопроса активирован. Отправьте ваш вопрос текстом.", reply_markup=mode_keyboard)


@router.message(F.text == "Загрузить документ")
async def select_upload_mode(message: Message, state: FSMContext):
    await state.set_state(BotStates.upload_mode)
    await message.answer("Режим загрузки активирован. Отправьте PDF файлы.", reply_markup=mode_keyboard)


@router.message(BotStates.ask_mode, F.text)
async def handle_ask(message: Message):
    question = message.text
    user_id = message.from_user.id

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
    if message.document.mime_type != "application/pdf":
        await message.answer("Отправьте файл в формате PDF.", reply_markup=mode_keyboard)
        return

    try:
        file_content = await message.bot.download(message.document)
    except Exception as e:
        await message.answer("Произошла ошибка при загрузке файла.", reply_markup=mode_keyboard)

    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', file_content,
                           filename=message.document.file_name)
            data.add_field('user_id', str(message.from_user.id))

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
