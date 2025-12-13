from aiogram import Router, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message

from src.bot.keyboards import mode_keyboard

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
    await message.answer(f"Ваш вопрос: {question}. (Ответ будет реализован позже)", reply_markup=mode_keyboard)


@router.message(BotStates.ask_mode, F.document)
async def handle_ask_document(message: Message):
    await message.answer("В режиме вопроса отправьте текст или переключитесь в режим загрузки.", reply_markup=mode_keyboard)


@router.message(BotStates.upload_mode, F.document)
async def handle_upload(message: Message):
    if message.document.mime_type != "application/pdf":
        await message.answer("Отправьте файл в формате PDF.", reply_markup=mode_keyboard)
        return

    import os
    junk_dir = "junk"
    os.makedirs(junk_dir, exist_ok=True)
    file_path = os.path.join(junk_dir, message.document.file_name)
    await message.bot.download(message.document, file_path)
    await message.answer(f"Файл {message.document.file_name} загружен в {junk_dir}", reply_markup=mode_keyboard)


@router.message(BotStates.upload_mode, F.text)
async def handle_upload_text(message: Message):
    await message.answer("В режиме загрузки отправьте PDF файл или переключитесь в режим вопроса.", reply_markup=mode_keyboard)
