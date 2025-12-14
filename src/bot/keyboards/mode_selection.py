from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

mode_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Задать вопрос"),
            KeyboardButton(text="Загрузить документ"),
        ],
        [
            # KeyboardButton(text="Прочитать документ"),
            KeyboardButton(text="Создать документ"),
        ],
    ],
    resize_keyboard=True,
    one_time_keyboard=False
)