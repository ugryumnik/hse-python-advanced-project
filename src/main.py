import asyncio
import logging

from aiogram import Bot, Dispatcher

from config import settings
from src.bot.handlers import router
from src.bot.middlewares import LoggingMiddleware


async def main():
    logging.basicConfig(level=logging.INFO)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()

    dp.message.middleware(LoggingMiddleware())

    dp.include_router(router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
