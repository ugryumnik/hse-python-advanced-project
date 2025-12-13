import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher

from config import settings
from bot.handlers import router
from bot.middlewares import LoggingMiddleware
from web import app


async def main():
    logging.basicConfig(level=logging.INFO)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()

    dp.message.middleware(LoggingMiddleware())
    dp.include_router(router)

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(server.serve())
        tg.create_task(dp.start_polling(bot))


if __name__ == "__main__":
    asyncio.run(main())