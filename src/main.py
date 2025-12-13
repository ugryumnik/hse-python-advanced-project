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

    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)

    tasks = [
        asyncio.create_task(server.serve()),
        asyncio.create_task(dp.start_polling(bot))
    ]

    done, pending = await asyncio.wait([*tasks], return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
