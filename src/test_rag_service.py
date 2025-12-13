"""Асинхронный тестовый скрипт для проверки RAG сервиса"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from core.services import RAGService


async def main():
    async with RAGService() as rag_service:
        await rag_service.health_check()
        answer, sources = await rag_service.query("как уволить сотрудника", 5)
        print(f"Ответ: {answer}")
        print(f"Источники: {sources}")


if __name__ == "__main__":
    asyncio.run(main())