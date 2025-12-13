"""Асинхронный тестовый скрипт для проверки RAG сервиса"""

from core.services import RAGService
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main():
    async with RAGService() as rag_service:
        await rag_service.health_check()
        # await rag_service.index_all(force=True)
        answer, sources = await rag_service.query("как уволить сотрудника", 5)
        print(f"Ответ: {answer}")
        print(f"Источники: {sources}")


if __name__ == "__main__":
    asyncio.run(main())
