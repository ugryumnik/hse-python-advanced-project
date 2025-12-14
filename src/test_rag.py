"""Тестовый скрипт для проверки RAG системы"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from infra.llm import LegalRAGAgent, RAGConfig, YandexGPTConfig, QdrantConfig


async def main():
    agent = LegalRAGAgent(
        config=RAGConfig(documents_dir=Path("./legal_docs")),
        yandex_config=YandexGPTConfig(),
        qdrant_config=QdrantConfig(),
    )

    print(f"Модель: {agent.yandex_config.model_uri}")

    # Проверка здоровья
    if not await agent.health_check():
        print("Ошибка подключения")
        return

    print("✓ Все сервисы работают")

    # Индексация если пусто
    stats = await agent.get_stats()
    if stats["total_chunks"] == 0:
        print("Индексация...")
        num = await agent.index_documents()
        print(f"✓ Проиндексировано {num} чанков")
    else:
        print(f"База: {stats['total_chunks']} чанков")

    # Запрос
    question = "Как уволить сотрудника?"
    print(f"\nВопрос: {question}")

    response = await agent.query(question)
    print(f"\nОтвет: {response.answer}")
    print(f"Токенов: {response.tokens_used}")

    if response.sources:
        print("\nИсточники:")
        for s in response.sources:
            print(f"  - {s['filename']}, стр. {s['page']}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())