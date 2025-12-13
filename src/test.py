"""Тестовый скрипт для проверки RAG системы"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from infra.llm import LegalRAGAgent, RAGConfig, YandexGPTConfig, QdrantConfig


def main():
    # Конфигурация (загружается из .env)
    agent = LegalRAGAgent(
        config=RAGConfig(documents_dir=Path("./legal_docs")),
        yandex_config=YandexGPTConfig(),
        qdrant_config=QdrantConfig(),
    )

    print(f"Модель: {agent.yandex_config.model_uri}")
    print(f"Qdrant: {agent.qdrant_config.host}:{agent.qdrant_config.port}")

    # Проверка здоровья
    print("\nПроверка подключения...")
    if not agent.health_check():
        print(" Ошибка подключения")
        return

    print(" Все сервисы работают!")

    # Индексация
    stats = agent.get_stats()
    if stats["total_chunks"] == 0:
        print("\nИндексация документов...")
        num_chunks = agent.index_documents()
        print(f" Проиндексировано {num_chunks} чанков")
    else:
        print(f"\nБаза содержит {stats['total_chunks']} чанков")

    # Тестовый запрос
    print("\n" + "=" * 50)
    question = "Как уволить сотрудника если он не выполняет работу?"
    print(f"Вопрос: {question}")
    print("=" * 50)

    response = agent.query(question)

    print("\nОТВЕТ:")
    print(response.answer)
    print(f"\n Токенов: {response.tokens_used}")
    
    if response.sources:
        print("\n ИСТОЧНИКИ:")
        for src in response.sources:
            score = f" ({src['score']:.3f})" if src.get('score') else ""
            print(f"  - {src['filename']}, стр. {src['page']}{score}")

    agent.close()
    print("\n Готово!")


if __name__ == "__main__":
    main()