from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

from infra.llm import LegalRAGAgent, RAGConfig, LLMConfig

# Конфигурация
config = RAGConfig(
    documents_dir=Path("./legal_docs"),
    llm=LLMConfig(
        local_model_path=Path("./models/llm/qwen2.5-7b-instruct-q6_k.gguf"),
        n_ctx=4096,
        max_tokens=512,
        temperature=0.1,
        repeat_penalty=1.25,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "Human"
        ],
    ),
)

# Инициализация
agent = LegalRAGAgent(config)

# Индексация (только если база пустая)
stats = agent.get_stats()
if stats["total_chunks"] == 0:
    num_chunks = agent.index_documents()
    print(f"Проиндексировано {num_chunks} чанков")
else:
    print(f"База уже содержит {stats['total_chunks']} чанков")

# Запрос
response = agent.query(
    "Как уволить сотрудника если он не выполняет рабту?"
)

print("=" * 50)
print("ОТВЕТ:")
print(response.answer)
print("\nИСТОЧНИКИ:")
for src in response.sources:
    print(f"  - {src['filename']}, стр. {src['page']}")

# Закрываем (убирает warning)
agent.close()