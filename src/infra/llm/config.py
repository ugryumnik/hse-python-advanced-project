from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class LLMConfig(BaseModel):
    """Конфигурация LLM модели"""

    # Путь к модели
    local_model_path: Path | None = None
    model_id: str | None = None
    model_file: str | None = None

    # Параметры llama-cpp
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 512

    # Параметры генерации
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40

    # Против повторений
    repeat_penalty: float = 1.1  # Штраф за повторение токенов

    # Stop токены для Qwen2.5-Instruct
    stop: list[str] = Field(default=[
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
    ])

    cache_dir: Path = Path("./models/llm")


class EmbeddingsConfig(BaseModel):
    """Конфигурация модели эмбеддингов"""
    model_id: str = "intfloat/multilingual-e5-large"
    device: DeviceType = DeviceType.MPS
    normalize_embeddings: bool = True
    cache_dir: Path = Path("./models/embeddings")


class VectorStoreConfig(BaseModel):
    """Конфигурация векторного хранилища"""
    persist_directory: Path = Path("./data/vectorstore")
    collection_name: str = "legal_documents"

    search_k: int = 5
    search_type: str = "mmr"
    mmr_lambda: float = 0.7


class ChunkingConfig(BaseModel):
    """Конфигурация разбиения текста"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(
        default=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )


class RAGConfig(BaseModel):
    """Общая конфигурация RAG пайплайна"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

    documents_dir: Path = Path("./data/documents")