"""Конфигурация компонентов RAG системы"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class YandexGPTModel(str, Enum):
    """Доступные модели Yandex GPT"""
    LITE = "yandexgpt-lite"
    PRO = "yandexgpt"
    PRO_32K = "yandexgpt-32k"


class YandexGPTConfig(BaseSettings):
    """Конфигурация Yandex GPT API"""
    
    model_config = SettingsConfigDict(
        env_prefix="YANDEX_GPT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = None
    folder_id: str

    model: YandexGPTModel = YandexGPTModel.LITE
    model_version: str = "latest"
    
    temperature: float = 0.1
    max_tokens: int = 1024
    
    api_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    embeddings_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
    timeout: int = 60
    max_retries: int = 3

    @property
    def model_uri(self) -> str:
        return f"gpt://{self.folder_id}/{self.model.value}/{self.model_version}"

    def get_auth_header(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("YANDEX_GPT_API_KEY не задан")
        return {"Authorization": f"Api-Key {self.api_key.get_secret_value()}"}


class QdrantConfig(BaseSettings):
    """Конфигурация Qdrant"""
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    host: str = "localhost"
    port: int = 6333
    
    # Для Qdrant Cloud
    url: str | None = None
    api_key: SecretStr | None = None
    
    collection_name: str = "legal_documents"
    embedding_dim: int = 256
    
    # Поиск
    search_k: int = 5
    use_mmr: bool = True
    mmr_lambda: float = 0.7


class ChunkingConfig(BaseModel):
    """Конфигурация разбиения текста"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(
        default=["\n\n", "\n", ".", "!", "?", ";", " "]
    )


class RAGConfig(BaseModel):
    """Общая конфигурация RAG"""
    documents_dir: Path = Path("./data/documents")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)