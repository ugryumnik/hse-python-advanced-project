from pathlib import Path
import logging
import torch
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EmbeddingsConfig, DeviceType

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Менеджер моделей эмбеддингов"""

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._embeddings = None

    def _detect_device(self) -> str:
        """Определить доступное устройство"""
        if self.config.device == DeviceType.MPS:
            if torch.backends.mps.is_available():
                logger.info("Apple Metal (MPS)")
                return "mps"
            logger.warning("CPU")
            return "cpu"
        elif self.config.device == DeviceType.CUDA:
            if torch.cuda.is_available():
                logger.info("CUDA")
                return "cuda"
            logger.warning("CPU")
            return "cpu"
        return "cpu"

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Получить или создать модель эмбеддингов (singleton)"""
        if self._embeddings is not None:
            return self._embeddings

        device = self._detect_device()
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Загружаем модель эмбеддингов: {self.config.model_id}")

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_id,
            model_kwargs={
                "device": device,
                "trust_remote_code": True,
            },
            encode_kwargs={
                "normalize_embeddings": self.config.normalize_embeddings,
                "batch_size": 32,
            },
            cache_folder=str(cache_dir),
        )

        logger.info("Модель эмбеддингов загружена успешно")
        return self._embeddings

    def embed_query(self, text: str) -> list[float]:
        """Получить эмбеддинг для запроса"""
        # Для E5 моделей нужен префикс "query: "
        if "e5" in self.config.model_id.lower():
            text = f"query: {text}"
        return self.get_embeddings().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для документов"""
        # Для E5 моделей нужен префикс "passage: "
        if "e5" in self.config.model_id.lower():
            texts = [f"passage: {t}" for t in texts]
        return self.get_embeddings().embed_documents(texts)