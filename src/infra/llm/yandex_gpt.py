"""Асинхронный клиент для Yandex GPT API"""

import json
import logging
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from .config import YandexGPTConfig

logger = logging.getLogger(__name__)


@dataclass
class YandexGPTMessage:
    """Сообщение для Yandex GPT"""
    role: str  # "system", "user", "assistant"
    text: str


@dataclass  
class YandexGPTResponse:
    """Ответ от Yandex GPT API"""
    text: str
    tokens_input: int
    tokens_output: int
    
    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class YandexGPTError(Exception):
    """Ошибка Yandex GPT API"""
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class YandexGPTClient:
    """Асинхронный клиент для Yandex GPT API"""

    def __init__(self, config: YandexGPTConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        logger.info(f"YandexGPT: {config.model_uri}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Ленивая инициализация клиента"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers={
                    "Content-Type": "application/json",
                    **self.config.get_auth_header(),
                },
            )
        return self._client

    async def complete(
        self,
        messages: list[YandexGPTMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> YandexGPTResponse:
        """Асинхронная генерация ответа"""
        body = {
            "modelUri": self.config.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature or self.config.temperature,
                "maxTokens": str(max_tokens or self.config.max_tokens),
            },
            "messages": [{"role": m.role, "text": m.text} for m in messages],
        }

        client = await self._get_client()

        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(self.config.api_url, json=body)
                response.raise_for_status()
                
                result = response.json()["result"]
                return YandexGPTResponse(
                    text=result["alternatives"][0]["message"]["text"],
                    tokens_input=int(result["usage"]["inputTextTokens"]),
                    tokens_output=int(result["usage"]["completionTokens"]),
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit, ждём {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                raise YandexGPTError(f"HTTP {e.response.status_code}", e.response.status_code)
            except httpx.RequestError as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise YandexGPTError(f"Ошибка соединения: {e}")

        raise YandexGPTError("Превышено количество попыток")

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()