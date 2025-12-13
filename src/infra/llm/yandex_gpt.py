"""Клиент для Yandex GPT API"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Iterator

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
    """Синхронный клиент для Yandex GPT API"""

    def __init__(self, config: YandexGPTConfig):
        self.config = config
        self._client = httpx.Client(
            timeout=config.timeout,
            headers={
                "Content-Type": "application/json",
                **config.get_auth_header(),
            },
        )
        logger.info(f"YandexGPT: {config.model_uri}")

    def complete(
        self,
        messages: list[YandexGPTMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> YandexGPTResponse:
        """Генерация ответа"""
        body = {
            "modelUri": self.config.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature or self.config.temperature,
                "maxTokens": str(max_tokens or self.config.max_tokens),
            },
            "messages": [{"role": m.role, "text": m.text} for m in messages],
        }

        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(self.config.api_url, json=body)
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
                    time.sleep(wait)
                    continue
                raise YandexGPTError(f"HTTP {e.response.status_code}", e.response.status_code)
            except httpx.RequestError as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise YandexGPTError(f"Ошибка соединения: {e}")

        raise YandexGPTError("Превышено количество попыток")

    def complete_stream(
        self,
        messages: list[YandexGPTMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Стриминговая генерация"""
        body = {
            "modelUri": self.config.model_uri,
            "completionOptions": {
                "stream": True,
                "temperature": temperature or self.config.temperature,
                "maxTokens": str(max_tokens or self.config.max_tokens),
            },
            "messages": [{"role": m.role, "text": m.text} for m in messages],
        }

        with self._client.stream("POST", self.config.api_url, json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if text := data.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text"):
                        yield text
                except json.JSONDecodeError:
                    continue

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()