from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .config import ModelConfig


class ChatClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str: ...


@dataclass(frozen=True)
class CacheRecord:
    key: str
    response: str


class SQLitePromptCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "cache_key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row else None

    def put(self, key: str, response: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses(cache_key, response, created_at) VALUES (?, ?, ?)",
                (key, response, time.time()),
            )


class OpenAIChatClient:
    def __init__(
        self,
        config: ModelConfig,
        *,
        cache: SQLitePromptCache | None = None,
        max_retries: int = 4,
        retry_base_seconds: float = 1.0,
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self.config = config
        self.cache = cache
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self._client = OpenAI()

    def _cache_key(self, prompt: str, max_tokens: int) -> str:
        payload = {
            **asdict(self.config),
            "max_tokens": max_tokens,
            "prompt": prompt,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        token_limit = max_tokens or self.config.max_tokens
        key = self._cache_key(prompt, token_limit)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        request = {
            "model": self.config.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": token_limit,
        }
        if self.config.seed is not None:
            request["seed"] = self.config.seed

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
                text = response.choices[0].message.content or ""
                if self.cache is not None:
                    self.cache.put(key, text)
                return text
            except Exception as error:  # SDK exception types differ across versions.
                last_error = error
                status = getattr(error, "status_code", None)
                if status in {400, 401, 403, 404} or attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_base_seconds * (2**attempt))
        assert last_error is not None
        raise last_error


class StaticChatClient:
    """Small deterministic client useful in tests and prompt previews."""

    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        return self.response

