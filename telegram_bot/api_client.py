"""Async HTTP client for the GigaMemory REST API.

Thin wrapper — one method per endpoint the bot needs. All bot business logic
lives here only as request shaping; the memory pipeline stays server-side.
"""

from __future__ import annotations

from typing import Any

import httpx


class GigaMemoryAPIError(RuntimeError):
    """Raised when the API is unreachable or returns a non-2xx status."""


class GigaMemoryClient:
    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base, timeout=self._timeout)

    async def health(self) -> bool:
        try:
            async with self._client() as c:
                r = await c.get("/health")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def send_message(
        self,
        dialogue_id: str,
        content: str,
        *,
        parallel_write: bool = False,
        prompt_language: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "content": content,
            "parallel_write": parallel_write,
        }
        if prompt_language:
            payload["prompt_language"] = prompt_language
        data = await self._post(f"/dialogue/{dialogue_id}/message", payload)
        return str(data.get("answer", ""))

    async def answer(
        self,
        dialogue_id: str,
        content: str,
        *,
        prompt_language: str | None = None,
    ) -> str:
        """Generate the answer only (no memory write)."""
        payload: dict[str, Any] = {"content": content}
        if prompt_language:
            payload["prompt_language"] = prompt_language
        data = await self._post(f"/dialogue/{dialogue_id}/answer", payload)
        return str(data.get("answer", ""))

    async def remember(
        self,
        dialogue_id: str,
        content: str,
        *,
        prompt_language: str | None = None,
    ) -> None:
        """Write the message to memory only (no answer)."""
        payload: dict[str, Any] = {"content": content}
        if prompt_language:
            payload["prompt_language"] = prompt_language
        await self._post(f"/dialogue/{dialogue_id}/remember", payload)

    async def graph_short(self, dialogue_id: str) -> dict[str, Any]:
        return await self._get_json(f"/dialogue/{dialogue_id}/graph_short")

    async def graph_image(self, dialogue_id: str) -> bytes:
        return await self._get_bytes(f"/dialogue/{dialogue_id}/graph/image")

    async def graph_html(self, dialogue_id: str) -> bytes:
        return await self._get_bytes(f"/dialogue/{dialogue_id}/graph/html")

    async def forget(self, dialogue_id: str) -> None:
        try:
            async with self._client() as c:
                r = await c.delete(f"/dialogue/{dialogue_id}")
                r.raise_for_status()
        except httpx.HTTPError as e:
            raise GigaMemoryAPIError(str(e)) from e

    # ── internals ──────────────────────────────────────────────────────────
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._client() as c:
                r = await c.post(path, json=payload)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as e:
            raise GigaMemoryAPIError(str(e)) from e

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            async with self._client() as c:
                r = await c.get(path)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as e:
            raise GigaMemoryAPIError(str(e)) from e

    async def _get_bytes(self, path: str) -> bytes:
        try:
            async with self._client() as c:
                r = await c.get(path)
                r.raise_for_status()
                return r.content
        except httpx.HTTPError as e:
            raise GigaMemoryAPIError(str(e)) from e
