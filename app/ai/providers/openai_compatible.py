"""Proveedor compatible con la API de OpenAI (``/chat/completions``).

Funciona con OpenAI y con cualquier servicio que replique su API:
OpenRouter, Groq, Together, Ollama (modo ``openai``), LM Studio, etc.
"""
import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from app.ai.providers.base import LLMError, LLMProvider
from app.ai.schemas import ChatMessage, ChatResponse, TokenUsage, ToolCall
from app.tools.schemas import ToolSpec

_CHAT_ENDPOINT = "/chat/completions"
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 60.0,
        retries: int = 2,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._retries = retries
        self._max_tokens = max_tokens
        self._extra_body = dict(extra_body) if extra_body else {}
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _parse_response(data: dict, provider: str, model: str) -> ChatResponse:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Formato de respuesta inesperado: {exc}") from exc

        content = message.get("content") or ""
        tool_calls = OpenAICompatibleProvider._parse_tool_calls(message.get("tool_calls"))

        usage_data = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return ChatResponse(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _parse_tool_calls(raw_calls: object) -> list[ToolCall] | None:
        if not isinstance(raw_calls, list) or not raw_calls:
            return None

        tool_calls: list[ToolCall] = []
        for raw in raw_calls:
            try:
                function = raw["function"]
                name = function["name"]
                arguments = json.loads(function.get("arguments") or "{}")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise LLMError(f"tool_calls mal formado: {exc}") from exc
            tool_calls.append(
                ToolCall(id=raw.get("id", ""), name=name, arguments=arguments)
            )
        return tool_calls

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict:
        """Serializa un mensaje al formato wire de OpenAI (tool_calls incluidos)."""
        data = message.model_dump(exclude_none=True)
        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return data

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResponse:
        payload = {
            "model": self._model,
            "messages": [self._message_payload(message) for message in messages],
        }
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        if self._extra_body:
            payload.update(self._extra_body)
        if tools:
            payload["tools"] = [tool.model_dump() for tool in tools]

        try:
            response = await self._post_with_retry(payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Error de conexión con el proveedor LLM: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(self._format_error(response))

        return self._parse_response(response.json(), self.name, self._model)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[tuple[str, object]]:
        """Streaming SSE real: emite ('delta', texto) y cierra con ('final', ...)."""
        payload = {
            "model": self._model,
            "messages": [self._message_payload(message) for message in messages],
            "stream": True,
        }
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        if self._extra_body:
            payload.update(self._extra_body)
        if tools:
            payload["tools"] = [tool.model_dump() for tool in tools]

        content_parts: list[str] = []
        raw_calls: dict[int, dict] = {}
        try:
            async with self._client.stream(
                "POST", _CHAT_ENDPOINT, headers=self._headers(), json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise LLMError(self._error_message(response.status_code, body))
                async for line in response.aiter_lines():
                    data = line.strip()
                    if not data.startswith("data:"):
                        continue
                    data = data[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        yield ("delta", piece)
                    self._accumulate_tool_call(
                        raw_calls, delta.get("tool_calls") or []
                    )
        except httpx.HTTPError as exc:
            raise LLMError(f"Error de conexión con el proveedor LLM: {exc}") from exc

        yield ("final", ChatResponse(
            content="".join(content_parts),
            model=self._model,
            provider=self.name,
            tool_calls=self._collect_tool_calls(raw_calls),
        ))

    @staticmethod
    def _accumulate_tool_call(slots: dict[int, dict], raw_calls: object) -> None:
        """Acumula los fragmentos parciales de tool_calls del stream."""
        if not isinstance(raw_calls, list):
            return
        for raw in raw_calls:
            index = raw.get("index", 0)
            slot = slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if raw.get("id"):
                slot["id"] = raw["id"]
            function = raw.get("function") or {}
            if function.get("name"):
                slot["name"] += function["name"]
            if function.get("arguments"):
                slot["arguments"] += function["arguments"]

    @staticmethod
    def _collect_tool_calls(slots: dict[int, dict]) -> list[ToolCall] | None:
        if not slots:
            return None
        calls: list[ToolCall] = []
        for _, slot in sorted(slots.items()):
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)
            )
        return calls

    async def _post_with_retry(self, payload: dict) -> httpx.Response:
        """POST con reintentos en 429/5xx (límites del free tier, servidores)."""
        for attempt in range(self._retries + 1):
            response = await self._client.post(
                _CHAT_ENDPOINT,
                headers=self._headers(),
                json=payload,
            )
            if response.status_code not in _RETRYABLE_STATUSES or attempt == self._retries:
                return response
            await asyncio.sleep(self._backoff_seconds(response, attempt))
        return response  # pragma: no cover

    @staticmethod
    def _backoff_seconds(response: httpx.Response, attempt: int = 0) -> float:
        """Espera antes de reintentar: Retry-After del servidor o 2^n segundos."""
        retry_after = response.headers.get("retry-after")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
        return 2**attempt

    @staticmethod
    def _format_error(response: httpx.Response) -> str:
        return OpenAICompatibleProvider._error_message(
            response.status_code, response.text
        )

    @staticmethod
    def _error_message(status_code: int, text: str) -> str:
        detail = ""
        try:
            body = json.loads(text)
            if isinstance(body, list):
                body = body[0] if body else {}
            detail = body.get("error", {}).get("message", "") or body.get("error", "")
        except ValueError:
            detail = text[:200]
        return f"El proveedor LLM respondió {status_code}: {detail}"

    async def aclose(self) -> None:
        await self._client.aclose()
