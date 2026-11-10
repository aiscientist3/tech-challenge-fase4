"""One chat completion against a local Ollama server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rag.generation.prompt import Prompt

DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


class OllamaError(Exception):
    """The local model did not return a completion."""


def complete(prompt: Prompt, *, model: str | None = None, timeout: float = 120) -> str:
    """Send the system and user messages and return the assistant text."""
    chosen = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    body = {
        "model": chosen,
        "stream": False,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
        "options": {"temperature": 0.2, "num_predict": 320},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OllamaError(
            "O Ollama não respondeu. Confira se o aplicativo está aberto e se o modelo foi baixado."
        ) from exc
    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("O Ollama devolveu uma resposta vazia.")
    return content.strip()
