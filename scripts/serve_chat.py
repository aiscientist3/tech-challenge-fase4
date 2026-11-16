"""Serve POST /chat for the local frontend.

Loads the embedding model once, searches the Chroma collection, and asks
Ollama to write the answer from those reviews. A thin retrieval skips the
model and returns the refusal.

    python scripts/serve_chat.py
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Settings, configure_logging
from rag.generation.chat import to_chat_response
from rag.generation.ollama import OllamaError, complete
from rag.generation.prompt import (
    build_generation,
    drop_repeated_opening,
    read_history,
    with_shared_opening,
    without_review_mentions,
)
from rag.ingestion.indexer import CollectionConfigError, build_embeddings
from rag.retrieval.payload import to_payload
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.relevance import RelevanceError, load_relevance
from rag.retrieval.retrieve import CollectionNotFoundError, retrieve
from rag.retrieval.select import SelectionPolicy

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    print("Carregando o modelo de embeddings...", flush=True)
    embeddings = build_embeddings(settings)
    print("Carregando o modelo que lê a pergunta...", flush=True)
    try:
        load_relevance()
    except RelevanceError as exc:
        print(str(exc), flush=True)
        raise SystemExit(1) from exc
    policy = SelectionPolicy()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/chat":
                self._send(404, {"answer": "Rota não encontrada.", "insufficient_evidence": True, "sources": []})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = json.loads(self.rfile.read(length) or b"{}")
                question = raw.get("question") if isinstance(raw, dict) else None
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("A pergunta não pode ficar vazia.")
                filters = _filters(raw.get("filters") if isinstance(raw, dict) else None)
                history = read_history(raw.get("history") if isinstance(raw, dict) else None)
                result = retrieve(question, settings, embeddings, filters=filters, policy=policy)
                answer = _generate(result, history)
            except (CollectionNotFoundError, CollectionConfigError, ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"answer": str(exc), "insufficient_evidence": True, "sources": []})
                return
            except (OllamaError, RelevanceError) as exc:
                self._send(503, {"answer": str(exc), "insufficient_evidence": True, "sources": []})
                return
            self._send(200, to_chat_response(result, answer=answer))

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[chat] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, status: int, body: dict[str, object]) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"POST http://{HOST}:{PORT}/chat", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando.", flush=True)
    finally:
        server.server_close()


def _generate(result: object, history: tuple[tuple[str, str], ...]) -> str | None:
    generated = build_generation(to_payload(result), history=history)
    if generated.prompt is None:
        return None
    answer = with_shared_opening(without_review_mentions(complete(generated.prompt)))
    return drop_repeated_opening(answer, history)


def _filters(raw: object) -> RetrievalFilters:
    if not isinstance(raw, dict):
        return RetrievalFilters()
    return RetrievalFilters(
        doc_type=_text(raw.get("doc_type")),
        review_score_min=_int(raw.get("review_score_min")),
        review_score_max=_int(raw.get("review_score_max")),
        product_category=_text(raw.get("product_category")),
        purchase_year=_int(raw.get("purchase_year")),
    )


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _int(value: object) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


if __name__ == "__main__":
    main()
