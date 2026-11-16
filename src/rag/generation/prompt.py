"""Assemble the system and user messages. No model is called."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.generation.context import build_context, build_grouped_context
from rag.generation.evidence import usable_reviews
from rag.retrieval.interpret import region_question

SYSTEM_INSTRUCTIONS = """\
Escreva em português do Brasil, como quem conversa, em um parágrafo curto.
A primeira frase começa com "As avaliações compartilham" e diz o que se repete, elogio ou reclamação.
Não descreva cada avaliação e não cite número entre colchetes.
Não mencione estado, UF ou região quando a pergunta não for sobre lugar.
Feche com o ponto principal que as avaliações sustentam, sem inventar causa, quantidade, UF ou região.
Quando houver grupos de região, diga qual lado aparece mais e o que os comentários relatam.
Se houver uma contagem por região, use esses números, sem colchetes, e não calcule outro.
Não copie o comentário e não repita este pedido.
Se as avaliações não mostrarem o que foi perguntado, diga isso com naturalidade.
"""


_INSTRUCTION_STARTS = (
    "escreva a resposta",
    "escreva em português",
    "responda em português",
    "leia todas as",
    "leia as ",
    "responda em um parágrafo",
    "agrupe as reclamações",
    "agrupe o que se repete",
    "começando por",
    "não descreva cada avaliação",
    "cite os números",
    "use a contagem",
    "a quantidade de cada grupo",
    "escreva essa quantidade",
    "[1] e [2] apontam",
    "diga quem aparece mais",
    "quando houver grupos",
    "não repita a pergunta",
    "não repita a resposta",
    "não use uma frase pronta",
    "responda em duas frases",
    "a primeira frase começa",
    "a primeira frase responde",
    "a primeira frase diz",
    "a segunda frase é só",
    "a segunda cita",
    "não responda com o caso",
    "não responda com uma avaliação",
    "não diga quantas",
    "não repita a primeira",
    "a segunda diz o que se repete",
    "use a nota, a uf",
    "não copie o comentário",
    "se as avaliações não mostrarem",
    "se os comentários não bastarem",
    "termine a frase",
    "termine a última frase",
)


def without_instruction_echo(answer: str, *, question: str = "") -> str:
    """Drop instructions and a leading copy of the question."""
    kept_lines: list[str] = []
    for line in answer.splitlines():
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", line.strip()) if part.strip()]
        kept = [sentence for sentence in sentences if not _is_instruction(sentence)]
        if kept:
            kept_lines.append(" ".join(kept))
    return _strip_leading_question("\n".join(kept_lines).strip(), question)


def _is_instruction(sentence: str) -> bool:
    folded = sentence.casefold()
    return folded.startswith(_INSTRUCTION_STARTS)


def _strip_leading_question(text: str, question: str) -> str:
    wanted = " ".join(question.split()).strip().rstrip("?")
    if len(wanted) < 8 or not text:
        return text
    leading = text.lstrip()
    if leading.casefold().startswith(wanted.casefold()):
        rest = leading[len(wanted) :].lstrip(" ?").strip()
        return rest
    return text


def _history_block(history: tuple[tuple[str, str], ...]) -> str:
    if not history:
        return ""
    lines = ["Conversa anterior, só para entender a continuação. Não repita a resposta anterior:"]
    for role, text in history:
        who = "Cliente" if role == "user" else "Assistente"
        shown = text if role == "user" else _first_sentence(text)
        lines.append(f"{who}: {shown}")
    return "\n".join(lines) + "\n\n"


def _first_sentence(text: str, limit: int = 160) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 3].rstrip() + "..."


_GROUPING_OPEN = re.compile(
    r"^\s*agrupando as reclamações semelhantes,\s*(?:observamos|vemos|notamos)\s+que\s+",
    re.IGNORECASE,
)


def with_shared_opening(answer: str) -> str:
    """Replace the copied 'Agrupando as reclamações' lead with the shared-voice opening."""
    return _GROUPING_OPEN.sub("As avaliações compartilham que ", " ".join(answer.split()).strip())


def without_review_mentions(answer: str) -> str:
    """Drop pointers to a single review. The paragraph stays about the group."""
    text = re.sub(r"\s*\[\d+\]", "", answer)
    text = re.sub(r",?\s*como no caso da avaliaç(?:ão|ões)(?:\s+e)*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\s+([.;])", r"\1", text)
    return " ".join(text.split()).strip()


def drop_repeated_opening(answer: str, history: tuple[tuple[str, str], ...]) -> str:
    """Drop a copied opening from the previous assistant reply."""
    previous = next((text for role, text in reversed(history) if role == "assistant"), "")
    current = " ".join(answer.split())
    earlier = " ".join(previous.split())
    if len(earlier) < 40 or len(current) < 40:
        return answer.strip()
    limit = min(len(current), len(earlier))
    shared = 0
    while shared < limit and current[shared].casefold() == earlier[shared].casefold():
        shared += 1
    if shared < 60:
        return answer.strip()
    shared = current.rfind(" ", 0, shared)
    rest = current[shared:].lstrip(" .,;").strip()
    if len(rest) < 40:
        return answer.strip()
    return rest


REFUSAL = (
    "Não há avaliações suficientes para responder com base nos comentários recuperados. "
    "Nenhuma resposta foi gerada."
)


@dataclass(frozen=True, slots=True)
class Prompt:
    """The two messages a later model call would send."""

    system: str
    user: str

    def render(self) -> str:
        """Return the full prompt as plain text."""
        return f"SYSTEM\n{self.system}\nUSER\n{self.user}\n"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Either a prompt ready for a model, or a refusal that skips the call."""

    insufficient_evidence: bool
    prompt: Prompt | None
    refusal: str | None


def read_history(raw: object, *, limit: int = 4, max_chars: int = 360) -> tuple[tuple[str, str], ...]:
    """Keep the last few chat turns. Each item is `(role, text)`."""
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    if not isinstance(raw, list):
        return ()
    turns: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        text = " ".join(content.split())
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        turns.append((role, text))
    return tuple(turns[-limit:])


def build_generation(
    payload: dict[str, object],
    *,
    history: tuple[tuple[str, str], ...] = (),
    min_reviews: int = 2,
    max_reviews: int = 10,
    max_context_chars: int = 12000,
) -> GenerationRequest:
    """Build the prompt from a retrieval payload.

    When the reviews cannot support an answer, `prompt` is `None` and
    `refusal` explains that in Portuguese. The model is not asked to invent one.
    """
    usable = usable_reviews(payload, min_reviews=min_reviews)
    if usable is None:
        return GenerationRequest(insufficient_evidence=True, prompt=None, refusal=REFUSAL)
    question, reviews = usable
    about_place = _about_place(question, payload)
    context = build_context(
        reviews,
        max_reviews=max_reviews,
        max_chars=max_context_chars,
        include_state=about_place,
    )
    if not context.strip():
        return GenerationRequest(insufficient_evidence=True, prompt=None, refusal=REFUSAL)
    tally = payload.get("score_tally")
    count = f"\n\n{tally}\n" if isinstance(tally, str) and tally.strip() else ""
    if count:
        left = tuple(payload.get("compare_left") or ())
        right = tuple(payload.get("compare_right") or ())
        context = build_grouped_context(reviews, left=left, right=right, max_chars=max_context_chars)
        closing = (
            "Responda em um parágrafo curto, começando por \"As avaliações compartilham\".\n"
            "Diga qual lado aparece mais, com a quantidade em algarismos e sem colchetes, "
            "e o que cada lado relata.\n"
            "Não descreva cada avaliação e não repita a resposta anterior."
        )
    else:
        place_line = "" if about_place else "Não cite estado, UF ou região.\n"
        closing = (
            "Responda em um parágrafo curto, começando por \"As avaliações compartilham\".\n"
            "Feche com o ponto principal.\n"
            f"{place_line}"
            "Não descreva cada avaliação e não repita a resposta anterior."
        )
    user = f"{_history_block(history)}Pergunta:\n{question}\n{count}\nAvaliações:\n{context}\n\n{closing}"
    return GenerationRequest(
        insufficient_evidence=False,
        prompt=Prompt(system=SYSTEM_INSTRUCTIONS, user=user),
        refusal=None,
    )


def _about_place(question: str, payload: dict[str, object]) -> bool:
    """True when the question, or the search filter, is about a state or region."""
    query = payload.get("query")
    filters = query.get("filters") if isinstance(query, dict) else None
    if isinstance(filters, dict) and (filters.get("customer_state") or filters.get("customer_states")):
        return True
    if region_question(question) is not None:
        return True
    return re.search(r"\b(estado|regi[aã]o|regio)\b", question, re.IGNORECASE) is not None
