"""Assemble the system and user messages. No model is called."""

from __future__ import annotations

from dataclasses import dataclass

from rag.generation.context import build_context
from rag.generation.evidence import usable_reviews

SYSTEM_INSTRUCTIONS = """\
You are a Voice of Customer Intelligence assistant.

Your task is to analyze customer reviews and answer questions based on the evidence provided.

Rules:

1. Use only the provided customer reviews as evidence.
2. Do not invent facts or information.
3. Identify recurring patterns when supported by evidence.
4. Reference the review IDs used to support your findings.
5. Do not assume that retrieved reviews represent the entire dataset.
6. If the provided evidence is insufficient, explicitly state that.
7. Always respond in Brazilian Portuguese.
"""

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


def build_generation(
    payload: dict[str, object],
    *,
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
    context = build_context(reviews, max_reviews=max_reviews, max_chars=max_context_chars)
    if not context.strip():
        return GenerationRequest(insufficient_evidence=True, prompt=None, refusal=REFUSAL)
    user = f"Pergunta:\n{question}\n\nAvaliações:\n{context}"
    return GenerationRequest(
        insufficient_evidence=False,
        prompt=Prompt(system=SYSTEM_INSTRUCTIONS, user=user),
        refusal=None,
    )
