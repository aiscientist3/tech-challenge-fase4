"""Check a written answer against the reviews that were shown.

A sentence that says one region has more problems than the others stays only
when the cited reviews actually show that. Otherwise the sentence is replaced
by a short correction. The rest of the wording is left alone.
"""

from __future__ import annotations

import re
import unicodedata

# Longer names first so "nordeste" is not read as "norte".
_REGIONS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("centro-oeste", "o Centro-Oeste", frozenset({"DF", "GO", "MS", "MT"})),
    ("centro oeste", "o Centro-Oeste", frozenset({"DF", "GO", "MS", "MT"})),
    ("nordeste", "o Nordeste", frozenset({"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"})),
    ("sudeste", "o Sudeste", frozenset({"ES", "MG", "RJ", "SP"})),
    ("norte", "o Norte", frozenset({"AC", "AM", "AP", "PA", "RO", "RR", "TO"})),
    ("sul", "o Sul", frozenset({"PR", "RS", "SC"})),
)


def ground_answer(answer: str, states: list[str]) -> str:
    """Return `answer`, or a correction when a regional comparison is not in the reviews."""
    text = answer.strip()
    if not text or not states:
        return text
    claimed = _regions_said_to_have_more(text)
    if not claimed:
        return text
    cited = _cited_positions(text, len(states))
    chosen = cited or list(range(1, len(states) + 1))
    more, other = _counts(chosen, states, claimed)
    if more >= 2 and other >= 2 and more > other:
        return text
    return _correction(claimed)


def _regions_said_to_have_more(answer: str) -> list[str]:
    folded = _fold(answer)
    match = re.search(r"\bmais\b", folded)
    if match is None:
        return []
    before = folded[max(0, match.start() - 12) : match.start()]
    if re.search(r"\bnao\b", before):
        return []
    after = folded[match.end() :]
    than = re.search(r"\b(?:do que|que)\b", after)
    # "o Norte tem mais problemas que o Sul" names the Norte, which sits before "mais".
    source = folded[: match.start()] if than is not None else re.split(r"[.!?]", after, maxsplit=1)[0]
    found: list[str] = []
    seen: set[str] = set()
    for name, label, _states in _REGIONS:
        if label in seen or re.search(rf"\b{re.escape(name)}\b", source) is None:
            continue
        seen.add(label)
        found.append(label)
    return found


def _cited_positions(answer: str, count: int) -> list[int]:
    numbers = [int(item) for item in re.findall(r"\[(\d+)\]", answer)]
    for match in re.finditer(r"\bem\s+((?:\d+\s*(?:,|e)?\s*)+)", answer, re.IGNORECASE):
        numbers.extend(int(item) for item in re.findall(r"\d+", match.group(1)))
    chosen: list[int] = []
    for number in numbers:
        if 1 <= number <= count and number not in chosen:
            chosen.append(number)
    return chosen


def _counts(positions: list[int], states: list[str], claimed: list[str]) -> tuple[int, int]:
    claimed_states: set[str] = set()
    known: set[str] = set()
    for _name, label, group in _REGIONS:
        known.update(group)
        if label in claimed:
            claimed_states.update(group)
    more = 0
    other = 0
    for position in positions:
        state = states[position - 1].strip().upper()
        if state in claimed_states:
            more += 1
        elif state in known:
            other += 1
    return more, other


def _correction(claimed: list[str]) -> str:
    if len(claimed) == 1:
        names = claimed[0]
    else:
        names = ", ".join(claimed[:-1]) + " e " + claimed[-1]
    return (
        f"Pelos comentários desta lista, não dá para dizer que {names} "
        "tenham mais problemas do que as outras regiões."
    )


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without.casefold()
