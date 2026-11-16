"""Group vector hits by region and score.

A question that asks which region has more low or high scores needs a count.
The count uses the reviews the vector search returned, split into notes 1–3
and notes 4–5. A region needs at least two reviews before it enters the
comparison, so a single review cannot become 100%.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_REGION_OF = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
}

_REGION_WORDS: tuple[tuple[str, str], ...] = (
    ("centro-oeste", "Centro-Oeste"),
    ("centro oeste", "Centro-Oeste"),
    ("nordeste", "Nordeste"),
    ("sudeste", "Sudeste"),
    ("norte", "Norte"),
    ("sul", "Sul"),
)

MIN_REGION_REVIEWS = 2


@dataclass(frozen=True, slots=True)
class RegionTally:
    """How the retrieved reviews of one region split between low and high scores."""

    region: str
    low: int
    high: int

    @property
    def total(self) -> int:
        return self.low + self.high

    def share(self, focus: str) -> float:
        if self.total < 1:
            return 0.0
        band = self.low if focus == "low" else self.high
        return band / self.total

    def band_count(self, focus: str) -> int:
        return self.low if focus == "low" else self.high


def region_name(state: str) -> str | None:
    """Return the Brazilian region for a customer UF."""
    return _REGION_OF.get(state.strip().upper())


def example_indexes(
    states: list[str],
    groups: tuple[RegionTally, ...],
    focus: str,
    left: tuple[str, ...] = (),
    right: tuple[str, ...] = (),
    order_ids: list[str] | None = None,
) -> list[int]:
    """Indexes of a few reviews from each side, nearest first.

    Two named sides contribute up to eight reviews each. An open comparison
    contributes up to two reviews from each region that showed up.
    """
    per_side = 8 if left and right else 2
    limit = per_side * 2 if left and right else 8
    plan = _example_plan(groups, focus, left, right)
    chosen: list[int] = []
    used: set[int] = set()
    seen_orders: set[str] = set()
    for names in plan:
        taken = 0
        for index, state in enumerate(states):
            if index in used:
                continue
            if region_name(state) not in names:
                continue
            order_id = order_ids[index] if order_ids and index < len(order_ids) else ""
            if order_id and order_id in seen_orders:
                continue
            chosen.append(index)
            used.add(index)
            if order_id:
                seen_orders.add(order_id)
            taken += 1
            if taken >= per_side or len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break
    return chosen


def _example_plan(
    groups: tuple[RegionTally, ...],
    focus: str,
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> list[set[str]]:
    if left and right:
        return [set(left), set(right)]
    if focus in {"low", "high"}:
        return [{group.region} for group in comparable(groups, focus)]
    ranked = [group for group in groups if group.total >= 1]
    ranked.sort(key=lambda group: (group.total, group.region), reverse=True)
    return [{group.region} for group in ranked]


def group_regions(pairs: list[tuple[str, int]]) -> tuple[RegionTally, ...]:
    """Count low and high scores for each region present in `pairs`."""
    counts: dict[str, list[int]] = {}
    for state, score in pairs:
        region = _REGION_OF.get(state.strip().upper())
        if region is None or score < 1 or score > 5:
            continue
        low, high = counts.setdefault(region, [0, 0])
        if score <= 3:
            counts[region] = [low + 1, high]
        else:
            counts[region] = [low, high + 1]
    tallies = tuple(
        RegionTally(region, low, high)
        for region, (low, high) in sorted(counts.items())
    )
    return tallies


def comparable(groups: tuple[RegionTally, ...], focus: str) -> list[RegionTally]:
    """Regions with enough reviews, highest share of `focus` first."""
    ranked = [group for group in groups if group.total >= MIN_REGION_REVIEWS]
    ranked.sort(key=lambda group: (group.share(focus), group.band_count(focus), group.region), reverse=True)
    return ranked


def tally_text(
    groups: tuple[RegionTally, ...],
    focus: str,
    left: tuple[str, ...] = (),
    right: tuple[str, ...] = (),
) -> str:
    """Portuguese lines the prompt can quote. Empty when there is nothing to compare."""
    if focus == "count":
        return _count_text(groups, left, right)
    ranked = comparable(groups, focus)
    if len(ranked) < 2:
        return ""
    note = "baixas (1 a 3)" if focus == "low" else "altas (4 e 5)"
    lines = [f"Parcela de notas {note} em cada região, nesta busca:"]
    for group in ranked:
        percent = round(100 * group.share(focus))
        lines.append(
            f"- {group.region}: {percent}% ({group.band_count(focus)} de {group.total})"
        )
    lines.append(f"A maior parcela é {ranked[0].region}. Use só estes percentuais.")
    return "\n".join(lines)


def prefer_counted(
    answer: str,
    groups: tuple[RegionTally, ...],
    focus: str,
    left: tuple[str, ...] = (),
    right: tuple[str, ...] = (),
) -> str:
    """Replace an answer that names a region other than the counted leader."""
    if focus == "count" and left and right:
        return _prefer_sides(answer, groups, left, right)
    ranked = comparable(groups, focus)
    if len(ranked) < 2:
        return answer
    leader = ranked[0]
    claimed = _claimed_regions(answer)
    if not claimed or leader.region in claimed:
        return answer
    note = "baixas" if focus == "low" else "altas"
    percent = round(100 * leader.share(focus))
    return (
        f"Nesta busca, {leader.region} é a região com a maior parcela de notas {note}: "
        f"{percent}% ({leader.band_count(focus)} de {leader.total})."
    )


def _count_text(
    groups: tuple[RegionTally, ...],
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> str:
    totals = {group.region: group.total for group in groups}
    if left and right:
        left_n = sum(totals.get(name, 0) for name in left)
        right_n = sum(totals.get(name, 0) for name in right)
        if left_n + right_n < 2:
            return ""
        left_name = _join(left)
        right_name = _join(right)
        lines = [
            f"Nesta busca, {left_name}: {left_n} avaliações.",
            f"{right_name}: {right_n} avaliações.",
        ]
        if left_n == right_n:
            lines.append("Os dois lados apareceram na mesma quantidade.")
        elif left_n > right_n:
            lines.append(f"O lado com mais resultados é {left_name}.")
        else:
            lines.append(f"O lado com mais resultados é {right_name}.")
        lines.append("Use só estas quantidades.")
        return "\n".join(lines)
    ranked = [group for group in groups if group.total >= MIN_REGION_REVIEWS]
    ranked.sort(key=lambda group: (group.total, group.region), reverse=True)
    if len(ranked) < 2:
        return ""
    lines = ["Quantidade de avaliações por região, nesta busca:"]
    for group in ranked:
        lines.append(f"- {group.region}: {group.total}")
    lines.append(f"A região com mais avaliações nesta busca é {ranked[0].region}. Use só estas quantidades.")
    return "\n".join(lines)


def _prefer_sides(
    answer: str,
    groups: tuple[RegionTally, ...],
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> str:
    totals = {group.region: group.total for group in groups}
    left_n = sum(totals.get(name, 0) for name in left)
    right_n = sum(totals.get(name, 0) for name in right)
    if left_n + right_n < 2 or left_n == right_n:
        return answer
    winner = left if left_n > right_n else right
    winner_n = left_n if left_n > right_n else right_n
    claimed = _more_side(answer)
    if "nao da para dizer" in _fold(answer):
        return (
            f"Nesta busca, {_join(winner)} aparece mais: "
            f"{winner_n} avaliações, contra {min(left_n, right_n)} do outro lado."
        )
    if not claimed or any(name in winner for name in claimed):
        return answer
    return (
        f"Nesta busca, {_join(winner)} aparece mais: "
        f"{winner_n} avaliações, contra {min(left_n, right_n)} do outro lado."
    )


def _more_side(answer: str) -> tuple[str, ...]:
    folded = _fold(answer)
    mais = re.search(r"\bmais\b", folded)
    if mais is None:
        return ()
    after = folded[mais.end() :]
    than = re.search(r"\b(?:do que|que)\b", after)
    if than is not None:
        return _labels(folded[: mais.start()])
    return _labels(re.split(r"[.!?]", after, maxsplit=1)[0])


def _labels(folded: str) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name, label in _REGION_WORDS:
        match = re.search(rf"\b{re.escape(name)}\b", folded)
        if match is None or label in seen:
            continue
        seen.add(label)
        found.append((match.start(), label))
    found.sort()
    return tuple(label for _start, label in found)


def _join(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " e " + names[-1]


def _claimed_regions(answer: str) -> set[str]:
    folded = _fold(answer)
    found: set[str] = set()
    for name, label in _REGION_WORDS:
        if re.search(rf"\b{re.escape(name)}\b", folded):
            found.add(label)
    for state, region in _REGION_OF.items():
        if re.search(rf"\b{state.lower()}\b", folded):
            found.add(region)
    return found


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without.casefold()
