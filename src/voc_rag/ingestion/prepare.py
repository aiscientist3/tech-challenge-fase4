"""Seleção, limpeza e montagem dos documentos indexados."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

UNKNOWN_DELAY_DAYS = -1
MISSING = "nao_informada"

DATASET_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "customers": "olist_customers_dataset.csv",
}

REQUIRED_COLUMNS = {
    "reviews": (
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
    ),
    "orders": (
        "order_id",
        "customer_id",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ),
    "items": ("order_id", "product_id"),
    "products": ("product_id", "product_category_name"),
    "translation": ("product_category_name", "product_category_name_english"),
    "customers": ("customer_id", "customer_city", "customer_state"),
}

# Substrings, já em minúsculas. "atraso" também cobre atrasada/atrasado.
ASPECT_TERMS = {
    "entrega": ("atraso", "entrega", "entregue"),
    "produto": ("defeito", "qualidade"),
    "atendimento": ("atendimento", "troca"),
}

METADATA_KEYS = (
    "review_score",
    "product_category",
    "customer_state",
    "customer_city",
    "delivery_delay_days",
    "late_delivery",
    "aspects",
    "review_date",
    "review_title",
    "order_id",
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ReviewDocument:
    review_id: str
    document: str
    review_score: int
    product_category: str
    customer_state: str
    customer_city: str
    delivery_delay_days: int
    late_delivery: bool
    aspects: str
    review_date: str
    review_title: str
    order_id: str

    def metadata(self) -> dict[str, str | int | bool]:
        payload = {key: getattr(self, key) for key in METADATA_KEYS}
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (str, int, float)):
                raise TypeError(f"Metadado {key} tem tipo {type(value).__name__}")
        return payload


@dataclass(frozen=True)
class PrepareResult:
    documents: list[ReviewDocument]
    rows_read: int
    dropped_duplicates: int
    dropped_empty: int
    dropped_without_score: int


def prepare_documents(raw_dir: Path) -> PrepareResult:
    frames = _load_frames(raw_dir)
    reviews = frames["reviews"]
    rows_read = len(reviews)

    reviews = reviews.copy()
    reviews["review_id"] = reviews["review_id"].map(_normalize_text)
    reviews["order_id"] = reviews["order_id"].map(_normalize_text)
    reviews["title_norm"] = reviews["review_comment_title"].map(_normalize_text)
    reviews["comment_norm"] = reviews["review_comment_message"].map(_normalize_text)
    reviews["title_len"] = reviews["title_norm"].str.len()
    reviews["comment_len"] = reviews["comment_norm"].str.len()
    reviews["source_order"] = range(len(reviews))
    reviews = reviews.sort_values(
        ["comment_len", "title_len", "source_order"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    reviews = reviews.drop_duplicates("review_id", keep="first")
    dropped_duplicates = rows_read - len(reviews)

    has_text = (reviews["title_norm"] != "") | (reviews["comment_norm"] != "")
    dropped_empty = int((~has_text).sum())
    reviews = reviews.loc[has_text]

    reviews["score_value"] = reviews["review_score"].map(_parse_score)
    has_score = reviews["score_value"].notna()
    dropped_without_score = int((~has_score).sum())
    reviews = reviews.loc[has_score]

    orders = _orders_with_customer(frames["orders"], frames["customers"])
    categories = _categories_by_order(
        frames["items"],
        frames["products"],
        frames["translation"],
    )
    reviews = reviews.merge(orders, on="order_id", how="left")
    reviews = reviews.merge(categories, on="order_id", how="left")

    documents: list[ReviewDocument] = []
    for row in reviews.itertuples(index=False):
        delay = _delay_days(getattr(row, "delivery_delay_days", None))
        category = _normalize_text(getattr(row, "product_category", "")) or MISSING
        state = _state(getattr(row, "customer_state", ""))
        city = _place(getattr(row, "customer_city", ""))
        title = row.title_norm
        comment = row.comment_norm
        documents.append(
            ReviewDocument(
                review_id=row.review_id,
                document=build_document_text(
                    review_score=int(row.score_value),
                    product_category=category,
                    customer_state=state,
                    delivery_delay_days=delay,
                    review_title=title,
                    review_comment=comment,
                ),
                review_score=int(row.score_value),
                product_category=category,
                customer_state=state,
                customer_city=city,
                delivery_delay_days=delay,
                late_delivery=delay != UNKNOWN_DELAY_DAYS and delay > 0,
                aspects=detect_aspects(title, comment),
                review_date=_format_date(row.review_creation_date),
                review_title=title,
                order_id=row.order_id,
            )
        )

    documents.sort(key=lambda document: document.review_id)
    return PrepareResult(
        documents=documents,
        rows_read=rows_read,
        dropped_duplicates=dropped_duplicates,
        dropped_empty=dropped_empty,
        dropped_without_score=dropped_without_score,
    )


def build_document_text(
    *,
    review_score: int,
    product_category: str,
    customer_state: str,
    delivery_delay_days: int,
    review_title: str,
    review_comment: str,
) -> str:
    lines = [
        f"Nota: {review_score}/5",
        f"Categoria: {product_category}",
        f"UF: {customer_state}",
    ]
    if delivery_delay_days != UNKNOWN_DELAY_DAYS:
        lines.append(_delivery_line(delivery_delay_days))
    if review_title:
        lines.append(f"Título: {review_title}")
    if review_comment:
        lines.append(f"Comentário: {review_comment}")
    return "\n".join(lines)


def detect_aspects(title: str, comment: str) -> str:
    blob = f"{title} {comment}".casefold()
    found = [
        aspect
        for aspect, terms in ASPECT_TERMS.items()
        if any(term in blob for term in terms)
    ]
    return ",".join(found)


def category_label(portuguese: object, english: object) -> str:
    return _normalize_text(portuguese) or _normalize_text(english)


def write_parquet(documents: list[ReviewDocument], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(document) for document in documents])
    frame.to_parquet(path, index=False)


def read_parquet(path: Path) -> list[ReviewDocument]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Não encontrei {path}. Rode a ingestão sem --from-processed antes."
        )
    frame = pd.read_parquet(path)
    documents: list[ReviewDocument] = []
    for row in frame.to_dict(orient="records"):
        documents.append(
            ReviewDocument(
                review_id=str(row["review_id"]),
                document=str(row["document"]),
                review_score=int(row["review_score"]),
                product_category=str(row["product_category"]),
                customer_state=str(row["customer_state"]),
                customer_city=str(row["customer_city"]),
                delivery_delay_days=int(row["delivery_delay_days"]),
                late_delivery=bool(row["late_delivery"]),
                aspects=str(row["aspects"]),
                review_date=str(row["review_date"]),
                review_title=str(row["review_title"]),
                order_id=str(row["order_id"]),
            )
        )
    return documents


def _load_frames(raw_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for key, filename in DATASET_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        frame = pd.read_csv(path)
        absent = [column for column in REQUIRED_COLUMNS[key] if column not in frame.columns]
        if absent:
            raise ValueError(f"{filename} sem as colunas: {', '.join(absent)}")
        frames[key] = frame
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"CSVs ausentes em {raw_dir}: {names}. "
            "Baixe o dataset Brazilian E-Commerce da Olist e coloque os arquivos em data/raw."
        )
    return frames


def _orders_with_customer(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    delivered = pd.to_datetime(orders["order_delivered_customer_date"], errors="coerce")
    estimated = pd.to_datetime(orders["order_estimated_delivery_date"], errors="coerce")
    delay = (delivered.dt.normalize() - estimated.dt.normalize()).dt.days

    prepared = orders[["order_id", "customer_id"]].copy()
    prepared["order_id"] = prepared["order_id"].map(_normalize_text)
    prepared["customer_id"] = prepared["customer_id"].map(_normalize_text)
    prepared["delivery_delay_days"] = delay

    people = customers[["customer_id", "customer_city", "customer_state"]].copy()
    people["customer_id"] = people["customer_id"].map(_normalize_text)
    people = people.drop_duplicates("customer_id", keep="first")
    prepared = prepared.merge(people, on="customer_id", how="left")
    prepared = prepared.drop_duplicates("order_id", keep="first")
    return prepared.drop(columns=["customer_id"])


def _categories_by_order(
    items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    catalog = products[["product_id", "product_category_name"]].merge(
        translation[["product_category_name", "product_category_name_english"]],
        on="product_category_name",
        how="left",
    )
    linked = items[["order_id", "product_id"]].merge(catalog, on="product_id", how="left")
    linked["order_id"] = linked["order_id"].map(_normalize_text)
    linked["category"] = [
        category_label(portuguese, english)
        for portuguese, english in zip(
            linked["product_category_name"],
            linked["product_category_name_english"],
            strict=True,
        )
    ]

    rows: list[dict[str, str]] = []
    if not linked.empty:
        for order_id, series in linked.groupby("order_id")["category"]:
            names = sorted({name for name in series if name})
            rows.append(
                {
                    "order_id": str(order_id),
                    "product_category": ", ".join(names) if names else MISSING,
                }
            )
    return pd.DataFrame(rows, columns=["order_id", "product_category"])


def _delivery_line(days: int) -> str:
    if days == 1:
        return "Entrega: atrasada em 1 dia"
    if days > 1:
        return f"Entrega: atrasada em {days} dias"
    return "Entrega: no prazo"


def _delay_days(value: object) -> int:
    if value is None or _is_missing(value):
        return UNKNOWN_DELAY_DAYS
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNKNOWN_DELAY_DAYS
    if math.isnan(number):
        return UNKNOWN_DELAY_DAYS
    return int(number)


def _parse_score(value: object) -> int | None:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _format_date(value: object) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if _is_missing(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _state(value: object) -> str:
    text = _normalize_text(value)
    return text.upper() if text else MISSING


def _place(value: object) -> str:
    return _normalize_text(value) or MISSING


def _normalize_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return _WHITESPACE.sub(" ", str(value).replace("\xa0", " ")).strip()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False
