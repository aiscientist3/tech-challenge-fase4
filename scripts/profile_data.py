"""Profile every Olist CSV and write docs/data_profile.md.

The report is generated from the files on disk. Re-run it after replacing the
raw extract so the documented cardinalities stay tied to that extract.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rag.config import KAGGLE_DATASET_URL, Settings, configure_logging
from rag.ingestion.extract import (
    EXPECTED_COLUMNS,
    TABLE_FILES,
    DatasetValidationError,
    load_raw_tables,
)

KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "olist_orders_dataset.csv": ("order_id", "customer_id", "order_status"),
    "olist_order_items_dataset.csv": ("order_id", "product_id", "seller_id"),
    "olist_order_payments_dataset.csv": ("order_id", "payment_type"),
    "olist_order_reviews_dataset.csv": ("review_id", "order_id", "review_score"),
    "olist_customers_dataset.csv": (
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    ),
    "olist_sellers_dataset.csv": ("seller_id", "seller_city", "seller_state"),
    "olist_products_dataset.csv": ("product_id", "product_category_name"),
    "olist_geolocation_dataset.csv": (
        "geolocation_zip_code_prefix",
        "geolocation_city",
        "geolocation_state",
    ),
    "product_category_name_translation.csv": (
        "product_category_name",
        "product_category_name_english",
    ),
}

DISTRIBUTION_LIMIT = 40
_MISSING_TOKENS = {"", "nan", "none", "<na>", "nat", "null"}


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    raw_dir = settings.resolve_path(settings.raw_dir)
    try:
        tables = load_raw_tables(raw_dir, include_geolocation=True)
    except DatasetValidationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    report = _render_report(tables)
    destination = ROOT / "docs" / "data_profile.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote profile to {destination}", file=sys.stderr)


def _render_report(tables: dict[str, pd.DataFrame]) -> str:
    generated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    sections = [
        "# Perfil dos dados Olist",
        "",
        "Este arquivo foi gerado por `scripts/profile_data.py` a partir dos CSV em `data/raw/`.",
        "Os números vêm dessa execução. Para regenerar: `make profile`.",
        "",
        f"Fonte: [Brazilian E-Commerce Public Dataset by Olist]({KAGGLE_DATASET_URL}).",
        "",
        f"Gerado em: {generated}.",
        "",
        _relations_section(tables),
        "",
    ]
    for key, filename in TABLE_FILES.items():
        sections.append(_file_section(filename, tables[key]))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def _file_section(filename: str, frame: pd.DataFrame) -> str:
    lines = [
        f"## `{filename}`",
        "",
        f"- Linhas: {_fmt_int(len(frame))}",
        f"- Colunas: {len(frame.columns)}",
        "",
        _markdown_table(
            ["Coluna", "Tipo", "Nulos (%)"],
            [
                [
                    str(column),
                    str(frame[column].dtype),
                    _fmt_pct(_null_pct(frame[column])),
                ]
                for column in frame.columns
            ],
        ),
        "",
        "Cardinalidade das colunas-chave:",
        "",
        _markdown_table(
            ["Coluna", "Valores distintos"],
            [
                [column, _fmt_int(int(frame[column].nunique(dropna=True)))]
                for column in KEY_COLUMNS[filename]
            ],
        ),
    ]
    if filename == "olist_order_reviews_dataset.csv":
        lines.extend(["", _review_fill_section(frame)])
    lines.extend(_distributions(frame))
    expected = list(EXPECTED_COLUMNS[filename])
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        lines.extend(["", f"Colunas ausentes em relação ao contrato: {', '.join(missing)}."])
    return "\n".join(lines)


def _review_fill_section(frame: pd.DataFrame) -> str:
    title_filled = int(_filled(frame["review_comment_title"]).sum())
    message_filled = int(_filled(frame["review_comment_message"]).sum())
    return "\n".join(
        [
            "Textos preenchidos (depois de remover vazio e tokens de nulo):",
            "",
            _markdown_table(
                ["Campo", "Preenchidos", "Preenchidos (%)"],
                [
                    [
                        "review_comment_title",
                        _fmt_int(title_filled),
                        _fmt_pct(_share(title_filled, len(frame))),
                    ],
                    [
                        "review_comment_message",
                        _fmt_int(message_filled),
                        _fmt_pct(_share(message_filled, len(frame))),
                    ],
                ],
            ),
        ]
    )


def _distributions(frame: pd.DataFrame) -> list[str]:
    blocks: list[str] = []
    for column in frame.columns:
        unique = int(frame[column].nunique(dropna=True))
        if unique == 0 or unique > DISTRIBUTION_LIMIT:
            continue
        counts = frame[column].astype("string").fillna("(nulo)").value_counts(dropna=False)
        rows = [
            [str(value), _fmt_int(int(count)), _fmt_pct(_share(int(count), len(frame)))]
            for value, count in counts.items()
        ]
        blocks.extend(
            [
                "",
                f"Distribuição de `{column}`:",
                "",
                _markdown_table(["Valor", "Linhas", "Linhas (%)"], rows),
            ]
        )
    return blocks


def _relations_section(tables: dict[str, pd.DataFrame]) -> str:
    orders = tables["orders"]
    items = tables["order_items"]
    payments = tables["payments"]
    reviews = tables["reviews"]
    products = tables["products"]
    translation = tables["category_translation"]
    customers = tables["customers"]
    geo = tables["geolocation"]

    item_sizes = items.groupby("order_id").size()
    payment_sizes = payments.groupby("order_id").size()
    review_sizes = reviews.groupby("order_id").size()
    delivered = orders["order_status"].eq("delivered")
    delivered_missing_date = int(orders.loc[delivered, "order_delivered_customer_date"].isna().sum())
    comments = int(_filled(reviews["review_comment_message"]).sum())
    product_categories = set(_present_strings(products["product_category_name"]))
    translated = set(_present_strings(translation["product_category_name"]))
    missing_translation = sorted(product_categories - translated)
    purchase = orders["order_purchase_timestamp"]
    prefix_count = int(geo["geolocation_zip_code_prefix"].nunique(dropna=True))
    lines = [
        "## Relações medidas",
        "",
        "Estas contagens justificam agregar itens, pagamentos e avaliações no pedido, "
        "em vez de indexar cada linha tabular.",
        "",
        f"- Pedidos: {_fmt_int(len(orders))}. Intervalo de `order_purchase_timestamp`: "
        f"{_fmt_ts(purchase.min())} até {_fmt_ts(purchase.max())}.",
        f"- Pedidos com mais de um item: {_fmt_int(int((item_sizes > 1).sum()))} "
        f"(máximo de {_fmt_int(int(item_sizes.max()))} itens em um pedido).",
        f"- Pedidos sem nenhuma linha em itens: {_fmt_int(int((~orders['order_id'].isin(items['order_id'])).sum()))}.",
        f"- Pedidos com mais de um pagamento: {_fmt_int(int((payment_sizes > 1).sum()))} "
        f"(máximo de {_fmt_int(int(payment_sizes.max()))}).",
        f"- Pedidos com mais de uma avaliação: {_fmt_int(int((review_sizes > 1).sum()))} "
        f"(máximo de {_fmt_int(int(review_sizes.max()))}).",
        f"- Avaliações com comentário preenchido: {_fmt_int(comments)} de {_fmt_int(len(reviews))} "
        f"({_fmt_pct(_share(comments, len(reviews)))}).",
        f"- Pedidos com status `delivered` e data de entrega ao cliente vazia: "
        f"{_fmt_int(delivered_missing_date)} de {_fmt_int(int(delivered.sum()))}.",
        f"- `customer_id` distintos em pedidos: {_fmt_int(int(orders['customer_id'].nunique()))}; "
        f"em clientes: {_fmt_int(int(customers['customer_id'].nunique()))}; "
        f"`customer_unique_id` distintos: {_fmt_int(int(customers['customer_unique_id'].nunique()))}.",
        f"- `order_id` duplicado em pedidos: {_fmt_int(int(orders['order_id'].duplicated().sum()))}.",
        f"- Produtos sem categoria: {_fmt_int(int(products['product_category_name'].isna().sum()))}.",
        f"- Categorias de produto sem tradução: {_fmt_int(len(missing_translation))}"
        + (f" ({', '.join(missing_translation)})." if missing_translation else "."),
        f"- Geolocalização: {_fmt_int(len(geo))} linhas para {_fmt_int(prefix_count)} prefixos de CEP "
        f"({_fmt_number(len(geo) / prefix_count if prefix_count else 0)} linhas por prefixo, em média).",
        "",
    ]
    return "\n".join(lines)


def _filled(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.notna() & ~text.str.lower().isin(_MISSING_TOKENS)


def _present_strings(series: pd.Series) -> list[str]:
    values: list[str] = []
    for value in series.astype("string").tolist():
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text.lower() in _MISSING_TOKENS:
            continue
        values.append(text)
    return values


def _null_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean() * 100)


def _share(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100


def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}".replace(".", ",") + "%"


def _fmt_number(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _fmt_ts(value: object) -> str:
    if value is None or pd.isna(value):
        return "(vazio)"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
