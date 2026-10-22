# Voz do cliente — ingestão no Chroma

Esta parte do Tech Challenge prepara as avaliações escritas do [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data) e as grava num banco vetorial [Chroma](https://www.trychroma.com/). O pipeline RAG (reescrita da pergunta, contexto, geração e recusa quando não houver evidência) fica com outra parte da equipe. O contrato da coleção, no final deste arquivo, é o que essa equipe usa para consultar os vetores.

## Arquitetura

Os CSVs passam por limpeza e enriquecimento. Cada avaliação com texto vira um documento. O modelo de embedding gera o vetor com o prefixo `passage: `. A inserção é feita pela API HTTP do Chroma, em lotes, com upsert pelo `review_id`. A coleção não usa o embedding padrão do Chroma: os vetores já chegam calculados.

```text
CSVs da Olist → limpeza e joins → documento + metadados → E5 (passage) → Chroma HTTP
```

O comando `search` só lista os vizinhos. Ele não chama modelo de linguagem e não produz resposta.

## Tecnologias

- Python 3.11+
- pandas e pyarrow, para a preparação e o parquet intermediário
- [sentence-transformers](https://www.sbert.net/) com `intfloat/multilingual-e5-small` (384 dimensões, textos em português)
- Chroma 1.5.9 em servidor Docker, cliente `HttpClient`
- pydantic-settings, para a configuração por ambiente

## Preparação da base

O foco são os comentários. A nota, a categoria, a UF e o atraso de entrega entram no texto e nos metadados para a busca e os filtros do RAG.

Arquivos usados, todos em `data/raw/`:

| Arquivo | Uso |
| --- | --- |
| `olist_order_reviews_dataset.csv` | título, comentário, nota e data |
| `olist_orders_dataset.csv` | atraso: data de entrega ao cliente menos a data estimada |
| `olist_order_items_dataset.csv` | itens do pedido |
| `olist_products_dataset.csv` | categoria em português |
| `product_category_name_translation.csv` | categoria em inglês, só quando o nome em português falta |
| `olist_customers_dataset.csv` | cidade e UF, via `customer_id` do pedido |

Limpeza:

- Avaliações com o mesmo `review_id` ficam numa só. Permanece o comentário mais longo; no empate, o título mais longo; no novo empate, a primeira linha do arquivo.
- Espaços repetidos e quebras de linha viram um espaço. O texto não é traduzido nem passado para minúsculas.
- Título ou comentário ausente vira string vazia. Se os dois ficarem vazios, a avaliação não entra no índice.
- Avaliação sem nota também fica de fora: a nota faz parte do documento e do filtro.
- Sem data de entrega ou sem data estimada, `delivery_delay_days` fica `-1`, `late_delivery` fica `false` e a linha de prazo não entra no texto.
- Categoria, cidade ou UF ausente vira `nao_informada`. Várias categorias no mesmo pedido são unidas, em ordem alfabética, separadas por vírgula e espaço.
- O atraso é a diferença de dias de calendário. Valor positivo é atraso. Zero ou negativo significa entrega no prazo; o número com sinal continua no metadado.

Cada documento é uma avaliação inteira. Os comentários da Olist são curtos, e fatiá-los pioraria a citação que o RAG vai montar.

```text
Nota: 2/5
Categoria: cama_mesa_banho
UF: SP
Entrega: atrasada em 12 dias
Título: Não recebi
Comentário: O prazo estourou e ninguém respondeu.
```

Um léxico marca aspectos no título e no comentário, só como metadado: `atraso`, `entrega` ou `entregue` → `entrega`; `defeito` ou `qualidade` → `produto`; `atendimento` ou `troca` → `atendimento`. Vários aspectos viram uma string separada por vírgula, sem espaço (`entrega,produto`).

A preparação também grava `data/processed/reviews.parquet`. A opção `--from-processed` reenvia esses documentos ao Chroma sem refazer os joins.

## Decisões

- **Chroma via API HTTP.** O servidor sobe no Docker e o cliente faz upsert. A coleção se chama `olist_reviews` e usa distância cosseno: quanto menor, mais parecido.
- **Embedding multilíngue E5.** O modelo padrão do Chroma é fraco em português. O E5 exige `passage: ` no documento e `query: ` na pergunta. Os prefixos entram só no vetor; o texto guardado na coleção continua legível.
- **Upsert pelo `review_id`.** Rodar a ingestão de novo atualiza os mesmos ids e não duplica avaliações.
- **Metadados escalares.** O Chroma não aceita lista nem valor nulo. Os campos são `review_score` (int), `product_category`, `customer_state`, `customer_city`, `delivery_delay_days` (int, `-1` se desconhecido), `late_delivery` (bool), `aspects`, `review_date` (`YYYY-MM-DD`), `review_title` e `order_id`.

## Como reproduzir

Na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
```

O Compose publica o Chroma na porta 8001 do host. A 8000 fica de fora porque costuma estar ocupada por outros serviços locais.

Baixe o dataset no Kaggle (é preciso `~/.kaggle/kaggle.json`) e copie os CSVs para `data/raw/`:

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('olistbr/brazilian-ecommerce'))"
```

O comando imprime o diretório do download. Copie para `data/raw/` os seis arquivos da tabela acima. Também dá para baixar o zip na página do dataset e extrair esses CSVs em `data/raw/`.

```bash
voc-rag ingest
voc-rag search "atraso na entrega"
```

A primeira ingestão baixa o modelo `intfloat/multilingual-e5-small` do Hugging Face. `voc-rag search` imprime id, distância cosseno, nota, categoria e o texto dos vizinhos. Não gera uma resposta.

Para reenviar o parquet já preparado:

```bash
voc-rag ingest --from-processed
```

Os testes cobrem a preparação e o formato dos metadados, sem baixar o modelo e sem subir o Chroma:

```bash
pytest
```

Variáveis de ambiente (veja `.env.example`): `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`, `EMBEDDING_MODEL`.

## Contrato para o pipeline RAG

Suba o mesmo `docker compose`. Use o cliente HTTP na mesma coleção e o mesmo modelo. Passe o vetor já calculado em `query_embeddings`. `query_texts` usaria o embedding padrão do Chroma e não acharia os vizinhos certos.

```python
import chromadb
from voc_rag.config import get_settings
from voc_rag.embeddings import embed_query

settings = get_settings()
client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
collection = client.get_collection(settings.chroma_collection)
vector = embed_query("atraso na entrega", settings.embedding_model)
found = collection.query(
    query_embeddings=[vector],
    n_results=8,
    include=["documents", "metadatas", "distances"],
)
```

Filtros que já estão nos metadados:

```python
collection.query(
    query_embeddings=[vector],
    n_results=8,
    where={"review_score": {"$lte": 2}},
    include=["documents", "metadatas", "distances"],
)
```

`late_delivery`, `product_category` e `customer_state` também servem em `where`. O campo `aspects` só casa por igualdade quando há um único rótulo (`entrega`). Com mais de um (`entrega,produto`), filtre a lista de metadados depois da busca.

A distância retornada é cosseno: menor é mais parecido. O texto em `documents` é o bloco mostrado acima, sem o prefixo `passage: `.
