# Camada de ingestão Olist

Esta pasta é a camada de extração e indexação de um projeto de RAG sobre o e-commerce brasileiro da Olist. Ela lê os CSV, monta documentos em linguagem natural e grava uma coleção persistente no Chroma. A consulta, o prompt, a geração com LLM e a interface ficam para a próxima etapa, que vai ler esta coleção.

## Visão geral

O fluxo cabe em quatro passos:

1. Validar os nove CSV e carregá-los com tipos explícitos.
2. Juntar pedidos, clientes, itens, produtos, tradução de categoria, vendedores, pagamentos e avaliações.
3. Renderizar um texto por pedido (e, por padrão, por avaliação com comentário e por resumo de categoria e de UF) e salvar `data/processed/documents.parquet`.
4. Quebrar só os textos longos, gerar embeddings e fazer upsert na coleção `olist_ecommerce` em `data/chroma/`.

O desenho das decisões está em [`docs/ingestion.md`](docs/ingestion.md). Os números do extrato estão em [`docs/data_profile.md`](docs/data_profile.md).

## Conjunto de dados

Fonte: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data).

Baixe o pacote e coloque os CSV em `data/raw/`. O passo a passo também está em [`data/raw/README.md`](data/raw/README.md). A execução para com a lista do que falta e com o link acima se algum arquivo ou coluna obrigatória não estiver lá.

Arquivos esperados:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

No extrato perfilado em 25/10/2026 há 99.441 pedidos, de 2016-09-04 a 2018-10-17, 112.650 itens, 103.886 pagamentos, 99.224 avaliações, 99.441 clientes, 3.095 vendedores, 32.951 produtos, 71 traduções de categoria e 1.000.163 linhas de geolocalização.

## Tecnologias e bibliotecas

Python 3.12.3 (o código pede 3.11+). As versões abaixo são as que `requirements.txt` fixa. O arquivo também trava as dependências transitivas, inclusive `torch==2.14.1`.

| Biblioteca | Versão | Função |
| --- | --- | --- |
| pandas | 3.1.0 | Leitura dos CSV, junções e agregações |
| pyarrow | 26.0.0 | Gravação de `documents.parquet` |
| langchain-core | 1.7.0 | Tipo `Document` |
| langchain-text-splitters | 1.1.3 | `RecursiveCharacterTextSplitter` |
| langchain-huggingface | 1.2.3 | Embeddings locais via Hugging Face |
| sentence-transformers | 5.7.0 | Modelo multilingual. A linha 5.x é a que o `langchain-huggingface` 1.2.3 aceita |
| langchain-chroma | 1.1.0 | Abre a coleção na verificação |
| chromadb | 1.5.9 | Índice vetorial persistente |
| pydantic-settings | 2.16.1 | Configuração e `.env` |
| python-dotenv | 1.2.4 | Leitura do `.env` |
| tqdm | 4.70.1 | Barra de progresso dos lotes |
| pytest | 9.1.1 | Testes com CSV mínimos em `tests/fixtures/` |

O modelo padrão é `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Para um modelo de API, defina `EMBEDDING_PROVIDER=openai` e instale `langchain-openai` à parte. A instalação padrão permanece local.

## Preparação dos dados

O perfil medido orienta a transformação. O resumo abaixo usa esses números; o detalhe coluna a coluna está em [`docs/data_profile.md`](docs/data_profile.md).

Junções, sempre à esquerda, para não perder um pedido quando faltar um cadastro:

- pedidos com clientes por `customer_id` (99.441 dos dois lados; a pessoa se repete em `customer_unique_id`, 96.096);
- itens com produtos, tradução de categoria e vendedores.

Campos derivados:

- atraso em dias de calendário entre a entrega ao cliente e a data estimada (positivo = atraso);
- valor total = soma de `payment_value`, ou preço + frete se o pedido não tiver pagamento;
- quantidade de itens = número de linhas de item.

Nulos saem da frase em vez de aparecerem como `nan`:

- 58,70% das avaliações não têm comentário; 88,34% não têm título;
- 610 produtos (1,85%) não têm categoria;
- as categorias `pc_gamer` e `portateis_cozinha_e_preparadores_de_alimentos` não têm tradução: o nome em português permanece;
- 8 pedidos `delivered` não têm data de entrega ao cliente. O metadado de atraso usa `-9999`;
- 2 pagamentos registram 0 parcelas. Essa parcela não entra no texto.

Há 9.803 pedidos com mais de um item (máximo 21), 2.961 com mais de um pagamento (máximo 29) e 547 com mais de uma avaliação (máximo 3). Tudo isso cabe em um documento por `order_id`. 775 pedidos não têm itens; o documento existe mesmo assim, sem a frase de itens.

Geolocalização: 1.000.163 linhas para 19.015 prefixos de CEP (média de 52,60 linhas por prefixo). O arquivo é validado e perfilado. A ingestão não o carrega. Cidade e UF já vêm de clientes e vendedores, e o texto do documento não usa coordenada.

## Modelagem dos documentos

A linha tabular crua é um mau documento para embedding. O pedido é a unidade que já reúne item, pagamento, prazo e avaliação.

Exemplo do texto, no formato que o código gera:

```text
Order {id} placed on {date} by a customer from {city}/{state}. Status: {status}. Items: {qty}x {category_en} ({category_pt}) sold by seller from {seller_city}/{seller_state} for R$ {price} + R$ {freight} freight. Payment: {type} in {installments} installments, total R$ {value}. Estimated delivery: {date}; delivered: {date} ({n} days late/early). Review score: {score}/5. Review title: ... Review comment: ...
```

As frases de título, comentário, data ou categoria somem quando o campo está vazio. O comentário e o nome de categoria em português ficam como estão no CSV. O nome em inglês vem do arquivo de tradução.

Por padrão também são gerados:

- `review`: uma avaliação com comentário preenchido (40.950 no extrato completo);
- `category_summary` e `state_summary`: contagem de pedidos, nota média, atraso médio e preço médio. Na categoria o preço é o do item; na UF, o valor do pedido.

`--doc-types order,review,category_summary,state_summary` escolhe o que entra naquela execução.

Metadados (somente `str`, `int`, `float` e `bool`):

| Campo | Conteúdo |
| --- | --- |
| `doc_type` | `order`, `review`, `category_summary` ou `state_summary` |
| `entity_id` | Id estável da entidade, usado também no id do chunk |
| `order_id` | Pedido, ou vazio nos resumos |
| `customer_state` | UF do cliente |
| `customer_city` | Cidade do cliente |
| `seller_state` | UF do vendedor do item mais caro |
| `product_category` | Categoria em inglês, ou o nome em português se não houver tradução |
| `order_status` | Status do pedido |
| `purchase_year` | Ano da compra, ou `0` |
| `purchase_month` | Mês da compra, ou `0` |
| `review_score` | Inteiro. No pedido, a menor nota. `-1` se não houver |
| `total_value` | Total pago, ou a média no resumo |
| `delivery_delay_days` | Dias de atraso, ou `-9999` se a entrega não puder ser comparada |
| `source_files` | CSV usados, separados por vírgula |
| `chunk_index` | `0` quando o documento permanece inteiro |

Strings ausentes ficam `""`. Um filtro posterior pode usar `{"customer_state": "SP", "review_score": {"$lte": 2}}`. A menor nota do pedido existe para esse filtro continuar achando um pedido que tenha ao menos uma avaliação ruim.

## Chunking e embeddings

Os comentários estão em português do Brasil, então o modelo precisa ser multilingual. O padrão roda na máquina, sem chave de API, com vetores normalizados e índice HNSW em cosseno. A inferência está fixada na CPU para o resultado não depender de GPU.

A maior parte dos pedidos é curta: 87,59% das linhas de item são o primeiro item do pedido. Cada documento com até 1000 caracteres permanece em um único chunk. O `RecursiveCharacterTextSplitter` (sobreposicão 150) só atua acima disso, e cada pedaço herda os metadados. Na amostra de 1.000 pedidos, os 1.495 documentos ficaram todos abaixo desse limite.

O id do chunk é o SHA-256 de `doc_type|entity_id|chunk_index`. O upsert torna a segunda execução idempotente. Se o texto encolher, até 20 ids seguintes da mesma entidade são apagados.

A coleção guarda modelo, provider, chunk, versão do dataset, maior data de compra da população ingerida, horário da ingestão e `sample_size` (`0` = tabela inteira). Modelo ou tamanho de amostra diferentes exigem `--reset`.

## Instalação e execução

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

O `make install` cria o virtualenv e instala o mesmo `requirements.txt`. Os CSV precisam já estar em `data/raw/`.

| Comando | Efeito |
| --- | --- |
| `make profile` | Regenera `docs/data_profile.md` |
| `make test` | Roda o pytest nos fixtures, sem baixar modelo e sem ler `data/raw/` |
| `make ingest-sample` | Ingere 1.000 pedidos (`--sample 1000`, semente 42) |
| `make ingest-full` | Ingere todos os pedidos |
| `make inspect` | Estatísticas da coleção e três buscas de sanidade |

Equivalentes diretos:

```bash
.venv/bin/python scripts/profile_data.py
.venv/bin/python -m pytest
.venv/bin/python scripts/ingest.py --sample 1000
.venv/bin/python scripts/ingest.py
.venv/bin/python scripts/inspect_collection.py
```

Nesta máquina, em 25/10/2026, `scripts/ingest.py --sample 1000` levou **36,6 s** de ponta a ponta, com o download do modelo na mesma execução. O embedding e o upsert dos 1.495 chunks levaram cerca de 16,6 s (três lotes, cerca de 90 chunks/s). A carga completa não foi executada. Uma conta a partir do perfil, se toda categoria nomeada aparecer em algum pedido, dá 99.441 pedidos + 40.950 avaliações com comentário + 73 categorias + 1 grupo sem categoria + 27 UFs = 140.492 documentos. Nessa taxa, só o embedding ficaria em torno de **26 minutos** com o modelo já em cache. É extrapolação, não medição. A primeira execução ainda inclui o download do modelo.

Depois da amostra, `make ingest-full` para de propósito: a coleção está com `sample_size=1000`. A carga inteira pede outra população:

```bash
.venv/bin/python scripts/ingest.py --reset
```

## Reprodutibilidade

- Python 3.12.3 no ambiente em que os pinos foram gerados.
- `requirements.txt` é o `pip freeze` desse virtualenv.
- Semente `RANDOM_SEED=42` para `--sample`.
- Modelo, lote, chunk e flags estão em [`.env.example`](.env.example).
- Comandos: `make test`, `make ingest-sample`, `make inspect`.

`make test` passou com 20 testes depois da implementação.

## Verificação da coleção

Saída padrão de `scripts/inspect_collection.py` sobre a amostra de 1.000 pedidos (os logs do carregamento do modelo vão para o stderr e não entram aqui). O `score` é a distância do Chroma no espaço cosseno: quanto menor, mais próximo. `dataset_as_of` é a compra mais recente **dessa amostra** (`2018-10-01T15:30:09`). No extrato completo a última compra é `2018-10-17 17:30:18`.

```text
collection: olist_ecommerce
persist_directory: /home/porto/Documents/estudos/tech-challenge-fase4/data/chroma
count: 1495
metadata:
  chunk_overlap: 150
  chunk_size: 1000
  dataset_as_of: 2018-10-01T15:30:09
  dataset_version: brazilian-ecommerce-olist
  distance_space: cosine
  embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  embedding_provider: huggingface
  hnsw:space: cosine
  ingested_at: 2026-10-25T16:05:13+00:00
  sample_size: 1000
documents per doc_type:
  order: 1000
  review: 418
  category_summary: 51
  state_summary: 26

query: entrega atrasada
  1. score=0.4271 doc_type=order order_id=0a472c47d928bf4cd8b6a7a30345eeb7 review_score=3
     Order 0a472c47d928bf4cd8b6a7a30345eeb7 placed on 2018-08-30 by a customer from sumare/SP. Status: canceled. Payment: voucher in 1 installment, total R$ 490.99. Estimated delivery: 2018-10-02. Review score: 3/5. Review title: Produto com ...
  2. score=0.4341 doc_type=state_summary order_id= review_score=5
     Customer state RN: 4 orders. Average review score: 4.50/5. Average delivery delay: -23.2 days. Average order value: R$ 302.66.
  3. score=0.4382 doc_type=state_summary order_id= review_score=4
     Customer state AL: 2 orders. Average review score: 4.00/5. Average delivery delay: 0.5 days. Average order value: R$ 167.27.

query: produto veio com defeito
  1. score=0.4775 doc_type=review order_id=e62c3201015050463bc2e3b6599a6bc5 review_score=1
     Review score: 1/5 on order e62c3201015050463bc2e3b6599a6bc5, placed on 2018-02-27 by a customer from rio de janeiro/RJ. Items: baby (bebes). Seller from sao paulo/SP. Review comment: Comprei o produto e até o momento não recebi e também ...
  2. score=0.4867 doc_type=review order_id=5075b9e252eb0b3f1350f282c94a82b0 review_score=1
     Review score: 1/5 on order 5075b9e252eb0b3f1350f282c94a82b0, placed on 2017-11-24 by a customer from mogi das cruzes/SP. Items: bed_bath_table (cama_mesa_banho). Seller from ibitinga/SP. Review comment: O produto veio errado não tive nen...
  3. score=0.5234 doc_type=review order_id=141d3bf5aa7128f40bf2a60cef620b93 review_score=1
     Review score: 1/5 on order 141d3bf5aa7128f40bf2a60cef620b93, placed on 2018-05-02 by a customer from sao paulo/SP. Items: computers_accessories (informatica_acessorios). Seller from maringa/PR. Review title: zero estrela. Review comment:...

query: pagamento com boleto
  1. score=0.3852 doc_type=order order_id=65c599e3998c4243f4e62affbf5bcdd7 review_score=5
     Order 65c599e3998c4243f4e62affbf5bcdd7 placed on 2018-08-19 by a customer from bambui/MG. Status: delivered. Items: 1x sports_leisure (esporte_lazer) sold by seller from santo andre/SP for R$ 112.49 + R$ 18.89 freight. Payment: credit_ca...
  2. score=0.3927 doc_type=order order_id=f9d6e4f177c33d17573279292f1473e4 review_score=4
     Order f9d6e4f177c33d17573279292f1473e4 placed on 2017-08-30 by a customer from rio de janeiro/RJ. Status: delivered. Items: 1x sports_leisure (esporte_lazer) sold by seller from ribeirao preto/SP for R$ 133.00 + R$ 16.69 freight. Payment...
  3. score=0.3932 doc_type=order order_id=0a472c47d928bf4cd8b6a7a30345eeb7 review_score=3
     Order 0a472c47d928bf4cd8b6a7a30345eeb7 placed on 2018-08-30 by a customer from sumare/SP. Status: canceled. Payment: voucher in 1 installment, total R$ 490.99. Estimated delivery: 2018-10-02. Review score: 3/5. Review title: Produto com ...
```

## Limitações e próximos passos

A próxima camada deve recuperar desta coleção e gerar a resposta. Este repositório para na indexação. `inspect_collection.py` só mostra que a coleção responde a uma busca por similaridade.

O que a verificação acima já mostra, e que a recuperação vai precisar tratar:

- A consulta `produto veio com defeito` devolveu avaliações com nota 1 e comentário em português, que é o comportamento útil desse tipo de documento.
- A consulta `entrega atrasada` colocou resumos de UF na 2ª e 3ª posições. Esses textos repetem a frase "Average delivery delay", então competem com pedidos individuais. Na recuperação, filtrar `doc_type` separa os resumos das reclamações.
- A consulta `pagamento com boleto` devolveu pedidos pagos com cartão e voucher. O tipo de pagamento está só no texto, e as frases são quase iguais. Um filtro por tipo de pagamento ainda não existe no metadado.
- Os agregados e a data `dataset_as_of` da coleção atual descrevem os 1.000 pedidos sorteados, não o extrato inteiro.
- A geolocalização não está no índice.
- O `torch` pinado é o wheel resolvido pelo pip nesta máquina e traz bibliotecas CUDA. O código pede `device=cpu`. A reinstalação baixa esse conjunto grande de pacotes.

Para indexar o extrato inteiro por cima da amostra, use `--reset`.
