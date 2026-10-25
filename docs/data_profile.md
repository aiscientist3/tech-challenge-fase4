# Perfil dos dados Olist

Este arquivo foi gerado por `scripts/profile_data.py` a partir dos CSV em `data/raw/`.
Os números vêm dessa execução. Para regenerar: `make profile`.

Fonte: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data).

Gerado em: 25/10/2026 13:03:46.

## Relações medidas

Estas contagens justificam agregar itens, pagamentos e avaliações no pedido, em vez de indexar cada linha tabular.

- Pedidos: 99.441. Intervalo de `order_purchase_timestamp`: 2016-09-04 21:15:19 até 2018-10-17 17:30:18.
- Pedidos com mais de um item: 9.803 (máximo de 21 itens em um pedido).
- Pedidos sem nenhuma linha em itens: 775.
- Pedidos com mais de um pagamento: 2.961 (máximo de 29).
- Pedidos com mais de uma avaliação: 547 (máximo de 3).
- Avaliações com comentário preenchido: 40.950 de 99.224 (41,27%).
- Pedidos com status `delivered` e data de entrega ao cliente vazia: 8 de 96.478.
- `customer_id` distintos em pedidos: 99.441; em clientes: 99.441; `customer_unique_id` distintos: 96.096.
- `order_id` duplicado em pedidos: 0.
- Produtos sem categoria: 610.
- Categorias de produto sem tradução: 2 (pc_gamer, portateis_cozinha_e_preparadores_de_alimentos).
- Geolocalização: 1.000.163 linhas para 19.015 prefixos de CEP (52,60 linhas por prefixo, em média).


## `olist_orders_dataset.csv`

- Linhas: 99.441
- Colunas: 8

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| order_id | string | 0,00% |
| customer_id | string | 0,00% |
| order_status | string | 0,00% |
| order_purchase_timestamp | datetime64[us] | 0,00% |
| order_approved_at | datetime64[us] | 0,16% |
| order_delivered_carrier_date | datetime64[us] | 1,79% |
| order_delivered_customer_date | datetime64[us] | 2,98% |
| order_estimated_delivery_date | datetime64[us] | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| order_id | 99.441 |
| customer_id | 99.441 |
| order_status | 8 |

Distribuição de `order_status`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| delivered | 96.478 | 97,02% |
| shipped | 1.107 | 1,11% |
| canceled | 625 | 0,63% |
| unavailable | 609 | 0,61% |
| invoiced | 314 | 0,32% |
| processing | 301 | 0,30% |
| created | 5 | 0,01% |
| approved | 2 | 0,00% |

## `olist_order_items_dataset.csv`

- Linhas: 112.650
- Colunas: 7

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| order_id | string | 0,00% |
| order_item_id | Int64 | 0,00% |
| product_id | string | 0,00% |
| seller_id | string | 0,00% |
| shipping_limit_date | datetime64[us] | 0,00% |
| price | float64 | 0,00% |
| freight_value | float64 | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| order_id | 98.666 |
| product_id | 32.951 |
| seller_id | 3.095 |

Distribuição de `order_item_id`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| 1 | 98.666 | 87,59% |
| 2 | 9.803 | 8,70% |
| 3 | 2.287 | 2,03% |
| 4 | 965 | 0,86% |
| 5 | 460 | 0,41% |
| 6 | 256 | 0,23% |
| 7 | 58 | 0,05% |
| 8 | 36 | 0,03% |
| 9 | 28 | 0,02% |
| 10 | 25 | 0,02% |
| 11 | 17 | 0,02% |
| 12 | 13 | 0,01% |
| 13 | 8 | 0,01% |
| 14 | 7 | 0,01% |
| 15 | 5 | 0,00% |
| 16 | 3 | 0,00% |
| 17 | 3 | 0,00% |
| 18 | 3 | 0,00% |
| 19 | 3 | 0,00% |
| 20 | 3 | 0,00% |
| 21 | 1 | 0,00% |

## `olist_order_payments_dataset.csv`

- Linhas: 103.886
- Colunas: 5

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| order_id | string | 0,00% |
| payment_sequential | Int64 | 0,00% |
| payment_type | string | 0,00% |
| payment_installments | Int64 | 0,00% |
| payment_value | float64 | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| order_id | 99.440 |
| payment_type | 5 |

Distribuição de `payment_sequential`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| 1 | 99.360 | 95,64% |
| 2 | 3.039 | 2,93% |
| 3 | 581 | 0,56% |
| 4 | 278 | 0,27% |
| 5 | 170 | 0,16% |
| 6 | 118 | 0,11% |
| 7 | 82 | 0,08% |
| 8 | 54 | 0,05% |
| 9 | 43 | 0,04% |
| 10 | 34 | 0,03% |
| 11 | 29 | 0,03% |
| 12 | 21 | 0,02% |
| 13 | 13 | 0,01% |
| 14 | 10 | 0,01% |
| 15 | 8 | 0,01% |
| 17 | 6 | 0,01% |
| 19 | 6 | 0,01% |
| 16 | 6 | 0,01% |
| 18 | 6 | 0,01% |
| 21 | 4 | 0,00% |
| 20 | 4 | 0,00% |
| 22 | 3 | 0,00% |
| 25 | 2 | 0,00% |
| 26 | 2 | 0,00% |
| 24 | 2 | 0,00% |
| 23 | 2 | 0,00% |
| 27 | 1 | 0,00% |
| 29 | 1 | 0,00% |
| 28 | 1 | 0,00% |

Distribuição de `payment_type`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| credit_card | 76.795 | 73,92% |
| boleto | 19.784 | 19,04% |
| voucher | 5.775 | 5,56% |
| debit_card | 1.529 | 1,47% |
| not_defined | 3 | 0,00% |

Distribuição de `payment_installments`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| 1 | 52.546 | 50,58% |
| 2 | 12.413 | 11,95% |
| 3 | 10.461 | 10,07% |
| 4 | 7.098 | 6,83% |
| 10 | 5.328 | 5,13% |
| 5 | 5.239 | 5,04% |
| 8 | 4.268 | 4,11% |
| 6 | 3.920 | 3,77% |
| 7 | 1.626 | 1,57% |
| 9 | 644 | 0,62% |
| 12 | 133 | 0,13% |
| 15 | 74 | 0,07% |
| 18 | 27 | 0,03% |
| 11 | 23 | 0,02% |
| 24 | 18 | 0,02% |
| 20 | 17 | 0,02% |
| 13 | 16 | 0,02% |
| 14 | 15 | 0,01% |
| 17 | 8 | 0,01% |
| 16 | 5 | 0,00% |
| 21 | 3 | 0,00% |
| 0 | 2 | 0,00% |
| 22 | 1 | 0,00% |
| 23 | 1 | 0,00% |

## `olist_order_reviews_dataset.csv`

- Linhas: 99.224
- Colunas: 7

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| review_id | string | 0,00% |
| order_id | string | 0,00% |
| review_score | Int64 | 0,00% |
| review_comment_title | string | 88,34% |
| review_comment_message | string | 58,70% |
| review_creation_date | datetime64[us] | 0,00% |
| review_answer_timestamp | datetime64[us] | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| review_id | 98.410 |
| order_id | 98.673 |
| review_score | 5 |

Textos preenchidos (depois de remover vazio e tokens de nulo):

| Campo | Preenchidos | Preenchidos (%) |
| --- | --- | --- |
| review_comment_title | 11.566 | 11,66% |
| review_comment_message | 40.950 | 41,27% |

Distribuição de `review_score`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| 5 | 57.328 | 57,78% |
| 4 | 19.142 | 19,29% |
| 1 | 11.424 | 11,51% |
| 3 | 8.179 | 8,24% |
| 2 | 3.151 | 3,18% |

## `olist_customers_dataset.csv`

- Linhas: 99.441
- Colunas: 5

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| customer_id | string | 0,00% |
| customer_unique_id | string | 0,00% |
| customer_zip_code_prefix | string | 0,00% |
| customer_city | string | 0,00% |
| customer_state | string | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| customer_id | 99.441 |
| customer_unique_id | 96.096 |
| customer_city | 4.119 |
| customer_state | 27 |

Distribuição de `customer_state`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| SP | 41.746 | 41,98% |
| RJ | 12.852 | 12,92% |
| MG | 11.635 | 11,70% |
| RS | 5.466 | 5,50% |
| PR | 5.045 | 5,07% |
| SC | 3.637 | 3,66% |
| BA | 3.380 | 3,40% |
| DF | 2.140 | 2,15% |
| ES | 2.033 | 2,04% |
| GO | 2.020 | 2,03% |
| PE | 1.652 | 1,66% |
| CE | 1.336 | 1,34% |
| PA | 975 | 0,98% |
| MT | 907 | 0,91% |
| MA | 747 | 0,75% |
| MS | 715 | 0,72% |
| PB | 536 | 0,54% |
| PI | 495 | 0,50% |
| RN | 485 | 0,49% |
| AL | 413 | 0,42% |
| SE | 350 | 0,35% |
| TO | 280 | 0,28% |
| RO | 253 | 0,25% |
| AM | 148 | 0,15% |
| AC | 81 | 0,08% |
| AP | 68 | 0,07% |
| RR | 46 | 0,05% |

## `olist_sellers_dataset.csv`

- Linhas: 3.095
- Colunas: 4

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| seller_id | string | 0,00% |
| seller_zip_code_prefix | string | 0,00% |
| seller_city | string | 0,00% |
| seller_state | string | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| seller_id | 3.095 |
| seller_city | 611 |
| seller_state | 23 |

Distribuição de `seller_state`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| SP | 1.849 | 59,74% |
| PR | 349 | 11,28% |
| MG | 244 | 7,88% |
| SC | 190 | 6,14% |
| RJ | 171 | 5,53% |
| RS | 129 | 4,17% |
| GO | 40 | 1,29% |
| DF | 30 | 0,97% |
| ES | 23 | 0,74% |
| BA | 19 | 0,61% |
| CE | 13 | 0,42% |
| PE | 9 | 0,29% |
| PB | 6 | 0,19% |
| RN | 5 | 0,16% |
| MS | 5 | 0,16% |
| MT | 4 | 0,13% |
| RO | 2 | 0,06% |
| SE | 2 | 0,06% |
| AC | 1 | 0,03% |
| PI | 1 | 0,03% |
| MA | 1 | 0,03% |
| AM | 1 | 0,03% |
| PA | 1 | 0,03% |

## `olist_products_dataset.csv`

- Linhas: 32.951
- Colunas: 9

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| product_id | string | 0,00% |
| product_category_name | string | 1,85% |
| product_name_lenght | float64 | 1,85% |
| product_description_lenght | float64 | 1,85% |
| product_photos_qty | float64 | 1,85% |
| product_weight_g | float64 | 0,01% |
| product_length_cm | float64 | 0,01% |
| product_height_cm | float64 | 0,01% |
| product_width_cm | float64 | 0,01% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| product_id | 32.951 |
| product_category_name | 73 |

Distribuição de `product_photos_qty`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| 1.0 | 16.489 | 50,04% |
| 2.0 | 6.263 | 19,01% |
| 3.0 | 3.860 | 11,71% |
| 4.0 | 2.428 | 7,37% |
| 5.0 | 1.484 | 4,50% |
| 6.0 | 968 | 2,94% |
| (nulo) | 610 | 1,85% |
| 7.0 | 343 | 1,04% |
| 8.0 | 192 | 0,58% |
| 9.0 | 105 | 0,32% |
| 10.0 | 95 | 0,29% |
| 11.0 | 46 | 0,14% |
| 12.0 | 35 | 0,11% |
| 13.0 | 9 | 0,03% |
| 15.0 | 8 | 0,02% |
| 17.0 | 7 | 0,02% |
| 14.0 | 5 | 0,02% |
| 18.0 | 2 | 0,01% |
| 20.0 | 1 | 0,00% |
| 19.0 | 1 | 0,00% |

## `product_category_name_translation.csv`

- Linhas: 71
- Colunas: 2

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| product_category_name | string | 0,00% |
| product_category_name_english | string | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| product_category_name | 71 |
| product_category_name_english | 71 |

## `olist_geolocation_dataset.csv`

- Linhas: 1.000.163
- Colunas: 5

| Coluna | Tipo | Nulos (%) |
| --- | --- | --- |
| geolocation_zip_code_prefix | string | 0,00% |
| geolocation_lat | float64 | 0,00% |
| geolocation_lng | float64 | 0,00% |
| geolocation_city | string | 0,00% |
| geolocation_state | string | 0,00% |

Cardinalidade das colunas-chave:

| Coluna | Valores distintos |
| --- | --- |
| geolocation_zip_code_prefix | 19.015 |
| geolocation_city | 8.011 |
| geolocation_state | 27 |

Distribuição de `geolocation_state`:

| Valor | Linhas | Linhas (%) |
| --- | --- | --- |
| SP | 404.268 | 40,42% |
| MG | 126.336 | 12,63% |
| RJ | 121.169 | 12,11% |
| RS | 61.851 | 6,18% |
| PR | 57.859 | 5,78% |
| SC | 38.328 | 3,83% |
| BA | 36.045 | 3,60% |
| GO | 20.139 | 2,01% |
| ES | 16.748 | 1,67% |
| PE | 16.432 | 1,64% |
| DF | 12.986 | 1,30% |
| MT | 12.031 | 1,20% |
| CE | 11.674 | 1,17% |
| PA | 10.853 | 1,09% |
| MS | 10.431 | 1,04% |
| MA | 7.853 | 0,79% |
| PB | 5.538 | 0,55% |
| RN | 5.041 | 0,50% |
| PI | 4.549 | 0,45% |
| AL | 4.183 | 0,42% |
| TO | 3.576 | 0,36% |
| SE | 3.563 | 0,36% |
| RO | 3.478 | 0,35% |
| AM | 2.432 | 0,24% |
| AC | 1.301 | 0,13% |
| AP | 853 | 0,09% |
| RR | 646 | 0,06% |
