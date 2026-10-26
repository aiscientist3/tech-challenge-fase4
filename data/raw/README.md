# Dados brutos

Os CSV do dataset público **Brazilian E-Commerce Public Dataset by Olist** não entram no Git. Baixe o pacote em:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data

Extraia os arquivos nesta pasta (`data/raw/`). A ingestão recusa a execução se algum arquivo ou coluna obrigatória faltar.

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

`olist_geolocation_dataset.csv` é validado e perfilado, mas não é carregado na coleção vetorial. No extrato perfilado há 1.000.163 linhas para 19.015 prefixos de CEP (média de 52,60 linhas por prefixo). Cidade e estado já vêm de clientes e vendedores.
