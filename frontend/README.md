# Frontend do chat

Interface de conversa para o RAG de Voice of Customer da Olist. O usuário faz uma pergunta em linguagem natural, pode filtrar por metadados e recebe a resposta com as avaliações usadas como evidência. As respostas vêm da rota `POST /chat`, que o backend ainda vai implementar. Enquanto ela não existe, o modo mock responde no lugar dela.

## Stack

- [Vite](https://vite.dev/) com TypeScript puro, sem framework
- [Tailwind CSS v4](https://tailwindcss.com/) pelo plugin `@tailwindcss/vite`
- [daisyUI 5](https://daisyui.com/) para os componentes (chat, collapse, alert, badge)
- `@tailwindcss/typography` para o texto da resposta
- Montserrat (`@fontsource-variable/montserrat`, servida localmente)

O visual é um tema próprio do daisyUI (`voc`, em `src/styles.css`): fundo preto, destaque magenta `#ED145B`, cantos retos e rótulos em caixa-alta com espaçamento largo, inspirado na linguagem visual do site da FIAP. Logo e nome da FIAP não são usados.

## Como rodar

Precisa de Node 20 ou mais recente (testado com o 24).

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Abra http://localhost:5173.

| Variável | Efeito |
| --- | --- |
| `VITE_USE_MOCK` | `true`: respostas de `src/mock.ts`, sem backend. `false`: chama `POST /chat` |
| `API_PROXY_TARGET` | Para onde o Vite repassa `/chat` no modo de desenvolvimento (padrão `http://localhost:8000`) |

Como o Vite faz o proxy, o backend não precisa liberar CORS durante o desenvolvimento.

Outros comandos: `npm run build` faz a checagem de tipos e gera `dist/`, e `npm run typecheck` só checa os tipos.

## Estrutura

```
frontend/
├── index.html          # layout: cabeçalho, filtros, chat
└── src/
    ├── main.ts         # eventos, estado da conversa, leitura dos filtros
    ├── api.ts          # contrato de /chat (tipos) e cliente fetch
    ├── mock.ts         # backend falso para desenvolvimento
    ├── render.ts       # mensagens e cards de evidência
    ├── categories.ts   # categorias para o filtro (do arquivo de tradução)
    └── styles.css      # Tailwind + daisyUI
```

## Contrato da rota `/chat`

Os tipos estão em [`src/api.ts`](src/api.ts). Os nomes dos filtros e das fontes seguem os metadados que a ingestão grava no Chroma (veja o [README da raiz](../README.md)).

Requisição:

```json
{
  "question": "O que os clientes relatam sobre a entrega?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "filters": {
    "doc_type": "review",
    "review_score_min": 1,
    "review_score_max": 2,
    "product_category": "bed_bath_table",
    "purchase_year": 2018
  }
}
```

- `history` leva até os últimos 10 turnos, sem a pergunta atual.
- Os campos de `filters` são todos opcionais. Um campo ausente significa que não há filtro.
- `product_category` usa o nome em inglês, ou o português quando não há tradução, igual ao metadado.

Resposta `200`:

```json
{
  "answer": "As reclamações se concentram em atraso... [1][2]",
  "insufficient_evidence": false,
  "sources": [
    {
      "id": "chunk-id",
      "doc_type": "review",
      "order_id": "e62c3201015050463bc2e3b6599a6bc5",
      "review_score": 1,
      "product_category": "baby",
      "purchase_date": "2018-02-27",
      "score": 0.4775,
      "excerpt": "Comprei o produto e até o momento não recebi..."
    }
  ]
}
```

- `sources` vem na ordem em que a interface numera as evidências. Assim, `[1]` na resposta aponta para o primeiro item.
- `score` é a distância de cosseno do Chroma: quanto menor, mais próximo.
- `review_score` vale `-1` quando não há nota (a mesma sentinela da ingestão).
- `purchase_date` aceita `YYYY-MM-DD`, `YYYY-MM` ou `""`.
- `insufficient_evidence: true` exibe o aviso de evidência insuficiente. Use para pergunta fora do escopo ou recuperação fraca. `sources` pode vir vazia nesse caso.

Qualquer status diferente de 2xx, timeout (60 s) ou corpo fora desse formato aparece como erro no chat, e a pergunta volta para a caixa de texto.

## Modo mock

`src/mock.ts` tem nove avaliações **fictícias**, marcadas como "Exemplo fictício", separadas em três temas: entrega, qualidade e elogios. O tema é escolhido por palavra-chave na pergunta e os filtros são aplicados de verdade. Uma pergunta fora do escopo, ou um filtro que deixa menos de duas fontes, devolve `insufficient_evidence: true`. Com o mock ativo, o cabeçalho mostra o selo "modo mock".
