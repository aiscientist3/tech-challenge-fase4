// Fake backend used while POST /chat does not exist (VITE_USE_MOCK=true).
// The reviews below are invented examples, not records from the dataset.
// Filters are applied for real so the filter panel can be exercised.

import type { ChatFilters, ChatRequest, ChatResponse, Source } from "./api.ts";

type Topic = "delivery" | "quality" | "praise";

interface MockSource extends Source {
  topic: Topic;
}

const POOL: MockSource[] = [
  {
    id: "mock-1", topic: "delivery", doc_type: "review", order_id: "mock-order-0001",
    review_score: 1, product_category: "bed_bath_table", purchase_date: "2018-03-12", score: 0.38,
    excerpt: "Exemplo fictício: o prazo passou há duas semanas e o pedido ainda não chegou. Ninguém responde no atendimento.",
  },
  {
    id: "mock-2", topic: "delivery", doc_type: "review", order_id: "mock-order-0002",
    review_score: 2, product_category: "health_beauty", purchase_date: "2017-11-28", score: 0.41,
    excerpt: "Exemplo fictício: recebi só um dos dois produtos comprados. O outro veio dias depois, separado.",
  },
  {
    id: "mock-3", topic: "delivery", doc_type: "order", order_id: "mock-order-0003",
    review_score: 1, product_category: "furniture_decor", purchase_date: "2018-02-05", score: 0.44,
    excerpt: "Exemplo fictício: pedido entregue 12 dias após a data estimada. Comentário: entrega muito atrasada.",
  },
  {
    id: "mock-4", topic: "quality", doc_type: "review", order_id: "mock-order-0004",
    review_score: 1, product_category: "computers_accessories", purchase_date: "2018-05-02", score: 0.39,
    excerpt: "Exemplo fictício: o produto veio com defeito e parou de funcionar no segundo dia de uso.",
  },
  {
    id: "mock-5", topic: "quality", doc_type: "review", order_id: "mock-order-0005",
    review_score: 2, product_category: "watches_gifts", purchase_date: "2017-08-19", score: 0.45,
    excerpt: "Exemplo fictício: a qualidade é bem inferior à das fotos do anúncio. Material frágil.",
  },
  {
    id: "mock-6", topic: "quality", doc_type: "review", order_id: "mock-order-0006",
    review_score: 1, product_category: "bed_bath_table", purchase_date: "2018-01-21", score: 0.47,
    excerpt: "Exemplo fictício: veio a cor errada e o tecido é diferente do descrito.",
  },
  {
    id: "mock-7", topic: "praise", doc_type: "review", order_id: "mock-order-0007",
    review_score: 5, product_category: "telephony", purchase_date: "2017-11-24", score: 0.36,
    excerpt: "Exemplo fictício: chegou antes do prazo, bem embalado e exatamente como no anúncio. Recomendo.",
  },
  {
    id: "mock-8", topic: "praise", doc_type: "review", order_id: "mock-order-0008",
    review_score: 5, product_category: "sports_leisure", purchase_date: "2018-06-30", score: 0.4,
    excerpt: "Exemplo fictício: ótimo custo-benefício e entrega rápida. Vendedor atencioso.",
  },
  {
    id: "mock-9", topic: "praise", doc_type: "review", order_id: "mock-order-0009",
    review_score: 4, product_category: "health_beauty", purchase_date: "2016-10-08", score: 0.46,
    excerpt: "Exemplo fictício: produto original e de boa qualidade, só a embalagem veio um pouco amassada.",
  },
];

const TOPIC_KEYWORDS: Record<Topic, string[]> = {
  delivery: ["entrega", "prazo", "atras", "chegou", "frete", "demora"],
  quality: ["qualidade", "defeito", "produto", "quebr", "errad"],
  praise: ["elogi", "positiv", "valoriz", "gost", "satisfeit", "bom", "melhor"],
};

// Questions that are in scope but not tied to one topic get every topic.
const GENERIC_KEYWORDS = ["problema", "reclama", "insatisf", "avalia", "cliente", "compra", "padr"];

const ANSWERS: Record<Topic, string> = {
  delivery:
    "As reclamações sobre entrega se concentram em atraso em relação ao prazo estimado e em pedidos entregues de forma parcial.",
  quality:
    "Os problemas de qualidade mais citados são defeito logo no início do uso, produto diferente do anúncio e item enviado errado.",
  praise:
    "Os clientes elogiam principalmente a entrega antes do prazo, o custo-benefício e a fidelidade ao anúncio.",
};

export async function mockChat(request: ChatRequest): Promise<ChatResponse> {
  await new Promise((resolve) => setTimeout(resolve, 700 + Math.random() * 600));

  const question = request.question.toLowerCase();
  const topics = detectTopics(question);

  if (topics.length === 0) {
    return {
      answer:
        "[mock] Não encontrei avaliações relacionadas a essa pergunta. A base cobre pedidos, entregas, produtos e avaliações da Olist entre 2016 e 2018.",
      insufficient_evidence: true,
      sources: [],
    };
  }

  const matched = POOL.filter((s) => topics.includes(s.topic) && matchesFilters(s, request.filters))
    .sort((a, b) => a.score - b.score);
  const sources = matched.map(({ topic: _topic, ...source }) => source);

  if (sources.length < 2) {
    return {
      answer:
        "[mock] Com os filtros atuais encontrei evidências insuficientes para uma conclusão. Tente ampliar o período ou a faixa de nota.",
      insufficient_evidence: true,
      sources,
    };
  }

  // Citations point at positions in `sources`, which is how the UI numbers them.
  const paragraphs = topics
    .map((topic) => {
      const refs = matched.flatMap((s, i) => (s.topic === topic ? [`[${i + 1}]`] : []));
      return refs.length > 0 ? `${ANSWERS[topic]} ${refs.join("")}` : null;
    })
    .filter((p): p is string => p !== null);

  return { answer: `[mock] ${paragraphs.join("\n\n")}`, insufficient_evidence: false, sources };
}

function detectTopics(question: string): Topic[] {
  const topics = (Object.keys(TOPIC_KEYWORDS) as Topic[]).filter((topic) =>
    TOPIC_KEYWORDS[topic].some((kw) => question.includes(kw)),
  );
  if (topics.length > 0) return topics;
  if (GENERIC_KEYWORDS.some((kw) => question.includes(kw))) return ["delivery", "quality"];
  return [];
}

function matchesFilters(source: Source, filters: ChatFilters): boolean {
  if (filters.doc_type && source.doc_type !== filters.doc_type) return false;
  if (filters.review_score_min !== undefined && source.review_score < filters.review_score_min) return false;
  if (filters.review_score_max !== undefined && source.review_score > filters.review_score_max) return false;
  if (filters.product_category && source.product_category !== filters.product_category) return false;
  if (filters.purchase_year && !source.purchase_date.startsWith(String(filters.purchase_year))) return false;
  return true;
}
