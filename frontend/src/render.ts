// DOM builders for the chat. Text from the API always goes through
// textContent, never innerHTML, because answers and reviews are untrusted.

import type { ChatResponse, DocType, Source } from "./api.ts";

const DOC_TYPE_LABELS: Record<DocType, string> = {
  review: "Avaliação",
  order: "Pedido",
  category_summary: "Resumo de categoria",
  state_summary: "Resumo de UF",
};

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = "",
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** A labelled message block; `side` aligns it like a chat. */
function messageBlock(side: "user" | "assistant", body: HTMLElement): HTMLElement {
  const wrapper = el(
    "article",
    side === "user"
      ? "flex max-w-[90%] flex-col items-end gap-1 self-end sm:max-w-[75%]"
      : "flex w-full max-w-full flex-col gap-1 self-start sm:max-w-[90%]",
  );
  wrapper.append(
    el("span", side === "user" ? "eyebrow text-base-content/50" : "eyebrow text-primary", side === "user" ? "Você" : "Assistente"),
  );
  wrapper.append(body);
  return wrapper;
}

export function userMessage(text: string): HTMLElement {
  const body = el("div", "bg-primary px-4 py-3 text-primary-content whitespace-pre-line", text);
  return messageBlock("user", body);
}

export function loadingMessage(): HTMLElement {
  const body = el("div", "border-l-2 border-primary bg-base-200 px-4 py-3");
  const dots = el("span", "loading loading-dots loading-sm text-primary");
  dots.setAttribute("aria-label", "Gerando resposta");
  body.append(dots);
  return messageBlock("assistant", body);
}

export function errorMessage(text: string): HTMLElement {
  const body = el("div", "border-l-2 border-error bg-base-200 px-4 py-3 text-error");
  body.setAttribute("role", "alert");
  body.textContent = text;
  return messageBlock("assistant", body);
}

export function assistantMessage(response: ChatResponse): HTMLElement {
  const insufficient = response.insufficient_evidence;
  const body = el(
    "div",
    insufficient
      ? "flex flex-col gap-4 border-l-2 border-warning bg-base-200 px-4 py-4"
      : "flex flex-col gap-4 border-l-2 border-primary bg-base-200 px-4 py-4",
  );

  if (insufficient) {
    const warning = el("span", "eyebrow text-warning", "⚠ Evidências insuficientes");
    warning.setAttribute("role", "status");
    body.append(warning);
  }

  body.append(answerText(response.answer));

  if (response.sources.length > 0) body.append(sourcesPanel(response.sources));

  return messageBlock("assistant", body);
}

/** Answer text with citation markers like [2] highlighted, built from text nodes only. */
function answerText(answer: string): HTMLElement {
  const node = el("div", "prose prose-sm prose-invert max-w-none whitespace-pre-line leading-relaxed");
  for (const part of answer.split(/(\[\d+\])/)) {
    if (!part) continue;
    node.append(/^\[\d+\]$/.test(part) ? el("sup", "font-mono font-bold text-primary", part) : part);
  }
  return node;
}

function sourcesPanel(sources: Source[]): HTMLElement {
  const panel = el("details", "collapse collapse-arrow border border-base-300 bg-base-100");
  panel.append(
    el("summary", "collapse-title eyebrow min-h-0 py-3 text-secondary", `Evidências utilizadas · ${sources.length}`),
  );
  const list = el("ol", "collapse-content flex flex-col gap-2 text-sm");
  sources.forEach((source, index) => list.append(sourceCard(source, index + 1)));
  panel.append(list);
  return panel;
}

function sourceCard(source: Source, number: number): HTMLElement {
  const card = el("li", "flex gap-3 border border-base-300 p-3 transition-colors hover:border-primary/60");

  card.append(el("span", "font-mono text-lg font-bold leading-none text-primary", String(number).padStart(2, "0")));

  const content = el("div", "flex min-w-0 flex-1 flex-col gap-2");

  const header = el("div", "flex flex-wrap items-center gap-x-3 gap-y-1");
  header.append(el("span", "eyebrow text-secondary", DOC_TYPE_LABELS[source.doc_type] ?? source.doc_type));
  header.append(stars(source.review_score));
  if (source.product_category) {
    header.append(el("span", "border border-base-300 px-1.5 py-0.5 text-xs", source.product_category));
  }
  if (source.purchase_date) {
    header.append(el("span", "text-xs text-base-content/50", source.purchase_date));
  }
  const distance = el("span", "ml-auto font-mono text-xs text-base-content/50", `dist ${source.score.toFixed(3)}`);
  distance.title = "Distância de cosseno no Chroma: quanto menor, mais próxima da pergunta.";
  header.append(distance);

  content.append(header);
  content.append(el("p", "whitespace-pre-line leading-relaxed", source.excerpt));
  if (source.order_id) {
    content.append(el("p", "truncate font-mono text-xs text-base-content/40", `pedido ${source.order_id}`));
  }

  card.append(content);
  return card;
}

function stars(score: number): HTMLElement {
  if (score < 1) return el("span", "text-xs text-base-content/50", "sem nota");
  const filled = Math.min(5, Math.round(score));
  const node = el("span", "text-sm tracking-tight");
  node.append(el("span", "text-primary", "★".repeat(filled)));
  node.append(el("span", "text-base-content/25", "★".repeat(5 - filled)));
  node.setAttribute("aria-label", `Nota ${filled} de 5`);
  return node;
}
