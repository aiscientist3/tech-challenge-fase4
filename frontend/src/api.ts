// Contract of POST /chat. The backend does not exist yet; these types are the
// spec it should follow. Metadata names match the Chroma collection built by
// src/rag/ingestion (see the root README).

import { mockChat } from "./mock.ts";

export type Role = "user" | "assistant";

export interface ChatTurn {
  role: Role;
  content: string;
}

export type DocType = "order" | "review" | "category_summary" | "state_summary";

export interface ChatFilters {
  review_score_min?: number;
  review_score_max?: number;
  product_category?: string;
  purchase_year?: number;
  doc_type?: DocType;
}

export interface ChatRequest {
  question: string;
  history: ChatTurn[];
  filters: ChatFilters;
}

export interface Source {
  id: string;
  doc_type: DocType;
  order_id: string;
  // -1 when the document has no review (ingestion sentinel).
  review_score: number;
  product_category: string;
  // "YYYY-MM-DD", "YYYY-MM" or "" when unknown.
  purchase_date: string;
  // Chroma cosine distance: lower is closer.
  score: number;
  excerpt: string;
}

export interface ChatResponse {
  answer: string;
  insufficient_evidence: boolean;
  sources: Source[];
}

export const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";

const TIMEOUT_MS = 60_000;

export class ChatError extends Error {}

export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  if (USE_MOCK) return mockChat(request);

  let response: Response;
  try {
    response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new ChatError("A API demorou demais para responder.");
    }
    throw new ChatError("Não foi possível conectar à API. O backend está rodando?");
  }

  if (!response.ok) {
    throw new ChatError(`A API respondeu com erro ${response.status}.`);
  }

  const body: unknown = await response.json().catch(() => null);
  if (!isChatResponse(body)) {
    throw new ChatError("A API devolveu uma resposta fora do formato esperado.");
  }
  return body;
}

function isChatResponse(value: unknown): value is ChatResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.answer === "string" &&
    typeof v.insufficient_evidence === "boolean" &&
    Array.isArray(v.sources)
  );
}
