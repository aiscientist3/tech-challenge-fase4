import "./styles.css";

import { ChatError, USE_MOCK, sendChat } from "./api.ts";
import type { ChatFilters, ChatTurn, DocType } from "./api.ts";
import { CATEGORIES } from "./categories.ts";
import { assistantMessage, errorMessage, loadingMessage, userMessage } from "./render.ts";

// Only the most recent turns go to the backend to keep the prompt small.
const HISTORY_LIMIT = 10;
const MAX_INPUT_HEIGHT = 160;

const messages = byId<HTMLDivElement>("messages");
const welcome = byId<HTMLElement>("welcome");
const chatForm = byId<HTMLFormElement>("chat-form");
const question = byId<HTMLTextAreaElement>("question");
const sendButton = byId<HTMLButtonElement>("send-btn");
const clearButton = byId<HTMLButtonElement>("clear-btn");
const filtersForm = byId<HTMLFormElement>("filters-form");
const filterCount = byId<HTMLSpanElement>("filter-count");
const filterError = byId<HTMLParagraphElement>("filter-error");

let history: ChatTurn[] = [];
let pending = false;

function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id} in index.html`);
  return node as T;
}

function init(): void {
  if (USE_MOCK) byId("mock-badge").classList.remove("hidden");
  // Sidebar starts open on wide screens (Tailwind lg) and collapsed on phones.
  byId<HTMLDetailsElement>("filters-panel").open = window.matchMedia("(min-width: 1024px)").matches;

  const categories = byId<HTMLDataListElement>("categories");
  for (const { value, label } of CATEGORIES) {
    const option = document.createElement("option");
    option.value = value;
    option.label = label;
    categories.append(option);
  }

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void ask(question.value);
  });

  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });
  question.addEventListener("input", autosize);

  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-example]")) {
    button.addEventListener("click", () => void ask(button.dataset.example ?? ""));
  }

  clearButton.addEventListener("click", resetConversation);
  filtersForm.addEventListener("input", updateFilterState);
  // reset fires before the fields are cleared.
  filtersForm.addEventListener("reset", () => setTimeout(updateFilterState));

  question.focus();
}

async function ask(text: string): Promise<void> {
  const content = text.trim();
  if (!content || pending) return;

  const filters = readFilters();
  if (filters === null) return;

  welcome.classList.add("hidden");
  messages.append(userMessage(content));
  question.value = "";
  autosize();

  const loading = loadingMessage();
  messages.append(loading);
  scrollToBottom();
  setPending(true);

  try {
    const response = await sendChat({
      question: content,
      history: history.slice(-HISTORY_LIMIT),
      filters,
    });
    loading.replaceWith(assistantMessage(response));
    history.push({ role: "user", content }, { role: "assistant", content: response.answer });
  } catch (error) {
    const message = error instanceof ChatError ? error.message : "Erro inesperado ao consultar a API.";
    loading.replaceWith(errorMessage(message));
    // Give the question back so the user can retry without retyping.
    if (!question.value) question.value = content;
    if (!(error instanceof ChatError)) console.error(error);
  } finally {
    setPending(false);
    scrollToBottom();
    question.focus();
  }
}

/** Reads the filter form. Returns null (and shows why) when it is invalid. */
function readFilters(): ChatFilters | null {
  const data = new FormData(filtersForm);
  const filters: ChatFilters = {};

  const docType = String(data.get("doc_type") ?? "");
  if (docType) filters.doc_type = docType as DocType;

  const min = Number(data.get("review_score_min") || 0);
  const max = Number(data.get("review_score_max") || 0);
  if (min) filters.review_score_min = min;
  if (max) filters.review_score_max = max;

  const category = String(data.get("product_category") ?? "").trim();
  if (category) filters.product_category = category;

  const year = Number(data.get("purchase_year") || 0);
  if (year) filters.purchase_year = year;

  if (min && max && min > max) {
    filterError.textContent = "A nota mínima não pode ser maior que a máxima.";
    filterError.classList.remove("hidden");
    return null;
  }
  if (category && !CATEGORIES.some((c) => c.value === category)) {
    filterError.textContent = "Escolha uma categoria da lista.";
    filterError.classList.remove("hidden");
    return null;
  }
  filterError.classList.add("hidden");
  return filters;
}

function updateFilterState(): void {
  const filters = readFilters();
  const count = filters ? Object.keys(filters).length : 0;
  filterCount.textContent = String(count);
  filterCount.classList.toggle("hidden", count === 0);
}

function resetConversation(): void {
  if (pending) return;
  history = [];
  messages.replaceChildren(welcome);
  welcome.classList.remove("hidden");
  question.focus();
}

function setPending(value: boolean): void {
  pending = value;
  sendButton.disabled = value;
  clearButton.disabled = value;
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-example]")) {
    button.disabled = value;
  }
}

function autosize(): void {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight, MAX_INPUT_HEIGHT)}px`;
  question.style.overflowY = question.scrollHeight > MAX_INPUT_HEIGHT ? "auto" : "hidden";
}

function scrollToBottom(): void {
  messages.scrollTop = messages.scrollHeight;
}

init();
