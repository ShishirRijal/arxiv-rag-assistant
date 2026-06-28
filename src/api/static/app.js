const state = {
  loading: false,
};

const form = document.querySelector("#ask-form");
const input = document.querySelector("#question-input");
const messages = document.querySelector("#messages");
const askButton = document.querySelector("#ask-button");
const modeSelect = document.querySelector("#rag-mode");
const hybridToggle = document.querySelector("#use-hybrid");
const categoryFilter = document.querySelector("#category-filter");
const healthStatus = document.querySelector("#health-status");
const refreshHealth = document.querySelector("#refresh-health");
const userTemplate = document.querySelector("#user-message-template");
const assistantTemplate = document.querySelector("#assistant-message-template");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || state.loading) return;

  input.value = "";
  appendUserMessage(question);
  await askQuestion(question);
});

refreshHealth.addEventListener("click", () => {
  loadHealth();
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.example;
    input.focus();
  });
});

modeSelect.addEventListener("change", () => {
  const agentic = modeSelect.value === "agentic";
  hybridToggle.disabled = agentic;
  categoryFilter.disabled = agentic;
});

async function askQuestion(question) {
  setLoading(true);
  const started = performance.now();
  const pending = appendThinkingMessage();

  try {
    const result = await callRagApi(question);
    replaceAssistantMessage(pending, result, Math.round(performance.now() - started));
  } catch (error) {
    replaceAssistantMessage(pending, {
      answer: `Request failed.\n\n${error.message}`,
      sources: [],
      search_mode: "error",
    });
  } finally {
    setLoading(false);
  }
}

async function callRagApi(question) {
  const agentic = modeSelect.value === "agentic";
  const url = agentic ? "/api/v1/ask-agentic/" : "/api/v1/ask/";
  const payload = agentic
    ? { question }
    : {
        question,
        use_hybrid: hybridToggle.checked,
        categories: categoryFilter.value ? [categoryFilter.value] : null,
      };

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `HTTP ${response.status}`);
  }
  return body;
}

function appendUserMessage(text) {
  const node = userTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".bubble").textContent = text;
  messages.appendChild(node);
  scrollToBottom();
}

function appendAssistantMessage(result) {
  const node = assistantTemplate.content.firstElementChild.cloneNode(true);
  renderAssistantNode(node, result);
  messages.appendChild(node);
  scrollToBottom();
  return node;
}

function appendThinkingMessage() {
  const node = assistantTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add("thinking-message");
  node.querySelector(".answer-text").textContent = "Thinking...";
  node.querySelector(".meta-row").innerHTML = "";
  node.querySelector(".sources").innerHTML = "";
  node.querySelector(".reasoning").hidden = true;
  messages.appendChild(node);
  scrollToBottom();
  return node;
}

function replaceAssistantMessage(node, result, elapsedMs) {
  renderAssistantNode(node, result, elapsedMs);
  scrollToBottom();
}

function renderAssistantNode(node, result, elapsedMs) {
  node.querySelector(".answer-text").textContent = result.answer || "No answer returned.";

  const meta = node.querySelector(".meta-row");
  meta.innerHTML = "";
  addPill(meta, result.search_mode && `mode: ${result.search_mode}`);
  addPill(meta, result.cached === true && "cached");
  addPill(meta, Number.isFinite(result.n_chunks) && `chunks: ${result.n_chunks}`);
  addPill(meta, Number.isFinite(result.retrieval_attempts) && `attempts: ${result.retrieval_attempts}`);
  addPill(meta, Number.isFinite(result.guardrail_score) && `guardrail: ${result.guardrail_score}`);
  addPill(meta, Number.isFinite(result.took_ms) && `api: ${result.took_ms}ms`);
  addPill(meta, Number.isFinite(elapsedMs) && `roundtrip: ${elapsedMs}ms`);

  renderSources(node.querySelector(".sources"), result.sources || []);
  renderReasoning(node.querySelector(".reasoning"), result.reasoning_steps || []);
}

function addPill(container, value) {
  if (!value) return;
  const pill = document.createElement("span");
  pill.className = "pill";
  pill.textContent = value;
  container.appendChild(pill);
}

function renderSources(container, sources) {
  container.innerHTML = "";
  const seen = new Set();

  sources.forEach((source) => {
    const key = source.arxiv_id || source.url || source.title;
    if (seen.has(key)) return;
    seen.add(key);

    const link = document.createElement("a");
    link.className = "source-card";
    link.href = source.url || `https://arxiv.org/abs/${source.arxiv_id}`;
    link.target = "_blank";
    link.rel = "noreferrer";

    const title = document.createElement("strong");
    title.textContent = source.title || "Untitled paper";

    const meta = document.createElement("span");
    meta.textContent = source.arxiv_id ? `arXiv:${source.arxiv_id}` : "source";

    link.append(title, meta);
    container.appendChild(link);
  });
}

function renderReasoning(details, steps) {
  const list = details.querySelector("ol");
  list.innerHTML = "";

  if (!steps.length) {
    details.hidden = true;
    return;
  }

  details.hidden = false;
  steps.forEach((step) => {
    const item = document.createElement("li");
    item.textContent = `${step.step}: ${step.message}`;
    list.appendChild(item);
  });
}

async function loadHealth() {
  healthStatus.textContent = "Checking services...";
  healthStatus.classList.add("muted");

  try {
    const response = await fetch("/health");
    const result = await response.json();
    const services = result.services || {};

    healthStatus.innerHTML = "";
    const overall = document.createElement("div");
    overall.innerHTML = `<span class="health-dot ${result.status === "healthy" ? "" : "bad"}"></span><strong>API ${result.status}</strong>`;
    healthStatus.appendChild(overall);

    Object.entries(services).forEach(([name, details]) => {
      const row = document.createElement("div");
      const status = details && details.status ? details.status : "unknown";
      row.innerHTML = `<span class="health-dot ${status === "healthy" || status === "enabled" ? "" : "bad"}"></span>${name}: ${status}`;
      healthStatus.appendChild(row);
    });
  } catch (error) {
    healthStatus.textContent = `Health check failed: ${error.message}`;
  }
}

function setLoading(loading) {
  state.loading = loading;
  askButton.disabled = loading;
  askButton.textContent = loading ? "Thinking..." : "Ask";
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

loadHealth();
