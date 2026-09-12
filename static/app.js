/* ==========================================================================
   Local LLM Document Q&A - front-end
   Dependency-free ES2020. Talks to the FastAPI backend over fetch() and
   renders streamed answers via the SSE endpoint.

   Structure: DOM cache -> utilities -> API client -> renderers -> actions ->
   bootstrap.
   ========================================================================== */
'use strict';

/* --------------------------------------------------------------------------
   DOM cache
   -------------------------------------------------------------------------- */
const $ = (id) => document.getElementById(id);

const dom = {
  statusPill: $('status-pill'),
  banner: $('banner'),
  themeToggle: $('theme-toggle'),
  sidebar: $('sidebar'),
  sidebarToggle: $('sidebar-toggle'),

  fileInput: $('file-input'),
  dropzone: $('dropzone'),
  pasteText: $('paste-text'),
  pasteName: $('paste-name'),
  pasteSubmit: $('paste-submit'),
  pathInput: $('path-input'),
  pathSubmit: $('path-submit'),

  documentList: $('document-list'),
  refreshDocs: $('refresh-docs'),
  resetIndex: $('reset-index'),
  runtimeMeta: $('runtime-meta'),

  topk: $('topk'),
  topkValue: $('topk-value'),
  useCache: $('use-cache'),
  useStream: $('use-stream'),
  useRerank: $('use-rerank'),

  messages: $('messages'),
  composer: $('composer'),
  input: $('composer-input'),
  send: $('send'),

  toasts: $('toasts'),
};

/* --------------------------------------------------------------------------
   Utilities
   -------------------------------------------------------------------------- */
const escapeHtml = (value) =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

/** Format a millisecond figure for the diagnostics row. */
const ms = (value) => `${Math.round(Number(value) || 0)} ms`;

/** Format a 0..1 score as a whole percentage. */
const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;

const isBlank = (value) => !value || !String(value).trim();

/** Show a transient message in the bottom-right corner. */
function toast(message, kind = 'info', timeout = 4200) {
  const node = document.createElement('div');
  node.className = `toast toast--${kind}`;
  node.textContent = message;
  dom.toasts.appendChild(node);
  setTimeout(() => node.remove(), timeout);
}

/* --------------------------------------------------------------------------
   API client
   -------------------------------------------------------------------------- */
const api = {
  async request(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = { detail: text };
    }
    if (!response.ok) {
      throw new Error(describeError(body, response.status));
    }
    return body;
  },

  health: () => api.request('/api/health'),
  config: () => api.request('/api/config'),
  stats: () => api.request('/api/stats'),
  documents: () => api.request('/api/documents'),
  resetIndex: () => api.request('/api/documents/reset', { method: 'POST' }),

  deleteDocument: (source) =>
    api.request(`/api/documents/${encodeURIComponent(source)}`, { method: 'DELETE' }),

  ingestText: (text, sourceName) =>
    api.request('/api/ingest/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, source_name: sourceName }),
    }),

  ingestPath: (path) =>
    api.request('/api/ingest/path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, recursive: true }),
    }),

  ingestFile: (file) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return api.request('/api/ingest/upload', { method: 'POST', body: form });
  },

  ask: (payload) =>
    api.request('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
};

/** Turn an error body into something worth showing a human. */
function describeError(body, status) {
  if (!body) return `Request failed (HTTP ${status}).`;
  const detail = body.detail;

  if (typeof detail === 'string') {
    return body.hint ? `${detail} ${body.hint}` : detail;
  }
  if (detail && typeof detail === 'object') {
    const parts = [detail.detail, body.hint, detail.hint].filter(Boolean);
    if (Array.isArray(detail.configuration_errors) && detail.configuration_errors.length) {
      parts.push(detail.configuration_errors.join(' '));
    }
    return parts.join(' ') || `Request failed (HTTP ${status}).`;
  }
  return `Request failed (HTTP ${status}).`;
}

/* --------------------------------------------------------------------------
   Rendering helpers
   -------------------------------------------------------------------------- */
/**
 * Render answer text as safe HTML, converting [n] markers into citation chips.
 * Input is escaped first, so model output can never inject markup.
 */
function renderAnswer(text, citationCount) {
  const escaped = escapeHtml(text);
  const body = escaped
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replaceAll('\n', '<br />')}</p>`)
    .join('');

  return body.replace(/\[(\d{1,2})\]/g, (match, digits) => {
    const index = Number(digits);
    if (index < 1 || index > citationCount) return match;
    return `<button type="button" class="cite" data-cite="${index}" title="Jump to source ${index}">${index}</button>`;
  });
}

/** Build the badge row that states how trustworthy the answer is. */
function renderBadges(result) {
  const badges = [];

  if (result.cached) {
    badges.push(`<span class="badge">⚡ cached · ${pct(result.cache_similarity)} similar</span>`);
  }
  if (result.insufficient_context) {
    badges.push('<span class="badge badge--warning">no supporting passages found</span>');
  } else if (result.uncited) {
    badges.push('<span class="badge badge--danger">uncited — verify before trusting</span>');
  } else {
    badges.push(`<span class="badge badge--ok">✓ grounded in ${result.citations.length} passage(s)</span>`);
  }
  if (result.model) {
    badges.push(`<span class="badge">${escapeHtml(result.model)}</span>`);
  }

  return `<div class="badges">${badges.join('')}</div>`;
}

/** Build one source card. Cited passages get the accent bar and are openable. */
function renderSource(source, citedIndices) {
  const isCited = citedIndices.has(source.index);
  const relevance = Number(source.relevance) || 0;
  const page = source.page != null ? ` · page ${escapeHtml(source.page)}` : '';
  const detail = source.rerank_score != null
    ? `rerank ${Number(source.rerank_score).toFixed(2)}`
    : `fusion ${Number(source.fusion_score || 0).toFixed(4)}`;

  return `
    <article class="source-card ${isCited ? 'is-cited' : ''}" data-source="${source.index}">
      <div class="source-head">
        <span class="source-rank">${isCited ? '[' + source.index + ']' : source.index}</span>
        <span class="source-label">${escapeHtml(source.label || source.source)}${page}</span>
        <span class="source-score">${pct(relevance)} · ${escapeHtml(detail)}</span>
      </div>
      <div class="relevance-bar" aria-hidden="true"><span style="width:${Math.max(2, Math.round(relevance * 100))}%"></span></div>
      <p class="source-snippet">${escapeHtml(source.snippet || '')}</p>
      <details class="source-full">
        <summary>Show full passage</summary>
        <pre>${escapeHtml(source.text || '')}</pre>
      </details>
    </article>`;
}

/** Compose the inner HTML of an assistant message (without the wrapper). */
function renderAssistantInner(result) {
  const citedIndices = new Set((result.citations || []).map((citation) => citation.index));
  const sources = result.retrieved || [];
  const sourcesHtml = sources.length
    ? `<section class="sources">
         <p class="sources-title">Sources considered (${sources.length})</p>
         ${sources.map((source) => renderSource(source, citedIndices)).join('')}
       </section>`
    : '';

  const timings = result.timings || {};
  const stats = result.stats || {};
  const diagnostics = [];
  if (timings.total_ms != null) diagnostics.push(`total ${ms(timings.total_ms)}`);
  if (timings.generation_ms != null) diagnostics.push(`generation ${ms(timings.generation_ms)}`);
  if (timings.dense_ms != null) diagnostics.push(`vector ${ms(timings.dense_ms)}`);
  if (timings.sparse_ms != null) diagnostics.push(`BM25 ${ms(timings.sparse_ms)}`);
  if (timings.rerank_ms != null) diagnostics.push(`rerank ${ms(timings.rerank_ms)}`);
  if (stats.completion_tokens) diagnostics.push(`${stats.completion_tokens} tokens`);
  if (stats.tokens_per_second) diagnostics.push(`${stats.tokens_per_second} tok/s`);
  if (result.trace_id) diagnostics.push(`trace ${escapeHtml(result.trace_id)}`);

  return `
      ${renderAnswer(result.answer || '', (result.citations || []).length)}
      ${renderBadges(result)}
      ${sourcesHtml}
      ${diagnostics.length ? `<div class="diagnostics"><span>${diagnostics.join('</span><span>')}</span></div>` : ''}`;
}

/** Compose the whole assistant message body. */
function renderAssistantMessage(result) {
  return `<div class="message-body">${renderAssistantInner(result)}</div>`;
}

/** Append a message node and scroll it into view. */
function appendMessage(role, html) {
  const article = document.createElement('article');
  article.className = `message message--${role}`;
  article.innerHTML =
    role === 'system' ? html : `<div class="message-role">${role === 'user' ? 'You' : 'Answer'}</div>${html}`;
  dom.messages.appendChild(article);
  dom.messages.scrollTop = dom.messages.scrollHeight;
  return article;
}

/** Wire citation chips inside a message to highlight their source card. */
function bindCitationChips(container) {
  container.querySelectorAll('.cite').forEach((chip) => {
    chip.addEventListener('click', () => {
      const index = chip.dataset.cite;
      const card = container.querySelector(`.source-card[data-source="${index}"]`);
      if (!card) return;

      container.querySelectorAll('.cite.is-active').forEach((node) => node.classList.remove('is-active'));
      container.querySelectorAll('.source-card.is-active').forEach((node) => node.classList.remove('is-active'));
      chip.classList.add('is-active');
      card.classList.add('is-active');

      const disclosure = card.querySelector('details');
      if (disclosure) disclosure.open = true;
      card.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
  });
}

/* --------------------------------------------------------------------------
   Status and sidebar data
   -------------------------------------------------------------------------- */
function setStatus(kind, label) {
  dom.statusPill.className = `pill pill--${kind}`;
  dom.statusPill.textContent = label;
}

function showBanner(title, detailHtml) {
  dom.banner.className = 'banner';
  dom.banner.innerHTML = `<strong>${escapeHtml(title)}</strong>${detailHtml}`;
  dom.banner.hidden = false;
}

function showErrorBanner(title, detailHtml) {
  dom.banner.className = 'banner banner--error';
  dom.banner.innerHTML = `<strong>${escapeHtml(title)}</strong>${detailHtml}`;
  dom.banner.hidden = false;
}

async function refreshHealth() {
  try {
    const health = await api.health();

    if (health.status === 'ok') {
      setStatus('ok', `ready · ${health.ollama.model}`);
      dom.banner.hidden = true;
    } else if (health.status === 'unconfigured') {
      setStatus('danger', 'LLM not configured');
      const items = (health.configuration_errors || [])
        .map((problem) => `<li><code>${escapeHtml(problem)}</code></li>`)
        .join('');
      showBanner(
        'The local LLM is not configured yet',
        `<ul>${items}</ul><p>Copy <code>.env.example</code> to <code>.env</code>, set the two values, then restart the server.</p>`
      );
    } else {
      setStatus('warning', 'model unavailable');
      showBanner('Ollama is not answering', `<p>${escapeHtml(health.ollama.error || 'Unknown error.')}</p>`);
    }
  } catch (error) {
    setStatus('danger', 'backend unreachable');
    showErrorBanner('Cannot reach the backend', `<p>${escapeHtml(error.message)}</p>`);
  }
}

async function refreshRuntimeMeta() {
  try {
    const [config, stats] = await Promise.all([api.config(), api.stats()]);
    const rows = [
      ['model', config.model],
      ['embedding', config.embedding_model],
      ['reranker', config.reranker_model],
      ['vectors', String(stats.vectors ?? 0)],
      ['lexical chunks', String(stats.lexical_chunks ?? 0)],
      ['cache entries', String(stats.cache?.entries ?? 0)],
      ['cache hit rate', pct(stats.cache?.hit_rate ?? 0)],
      ['traces', config.ragobserve_enabled ? 'RAGObserve on' : 'off'],
    ];
    dom.runtimeMeta.innerHTML = rows
      .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`)
      .join('');

    dom.useRerank.checked = Boolean(config.rerank_enabled);
    dom.useCache.checked = Boolean(config.cache_enabled);
  } catch (error) {
    dom.runtimeMeta.innerHTML = `<dt>error</dt><dd>${escapeHtml(error.message)}</dd>`;
  }
}

async function refreshDocuments() {
  try {
    const payload = await api.documents();
    const documents = payload.documents || [];

    if (!documents.length) {
      dom.documentList.innerHTML = '<li class="empty-state">Nothing indexed yet.</li>';
      return;
    }

    dom.documentList.innerHTML = documents
      .map(
        (document) => `
        <li>
          <span class="document-name" title="${escapeHtml(document.source)}">${escapeHtml(document.source)}</span>
          <span class="document-meta">${document.chunks} chunk${document.chunks === 1 ? '' : 's'}</span>
          <button class="text-button text-button--danger" type="button"
                  data-delete="${escapeHtml(document.source)}" title="Remove from index">✕</button>
        </li>`
      )
      .join('');
  } catch (error) {
    dom.documentList.innerHTML = `<li class="empty-state">${escapeHtml(error.message)}</li>`;
  }
}

/* --------------------------------------------------------------------------
   Actions
   -------------------------------------------------------------------------- */
/** Report the outcome of an ingestion call consistently. */
function reportIngestion(result, label) {
  const chunks = result.chunks ?? 0;
  const failures = result.failures || [];
  if (chunks) {
    toast(`${label}: ${chunks} chunk(s) from ${result.documents ?? '?'} document(s)`, 'success');
  } else {
    toast(`${label}: nothing indexed (no extractable text)`, 'error');
  }
  if (failures.length) {
    toast(`${failures.length} file(s) skipped: ${failures[0]}`, 'error', 7000);
  }
  refreshDocuments();
  refreshRuntimeMeta();
}

async function handleFiles(files) {
  const list = Array.from(files || []);
  if (!list.length) return;

  for (const file of list) {
    toast(`Indexing ${file.name}…`);
    try {
      reportIngestion(await api.ingestFile(file), file.name);
    } catch (error) {
      toast(`${file.name}: ${error.message}`, 'error', 8000);
    }
  }
}

async function handlePaste() {
  const text = dom.pasteText.value;
  if (isBlank(text)) {
    toast('Nothing to index — paste some text first.', 'error');
    return;
  }
  const name = isBlank(dom.pasteName.value) ? 'pasted-text.txt' : dom.pasteName.value.trim();

  dom.pasteSubmit.disabled = true;
  try {
    reportIngestion(await api.ingestText(text, name), name);
    dom.pasteText.value = '';
    dom.pasteName.value = '';
  } catch (error) {
    toast(error.message, 'error', 8000);
  } finally {
    dom.pasteSubmit.disabled = false;
  }
}

async function handlePath() {
  const path = dom.pathInput.value;
  if (isBlank(path)) {
    toast('Enter a path under data/documents.', 'error');
    return;
  }

  dom.pathSubmit.disabled = true;
  try {
    reportIngestion(await api.ingestPath(path.trim()), path.trim());
  } catch (error) {
    toast(error.message, 'error', 8000);
  } finally {
    dom.pathSubmit.disabled = false;
  }
}

async function handleDelete(source) {
  if (!window.confirm(`Remove "${source}" from the index?`)) return;
  try {
    await api.deleteDocument(source);
    toast(`Removed ${source}`, 'success');
    refreshDocuments();
    refreshRuntimeMeta();
  } catch (error) {
    toast(error.message, 'error', 8000);
  }
}

async function handleReset() {
  const confirmed = window.confirm(
    'Delete the entire index? Every document must be re-ingested. This cannot be undone.'
  );
  if (!confirmed) return;

  try {
    await api.resetIndex();
    toast('Index cleared.', 'success');
    refreshDocuments();
    refreshRuntimeMeta();
  } catch (error) {
    toast(error.message, 'error', 8000);
  }
}

/* --------------------------------------------------------------------------
   Asking questions
   -------------------------------------------------------------------------- */
function currentPayload(question) {
  return {
    question,
    top_k: Number(dom.topk.value),
    use_cache: dom.useCache.checked,
  };
}

async function askOnce(question) {
  const message = appendMessage('assistant', '<div class="message-body"><span class="thinking"><span></span><span></span><span></span></span></div>');
  const body = message.querySelector('.message-body');

  try {
    const result = await api.ask(currentPayload(question));
    body.innerHTML = renderAssistantInner(result);
  } catch (error) {
    body.innerHTML = `<p><strong>Could not answer.</strong> ${escapeHtml(error.message)}</p>`;
  }
  return message;
}

/** Stream an answer, replacing the placeholder text as tokens arrive. */
async function askStreaming(question) {
  const message = appendMessage('assistant', '<div class="message-body"><span class="thinking"><span></span><span></span><span></span></span></div>');
  const body = message.querySelector('.message-body');

  let answerText = '';
  let finalResult = null;
  let retrievalSources = [];

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentPayload(question)),
    });

    if (!response.ok) {
      const text = await response.text();
      let parsed = null;
      try { parsed = JSON.parse(text); } catch { /* non-JSON error body */ }
      throw new Error(describeError(parsed, response.status));
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf('\n\n');

        if (!frame.startsWith('data:')) continue;
        const payload = frame.slice(5).trim();
        if (payload === '[DONE]') continue;

        let event;
        try { event = JSON.parse(payload); } catch { continue; }

        if (event.type === 'sources') {
          retrievalSources = event.retrieved || [];
          body.innerHTML = '<p class="source-snippet">retrieving…</p>';
        } else if (event.type === 'token') {
          answerText += event.content;
          body.innerHTML = renderAnswer(answerText, 99);
          dom.messages.scrollTop = dom.messages.scrollHeight;
        } else if (event.type === 'done') {
          finalResult = event;
        } else if (event.type === 'error') {
          throw new Error(event.detail || 'Streaming failed.');
        }
      }
    }

    if (finalResult) {
      body.innerHTML = renderAssistantInner({
        ...finalResult,
        retrieved: finalResult.retrieved || retrievalSources,
      });
    } else {
      body.innerHTML = renderAssistantInner({
        answer: answerText,
        citations: [],
        retrieved: retrievalSources,
        uncited: true,
      });
    }
  } catch (error) {
    body.innerHTML = `<p><strong>Could not answer.</strong> ${escapeHtml(error.message)}</p>`;
  }

  return message;
}

async function handleAsk(event) {
  event.preventDefault();
  const question = dom.input.value.trim();
  if (!question) return;

  appendMessage('user', `<div class="message-body"><p>${escapeHtml(question)}</p></div>`);
  dom.input.value = '';
  autoGrow();
  dom.send.disabled = true;

  try {
    const node = dom.useStream.checked ? await askStreaming(question) : await askOnce(question);
    bindCitationChips(node);
  } finally {
    dom.send.disabled = false;
    dom.input.focus();
  }
}

/* --------------------------------------------------------------------------
   Small UI behaviours
   -------------------------------------------------------------------------- */
function autoGrow() {
  dom.input.style.height = 'auto';
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 180)}px`;
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('rag-theme', theme); } catch { /* private mode */ }
}

function initTheme() {
  let stored = null;
  try { stored = localStorage.getItem('rag-theme'); } catch { /* private mode */ }
  const preferred = stored ||
    (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(preferred);
}

/* --------------------------------------------------------------------------
   Bootstrap
   -------------------------------------------------------------------------- */
function bindEvents() {
  dom.themeToggle.addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });

  dom.sidebarToggle.addEventListener('click', () => {
    const open = dom.sidebar.classList.toggle('is-open');
    dom.sidebarToggle.setAttribute('aria-expanded', String(open));
  });

  dom.composer.addEventListener('submit', handleAsk);
  dom.input.addEventListener('input', autoGrow);
  dom.input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      dom.composer.requestSubmit();
    }
  });

  dom.fileInput.addEventListener('change', (event) => {
    handleFiles(event.target.files);
    event.target.value = '';
  });

  ['dragenter', 'dragover'].forEach((type) =>
    dom.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dom.dropzone.classList.add('is-over');
    })
  );
  ['dragleave', 'drop'].forEach((type) =>
    dom.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dom.dropzone.classList.remove('is-over');
    })
  );
  dom.dropzone.addEventListener('drop', (event) => handleFiles(event.dataTransfer.files));

  dom.pasteSubmit.addEventListener('click', handlePaste);
  dom.pathSubmit.addEventListener('click', handlePath);
  dom.refreshDocs.addEventListener('click', refreshDocuments);
  dom.resetIndex.addEventListener('click', handleReset);

  dom.documentList.addEventListener('click', (event) => {
    const button = event.target.closest('[data-delete]');
    if (button) handleDelete(button.dataset.delete);
  });

  dom.topk.addEventListener('input', () => {
    dom.topkValue.textContent = dom.topk.value;
  });
}

async function init() {
  initTheme();
  bindEvents();

  dom.topkValue.textContent = dom.topk.value;
  autoGrow();

  await refreshHealth();
  await Promise.all([refreshDocuments(), refreshRuntimeMeta()]);

  dom.input.focus();
  // Poll health so the banner clears on its own once Ollama comes up.
  setInterval(refreshHealth, 30000);
}

document.addEventListener('DOMContentLoaded', init);
