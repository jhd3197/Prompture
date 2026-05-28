// Data model. Each item has:
//   id       — unique key
//   label    — display name
//   sub      — optional short description
//   extra    — pip extra name (or null if base install is enough)
//   env      — array of lines for .env (or [] if nothing to configure)
//
// The pip extras come from pyproject.toml [project.optional-dependencies].
// The env keys come from .env.copy. We deliberately only model what's in
// those files so the page can't drift from reality.

const SECTIONS = [
  {
    id: "cloud-llm",
    title: "Cloud LLM Providers",
    desc: "Hosted chat / completion APIs.",
    items: [
      {
        id: "openai",
        label: "OpenAI",
        sub: "GPT-4o, GPT-4, GPT-3.5",
        extra: "openai",
        env: ["OPENAI_API_KEY=", "OPENAI_MODEL=gpt-4o-mini"],
      },
      {
        id: "anthropic",
        label: "Anthropic Claude",
        sub: "Claude 4 family",
        extra: "anthropic",
        env: ["CLAUDE_API_KEY=", "CLAUDE_MODEL=claude-haiku-4-5-20251001"],
      },
      {
        id: "google",
        label: "Google Gemini",
        sub: "Gemini 2.5",
        extra: "google",
        env: ["GOOGLE_API_KEY=", "GOOGLE_MODEL=gemini-2.5-flash"],
      },
      {
        id: "groq",
        label: "Groq",
        sub: "Fast Llama / Mixtral inference",
        extra: "groq",
        env: ["GROQ_API_KEY=", "GROQ_MODEL=llama-3.3-70b-versatile"],
      },
      {
        id: "openrouter",
        label: "OpenRouter",
        sub: "Multi-provider router",
        extra: null,
        env: ["OPENROUTER_API_KEY=", "OPENROUTER_MODEL=openai/gpt-4o-mini"],
      },
      {
        id: "grok",
        label: "Grok (xAI)",
        sub: "Grok 4 + Imagine video",
        extra: null,
        env: [
          "GROK_API_KEY=",
          "# XAI_API_KEY is also supported for xAI docs compatibility",
          "XAI_API_KEY=",
          "GROK_MODEL=grok-4-fast-reasoning",
          "GROK_VIDEO_MODEL=grok-imagine-video",
        ],
      },
      {
        id: "azure",
        label: "Azure OpenAI",
        sub: "Plus Azure Claude / Mistral backends",
        extra: null,
        env: [
          "AZURE_API_KEY=",
          "AZURE_API_ENDPOINT=",
          "AZURE_DEPLOYMENT_ID=",
          "AZURE_API_VERSION=2025-01-01-preview",
          "# Optional: Azure Claude backend",
          "AZURE_CLAUDE_API_KEY=",
          "AZURE_CLAUDE_ENDPOINT=",
          "AZURE_CLAUDE_API_VERSION=",
          "# Optional: Azure Mistral backend",
          "AZURE_MISTRAL_API_KEY=",
          "AZURE_MISTRAL_ENDPOINT=",
          "AZURE_MISTRAL_API_VERSION=",
        ],
      },
      {
        id: "moonshot",
        label: "Moonshot (Kimi)",
        extra: null,
        env: [
          "MOONSHOT_API_KEY=",
          "MOONSHOT_MODEL=kimi-k2-0905-preview",
          "MOONSHOT_ENDPOINT=https://api.moonshot.ai/v1",
        ],
      },
      {
        id: "zai",
        label: "Z.ai (Zhipu)",
        sub: "GLM-4 family",
        extra: null,
        env: [
          "ZHIPU_API_KEY=",
          "ZHIPU_MODEL=glm-4.7",
          "ZHIPU_ENDPOINT=https://api.z.ai/api/paas/v4",
        ],
      },
      {
        id: "modelscope",
        label: "ModelScope",
        sub: "Alibaba Cloud — Qwen models",
        extra: null,
        env: [
          "MODELSCOPE_API_KEY=",
          "MODELSCOPE_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507",
          "MODELSCOPE_ENDPOINT=https://api-inference.modelscope.cn/v1",
        ],
      },
      {
        id: "mistral",
        label: "Mistral AI",
        extra: null,
        env: ["MISTRAL_API_KEY=", "MISTRAL_MODEL=mistral-large-latest"],
      },
      {
        id: "deepseek",
        label: "DeepSeek",
        extra: null,
        env: ["DEEPSEEK_API_KEY=", "DEEPSEEK_MODEL=deepseek-chat"],
      },
      {
        id: "cohere",
        label: "Cohere",
        sub: "Command-R + embeddings + rerank",
        extra: null,
        env: [
          "COHERE_API_KEY=",
          "COHERE_MODEL=command-r-plus",
          "COHERE_EMBEDDING_MODEL=embed-v4.0",
        ],
      },
      {
        id: "bedrock",
        label: "AWS Bedrock",
        sub: "Anthropic / Llama / Mistral on AWS",
        extra: "bedrock",
        env: [
          "# boto3's standard credential chain is also honoured",
          "AWS_ACCESS_KEY_ID=",
          "AWS_SECRET_ACCESS_KEY=",
          "AWS_REGION=us-east-1",
          "BEDROCK_MODEL=anthropic.claude-3-5-haiku-20241022-v1:0",
        ],
      },
      {
        id: "minimax",
        label: "MiniMax / Hailuo",
        sub: "LLM + video",
        extra: null,
        env: [
          "MINIMAX_API_KEY=",
          "# HAILUO_API_KEY is also supported",
          "HAILUO_API_KEY=",
          "MINIMAX_ENDPOINT=https://api.minimax.io/v1",
          "MINIMAX_MODEL=MiniMax-Text-01",
          "MINIMAX_VIDEO_MODEL=MiniMax-Hailuo-2.3",
        ],
      },
      {
        id: "cachibot",
        label: "CachiBot",
        extra: null,
        env: ["CACHIBOT_API_KEY="],
      },
      {
        id: "github_models",
        label: "GitHub Models",
        sub: "Via openai_compatible driver",
        extra: null,
        env: ["GITHUB_TOKEN="],
      },
    ],
  },
  {
    id: "local-llm",
    title: "Local & Self-hosted LLMs",
    desc: "Run models on your own hardware or a private endpoint.",
    items: [
      {
        id: "ollama",
        label: "Ollama",
        sub: "Default local backend",
        extra: null,
        env: ["OLLAMA_ENDPOINT=http://localhost:11434", "OLLAMA_MODEL=llama3.1:8b"],
      },
      {
        id: "lmstudio",
        label: "LM Studio",
        extra: null,
        env: [
          "LMSTUDIO_ENDPOINT=http://127.0.0.1:1234/v1/chat/completions",
          "LMSTUDIO_MODEL=deepseek/deepseek-r1-0528-qwen3-8b",
          "LMSTUDIO_API_KEY=",
        ],
      },
      {
        id: "huggingface",
        label: "HuggingFace Endpoints",
        extra: null,
        env: ["HF_ENDPOINT=", "HF_TOKEN="],
      },
      {
        id: "local_http",
        label: "Generic HTTP",
        sub: "Any OpenAI-shaped local server",
        extra: null,
        env: [
          "LOCAL_HTTP_ENDPOINT=http://localhost:8000/generate",
          "LOCAL_HTTP_MODEL=local-model",
        ],
      },
      {
        id: "airllm",
        label: "AirLLM",
        sub: "Layer-by-layer inference",
        extra: "airllm",
        env: ["AIRLLM_MODEL=meta-llama/Llama-2-7b-hf", "AIRLLM_COMPRESSION="],
      },
    ],
  },
  {
    id: "openai-compat",
    title: "OpenAI-compatible Aggregators",
    desc: "Use with the openai_compatible driver and a profile (e.g. openai_compatible/fireworks/&lt;model&gt;).",
    items: [
      { id: "fireworks", label: "Fireworks AI", extra: null, env: ["FIREWORKS_API_KEY="] },
      { id: "together", label: "Together AI", extra: null, env: ["TOGETHER_API_KEY="] },
      { id: "cerebras", label: "Cerebras", extra: null, env: ["CEREBRAS_API_KEY="] },
      { id: "sambanova", label: "SambaNova", extra: null, env: ["SAMBANOVA_API_KEY="] },
      { id: "perplexity", label: "Perplexity", extra: null, env: ["PERPLEXITY_API_KEY="] },
      { id: "nvidia", label: "NVIDIA NIM", extra: null, env: ["NVIDIA_API_KEY="] },
      { id: "deepinfra", label: "DeepInfra", extra: null, env: ["DEEPINFRA_API_KEY="] },
      { id: "siliconflow", label: "SiliconFlow", extra: null, env: ["SILICONFLOW_API_KEY="] },
    ],
  },
  {
    id: "image-gen",
    title: "Image Generation",
    items: [
      { id: "img-openai", label: "OpenAI (DALL·E / gpt-image)", sub: "Uses the openai extra", extra: "openai", env: ["OPENAI_API_KEY="] },
      { id: "img-google", label: "Google Imagen", extra: "google", env: ["GOOGLE_API_KEY="] },
      { id: "img-grok", label: "Grok Image", extra: null, env: ["GROK_API_KEY="] },
      { id: "img-stability", label: "Stability AI", extra: null, env: ["STABILITY_API_KEY="] },
      {
        id: "img-runway",
        label: "Runway",
        sub: "Image + video + audio under one key",
        extra: null,
        env: [
          "# Either RUNWAY_API_KEY or RUNWAYML_API_SECRET works",
          "RUNWAY_API_KEY=",
        ],
      },
      {
        id: "img-kling",
        label: "Kling AI",
        sub: "Image + video",
        extra: null,
        env: [
          "KLING_ACCESS_KEY=",
          "KLING_SECRET_KEY=",
          "KLING_ENDPOINT=",
          "KLING_IMAGE_MODEL=kling-v2-1",
          "KLING_VIDEO_MODEL=kling-v2-1",
        ],
      },
      {
        id: "img-fal",
        label: "Fal.ai",
        sub: "Image + video aggregator",
        extra: null,
        env: [
          "FAL_API_KEY=",
          "FAL_ENDPOINT=",
          "FAL_IMAGE_MODEL=fal-ai/flux/dev",
          "FAL_VIDEO_MODEL=fal-ai/kling-video/v2.6/pro/image-to-video",
        ],
      },
      {
        id: "img-ideogram",
        label: "Ideogram",
        sub: "Strong typography rendering",
        extra: null,
        env: ["IDEOGRAM_API_KEY=", "IDEOGRAM_IMAGE_MODEL=ideogram-v3"],
      },
      {
        id: "img-bfl",
        label: "Black Forest Labs",
        sub: "Direct Flux access",
        extra: null,
        env: ["BFL_API_KEY=", "BFL_IMAGE_MODEL=flux-pro-1.1"],
      },
    ],
  },
  {
    id: "video-gen",
    title: "Video Generation",
    items: [
      { id: "vid-luma", label: "Luma Dream Machine", extra: null, env: ["LUMA_API_KEY=", "LUMA_VIDEO_MODEL=ray-2"] },
      { id: "vid-pika", label: "Pika Labs", extra: null, env: ["PIKA_API_KEY=", "PIKA_VIDEO_MODEL=pika-2.2"] },
    ],
  },
  {
    id: "audio",
    title: "Audio (TTS / STT)",
    items: [
      {
        id: "audio-cartesia",
        label: "Cartesia",
        sub: "Sonic TTS, ultra-low-latency",
        extra: null,
        env: ["CARTESIA_API_KEY=", "CARTESIA_TTS_MODEL=sonic-2"],
      },
      {
        id: "audio-deepgram",
        label: "Deepgram",
        sub: "Nova-3 STT + Aura-2 TTS",
        extra: null,
        env: [
          "DEEPGRAM_API_KEY=",
          "DEEPGRAM_STT_MODEL=nova-3",
          "DEEPGRAM_TTS_MODEL=aura-2-thalia-en",
        ],
      },
      {
        id: "audio-assemblyai",
        label: "AssemblyAI",
        sub: "Universal-2 STT",
        extra: null,
        env: ["ASSEMBLYAI_API_KEY=", "ASSEMBLYAI_STT_MODEL=universal"],
      },
    ],
  },
  {
    id: "embed-rerank",
    title: "Embeddings & Rerank",
    items: [
      { id: "emb-voyage", label: "Voyage AI", extra: null, env: ["VOYAGE_API_KEY=", "VOYAGE_EMBEDDING_MODEL=voyage-3.5"] },
      { id: "emb-jina", label: "Jina AI", extra: null, env: ["JINA_API_KEY=", "JINA_EMBEDDING_MODEL=jina-embeddings-v3"] },
      { id: "emb-nomic", label: "Nomic Atlas", extra: null, env: ["NOMIC_API_KEY=", "NOMIC_EMBEDDING_MODEL=nomic-embed-text-v1.5"] },
      {
        id: "emb-mixedbread",
        label: "Mixedbread (mxbai)",
        sub: "Embeddings + rerank",
        extra: null,
        env: [
          "MIXEDBREAD_API_KEY=",
          "MXBAI_EMBEDDING_MODEL=mxbai-embed-large-v1",
          "MXBAI_RERANK_MODEL=mxbai-rerank-large-v1",
        ],
      },
    ],
  },
  {
    id: "web-search",
    title: "Web Search Backends",
    desc: "Used by prompture.tools.WebSearchTool.",
    items: [
      { id: "search-tavily", label: "Tavily", sub: "Default, AI-friendly snippets", extra: null, env: ["TAVILY_API_KEY="] },
      { id: "search-serper", label: "Serper", sub: "Google Search API wrapper", extra: null, env: ["SERPER_API_KEY="] },
      { id: "search-brave", label: "Brave Search", sub: "Independent index", extra: null, env: ["BRAVE_SEARCH_API_KEY="] },
      { id: "search-searxng", label: "SearXNG", sub: "Self-hosted, no key", extra: null, env: ["SEARXNG_ENDPOINT="] },
    ],
  },
  {
    id: "features",
    title: "Features",
    desc: "Optional capabilities. No API keys required.",
    items: [
      { id: "feat-toon", label: "TOON", sub: "Compact token-saving I/O format", extra: "toon", env: [] },
      { id: "feat-pandas", label: "Pandas integration", sub: "extract_from_pandas()", extra: "pandas", env: [] },
      { id: "feat-ingest", label: "Document ingest", sub: "PDF / DOCX / HTML / XLSX readers", extra: "ingest", env: [] },
      { id: "feat-redis", label: "Redis", sub: "Caching backend", extra: "redis", env: [] },
      { id: "feat-serve", label: "Serve (FastAPI)", sub: "Built-in HTTP server", extra: "serve", env: [] },
      { id: "feat-scaffold", label: "Scaffold", sub: "Jinja2 templates for codegen", extra: "scaffold", env: [] },
      { id: "feat-sandbox", label: "Sandbox tools", sub: "tukuy for tool execution", extra: "sandbox", env: [] },
    ],
  },
  {
    id: "rag",
    title: "RAG Components",
    desc: "Mix and match document readers, tokenizers, and vector stores.",
    items: [
      { id: "rag-pdf", label: "PDF reader", extra: "rag-pdf", env: [] },
      { id: "rag-docx", label: "DOCX reader", extra: "rag-docx", env: [] },
      { id: "rag-html", label: "HTML reader", extra: "rag-html", env: [] },
      { id: "rag-epub", label: "EPUB reader", extra: "rag-epub", env: [] },
      { id: "rag-xlsx", label: "XLSX reader", extra: "rag-xlsx", env: [] },
      { id: "rag-token", label: "tiktoken counting", extra: "rag-token", env: [] },
      { id: "rag-semantic", label: "Semantic chunking", sub: "numpy", extra: "rag-semantic", env: [] },
      { id: "rag-vs-chroma", label: "Chroma vector store", extra: "rag-vs-chroma", env: [] },
      { id: "rag-vs-pinecone", label: "Pinecone vector store", extra: "rag-vs-pinecone", env: [] },
      { id: "rag-vs-qdrant", label: "Qdrant vector store", extra: "rag-vs-qdrant", env: [] },
      { id: "rag-vs-pgvector", label: "pgvector (Postgres)", extra: "rag-vs-pgvector", env: [] },
      { id: "rag-vs-faiss", label: "FAISS (in-process)", extra: "rag-vs-faiss", env: [] },
      { id: "rag-vs-weaviate", label: "Weaviate vector store", extra: "rag-vs-weaviate", env: [] },
      { id: "rag-hybrid", label: "Hybrid retrieval (BM25)", extra: "rag-hybrid", env: [] },
    ],
  },
  {
    id: "dev",
    title: "Development",
    desc: "Test runner, linting, and the kitchen-sink meta-extra.",
    items: [
      { id: "meta-all", label: "all — every provider extra", sub: "openai + anthropic + google + groq + toon + pandas + sandbox", extra: "all", env: [] },
      { id: "meta-test", label: "test", sub: "pytest + asyncio + all", extra: "test", env: [] },
      { id: "meta-dev", label: "dev", sub: "test + ruff", extra: "dev", env: [] },
    ],
  },
];

// ---- State ----

const state = {
  selected: new Set(),
  filter: "",
};

const STORAGE_KEY = "prompture-installer.selected.v1";

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const ids = JSON.parse(raw);
    if (Array.isArray(ids)) {
      for (const id of ids) state.selected.add(id);
    }
  } catch {
    // ignore corrupt state
  }
}

function persistState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...state.selected]));
  } catch {
    // ignore quota errors
  }
}

// ---- Lookups ----

const ITEM_INDEX = new Map();
for (const section of SECTIONS) {
  for (const item of section.items) {
    ITEM_INDEX.set(item.id, { ...item, sectionId: section.id, sectionTitle: section.title });
  }
}

// ---- Rendering ----

function renderSections() {
  const root = document.getElementById("sections");
  root.innerHTML = "";
  for (const section of SECTIONS) {
    const sec = document.createElement("div");
    sec.className = "section";
    sec.dataset.sectionId = section.id;
    sec.innerHTML = `
      <h3>${section.title}</h3>
      ${section.desc ? `<p class="desc">${section.desc}</p>` : ""}
      <div class="grid"></div>
    `;
    const grid = sec.querySelector(".grid");
    for (const item of section.items) {
      const el = document.createElement("label");
      el.className = "item";
      el.dataset.itemId = item.id;
      el.dataset.search = `${item.label} ${item.sub || ""} ${item.extra || ""}`.toLowerCase();
      const tag = item.extra ? `<span class="tag">[${item.extra}]</span>` : "";
      el.innerHTML = `
        <input type="checkbox" />
        <span>
          <span class="label">${item.label}${tag}</span>
          ${item.sub ? `<span class="sub">${item.sub}</span>` : ""}
        </span>
      `;
      const cb = el.querySelector("input");
      cb.addEventListener("change", () => toggle(item.id, cb.checked));
      grid.appendChild(el);
    }
    root.appendChild(sec);
  }
  syncSelection();
}

function syncSelection() {
  for (const el of document.querySelectorAll(".item")) {
    const id = el.dataset.itemId;
    const on = state.selected.has(id);
    el.classList.toggle("selected", on);
    const cb = el.querySelector("input");
    if (cb.checked !== on) cb.checked = on;
  }
}

function applyFilter() {
  const q = state.filter.trim().toLowerCase();
  for (const sec of document.querySelectorAll(".section")) {
    let visible = 0;
    for (const item of sec.querySelectorAll(".item")) {
      const match = !q || item.dataset.search.includes(q);
      item.classList.toggle("hidden", !match);
      if (match) visible++;
    }
    sec.style.display = visible === 0 && q ? "none" : "";
  }
}

// ---- Output generation ----

function buildInstall() {
  const extras = new Set();
  for (const id of state.selected) {
    const item = ITEM_INDEX.get(id);
    if (item && item.extra) extras.add(item.extra);
  }
  const sorted = [...extras].sort();
  if (sorted.length === 0) {
    return { cmd: "pip install prompture", note: "Base install. No optional extras selected." };
  }
  const joined = sorted.join(",");
  return {
    cmd: `pip install "prompture[${joined}]"`,
    note: `${sorted.length} extra${sorted.length === 1 ? "" : "s"}: ${joined}`,
  };
}

function buildEnv() {
  // Group selected items by section, preserving section order.
  const bySection = new Map();
  for (const section of SECTIONS) bySection.set(section.id, []);
  for (const id of state.selected) {
    const item = ITEM_INDEX.get(id);
    if (!item || item.env.length === 0) continue;
    bySection.get(item.sectionId).push(item);
  }

  // Track keys already emitted so we don't duplicate when the same provider
  // appears under multiple modalities (e.g. OpenAI for chat + image).
  const seenKeys = new Set();

  const lines = [];
  let any = false;
  for (const section of SECTIONS) {
    const items = bySection.get(section.id);
    if (!items || items.length === 0) continue;

    // Filter each item's env lines, dropping assignment lines whose key was
    // already emitted. Keep comment lines as-is but skip blocks that end up
    // empty after dedupe.
    const itemBlocks = [];
    for (const item of items) {
      const kept = [];
      for (const line of item.env) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) {
          kept.push(line);
          continue;
        }
        const eq = trimmed.indexOf("=");
        const key = eq === -1 ? trimmed : trimmed.slice(0, eq);
        if (seenKeys.has(key)) continue;
        seenKeys.add(key);
        kept.push(line);
      }
      // Drop blocks that have no assignment lines left (only comments).
      const hasAssignment = kept.some((l) => {
        const t = l.trim();
        return t && !t.startsWith("#");
      });
      if (hasAssignment) itemBlocks.push({ item, kept });
    }

    if (itemBlocks.length === 0) continue;

    if (any) lines.push("");
    lines.push(`# ─── ${section.title} ${"─".repeat(Math.max(2, 60 - section.title.length))}`);
    for (const { item, kept } of itemBlocks) {
      lines.push(`# ${item.label}`);
      for (const line of kept) lines.push(line);
      lines.push("");
    }
    if (lines[lines.length - 1] === "") lines.pop();
    any = true;
  }
  return any ? lines.join("\n") + "\n" : "";
}

function buildSnippet() {
  // Pick a single representative model to seed the example.
  const preferenceOrder = [
    ["openai", "openai/gpt-4o-mini"],
    ["anthropic", "claude/claude-haiku-4-5-20251001"],
    ["google", "google/gemini-2.5-flash"],
    ["groq", "groq/llama-3.3-70b-versatile"],
    ["openrouter", "openrouter/openai/gpt-4o-mini"],
    ["ollama", "ollama/llama3.1:8b"],
    ["lmstudio", "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"],
    ["grok", "grok/grok-4-fast-reasoning"],
    ["mistral", "mistral/mistral-large-latest"],
    ["deepseek", "deepseek/deepseek-chat"],
    ["bedrock", "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"],
  ];
  let model = "ollama/llama3.1:8b";
  for (const [id, m] of preferenceOrder) {
    if (state.selected.has(id)) {
      model = m;
      break;
    }
  }

  return `from prompture import extract_with_model
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int | None = None


result = extract_with_model(
    text="Maria is 32 and lives in Lisbon.",
    model_cls=Person,
    model="${model}",
)
print(result.data)
`;
}

function refreshOutput() {
  const { cmd, note } = buildInstall();
  document.getElementById("install-cmd").textContent = cmd;
  document.getElementById("install-note").textContent = note;

  const env = buildEnv();
  document.getElementById("env-file").textContent = env;
  const envNote = document.getElementById("env-note");
  if (!env) {
    envNote.textContent = "Pick a provider to populate .env.";
  } else {
    const lines = env.split("\n").filter((l) => l && !l.startsWith("#")).length;
    envNote.textContent = `${lines} environment variable${lines === 1 ? "" : "s"} for your selected providers.`;
  }

  document.getElementById("snippet").textContent = buildSnippet();
}

// ---- Interactions ----

function toggle(id, on) {
  if (on) state.selected.add(id);
  else state.selected.delete(id);
  persistState();
  syncSelection();
  refreshOutput();
}

function clearAll() {
  state.selected.clear();
  persistState();
  syncSelection();
  refreshOutput();
}

async function copyTarget(targetId, btn) {
  const text = document.getElementById(targetId).textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Fallback: select-and-copy
    const range = document.createRange();
    range.selectNodeContents(document.getElementById(targetId));
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    document.execCommand("copy");
    sel.removeAllRanges();
  }
  const originalLabel = btn.textContent;
  btn.textContent = "Copied";
  btn.classList.add("copied");
  setTimeout(() => {
    btn.textContent = originalLabel;
    btn.classList.remove("copied");
  }, 1200);
}

function downloadEnv() {
  const content = document.getElementById("env-file").textContent;
  if (!content) return;
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = ".env";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---- Bootstrap ----

function init() {
  loadState();
  renderSections();
  refreshOutput();

  document.getElementById("search").addEventListener("input", (e) => {
    state.filter = e.target.value;
    applyFilter();
  });
  document.getElementById("clear-btn").addEventListener("click", clearAll);
  document.getElementById("download-env").addEventListener("click", downloadEnv);
  for (const btn of document.querySelectorAll(".copy-btn")) {
    btn.addEventListener("click", () => copyTarget(btn.dataset.target, btn));
  }
}

init();
