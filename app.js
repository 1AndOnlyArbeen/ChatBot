// =============================================================================
// Matat Portfolio - Documentation Chatbot (Ultron)
// -----------------------------------------------------------------------------
// - Express server on http://localhost:3000
// - Engine: Anthropic Claude when ANTHROPIC_API_KEY is set, else local Ollama
// - Loads docs.txt ONCE at startup and keeps it in memory
// - Persistent conversation memory in conversation_history.json
//   stored as SESSIONS (Claude-style threads), not a flat list
// - Detects Hebrew vs English questions and replies in the same language
// =============================================================================

import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { ChatOllama } from "@langchain/ollama";
import Anthropic from "@anthropic-ai/sdk";
import { searchWeb, formatResultsForPrompt } from "./webSearch.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
const PORT = process.env.PORT || 3000;

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "mistral";

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || "claude-haiku-4-5";

const MAX_SESSIONS = 100;       // sessions kept on disk
const TITLE_MAX_LEN = 60;       // sidebar title truncation

const DOCS_FILE = path.join(__dirname, "docs.txt");
const HISTORY_FILE = path.join(__dirname, "conversation_history.json");

// -----------------------------------------------------------------------------
// In-memory state (loaded once at startup)
// -----------------------------------------------------------------------------
let documentation = "";

// sessions = [{ id, title, language, createdAt, updatedAt,
//               messages: [{ id, role, content, timestamp, language }] }]
let sessions = [];

// -----------------------------------------------------------------------------
// Documentation loader
// -----------------------------------------------------------------------------
function loadDocumentation() {
  try {
    documentation = fs.readFileSync(DOCS_FILE, "utf8");
    console.log(`[docs] Loaded ${documentation.length} chars from docs.txt`);
  } catch (err) {
    console.error("[docs] Failed to read docs.txt:", err.message);
    documentation = "(no documentation loaded)";
  }
}

// -----------------------------------------------------------------------------
// Sessions persistence (with one-time migration from flat history)
// -----------------------------------------------------------------------------
function loadSessions() {
  if (!fs.existsSync(HISTORY_FILE)) {
    console.log("[history] No prior history file - starting fresh");
    return;
  }
  try {
    const raw = fs.readFileSync(HISTORY_FILE, "utf8");
    const parsed = JSON.parse(raw);

    // New format: { sessions: [...] }
    if (parsed && Array.isArray(parsed.sessions)) {
      sessions = parsed.sessions;
      console.log(`[history] Restored ${sessions.length} sessions`);
      return;
    }

    // Legacy flat array - migrate each entry into its own one-turn session
    if (Array.isArray(parsed)) {
      sessions = parsed.map((c) => ({
        id: c.id || `ses_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        title: truncateTitle(c.question || "Conversation"),
        language: c.language || "en",
        createdAt: c.timestamp || new Date().toISOString(),
        updatedAt: c.timestamp || new Date().toISOString(),
        messages: [
          {
            id: `msg_${Math.random().toString(36).slice(2, 10)}`,
            role: "user",
            content: c.question || "",
            timestamp: c.timestamp || new Date().toISOString(),
            language: c.language || "en",
          },
          {
            id: `msg_${Math.random().toString(36).slice(2, 10)}`,
            role: "assistant",
            content: c.answer || "",
            timestamp: c.timestamp || new Date().toISOString(),
            language: c.language || "en",
          },
        ],
      }));
      console.log(`[history] Migrated ${sessions.length} legacy entries to sessions`);
      saveSessions();
      return;
    }

    console.warn("[history] Unrecognized format - starting fresh");
  } catch (err) {
    console.error("[history] Failed to load history:", err.message);
    sessions = [];
  }
}

function saveSessions() {
  try {
    fs.writeFileSync(
      HISTORY_FILE,
      JSON.stringify({ version: 2, sessions }, null, 2),
      "utf8",
    );
  } catch (err) {
    console.error("[history] Failed to save:", err.message);
  }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
const HEBREW_REGEX = /[֐-׿]/;
function detectLanguage(text = "") {
  return HEBREW_REGEX.test(text) ? "he" : "en";
}

function truncateTitle(text = "") {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > TITLE_MAX_LEN ? t.slice(0, TITLE_MAX_LEN - 1) + "…" : t;
}

function findSession(id) {
  return sessions.find((s) => s.id === id);
}

function newSession(language = "en") {
  const now = new Date().toISOString();
  const session = {
    id: `ses_${randomUUID().slice(0, 12)}`,
    title: "New chat",
    language,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
  sessions.push(session);
  if (sessions.length > MAX_SESSIONS) {
    // drop the oldest by createdAt
    sessions.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
    sessions = sessions.slice(sessions.length - MAX_SESSIONS);
  }
  saveSessions();
  return session;
}

// Sidebar-friendly summary (no full message bodies)
function summarizeSession(s) {
  return {
    id: s.id,
    title: s.title,
    language: s.language,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
    messageCount: s.messages.length,
  };
}

// -----------------------------------------------------------------------------
// Prompt builders
//   `mode` is "docs" (default - use docs.txt only)
//             "web"  (use web search results, ignore docs)
//   `webContext` is a string of search-result snippets, only used in web mode
// -----------------------------------------------------------------------------
function ollamaSystemRules(language, mode = "docs", webContext = "") {
  const langRule =
    language === "he"
      ? `The user's question is in HEBREW. Reply in HEBREW.
Use polite, friendly, professional Israeli Hebrew - modern but no street slang.
Avoid: יא פנים, סבבה, טוב אחי, אדם, יא חבר.`
      : `The user's question is in ENGLISH. Reply in ENGLISH.
Use a friendly but professional tone - like a helpful colleague.
Avoid: "yo", "bro", "no cap", "lol", "haha".`;

  const personaCommon = `You are "Ultron" - a chatbot.

IDENTITY:
- Your name is Ultron. Never say you are an AI assistant, language model, Claude, Mistral, or anything else.

RESPONSE FORMAT (strict):
- NO preamble. Do not start with "Hey", "Sure", "Happy to help", "Great question".
- Start with the actual answer immediately.
- Keep it SHORT - one or two sentences usually.
- No emoji unless genuinely needed (almost never).
- Tone: helpful colleague. Direct, friendly, professional.

VERBATIM NAMES:
- Copy proper nouns (people, products, brands) EXACTLY as in the source.
- Do not auto-correct or normalize names.

LANGUAGE RULE:
${langRule}`;

  if (mode === "web") {
    return `${personaCommon}

WEB MODE:
- You are answering using LIVE WEB SEARCH RESULTS below.
- Use ONLY the search results as your source. Do NOT use training or general knowledge.
- If the results don't contain the answer, say so honestly.
- Cite sources by adding "[1]", "[2]" etc. after facts, matching the result numbers below.
- Keep the answer concise — synthesize, don't dump.

WEB SEARCH RESULTS:
"""
${webContext || "(no results)"}
"""`;
  }

  // default: docs mode
  return `${personaCommon}

DOCS-ONLY RULE:
- Your ONLY source is the DOCUMENTATION below. Do not use training, common sense, or outside knowledge.
- If the answer isn't in the docs, reply EXACTLY (in the right language):
    EN: "That's not in my documentation, so I can't answer that."
    HE: "המידע הזה לא נמצא בתיעוד שלי, אז אני לא יכול לענות על זה."

DOCUMENTATION:
"""
${documentation}
"""`;
}

// Flatten current session + new question into a single Ollama prompt.
function buildOllamaPrompt(session, question, language, mode = "docs", webContext = "") {
  const transcript = session.messages
    .map((m) => (m.role === "user" ? `User: ${m.content}` : `Ultron: ${m.content}`))
    .join("\n\n");

  return `${ollamaSystemRules(language, mode, webContext)}

CONVERSATION SO FAR:
${transcript || "(this is the start of the conversation)"}

NEW QUESTION:
${question}

Now answer.`;
}

// Claude system: stable persona/rules go first (cacheable),
// language + per-request context (docs OR web results) go after.
function claudeSystemBlocks(language, mode = "docs", webContext = "") {
  const stablePersona = `You are "Ultron" - a chatbot.

IDENTITY:
- Your name is Ultron. Never say you are an AI assistant, language model, Claude, Mistral, or anything else.

RESPONSE FORMAT (strict):
- NO preamble. Do not start with "Hey", "Sure", "Happy to help", "Great question".
- Start with the actual answer immediately.
- Keep it SHORT - one or two sentences usually.
- No emoji unless genuinely needed.
- Tone: helpful colleague. Direct, friendly, professional.

VERBATIM NAMES:
- Copy proper nouns (people, products, brands) EXACTLY as in the source.
- Do not auto-correct or normalize names.`;

  const langRule =
    language === "he"
      ? `The user's question is in HEBREW. Reply in HEBREW.
Polite, friendly, professional Israeli Hebrew - no street slang.`
      : `The user's question is in ENGLISH. Reply in ENGLISH.
Friendly but professional - like a helpful colleague.`;

  if (mode === "web") {
    const webBlock = `WEB MODE:
- You are answering from LIVE WEB SEARCH RESULTS below.
- Use ONLY these results. Do not use training/general knowledge.
- If results don't contain the answer, say so.
- Cite sources with "[1]", "[2]" etc. matching result numbers.
- Synthesize; don't dump.

WEB SEARCH RESULTS:
"""
${webContext || "(no results)"}
"""`;

    return [
      { type: "text", text: stablePersona, cache_control: { type: "ephemeral" } },
      { type: "text", text: langRule },
      { type: "text", text: webBlock },
    ];
  }

  // default: docs mode (cacheable since docs.txt is stable)
  const docsBlock = `DOCS-ONLY RULE:
- Your ONLY source is the DOCUMENTATION below. Do not use training, common sense, or outside knowledge.
- If the answer isn't in the docs, reply EXACTLY (in the right language):
    EN: "That's not in my documentation, so I can't answer that."
    HE: "המידע הזה לא נמצא בתיעוד שלי, אז אני לא יכול לענות על זה."

DOCUMENTATION:
"""
${documentation}
"""`;

  return [
    { type: "text", text: stablePersona, cache_control: { type: "ephemeral" } },
    { type: "text", text: docsBlock, cache_control: { type: "ephemeral" } },
    { type: "text", text: langRule },
  ];
}

// -----------------------------------------------------------------------------
// LLM clients
// -----------------------------------------------------------------------------
const ollama = new ChatOllama({
  baseUrl: OLLAMA_URL,
  model: OLLAMA_MODEL,
  temperature: 0.5,
});

const claude = ANTHROPIC_API_KEY ? new Anthropic({ apiKey: ANTHROPIC_API_KEY }) : null;

async function askOllama(prompt) {
  const response = await ollama.invoke(prompt);
  return typeof response === "string" ? response : response.content;
}

async function askClaude(session, question, language, mode = "docs", webContext = "") {
  const messages = [];
  for (const m of session.messages) {
    messages.push({ role: m.role, content: m.content });
  }
  messages.push({ role: "user", content: question });

  const response = await claude.messages.create({
    model: ANTHROPIC_MODEL,
    max_tokens: 1024,
    system: claudeSystemBlocks(language, mode, webContext),
    messages,
  });

  const answer = response.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n")
    .trim();

  const u = response.usage || {};
  if (u.cache_read_input_tokens || u.cache_creation_input_tokens) {
    console.log(
      `[claude] cache: read=${u.cache_read_input_tokens || 0} write=${u.cache_creation_input_tokens || 0} fresh=${u.input_tokens || 0} out=${u.output_tokens || 0}`,
    );
  }
  return answer;
}

// -----------------------------------------------------------------------------
// Express
// -----------------------------------------------------------------------------
const app = express();
app.use(express.json({ limit: "1mb" }));

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

app.use(express.static(path.join(__dirname, "public")));

app.get("/healthz", (_req, res) => {
  res.json({
    ok: true,
    engine: claude ? "anthropic" : "ollama",
    model: claude ? ANTHROPIC_MODEL : OLLAMA_MODEL,
    docsLoaded: documentation.length > 0,
    sessionCount: sessions.length,
  });
});

// List sessions for the sidebar (newest first, summary only)
app.get("/sessions", (_req, res) => {
  const list = [...sessions]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .map(summarizeSession);
  res.json({ sessions: list });
});

// Get one session with all its messages
app.get("/sessions/:id", (req, res) => {
  const s = findSession(req.params.id);
  if (!s) return res.status(404).json({ error: "Session not found" });
  res.json({ session: s });
});

// Create a brand-new (empty) session
app.post("/sessions", (_req, res) => {
  const s = newSession();
  res.json({ session: s });
});

// Delete a single session
app.delete("/sessions/:id", (req, res) => {
  const before = sessions.length;
  sessions = sessions.filter((s) => s.id !== req.params.id);
  if (sessions.length === before) return res.status(404).json({ error: "Session not found" });
  saveSessions();
  res.json({ ok: true });
});

// Wipe ALL sessions
app.post("/clear-history", (_req, res) => {
  sessions = [];
  try {
    if (fs.existsSync(HISTORY_FILE)) fs.unlinkSync(HISTORY_FILE);
  } catch (err) {
    console.error("[history] Failed to delete file:", err.message);
  }
  console.log("[history] All sessions cleared");
  res.json({ ok: true });
});

// Send a message into a session (creates a session if sessionId not provided)
//   body: { question, sessionId?, mode? }   mode: "docs" (default) | "web"
app.post("/chat", async (req, res) => {
  const question = (req.body?.question || "").trim();
  if (!question) {
    return res.status(400).json({ error: "Missing 'question' in request body" });
  }
  const language = detectLanguage(question);
  const mode = req.body?.mode === "web" ? "web" : "docs";

  // get or create session
  let session = req.body?.sessionId ? findSession(req.body.sessionId) : null;
  if (!session) {
    session = newSession(language);
  }

  // record the user message before calling the LLM
  const userMsg = {
    id: `msg_${randomUUID().slice(0, 10)}`,
    role: "user",
    content: question,
    timestamp: new Date().toISOString(),
    language,
    mode,
  };

  // If web mode: run a search BEFORE prompting the model
  let webResults = [];
  let webContext = "";
  if (mode === "web") {
    console.log(`[chat] web search: "${question.slice(0, 60)}"`);
    webResults = await searchWeb(question, 5);
    webContext = formatResultsForPrompt(webResults);
    console.log(`[chat] got ${webResults.length} web results`);
  }

  try {
    const answer = claude
      ? await askClaude(session, question, language, mode, webContext)
      : await askOllama(buildOllamaPrompt(session, question, language, mode, webContext));

    const assistantMsg = {
      id: `msg_${randomUUID().slice(0, 10)}`,
      role: "assistant",
      content: answer,
      timestamp: new Date().toISOString(),
      language,
      mode,
      // include sources only in web mode so the UI can render them
      sources: mode === "web" ? webResults.map((r) => ({ title: r.title, url: r.url })) : undefined,
    };

    // commit both messages atomically into the session
    session.messages.push(userMsg, assistantMsg);
    session.updatedAt = assistantMsg.timestamp;
    // first user message becomes the title
    if (session.messages.length === 2 && session.title === "New chat") {
      session.title = truncateTitle(question);
      session.language = language;
    }
    saveSessions();

    const engine = claude ? "claude" : "ollama";
    console.log(`[chat:${engine}:${mode}] (${language}) ${session.id} Q: ${question.slice(0, 60)}`);
    return res.json({
      sessionId: session.id,
      title: session.title,
      userMessage: userMsg,
      assistantMessage: assistantMsg,
    });
  } catch (err) {
    let errorKind = "generic";
    if (claude) {
      if (err instanceof Anthropic.AuthenticationError) errorKind = "auth";
      else if (err instanceof Anthropic.RateLimitError) errorKind = "rate_limit";
      else if (err instanceof Anthropic.APIError) errorKind = "api";
    }
    console.error(`[chat] error (${errorKind}):`, err.message);

    const fallback =
      language === "he"
        ? "אני (Ultron) לא מצליח להתחבר למוח שלי כרגע. נסו שוב עוד רגע."
        : "I (Ultron) can't reach my brain right now. Please try again in a moment.";
    return res.status(500).json({
      error: err.message,
      assistantMessage: {
        id: `msg_${randomUUID().slice(0, 10)}`,
        role: "assistant",
        content: fallback,
        timestamp: new Date().toISOString(),
        language,
      },
      language,
      errorKind,
    });
  }
});

// -----------------------------------------------------------------------------
// Boot
// -----------------------------------------------------------------------------
loadDocumentation();
loadSessions();

app.listen(PORT, () => {
  console.log("===========================================");
  console.log(`  Ultron chatbot running on :${PORT}`);
  if (claude) {
    console.log(`  Engine : Anthropic Claude (${ANTHROPIC_MODEL})`);
  } else {
    console.log(`  Engine : Ollama (${OLLAMA_URL}, model: ${OLLAMA_MODEL})`);
  }
  console.log(`  Sessions loaded: ${sessions.length}`);
  console.log(`  UI     : http://localhost:${PORT}`);
  console.log("===========================================");
});
