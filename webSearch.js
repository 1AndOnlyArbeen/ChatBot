// =============================================================================
// Web search module - DuckDuckGo HTML scraping
// -----------------------------------------------------------------------------
// - No API key, no signup, no rate-limit headache
// - Uses html.duckduckgo.com which renders results server-side as HTML
// - Returns top N results with { title, url, snippet }
// - Pure Node, no external deps (uses Node 20+ native fetch)
// =============================================================================

const UA =
  "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0";

// Strip HTML tags and decode common entities
function stripTags(html = "") {
  return html
    .replace(/<[^>]*>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// DuckDuckGo wraps result links in a redirect. Unwrap to the real URL.
//   //duckduckgo.com/l/?uddg=ENCODED_REAL_URL&...
function unwrapDdgRedirect(href = "") {
  try {
    if (!href) return href;
    const m = href.match(/[?&]uddg=([^&]+)/);
    if (m) return decodeURIComponent(m[1]);
    if (href.startsWith("//")) return "https:" + href;
    return href;
  } catch {
    return href;
  }
}

/**
 * Search DuckDuckGo and return top N results.
 * @param {string} query
 * @param {number} maxResults
 * @returns {Promise<Array<{title: string, url: string, snippet: string}>>}
 */
export async function searchWeb(query, maxResults = 3) {
  const q = (query || "").trim();
  if (!q) return [];

  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`;

  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 8000);

    const res = await fetch(url, {
      method: "POST",
      headers: {
        "User-Agent": UA,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: `q=${encodeURIComponent(q)}`,
      signal: ctrl.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      console.error("[websearch] HTTP", res.status);
      return [];
    }
    const html = await res.text();

    // DDG HTML uses repeating result blocks like:
    //   <div class="result results_links ...">
    //     ... <a class="result__a" href="...">TITLE</a> ...
    //     ... <a class="result__snippet" ...>SNIPPET</a> ...
    //   </div>
    const blockRe =
      /<div[^>]+class="[^"]*result[^"]*results_links[^"]*"[^>]*>([\s\S]*?)(?=<div[^>]+class="[^"]*result[^"]*results_links|<\/div>\s*<\/div>\s*<div[^>]+class="result__hidden|$)/g;

    const titleRe =
      /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/;
    const snippetRe =
      /<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/;

    const results = [];
    let m;
    while ((m = blockRe.exec(html)) !== null && results.length < maxResults) {
      const block = m[1];
      const t = titleRe.exec(block);
      if (!t) continue;
      const s = snippetRe.exec(block);

      const title = stripTags(t[2]);
      const realUrl = unwrapDdgRedirect(t[1]);
      const snippet = s ? stripTags(s[1]) : "";

      // skip ads / empty entries
      if (!title || !realUrl || realUrl.includes("/y.js?")) continue;

      results.push({ title, url: realUrl, snippet });
    }

    return results;
  } catch (err) {
    console.error("[websearch] error:", err.message);
    return [];
  }
}

/**
 * Format search results as a plain-text block to inject into LLM prompts.
 */
export function formatResultsForPrompt(results) {
  if (!results || !results.length) {
    return "(no web results found)";
  }
  return results
    .map(
      (r, i) =>
        `[${i + 1}] ${r.title}\n    URL: ${r.url}\n    ${r.snippet}`,
    )
    .join("\n\n");
}
