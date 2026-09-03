# Shimmi Changelog

---

## v3.17.4 — 2026-09-03

> **Note:** this changelog was not kept up to date between v3.2.0 and v3.17.3
> (the code advanced through ~15 versions — see each module's own header
> comment for its version, e.g. `app/agent_engine.py`, `app/main.py` — without
> a matching entry here). This entry resumes changelog tracking; the gap
> itself was not reconstructed since the intermediate change history wasn't
> available to fill in accurately.

### 🟠 Bugs Fixed

**FIX-DEBOUNCE — Chat-wide debounce could silently starve busy chats**

`CHAT_LAST_MSG_TS` (`app/main.py`) was keyed by `chat_id` alone and the
timestamp was advanced on *every* check, including a debounced one. In a
group chat busier than `MESSAGE_DEBOUNCE_MS` (or any chat with
`ALLOW_NLP_WITHOUT_PREFIX=1`), this degenerated into a sliding window that
could suppress every message indefinitely — not just genuine duplicate
resends. Now keyed by `(chat_id, sender_key)` and only advanced when a
message is actually accepted.

**FIX-7b — Non-rate-limit worker errors left the user with total silence**

`process_message` only sends a user-facing reply when the failure is a
rate-limit (handled internally); any other unhandled exception (a DB error,
a WAHA outage, a bug) was caught in `_chat_worker`, logged, and otherwise
silently dropped — no reply at all. The worker now sends a best-effort
"something went wrong" message on any unhandled exception.

**FIX-VERSION — `/healthz` reported a stale hardcoded version**

`GET /healthz` returned `"version": "3.3.0"` regardless of the code actually
running (every module header was already at v3.17.x). Added `APP_VERSION` in
`app/main.py` as the single source of truth for the reported version.

---

## v3.2.0 — 2026-03-15

### 🔴 Critical Bug Fix

**FIX-CHAIN — Groq 8B never reached when Gemini + Groq 70B both fail**

The `_groq_raw` fallback loop iterated providers but used `return await _call_llm(...)` 
for each fallback. When Groq 70B also hit its 429 limit, that exception escaped the 
loop immediately — Groq 8B (with 500K tokens/day remaining) was never tried. The user 
received a fatal error with a full Python traceback instead of a response.

Fix: each candidate is wrapped in its own `try/except`. On rate-limit or timeout the 
loop logs "fallback.exhausted" and continues to the next provider. Only raises when 
the entire candidate list is empty.

**Impact on the log you sent:** The `worker.msg_error` stack trace ending with 
`groq.RateLimitError: Rate limit reached (TPD): Limit 100000, Used 96474` was caused 
entirely by this bug. Groq 8B had 500K/day available and was being skipped.

---

### 🟠 Bugs Fixed

**FIX-RPD — Gemini daily quota sets 2-hour cooldown (not 5 minutes)**

`_parse_retry_after()` now detects the Gemini RPD error signature 
(`"You exceeded your current quota"`) and sets a 2-hour cooldown. Previously the 
default 300s cooldown meant Gemini was retried and immediately failed on every 
message for the rest of the day, emitting a `WARNING` log pair per request.

**FIX-NOISE — 15 ephemeral keys stripped from LLM prompt**

`_clean_facts()` now filters out keys that are transient activity records rather 
than durable personal facts: `result_*`, `recent_activity`, `next_meeting_*`, 
`semester`, `year`, `course`, `online_courses`, `friend_since`, 
`previous_startup_status`, `previous_employer`, `next_trip_*`. These stay in 
SQLite for audit/consolidation but stop being injected into every orchestrator 
call. Saves 200–500 tokens per request.

**FIX-TOOL — Keyword routing when Groq omits `tool_call`**

When Groq 70B acts as fallback orchestrator it frequently omits the structured 
`tool_call` JSON block. The new `_keyword_tool_from_query()` detects weather / 
stocks / news / currency / timezone queries from the query text and routes them 
to the correct MCP endpoint. Structured live-data tools now work correctly even 
when Gemini is fully exhausted for the day.

**FIX-TIME — "What time is it?" answered from server clock**

The LLM reads stale timestamps from conversation context and hallucinates the 
current time. `_try_time_shortcut()` intercepts short time/date queries before 
any LLM call and answers directly from the server clock — zero tokens, always 
accurate, honours `APP_TIMEZONE`.

**FIX-MCP-LOG — MCP error logging now shows actual errors**

`mcp.error  path=/stocks  err=` was logging empty strings because 
`str(httpx.HTTPStatusError)` is empty. Now logs `repr(exc)` and the exception 
type name, so `/stocks` failures are diagnosable.

**FIX-MCP-DUP — Duplicate `mcp_format()` removed from mcp_client.py**

`mcp_format()` was defined twice. The duplicate definition silently replaced the 
first one; the first (correct) definition with the `timeout=3.0` parameter was 
discarded. Cleaned up to a single correct definition.

**FIX-HTTP — Port-scanner noise suppressed**

`_InvalidHttpFilter` added to `logging_setup.py` suppresses uvicorn's 
`"Invalid HTTP request received"` WARNING, which fires on every port-scan probe 
or TLS health-check hitting the plain-HTTP port. Harmless but previously 
cluttered logs on every external scan.

---

### 🟢 MCP Server v2.0.0

**CACHE — TTL response cache for all external-API endpoints**

| Endpoint  | TTL    | Rationale |
|-----------|--------|-----------|
| `/weather`  | 10 min | Weather changes slowly |
| `/stocks`   | 3 min  | Data is 15-min delayed anyway |
| `/news`     | 5 min  | Headlines don't change per-minute |
| `/currency` | 1 hour | ECB rates update once daily |
| `/timezone` | 24 hr  | City→TZ mapping is static |

Eliminates redundant external API calls when multiple users ask about the same 
city or stock within the TTL window.

**FORMAT — `POST /format` endpoint (zero LLM tokens)**

Deterministic WhatsApp formatting in pure Python. Replaces the Groq 8B 
`_format_whatsapp()` LLM call for routine formatting. Saves ~50–100K Groq 
tokens/day. Rules: `**bold**` → `*bold*`, bullet normalisation, table→bullets, 
code-fence removal, filler phrase stripping, 3800-char hard cap. `agent_engine` 
tries MCP `/format` first; falls back to LLM only if MCP is unreachable.

**STOCKS-2 — Per-ticker timeout guard**

A single slow or hung yfinance ticker no longer stalls the entire `/stocks` 
response. Each ticker is fetched with `asyncio.wait_for(timeout=8s)`; hung 
tickers return `{"symbol": "...", "error": "timeout"}` instead of blocking.

**HTTP-1 — httpx client timeout 30s → 12s**

Stocks calls were occasionally hanging for 25+ seconds per ticker. Reduced 
global timeout to 12s to fail fast and let the bot send a partial result.

---

### MCP Server — what it offloads and why

| What used to run in the bot | Now runs in MCP | Saving |
|-----------------------------|-----------------|--------|
| `_format_whatsapp()` Groq 8B call | `POST /format` pure Python | ~50–100K tokens/day |
| Repeated weather fetches | Cached 10 min | External API calls |
| Repeated stock fetches | Cached 3 min | yfinance latency |
| Repeated news fetches | Cached 5 min | GNews/RSS quota |
| `/datetime` (already existed) | Available for time queries | 0 LLM calls |

The MCP server is the right place for: deterministic transformations, external 
API calls with caching, and any computation that doesn't need LLM reasoning. 
The bot should remain thin — route everything it can to MCP, keep LLM calls 
for reasoning and memory tasks only.

---

## v3.1.0 — 2026-03-12 (previous session)

- Gemini added as primary orchestrator (Groq fallback unchanged)
- Per-provider circuit breakers
- Token budget tracker
- Facts shortcut v2 (30+ patterns, zero tokens)
- Junk fact filter (strips unknown/none/null before prompt)
- Fire-and-forget exception handling fixed
- asyncio.get_event_loop() → get_running_loop() (Python 3.12+)
- Retry-after parsing from Groq/Gemini error messages
- Live search 413 handling with automatic query truncation
- User-facing error messages when all providers exhausted
