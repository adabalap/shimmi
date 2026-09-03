# 🤖 Shimmi — Stateful, Memory-Intelligent WhatsApp AI Bot

> *"Your personal AI companion on WhatsApp — remembers you, searches the world, reminds you."*

Shimmi (also known as **Spock** / **Chitti**) is a production-grade, **stateful AI chatbot delivered over WhatsApp**. It combines a multi-model Groq LLM orchestration engine with a persistent per-user memory layer, live web search, semantic conversation recall, and a proactive reminder system — all accessible from any ordinary WhatsApp chat.

---

## ✨ Feature Highlights

| Capability | What it does |
|---|---|
| 🧠 **Persistent Memory** | Remembers facts per user (name, city, preferences, lists) across all sessions via SQLite |
| 📚 **Semantic Context** | Pulls relevant past conversation snippets from ChromaDB before every reply |
| 🗣️ **Multi-turn History** | Injects ordered conversation turns so the LLM always has proper dialogue context |
| 🔍 **Live Web Search** | Answers weather, news, prices, specs, and current events via Groq compound model |
| ⏰ **Reminder Scheduler** | Users set natural-language reminders; bot proactively pings them at the right time |
| 🔄 **Multi-Model Fallback** | Rotates across a Groq model pool with per-model circuit breakers |
| 🔒 **Allowlist Security** | Only whitelisted JIDs can interact — all others are silently dropped |
| 📋 **Custom Lists** | Manages shopping lists, book lists, and any other named user list |
| 📝 **Conversation Summary** | Summarises prior conversation on demand |
| 🌐 **WhatsApp Native** | Works on real WhatsApp via WAHA — no new app required for users |

---

## 🏗️ Architecture Overview

```
WhatsApp User
      │
      ▼
  WAHA Bridge                    (self-hosted WhatsApp HTTP API)
      │  POST /webhook
      ▼
  FastAPI App  :6000             (app/main.py)
      │
      ├── Signature verify  (HMAC-SHA256, optional)
      ├── Allowlist check
      ├── Echo / debounce guard
      ├── _ambient_store         → SQLite + ChromaDB (inbound)
      └── Per-chat async worker queue
              │
              ▼
       process_message()
              │
              ├─ facts_load       → SQLite: all user facts
              ├─ reminders_load   → SQLite: pending scheduled reminders
              ├─ context_build    → ChromaDB: semantic search + recent window
              └─ history_load     → SQLite: ordered conversation turns (N turns)
                      │
                      ▼
             agent_engine.run_agent()    (app/agent_engine.py)
                      │
                      ├─ [parallel] _plan() + _extract_memory()
                      │         └─ Groq LLM × 2 simultaneously
                      │
                      ├─ PlannerResult.mode:
                      │     "answer"      → _groq_messages() with full history
                      │     "live_search" → groq_live_search() (compound-beta-mini)
                      │     "ask_facts"   → ask user for missing locale/facts
                      │
                      ├─ [single] _verify_updates()  → Groq: confidence gate
                      ├─ _format_whatsapp()           → Groq: WA markdown rewrite
                      └─ AgentResult { reply, memory_updates }
                      │
                      ▼
       send_text()    → WAHA API → WhatsApp
       memory_save()  → SQLite: upsert facts
       chroma_store() → ChromaDB: log outbound turn

  Scheduler (60s tick)
      └── fire due reminders → send_text() via WAHA
```

---

## 🗂️ Project Structure

```
shimmi/
├── app/
│   ├── main.py            # FastAPI app, webhook, per-chat workers
│   ├── agent_engine.py    # LLM pipeline: planner + extractor + verifier
│   ├── database.py        # SQLite (WAL) facts/history + ChromaDB ambient
│   ├── waha_provider.py   # Async WAHA HTTP client, echo cache, typing keepalive
│   ├── config.py          # Settings from .env via dataclass
│   ├── prompts.py         # All LLM system prompts (single source of truth)
│   ├── utils.py           # Signature verify, prefix parse, WA sanitizer
│   ├── retry.py           # Async retry with exponential backoff
│   ├── logging_setup.py   # Structured logging config
│   └── __init__.py
├── data/                  # Runtime data (auto-created)
│   ├── shimmi.sqlite      # Persistent user facts & message log
│   └── chroma/            # ChromaDB vector store
├── .env.example           # Config template — copy to .env
├── requirements.txt
├── shimmi-bot.service     # systemd unit file
└── main.py                # Entry shim: `python -m uvicorn app.main:app …`
```

---

## ⚙️ Configuration — `.env`

Copy `.env.example` to `.env` and fill in your values.

```dotenv
# ── Server ─────────────────────────────────────────────────
DATA_DIR=./data                    # Where SQLite + ChromaDB live
APP_TIMEZONE=Asia/Kolkata          # Used for reminder scheduling (e.g. "UTC", "US/Eastern")

# ── Bot identity ───────────────────────────────────────────
BOT_PERSONA_NAME=Shimmi
BOT_COMMAND_PREFIX=@shimmi,shimmi,@spock,spock,చిట్టి

# ── Message routing ────────────────────────────────────────
ALLOW_NLP_WITHOUT_PREFIX=false     # true = respond to any message; false = require prefix
ALLOW_FROMME=1                     # 1 = process messages sent FROM this session (own phone)

# ── WAHA (WhatsApp HTTP API) ────────────────────────────────
WAHA_API_URL=http://localhost:3000/api
WAHA_API_KEY=                      # Optional WAHA auth key
WAHA_SESSION=default
WEBHOOK_SECRET=                    # Optional HMAC-SHA256 secret for webhook signature

# ── Access control ─────────────────────────────────────────
# Comma-separated WhatsApp JIDs (e.g. 919..@c.us or group@g.us)
ALLOWED_GROUP_JIDS=919000000000@c.us,another@c.us

# ── Groq LLM ───────────────────────────────────────────────
GROQ_API_KEY=gsk_...
# IMPORTANT: put your preferred/largest model FIRST
GROQ_MODEL_POOL=llama-3.3-70b-versatile,llama-3.1-8b-instant
GROQ_TIMEOUT=60                    # Per-request timeout (seconds)
GROQ_MAX_INFLIGHT=5                # Concurrent in-flight Groq requests

# ── Live search ────────────────────────────────────────────
LIVE_SEARCH_ENABLED=1
LIVE_SEARCH_MODEL=compound-beta-mini   # Groq compound model with built-in web search

# ── ChromaDB semantic context ──────────────────────────────
CHROMA_ENABLED=1
CHROMA_COLLECTION=shimmi_conversations
CHROMA_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_TOP_K=10                    # Semantic hits per query
CHROMA_RECENT_K=10                 # Recent messages to always include

# ── Per-user memory ────────────────────────────────────────
FACTS_VERIFICATION=1               # Run LLM confidence check before saving facts
FACTS_MIN_CONF=0.85                # Minimum confidence to accept a proposed fact
ALLOW_FREEFORM_MEMORY_KEYS=1       # Allow agent to invent new fact keys

# ── Performance / queue ────────────────────────────────────
MESSAGE_DEBOUNCE_MS=800            # Drop duplicate messages within this window
LLM_MAX_QUEUE_PER_CHAT=3          # Max backlog per chat before rejecting
LLM_QUEUE_WAIT_SEC=20              # Seconds to wait for queue slot before dropping
WORKER_IDLE_TTL_SEC=300            # Worker self-destructs after N seconds of silence
HISTORY_TURNS=10                   # Conversation turns to inject as LLM context

# ── Logging ────────────────────────────────────────────────
LOG_LEVEL=INFO
DEBUG_AGENT=0                      # Set to 1 for verbose LLM trace logs
ANONYMIZED_TELEMETRY=False         # Disable ChromaDB anonymous telemetry
```

---

## 🚀 Installation & Deployment

### Prerequisites

- Python 3.12+
- A running [WAHA](https://waha.devlike.pro/) instance with an active WhatsApp session
- Groq API key — free tier: 100K tokens/day per model (upgrade for production)

### Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/shimmi.git
cd shimmi

# 2. Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Fill in WAHA_API_URL, GROQ_API_KEY, ALLOWED_GROUP_JIDS

# 5. Run
python -m uvicorn app.main:app --host 0.0.0.0 --port 6000
```

### Running the tests

Test-only dependencies live in `requirements-dev.txt` (which includes the
runtime ones):

```bash
pip install -r requirements-dev.txt
pytest                # offline suite — no network, no LLM quota
```

### systemd (production)

```bash
# Edit shimmi-bot.service to set correct paths
sudo cp shimmi-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now shimmi-bot
sudo journalctl -u shimmi-bot -f   # tail logs
```

### Configure WAHA Webhook

In WAHA dashboard → Settings → Webhooks, add:

```
URL:    http://<shimmi-host>:6000/webhook
Events: message.received
```

---

## 🧠 Memory System

Shimmi maintains a **per-user key-value facts store** in SQLite. The agent automatically extracts and persists personal information from natural conversation — no explicit commands needed.

### Well-known Facts

| Key | Example | Extracted when… |
|---|---|---|
| `name` | `Phani Adabala` | User introduces themselves |
| `city` | `Hyderabad` | User mentions location |
| `country` | `India` | Stated or inferred |
| `postal_code` | `500083` | User shares postal code |
| `favorite_drink` | `beetroot juice` | User states a preference |
| `shopping_list` | `bread, jam, milk` | User creates/updates a list |
| `reminder_notes` | `5pm reminder` | User sets a reminder |

With `ALLOW_FREEFORM_MEMORY_KEYS=1`, the agent can also create new keys it deems relevant (e.g., `favorite_sport`, `dietary_preference`).

Facts persist across all sessions and are loaded at the top of every message, giving Shimmi full personal context without re-asking.

### Memory Pipeline

Each incoming message goes through a 3-phase memory pipeline:

```
1. _extract_memory()   → LLM scans user text for explicit facts
2. _verify_updates()   → second LLM pass assigns confidence score (≥ FACTS_MIN_CONF to pass)
3. upsert_fact()       → write survivors to SQLite
```

The extraction and planning steps run in **parallel** (`asyncio.gather`) to minimise latency.

---

## 🔍 Live Search

The planner automatically routes queries requiring current information to Groq's `compound-beta-mini` model, which has built-in web search capabilities.

**Example queries handled via live search:**

| Query | Routed as |
|---|---|
| *"What's the weather tomorrow?"* | `live_search` — locale from facts |
| *"What's the latest news?"* | `live_search` — world news |
| *"PAYTM stock price"* | `live_search` — financial data |
| *"Gold price in India today"* | `live_search` — commodity price |
| *"Samsung Galaxy Note 9 specs"* | `live_search` — product research |

The planner enforces a **locale gate**: if the query depends on location (weather, local prices, timezone) but no locale facts are stored, it asks for location first rather than guessing.

---

## 🤖 LLM Orchestration

```
Incoming message
       │
       ├─ [parallel] ──────────────────────────────────────────┐
       │  _plan(user, facts, context)                           │
       │  → PlannerResult { mode, requires_locale, query, q }   │
       │                                                        │
       │  _extract_memory(user)                                 │
       │  → List[MemoryUpdate]                                  │
       └────────────────────────────────────────────────────────┘
       │
       ├─ mode = "answer"      → _groq_messages() full history context
       ├─ mode = "live_search" → groq_live_search() + compound-beta-mini
       └─ mode = "ask_facts"   → return clarifying question
       │
       ├─ _verify_updates()    → confidence-gated memory approval
       ├─ _format_whatsapp()   → WhatsApp markdown rewrite
       └─ AgentResult
```

### Model Pool & Circuit Breaker

- Models are tried in order from `GROQ_MODEL_POOL`
- A model is tripped into cooldown if any call raises an exception
- Cooldown is 10–14 seconds (with jitter)
- Each chat keeps a **sticky model** preference to avoid thrashing
- `compound-beta-mini` should be set as `LIVE_SEARCH_MODEL` (separate pool slot)

> ⚠️ **Critical config:** Always put `llama-3.3-70b-versatile` **first** in `GROQ_MODEL_POOL`. Putting the 8B model first causes slower, lower-quality responses and triggers the circuit breaker more frequently under Groq's shared infrastructure load.

---

## 📊 Observability

### Health & Status Endpoints

The app serves three routes: `POST /webhook`, `GET /healthz`, and `GET /metrics`.
(Earlier revisions of this README documented a `GET /status` endpoint — it does
not exist; its content was folded into `/healthz`.)

```
GET /healthz   → {
  "status":  "ok",
  "version": "3.17.5",
  "workers": 2,
  "queues":  { "<chat_id>": 0 },
  "providers": {
    "gemini": { "enabled": true, "orchestrator": "...", "extraction": "..." },
    "groq":   { "orchestrator": "...", "extraction": "...", "pool": [...] }
  },
  "live_search":       true,
  "chroma":            true,
  "model_circuits":    { "llama-3.1-8b-instant": "tripped (8s remaining)" },
  "provider_circuits": { "groq_70b": "open" },
  "token_budget":      { "groq_70b": "12.4% of 100,000/day", "groq_8b": "…" },
  "reminder_task":     true
}
```

### Prometheus Metrics

```
GET /metrics   → text/plain; version=0.0.4
```

Counters live in `app/metrics.py` (stdlib only — no `prometheus_client`
dependency); live state is snapshotted per scrape. Every counter is seeded at
zero on startup, so a dashboard shows a real `0` rather than a gap before the
first occurrence.

| Metric | Type | Labels |
|---|---|---|
| `shimmi_messages_received_total` | counter | — |
| `shimmi_messages_skipped_total` | counter | `reason` (allowlist, empty, echo, duplicate, from_me, no_prefix, debounced) |
| `shimmi_messages_enqueued_total` | counter | — |
| `shimmi_messages_dropped_total` | counter | `reason` (queue_timeout) |
| `shimmi_messages_processed_total` | counter | `outcome` (ok, error) |
| `shimmi_replies_sent_total` | counter | — |
| `shimmi_rate_limit_replies_total` | counter | — |
| `shimmi_memory_facts_total` | counter | `op` (created, updated, unchanged, deleted) |
| `shimmi_reminders_total` | counter | `outcome` (sent, retry, failed, stale, bad_trigger) |
| `shimmi_webhook_auth_failures_total` | counter | — |
| `shimmi_webhook_invalid_payload_total` | counter | — |
| `shimmi_active_workers` | gauge | — |
| `shimmi_queue_depth_total` / `_max` | gauge | — |
| `shimmi_model_circuit_tripped` | gauge | `model` |
| `shimmi_provider_circuit_tripped` | gauge | `provider` |
| `shimmi_provider_circuit_cooldown_seconds` | gauge | `provider` |
| `shimmi_token_budget_fraction` | gauge | `provider` |
| `shimmi_reminder_task_up` | gauge | — |
| `shimmi_build_info` | gauge | `version` |

Queue depth is deliberately exposed as a sum and a max rather than per-chat:
`chat_id` is unbounded and would blow up scrape cardinality. For the same
reason no metric is ever labelled with a chat, sender, or event id.

Example scrape config:

```yaml
scrape_configs:
  - job_name: shimmi
    static_configs:
      - targets: ["<shimmi-host>:6000"]
```

### Structured Logging

Every message produces a full step-level trace log:

```
INFO app.trace:
╔══ MSG TRACE ══════════════════════════════════════════════
║  event    true_<chat>_<id>
║  chat     <chat_id>
║  outcome  ✓ OK  (1503.5 ms total)
╠══ STEPS ══════════════════════════════════════════════════
║  ✓ facts_load        1.6ms    facts_count=7
║  ✓ reminders_load    1.3ms    reminders_pending=1
║  ✓ context_build    89.0ms    context_total=20
║  ✓ agent_run      1378.3ms
║  ✓ memory_save      14.8ms    reminders_saved=1  facts_updated=1
║  ✓ send             14.9ms    sent=True  msg_id=3EB0CF...
╚═══════════════════════════════════════════════════════════
```

---

## 🛡️ Security

| Concern | Mitigation |
|---|---|
| Unauthorised access | `ALLOWED_GROUP_JIDS` allowlist — unknown JIDs silently dropped |
| Webhook spoofing | Optional HMAC-SHA256 signature validation (`WEBHOOK_SECRET`) |
| Bot echo loops | Outbound message cache (IDs + content hash, 300s TTL) |
| Message floods | Per-chat debounce (800ms default) + queue backpressure (max 3) |
| Data residency | All data stays on your infrastructure (SQLite + ChromaDB on-disk) |
| WAHA credentials | Never sent to Groq or any third party |

---

## 🐛 Integrity Report & Known Issues

### Issues Fixed in v3.0.0 (from live log analysis)

All 13 bugs observed in the prior production log have been resolved in v3.0.0:

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | Critical | `AttributeError: 'ReplyPayload' object has no attribute 'action'` on every message | Removed stale `result.reply.action` check in `process_message` |
| 2 | High | `_verify_updates` called twice per turn — wasted API call | Single verify over deduplicated union of all proposed updates |
| 3 | High | `_extract_memory` + `_plan` ran sequentially (+400–800ms) | `asyncio.gather` parallelisation |
| 4 | High | `CHAT_QUEUES` / `CHAT_WORKERS` memory leak — workers ran forever | Idle TTL with `asyncio.wait_for` self-destruct |
| 5 | Medium | `_ambient_store` fired for echo messages and filtered-out traffic | Store only called after all guards pass |
| 6 | Medium | Memory persisted even on failed `send_text` | Early-return guard after send |
| 7 | Medium | `LIVE_SEARCH_MODEL` defaulted to invalid `"groq/compound-mini"` | Fixed to `"compound-beta-mini"` |
| 8 | Medium | Single SQLite lock serialised all reads | Separate `_write_lock`; reads run concurrently under WAL |
| 9 | Medium | No ordered conversation history — LLM re-inferred state from embeddings | `get_recent_messages()` injects N ordered turns |
| 10 | Low | Chroma `recent_window` fetched `max(50, k*5)` docs — O(N) sort | Capped at `min(k*2, 100)` |
| 11 | Low | Worker tasks blocked on `queue.get()` indefinitely | Resolved by Fix #4 |
| 12 | Low | No `/status` endpoint for operational visibility | Added `GET /status` |
| 13 | Low | `ALLOW_NLP_WITHOUT_PREFIX` code default (`True`) vs `.env.example` (`false`) mismatch | Aligned to `False` |

---

### Remaining Issues *(open — PRs welcome)*

> ⚠️ This section previously described v3.0.0. The code has since moved on to
> v3.17.4 (`CHANGELOG.md`) without this section being refreshed, so it was
> understating three items as unimplemented that had actually been built
> (memory deletion, the reminder scheduler, and `asyncio.get_running_loop()`)
> and was missing three real, currently-open bugs. Corrected below.

#### ✅ RESOLVED — Memory Deletion

Implemented: `delete_fact()` / `delete_facts_batch()` in `app/database.py`,
with a `DeleteOutcome` enum (`DELETED` / `NEEDS_CONFIRM` / `NOT_FOUND` /
`BLOCKED` / `EMPTY_KEY`) and a confirmation flow for high-stakes keys
(shopping/grocery/todo lists) handled in `app/main.py`'s `memory_save` step.

#### ✅ RESOLVED — Proactive Reminder Delivery

Implemented in `app/scheduler.py`: a 60s asyncio poll loop
(`run_reminder_loop`) with a `reminders` table, missed-reminder handling
(dropped if >2h overdue), and retry with exponential backoff (30s/60s/120s,
max 3 attempts) on send failure. Note this is a hand-rolled asyncio loop, not
APScheduler — the Tech Stack table below has been corrected to match.

#### ✅ RESOLVED — `asyncio.get_event_loop()` Deprecation

No occurrences remain in `app/*.py`; all executor calls use
`asyncio.get_running_loop()`.

#### 🟠 FIXED IN v3.17.4 — Debounce Could Silently Starve Busy Chats

**Location:** `app/main.py` `CHAT_LAST_MSG_TS`

Keyed by `chat_id` alone and re-armed on every check (even a debounced one).
In a group chat busier than `MESSAGE_DEBOUNCE_MS`, or with
`ALLOW_NLP_WITHOUT_PREFIX=1`, this became a sliding window that could
suppress every message indefinitely rather than just deduping genuine
resends. Fixed by keying on `(chat_id, sender_key)` and only advancing the
timestamp when a message is accepted. See `CHANGELOG.md`.

#### 🟠 FIXED IN v3.17.4 — No User-Facing Error on Non-Rate-Limit Failures

**Location:** `app/main.py` `_chat_worker()`

`process_message` only sent a user-facing reply for rate-limit failures
(handled internally); any other unhandled exception — a DB error, a WAHA
outage, a bug — was caught in the worker, logged, and the user got no reply
at all. The worker now sends a best-effort "something went wrong" message on
any unhandled exception.

#### 🟢 FIXED IN v3.17.4 — Stale `/healthz` Version

`GET /healthz` reported a hardcoded `"3.3.0"` unrelated to the code actually
running. Added `APP_VERSION` in `app/main.py` as the single source of truth.

#### 🔴 FIXED IN v3.17.5 — App Could Not Start From A Clean Install

`app/agent_engine.py` imports `AsyncOpenAI` unconditionally at module level
(the Gemini/Mistral OpenAI-compatible endpoints), but `openai` was missing from
`requirements.txt` — which actively stated "No extra SDK". A fresh
`pip install -r requirements.txt` therefore produced an app that died on import
with `ModuleNotFoundError: No module named 'openai'`. Added to requirements.

#### 🔴 FIXED IN v3.17.5 — Fact Consolidation Never Ran For An Hour After Boot

**Location:** `app/agent_engine.py` `consolidate_user_facts()`

The cooldown gate used `0.0` as the "never run" sentinel and compared it
against `time.monotonic()`, which counts from system boot. For the first hour
of uptime `now - 0.0 < 3600` was true for *every* user, so LLM-driven key
deduplication was silently skipped after every restart — on a container or a
rebooted VM, after every deploy. `None` is now the sentinel. The existing test
`test_consolidation_cooldown_prevents_frequent_runs` had been failing on this
the whole time; it passes now without being weakened.

---

#### 🟡 MEDIUM — Groq Daily Token Limit (100K TPD on Free Tier)

The free Groq tier allows only 100,000 tokens/day per model. With a full pipeline (planner + extractor + verifier + formatter + answer = 5 LLM calls per message), a moderately busy bot can exhaust this in a few hours.

**Mitigations:**
- Upgrade to Groq Dev Tier for higher limits
- Set `FACTS_VERIFICATION=0` to skip the verifier call when not needed
- Set `HISTORY_TURNS=5` to reduce context size
- Add `compound-beta-mini` as a third model in `GROQ_MODEL_POOL` as emergency fallback

---

#### 🟢 LOW — `sentence-transformers` Requires PyTorch

Installing `sentence-transformers` pulls in PyTorch (~2GB on CPU). On memory-constrained servers this can be slow and surprising.

**Mitigations:**
- Pre-install PyTorch CPU-only before running `pip install -r requirements.txt`:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- Or use a lighter embedding backend (e.g. `chromadb`'s default `all-MiniLM-L6-v2` via ONNX)

---

## 📦 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Web framework | FastAPI | 0.115.6 |
| ASGI server | Uvicorn | 0.30.6 |
| LLM inference | Groq SDK | 0.13.1 |
| LLM models | llama-3.3-70b-versatile, llama-3.1-8b-instant, compound-beta-mini | — |
| WhatsApp bridge | WAHA (self-hosted) | — |
| HTTP client | httpx (async) | 0.27.2 |
| Persistent memory | SQLite (WAL mode) + stdlib `sqlite3` | built-in |
| Semantic search | ChromaDB | 0.5.20 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 2.7.0 |
| Data validation | Pydantic v2 | 2.9.2 |
| Config | python-dotenv | 1.0.1 |
| Scheduler | hand-rolled `asyncio` poll loop (`app/scheduler.py`) | — |

---

## 🗺️ Roadmap

- [x] Re-add proactive reminder scheduler
- [x] Memory deletion support (`delete_fact` / `delete_facts_batch` + confirmation flow)
- [x] User-facing error message on total LLM failure (rate-limit path; v3.17.4 closed the remaining gap for other failure types)
- [x] `/metrics` endpoint (Prometheus-compatible)
- [ ] Multi-language support (currently Telugu prefixes already supported)
- [ ] Group chat support (two-tier context strategy already implemented — see `app/main.py` `context_build`; needs broader real-world testing)
- [ ] Voice message transcription (Groq Whisper)
- [ ] Backfill `CHANGELOG.md` history between v3.2.0 and v3.17.3, or otherwise reconcile version numbering

---

## 📄 License

MIT — see `LICENSE` for details.
