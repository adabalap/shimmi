# Shimmi – Stateful WhatsApp Assistant

Shimmi is a production‑lean, privacy‑aware WhatsApp assistant with **durable memory**, **time‑windowed RAG**, and **WhatsApp‑friendly responses**. It supports **strict prefix gating** in groups/DMs, **emoji branding**, **profile snapshot recall**, and **skinny retries** on provider hiccups. Optional **ambient observation** (opt‑in) captures meaningful group chatter for later catch‑ups and suggestions—**without** background LLM cost.

> **Highlights**
> - Fast startup: embeddings pre‑warmed.
> - Clean output: short paragraphs, minimal emojis, no auto‑bullets.
> - Safety: prefix/allowlist gates; no unsolicited group messages.
> - Memory: judge‑hardened facts in SQLite; unified recall via PROFILE snapshot + RAG.
> - Observability: `/diag/rag`, `/diag/profile`.

---

## Architecture

- **Inbound**: WhatsApp → WAHA → FastAPI **/webhook**
- **Gatekeeping**: signature, allowlist, prefix, echo‑filter, debounce
- **Processing**: per‑chat queue → extract+judge → persist (SQLite) → profile snapshot → RAG add (if long)
- **Context**: assemble **LOCATION / PROFILE_FACTS / FACTS / SNIPPETS (time‑window)**
- **LLM**: reply with skinny‑retry; second‑person recall; no prefix‑as‑name
- **Outbound**: policy (emoji prefix, cap) → WAHA send

### System Overview (Mermaid)
```mermaid
flowchart LR
  U[WhatsApp User] -->|messages| W(WAHA API)
  W -->|webhook| F[FastAPI Webhook]
  F --> G{Gatekeeping
allowlist/prefix/echo}
  G -->|enqueue| Q[Per-chat Queue]
  Q --> P[Process
extract+judge→persist→snapshot→RAG add]
  P --> C[Assemble Context
LOCATION/PROFILE/FACTS/SNIPPETS]
  C --> L[LLM Reply
skinny retry]
  L --> O[Output Policy
emoji prefix/minimal emojis]
  O -->|sendText| W
  F -.-> D[/diag]
  P --> S[(SQLite + FAISS)]
```

### Ambient Observation (Opt-in)
```mermaid
flowchart LR
  GM[Group message (no prefix)] --> OBS{observe enabled?}
  OBS -- no --> X[ignore]
  OBS -- yes --> R[Redaction]
  R --> F[Filters
min length, not link-only]
  F --> T[Topic Tag
MOVIES/TRIP/MUSIC]
  T --> E[Embed→RAG (no LLM)]
```

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- WAHA (WhatsApp host API) reachable from Shimmi
- API key & model access for your LLM provider

### 2. Environment (`.env` excerpt)

```ini
BOT_PERSONA_NAME=Shimmi
BOT_COMMAND_PREFIX=@shimmi,shimmi,@spock,spock,చిట్టి,shichitti
ALLOW_NLP_WITHOUT_PREFIX=false

BOT_EMOJI=🤖
BOT_EMOJI_PREFIX_ENABLED=1
EMOJI_POLICY=minimal
EMOJI_MAX_PER_MSG=2

WA_STRIP_MARKDOWN=1
WA_NORMALIZE_BULLETS=0
WA_WRAP_COL=0

CHROMA_ENABLED=1
FACTS_EXTRACTION_MODE=hybrid
FACTS_VERIFICATION=1
FACTS_MIN_CONF=0.80
```

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 6000
```

### 4. Health & Diagnostics

```bash
curl -s http://localhost:6000/healthz
curl -s http://localhost:6000/diag/rag | jq
curl -s "http://localhost:6000/diag/profile?sender_id=<id>" | jq
```

---

## Features

- **Prefix Gating**: groups & DMs respect `BOT_COMMAND_PREFIX`; DMs can allow/deny non‑prefix via `ALLOW_NLP_WITHOUT_PREFIX`.
- **Style Policy**: short paragraphs; minimal emojis; WhatsApp clean text.
- **Durable Memory**: facts stored in `user_facts`; judge fallback for simple declaratives.
- **Unified Recall**: `build_profile_snapshot_text()` → appended to RAG for holistic Q&A.
- **Time‑Windowed RAG**: queries support “today”, “yesterday”, “last 7 days”, etc.
- **Resilient LLM Calls**: skinny retry; future rate‑limit header logging.
- **/diag**: quick counts and profile preview.

---

## Ambient Observation (Opt-in)

- Disabled by default. Admin commands:

```text
/observe on
/observe off
/observe status
/observe retention 30d
/observe redaction on
```

- Pipeline: **Redaction → Filters → Topic Tag → Embed** (no LLM)
- Retention configurable per chat (default: 30 days).  
  Supports `/forget me` and `/purge` (if implemented).

---

## Troubleshooting

- **No replies in group** → missing prefix or not in `ALLOWED_GROUP_JIDS`.
- **Emoji spam** → set `EMOJI_POLICY=none` or reduce `EMOJI_MAX_PER_MSG`.
- **Cold first query** → confirm warm‑up log `chroma.warmup dim=384`.
- **LLM failures** → check logs for `llm.reply.end ok=False`.
- **RAG appears empty** → vectors stored in `bot_memory.db`; use `/diag/rag`.

---

## Roadmap

- Instrument & log `x-ratelimit-remaining-*` headers; adaptive backoff.
- Optional **local TPM/RPM shaping** + global semaphore.
- Consent card for `/observe on`; admin FAQ.
- Optional ChromaDB persistent mode + migration.

---

## License
MIT (or your org’s standard). Replace as needed.
