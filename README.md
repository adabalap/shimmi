# Shimmi – Stateful WhatsApp Assistant

Shimmi is a production‑lean, privacy‑aware WhatsApp assistant with **durable memory**, **time‑windowed RAG**, and **WhatsApp‑friendly responses**. It supports **strict prefix gating**, **emoji branding**, **profile snapshot recall**, and **skinny retries**. Optional ambient observation captures meaningful group chatter for catch‑ups—without LLM background cost.

---

## Highlights
- Fast startup: embeddings pre‑warmed.
- Clean output: short paragraphs, minimal emojis.
- Safety: prefix/allowlist gates; no unsolicited group messages.
- Memory: judge‑hardened facts in SQLite; unified recall via PROFILE snapshot + RAG.
- Observability: `/diag/rag`, `/diag/profile`.

---

## Architecture Overview

```
 WhatsApp User
       |
       v
   +--------+        +-------------+
   | WAHA   | -----> | FastAPI     |
   |  API   | <----- | /webhook    |
   +--------+        +-------------+
                          |
                          v
                 +--------------------+
                 | Gatekeeping        |
                 | (allowlist/prefix) |
                 +--------------------+
                          |
                          v
                 +--------------------+
                 | Per‑chat Queue     |
                 +--------------------+
                          |
                          v
      +----------------------------------------------+
      | Process:                                      |
      | extract + judge → persist → snapshot → RAG    |
      +----------------------------------------------+
                          |
                          v
              +---------------------------+
              | Assemble Context          |
              | (LOCATION/PROFILE/FACTS)  |
              +---------------------------+
                          |
                          v
                    +-----------+
                    |   LLM     |
                    +-----------+
                          |
                          v
               +------------------------+
               | Output Policy          |
               | emoji/minimal styling  |
               +------------------------+
                          |
                          v
                        WAHA
```

### Ambient Observation (Opt-in)
```
 Group Message ---> [Observe Enabled?] --[no]--> ignore
                                 [yes]
                                   |
                                   v
                              [Redaction]
                                   |
                                   v
                               [Filters]
                                   |
                                   v
                              [Topic Tag]
                                   |
                                   v
                              [Embed → RAG]
```

---

## Quick Start

### Requirements
- Python 3.11+
- WAHA (WhatsApp Host API)
- LLM provider API key

### Environment (`.env` excerpt)
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

### Run Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 6000
```

### Diagnostics
```bash
curl -s http://localhost:6000/healthz
curl -s http://localhost:6000/diag/rag | jq
curl -s "http://localhost:6000/diag/profile?sender_id=<id>" | jq
```

---

## Features
- Prefix‑gated NLP.
- WhatsApp‑friendly output formatting.
- Durable memory via SQLite.
- Unified recall pipeline.
- Time‑windowed RAG queries.
- Skinny‑retry LLM calls.
- Diagnostics endpoints.

---

## Ambient Observation
Admin commands:
```
/observe on
/observe off
/observe status
/observe retention 30d
/observe redaction on
```
Pipeline: Redaction → Filters → Topics → Embedding (no LLM).

---

## Troubleshooting
- No replies in group → prefix missing or group not allowed.
- Emoji noise → set `EMOJI_POLICY=none`.
- Cold start → ensure chroma warm‑up log appears.
- RAG empty → vectors stored in `bot_memory.db`.

---

## Roadmap
- Rate-limit header tracking + adaptive backoff.
- Local TPM/RPM shaping.
- `/observe` onboarding card.
- Optional ChromaDB persistent mode.

---

## License (Updated MIT)

```
MIT License

Copyright (c) 2026 <Your Name or Organization>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

