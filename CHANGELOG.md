# Shimmi v2.8.0 — Changelog

## 🔴 Bugs Fixed

### "Good morning" at 8:27 PM (time-of-day blindness)
`agent_engine.py` now injects `current_time` (HH:MM in local timezone), `today`
(YYYY-MM-DD), and `utc_offset` (±HH:MM) into every orchestrator call.
The prompt's `TIME-OF-DAY GREETINGS` section maps exact time ranges to correct
greetings. The LLM can no longer say "Good morning" at 8 PM.

**Required env var**: `APP_TIMEZONE=Asia/Kolkata`

### "I've already shared the news" — hallucination with no prior reply
The `MANDATORY SEARCH` section now explicitly blocks the LLM from claiming it
already answered if no bot reply is visible in context. It must always re-search
for any "news / latest / scores / weather" request.

### India news hallucinated (no live search used)
Stronger mandatory search trigger covers all news/current-events requests.
The `LIVE_SEARCH_PROMPT` requires source citation per bullet.

## 🟠 New Feature: Background Reminder System

### scheduler.py (new background task)
- Runs every 60 seconds via `asyncio.create_task()` in lifespan
- Queries `reminders` table for `trigger_iso <= now UTC AND sent_at IS NULL`
- Sends WhatsApp ping via WAHA with formatted reminder message
- Misses up to 2 hours old are sent; older are silently dropped
- Per-reminder error isolation — one failure never aborts the check loop

### database.py — reminders table
New `reminders` table: `id, whatsapp_id, chat_id, reminder_text, trigger_iso,
created_at, sent_at, cancelled, failed`

Methods: `add_reminder()`, `get_user_reminders()`, `get_due_reminders()`,
`mark_reminder_sent()`, `mark_reminder_failed()`, `cancel_reminder()`

### prompts.py — REMINDER SYSTEM section
The orchestrator knows how to:
- Create reminders via `{"key": "_reminder", "value": "ISO_DATETIME|text"}`
- Cancel by ID via `{"key": "_cancel_reminder", "value": "ID"}`
- Display pending reminders from the `reminders` input field

### main.py — _handle_special_memory_keys()
Routes `_reminder` / `_cancel_reminder` keys to the reminders table instead
of the key-value facts store. Regular facts are unaffected.

### No more confirmation round-trips for alarms
Old: "Would you like me to save a reminder note?" ← extra turn, bad UX
New: Save immediately, confirm in the same reply, suggest phone Clock app.

## 🟠 WhatsApp UX Improvements

### List display as bullets
`LIST DISPLAY FORMAT` section in prompt mandates:
```
Your grocery list 🛒
• milk
• bread
• cheese
```
Never: "Your list has the following items: milk, bread, and cheese."

### Emoji hygiene
Anti-patterns section: no ☕ on unrelated messages, no ☀️ at night,
max 2 emojis per message. Matched-topic emojis only.

### News with source citations
`LIVE_SEARCH_PROMPT` now requires per-bullet source name ("per *The Hindu*,")
and forbids "according to the search results" phrasing.

## 🟡 Visibility

### Startup listener log
`logging_setup.py` sets `uvicorn.error` to INFO (was WARNING), making the
uvicorn "Uvicorn running on http://0.0.0.0:6000" line appear in logs.
Additionally `main.py` logs its own:
`🚀 startup.ready  http://0.0.0.0:6000  allowlist=7  ...`

Port is read from env var `PORT` (default `6000`).

### Reminder scheduler in /healthz
`GET /healthz` now includes `"reminder_task": true/false`.

## 🟡 .env Changes Required

| Variable | Old Default | New Required Value |
|---|---|---|
| `APP_TIMEZONE` | `UTC` | `Asia/Kolkata` (or your tz) |
| `GROQ_MODEL_POOL` | (broken) | Remove `groq/compound` entry |
| `PORT` | — | `6000` (or your port) |
