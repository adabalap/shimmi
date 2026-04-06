# Shimmi Test Suite

Three tiers. The goal: **maximum confidence at minimum quota cost.**

```
TIER 0  offline  →  pytest, zero tokens, zero network, always safe
TIER 1  smoke    →  6 messages, ~5 K tokens, after every deploy
TIER 2  full     →  46 messages, ~40 K tokens, before each release
```

The pyramid: most tests are at the bottom. Only go up a tier when the tier below passes.

---

## When to run what

| Event | Command | Cost |
|-------|---------|------|
| Every code change | `make test` | 0 tokens |
| After `git push` to server | `make smoke` | ~5K tokens |
| Before a production release | `make full` | ~40K tokens |
| Debugging a specific feature | `make full --groups memory_write` | varies |

---

## TIER 0 — Offline tests  (`make test`)

**251 tests across 10 files. Run in ~8 seconds. Zero quota.**

Every logic path in the codebase is covered deterministically:

| File | What it covers |
|------|----------------|
| `unit/test_shortcuts_and_routing.py` | Null-field fix, time shortcut, update guard, word-overlap fallback, keyword tool routing, RPD cooldown |
| `unit/test_delete_guardrails.py` | Deletion allowlist, confirmation flow, sender isolation, prompt completeness |
| `unit/test_memory_quality.py` | Ephemeral key filter, consolidation safety, upsert counter |
| `unit/test_reminders.py` | Reminder dedup, timezone fix, ISO parsing, webhook dedup |
| `unit/test_tools.py` | All 6 tool types: parse, validate, dispatch |
| `unit/test_agent_p1.py` | Pydantic models, dispatch chain, JSON repair |
| `unit/test_database_p1.py` | SQLite upsert, delete, batch operations |
| `integration/test_resilience.py` | Fallback chain (groq_8b reached), reply_extract shortcut skip |
| `integration/test_tool_dispatch.py` | Full weather/news/stocks dispatch with mocked HTTP |
| `integration/test_confirmation_flow.py` | Full list-delete confirm/cancel cycle |

---

## TIER 1 — Smoke test  (`make smoke`)

**6 messages. Each exercises one critical path. Total: ~5K tokens.**

The 6 messages are carefully chosen so no two test the same thing:

| Label | What it proves |
|-------|----------------|
| `greeting` | Webhook alive; Groq fallback works (Gemini RPD circuit open) |
| `memory_write` | LLM extraction pipeline → fact persisted to SQLite |
| `memory_recall` | Shortcut fires instantly (zero LLM tokens) |
| `memory_correction` | Update guard blocks shortcut; LLM saves corrected value |
| `live_weather` | MCP /weather dispatched → Open-Meteo → formatted reply |
| `opinion_query` | `question=null` no longer crashes (OrchestratorResult fix) |

---

## TIER 2 — Full suite  (`make full`)

**46 messages across 16 feature groups. Total: ~40K tokens.**

Run specific groups to save quota:

```bash
make full --groups memory_write memory_recall memory_update   # memory only
make full --groups live_weather live_news live_stocks         # live data only
make full --groups null_field_resilience context_switch       # robustness only
```

| Group | Messages | Cost |
|-------|----------|------|
| `memory_write` | 6 | medium |
| `memory_recall` | 6 | **minimal** (shortcuts fire, near-zero tokens) |
| `memory_update` | 3 | low |
| `lists` | 4 | medium |
| `memory_delete` | 4 | medium |
| `reminders` | 2 | low |
| `live_weather` | 2 | low (MCP cached 10 min) |
| `live_news` | 2 | low (MCP cached 5 min) |
| `live_stocks` | 2 | low (MCP cached 3 min) |
| `live_currency` | 1 | **minimal** (MCP cached 1 hr) |
| `time_shortcuts` | 3 | **ZERO** (server clock, no LLM) |
| `general_knowledge` | 3 | medium |
| `null_field_resilience` | 2 | low |
| `context_switch` | 3 | low |
| `unknown_info` | 2 | low |
| `privacy` | 3 | low |
| `profile_dump` | 1 | high |

---

## Token economy

```
Groq free tier:  100K tokens/day  (llama-3.3-70b)
                 500K tokens/day  (llama-3.1-8b extraction)

smoke  → ~5K tokens   = 0.05% of daily budget  ✅ always safe
full   → ~40K tokens  = 40% of daily budget     ⚠️  once per release only

KEY: Tier 0 (offline) covers ALL logic paths at zero cost.
     Tier 1 (smoke) confirms the live system works after deploy.
     Tier 2 (full) validates UX quality before a release.
     Never run Tier 2 for routine regression.
```

---

## Adding tests

**New bug fix** → add a unit test in `tests/unit/test_*.py`. No tokens.

**New feature** → add to the appropriate group in `tests/shimmi_tester.py` `FULL_SCENARIOS`. Document the token cost.

**New zero-token path** → add to `time_shortcuts` or `memory_recall` group (these are free or near-free).
