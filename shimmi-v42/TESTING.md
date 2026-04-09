# Shimmi — Testing Guide

## The Core Principle

Tests exist at three levels. The only way to keep costs sustainable is to test at the lowest level that can catch a given bug.

```
TIER 0  offline pytest   →  zero tokens   →  run on every code change
TIER 1  smoke (6 msgs)   →  ~5K tokens    →  run after every deployment
TIER 2  full  (46 msgs)  →  ~40K tokens   →  run once before a release
```

**Never use TIER 2 for regression.** That is what TIER 0 is for. TIER 2 exists only to verify user-facing behaviour end-to-end before shipping.

---

## Quick Reference

```bash
# Every code change (zero cost)
python3 tests/shimmi_tester.py offline

# After deploying to server (~5K Groq tokens)
python3 tests/shimmi_tester.py smoke --url http://YOUR_SERVER:6000/webhook

# Before a production release (~40K Groq tokens)
python3 tests/shimmi_tester.py full --url http://YOUR_SERVER:6000/webhook

# Run specific feature groups only
python3 tests/shimmi_tester.py full --groups memory live_weather

# List all groups with token costs
python3 tests/shimmi_tester.py list
```

---

## TIER 0 — Offline Tests (Zero Quota)

Run with: `python3 tests/shimmi_tester.py offline`

These are standard pytest tests that mock every external dependency. No network calls, no LLM calls, no API keys needed. They run in ~8 seconds and cover every code path deterministically.

### What is covered

| File | Class | What it tests |
|------|-------|---------------|
| `unit/test_agent_p1.py` | `TestMemoryUpdateDeleteFlag` | MemoryUpdate Pydantic model field validation |
| | `TestOrchestratorResultToolCall` | OrchestratorResult model, tool_call field |
| | `TestDispatchTool` | _dispatch_tool routing including keyword fallback |
| | `TestCleanFacts` | Junk value filtering |
| | `TestParseJson` | JSON parsing with fence removal |
| `unit/test_database_p1.py` | `TestDeleteFact` | Fact deletion, key normalisation |
| | `TestDeleteFactsBatch` | Batch deletion |
| | `TestDeleteUpsertInteraction` | Write after delete, junk value migration |
| `unit/test_delete_guardrails.py` | `TestIsKeyDeletable` | Allowlist, confirmation gates |
| | `TestDeleteOutcome` | Enum values |
| | `TestDeleteFactGuardrails` | End-to-end delete with SQLite |
| | `TestSenderIsolation` | Cross-user data isolation |
| | `TestPendingDeleteCache` | TTL cache, confirm/cancel words |
| | `TestRunAgentPendingDeleteIntercept` | Zero-token confirm/cancel flow |
| | `TestMainBranchLogic` | main.py memory save branching |
| | `TestPromptCompleteness` | Prompt contains required delete syntax |
| `unit/test_tools.py` | `TestParseToolCall` | All tool parsing: weather/news/stocks/currency/timezone |
| | `TestToolDispatcher` | All tool dispatch routing |
| `unit/test_shortcuts_and_routing.py` | `TestOrchestratorNullCoercion` | **v3.3.0**: null field crash fix |
| | `TestTimeShortcut` | Zero-token time/date intercept |
| | `TestFactsShortcutUpdateGuard` | **v3.3.0**: declaration vs recall distinction |
| | `TestFactsShortcutWordOverlap` | **v3.3.0**: word-overlap fallback (no pattern bloat) |
| | `TestEphemeralKeyFilter` | **v3.3.0**: last_summary/conversation_since filtered |
| | `TestKeywordToolRouting` | **v3.2.0**: keyword routing for Groq-fallback orchestrator |
| `integration/test_confirmation_flow.py` | all | Pending-delete confirm/cancel end-to-end |
| `integration/test_memory_deletion.py` | all | Delete pipeline with mocked LLM |
| `integration/test_tool_dispatch.py` | all | All tool dispatch paths, mocked HTTP |
| `integration/test_resilience.py` | `TestFallbackChain` | **v3.2.0 CRITICAL**: groq_8b reached when Gemini+70b fail |
| | `TestConsolidationSafety` | **v3.3.0**: phantom keys, cooldown, valid merge |
| | `TestReplyExtractShortcutSkip` | **v3.3.0**: reply_extract skipped on shortcut |

### Running individual test files

```bash
pytest tests/unit/test_shortcuts_and_routing.py -v   # shortcut + routing
pytest tests/integration/test_resilience.py -v        # fallback chain + consolidation
pytest tests/ -k "null" -v                             # just the null-field fix tests
```

---

## TIER 1 — Smoke Test (~5K tokens)

Run with: `python3 tests/shimmi_tester.py smoke --url http://SERVER:6000/webhook`

Six messages that together cover the entire critical path from webhook to LLM to reply.

| # | Label | Message | What it validates |
|---|-------|---------|-------------------|
| 1 | `greeting` | `ping` | Webhook alive; Groq fallback fires (Gemini RPD down) |
| 2 | `memory_write` | `my name is TestUser and I live in Hyderabad` | LLM extracts name+city; both saved to SQLite |
| 3 | `memory_recall` | `what's my name?` | Zero-token shortcut fires; NO orchestration call |
| 4 | `memory_correction` | `actually I made a mistake — my name is Phani` | Update guard blocks shortcut; LLM saves corrected value |
| 5 | `live_weather` | `what's the weather forecast for Hyderabad today?` | MCP weather tool dispatched; Open-Meteo called |
| 6 | `opinion_query` | `what do you think about Red Tape running shoes?` | Null-field fix: `question=null` no longer crashes |

**Interpreting results:** HTTP 200 = webhook accepted (message enqueued). Check bot logs to verify LLM replies and memory writes. A 200 with no bot reply in logs = LLM error, not a webhook error.

---

## TIER 2 — Full Scenario Suite (~40K tokens)

Run with: `python3 tests/shimmi_tester.py full --url http://SERVER:6000/webhook`

Sixteen independent groups covering every user-facing feature. Run specific groups with `--groups`.

### Group overview

| Group | Msgs | Token cost | What it validates |
|-------|------|------------|-------------------|
| `memory_write` | 6 | ~5K | Fact extraction and persistence (name, age, city, drink, car, allergy) |
| `memory_recall` | 6 | ~100 ⚡ | Shortcut fires for all key types; word-overlap for "current age" |
| `memory_update` | 3 | ~3K | Age correction, promotion update, verify updated value |
| `lists` | 4 | ~4K | Create list, add item, remove item, recall |
| `memory_delete` | 4 | ~4K | Car delete, list delete with confirmation flow |
| `reminders` | 2 | ~2K | Create reminder, list pending reminders |
| `live_weather` | 2 | ~2K | MCP → Open-Meteo, today + tomorrow |
| `live_news` | 2 | ~2K | MCP → GNews/RSS, India + tech |
| `live_stocks` | 2 | ~2K | MCP → yfinance, Nifty + specific symbols |
| `live_currency` | 1 | ~1K | MCP → Frankfurter ECB, USD/INR |
| `time_shortcuts` | 3 | **ZERO** ⚡ | Server clock, no LLM call at all |
| `general_knowledge` | 3 | ~3K | Brand facts, math, explanations from LLM training |
| `null_field_resilience` | 2 | ~1K | Opinion queries that previously crashed (v3.3.0 fix) |
| `context_switch` | 3 | ~2K | Topic switching + memory still intact |
| `unknown_info` | 2 | ~2K | Honest "I don't know" without hallucinating |
| `privacy` | 3 | ~2K | SSN not stored, safe tool names are stored |
| `profile_dump` | 1 | ~4K | Full profile summary from all facts |

⚡ = these groups cost essentially zero tokens (shortcuts fire, no LLM orchestration)

### Running groups by category

```bash
# All memory-related groups
python3 tests/shimmi_tester.py full --groups memory

# All live-data groups
python3 tests/shimmi_tester.py full --groups live

# Only the fixes validated in v3.3.0
python3 tests/shimmi_tester.py full --groups null_field_resilience memory_update memory_recall

# Free groups first (zero or near-zero tokens)
python3 tests/shimmi_tester.py full --groups time_shortcuts memory_recall
```

---

## Token Economy — by the numbers

| Tier | Messages | Groq tokens | % of 100K/day | Frequency |
|------|----------|-------------|----------------|-----------|
| TIER 0 offline | 0 | 0 | 0% | Every code change |
| TIER 1 smoke | 6 | ~5,000 | 5% | Every deployment |
| TIER 2 full | 46 | ~41,000 | 41% | Once per release |

Running `offline` + `smoke` together uses only 5% of the daily Groq budget. You can run this combination 20 times per day with budget to spare. Only run `full` the day before a production release.

---

## Checklist — Pre-release validation

Before deploying a new version to production:

```
□ python3 tests/shimmi_tester.py offline                    (must be 100% pass)
□ Deploy to server
□ python3 tests/shimmi_tester.py smoke --url http://...     (must be 6/6 HTTP 200)
□ Check bot logs: 5 messages should have bot replies
□ Confirm message 3 (memory_recall) shows "facts.shortcut" in logs (zero tokens)
□ python3 tests/shimmi_tester.py full --url http://...      (run overnight if needed)
□ Review test_results.json for any HTTP errors
□ Check bot logs for any worker.msg_error entries
```

---

## On the "don't grow pattern lists" principle

The test suite itself follows this principle. The `memory_recall` group includes `"what's my current age?"` — a variation that doesn't match any explicit signal phrase. This is not added as a new pattern; instead, the word-overlap fallback in `_try_facts_shortcut` handles it. The test validates the mechanism, not the specific phrase.

When you add new memory keys or user queries to the system, you do **not** need to add new test cases for every phrasing variation. The offline unit tests in `TestFactsShortcutWordOverlap` already prove the mechanism works for any key. Add an E2E scenario only if you're testing a new **feature path**, not a new **phrase**.
