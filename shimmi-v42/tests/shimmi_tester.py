#!/usr/bin/env python3
"""
shimmi_tester.py — Shimmi Test Runner v4.0
Token-economy-aware, three-tier test harness.

━━━ PHILOSOPHY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The test pyramid: most tests at the bottom (free), fewest at the top (costly).

  TIER 0  offline  — pytest unit + integration, zero LLM tokens, always safe.
                     Covers every code path deterministically. Run on every
                     code change.

  TIER 1  smoke    — 6 webhook messages, 1 LLM call each, ~5K Groq tokens.
                     Confirms the live system is up and the critical path works
                     after a deployment. Run after every release.

  TIER 2  full     — 16 scenario groups, ~46 messages, ~40K Groq tokens.
                     Validates every user-facing feature end-to-end before a
                     production release. Run once per version.

━━━ USAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python3 shimmi_tester.py offline                   # always safe, zero quota
  python3 shimmi_tester.py smoke                     # 6 msgs, ~5K tokens
  python3 shimmi_tester.py smoke --url http://SERVER:6000/webhook
  python3 shimmi_tester.py full                      # all groups, ~40K tokens
  python3 shimmi_tester.py full --groups memory live # specific groups only
  python3 shimmi_tester.py list                      # show all groups + costs

━━━ TOKEN COST GUIDE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Groq free tier: 100K tokens/day (llama-3.3-70b) + 500K/day (llama-3.1-8b)
  Gemini free:    1,500 req/day

  smoke  → 6 messages × ~800 tokens ≈ 5K tokens (0.05% of daily Groq budget)
  full   → 46 messages × ~900 tokens ≈ 40K tokens (40% of daily Groq budget)
  
  KEY: offline tests (pytest) cover ALL code paths at ZERO cost.
       Only use smoke/full for live integration and UX validation.
       Never run `full` for routine regression — that is what `offline` is for.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_URL     = "http://localhost:6000/webhook"
DEFAULT_PHONE   = "4930656034916@lid"
DEFAULT_BOT     = "919573717667@c.us"
DEFAULT_SESSION = "default"
DEFAULT_DELAY   = 3      # seconds between sends (not LLM response time)
DEFAULT_TIMEOUT = 30     # HTTP request timeout

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — SMOKE SCENARIOS
#
# Exactly 6 messages. Each exercises ONE specific system capability.
# No redundancy. Together they cover the full critical path.
#
# Cost: 6 × ~800 tokens ≈ 5K Groq tokens (always well within free tier)
#
# What each test validates:
# ┌──────────────────────┬────────────────────────────────────────────────────┐
# │ Label                │ System path tested                                 │
# ├──────────────────────┼────────────────────────────────────────────────────┤
# │ greeting             │ Webhook accepts → Groq fallback (Gemini RPD down) │
# │ memory_write         │ LLM extracts name+city → persisted to SQLite      │
# │ memory_recall        │ Shortcut fires (ZERO LLM tokens, instant reply)   │
# │ memory_correction    │ Update guard fires → LLM saves corrected value    │
# │ live_weather         │ MCP weather tool dispatched → Open-Meteo → reply  │
# │ opinion_query        │ Null-field robustness (OrchestratorResult fix)    │
# └──────────────────────┴────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────

SMOKE_SCENARIOS: List[Tuple[str, str, str]] = [
    (
        "ping",
        "greeting",
        "Webhook alive + Groq fallback greeting (Gemini RPD down → Groq fires)",
    ),
    (
        "my name is TestUser and I live in Hyderabad",
        "memory_write",
        "LLM extracts name+city, both saved to SQLite",
    ),
    (
        "what's my name?",
        "memory_recall",
        "Shortcut fires — zero LLM tokens, instant reply from DB",
    ),
    (
        "actually I made a mistake — my name is Phani",
        "memory_correction",
        "Update guard blocks shortcut; LLM updates the existing fact",
    ),
    (
        "what's the weather forecast for Hyderabad today?",
        "live_weather",
        "MCP weather tool dispatched; Open-Meteo called (no API key needed)",
    ),
    (
        "what do you think about Red Tape running shoes?",
        "opinion_query",
        "OrchestratorResult null-field fix: question=null no longer crashes",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — FULL SCENARIO GROUPS
#
# 16 independent groups. Each group can be run in isolation with --groups.
# Messages within a group build on each other (order matters per group).
# Groups are designed to be independent (no cross-group state assumptions).
#
# Total: 46 messages × ~900 tokens avg ≈ 40K tokens
#
# Token cost per group is annotated. Groups marked (MCP) use free external
# APIs via MCP — these still trigger LLM orchestration (1 call each) but
# the data fetch itself is free and cached.
#
# Format: (message_text, label, notes)
# ─────────────────────────────────────────────────────────────────────────────

FULL_SCENARIOS: Dict[str, Dict] = {

    # ── Memory: Core write pipeline ──────────────────────────────────────────
    "memory_write": {
        "name": "Memory — Write",
        "description": "LLM extracts and persists personal facts from declarations",
        "token_cost": "medium (~5K)",
        "scenarios": [
            ("my name is Sarah",
             "write:name",
             "Explicit declaration → name saved"),
            ("I'm 28 years old",
             "write:age",
             "Explicit declaration → age saved"),
            ("I live in Hyderabad, India",
             "write:city+country",
             "Compound fact → city and country both saved"),
            ("my coffee order is a medium oat milk latte with one pump of vanilla",
             "write:drink",
             "Long preference declaration → favorite_drink saved"),
            ("I drive a Renault Duster SUV",
             "write:car",
             "Vehicle declaration → car saved"),
            ("I'm allergic to peanuts",
             "write:allergy",
             "Health fact declaration → allergies saved"),
        ],
    },

    # ── Memory: Recall (shortcut path — zero LLM tokens) ─────────────────────
    "memory_recall": {
        "name": "Memory — Recall (Shortcut)",
        "description": "All recall questions hit the zero-token shortcut, not the LLM",
        "token_cost": "minimal (~100 tokens — shortcut fires, NO orchestration)",
        "scenarios": [
            ("what's my name?",
             "recall:name",
             "Signal phrase match → instant shortcut reply"),
            ("what's my coffee order?",
             "recall:drink",
             "Signal phrase match → instant shortcut reply"),
            ("how old am I?",
             "recall:age",
             "Signal phrase match → instant shortcut reply"),
            ("what city do I live in?",
             "recall:city",
             "Signal phrase match → instant shortcut reply"),
            ("what am I allergic to?",
             "recall:allergy",
             "Signal phrase match → instant shortcut reply"),
            ("what's my current age?",
             "recall:age_variant",
             "Word-overlap fallback (no 'current age' signal) → still shortcuts"),
        ],
    },

    # ── Memory: Corrections (update guard path) ───────────────────────────────
    "memory_update": {
        "name": "Memory — Update & Correction",
        "description": "Corrections must bypass shortcut and go to LLM to save new value",
        "token_cost": "medium (~3K)",
        "scenarios": [
            ("actually, I made a mistake — I'm 29 years old, not 28",
             "update:age_correction",
             "Correction → update guard blocks shortcut → LLM saves age=29"),
            ("I just got promoted — I'm now a lead engineer at DataFlow Solutions",
             "update:occupation",
             "Life update → LLM saves new occupation"),
            ("what's my current age?",
             "verify:age_updated",
             "Recall after update → shortcut returns 29 (not 28)"),
        ],
    },

    # ── Memory: Lists (create, modify, recall) ────────────────────────────────
    "lists": {
        "name": "Memory — List Management",
        "description": "Shopping/grocery/todo list create, add-item, remove-item, recall",
        "token_cost": "medium (~4K)",
        "scenarios": [
            ("my shopping list: milk, bread, eggs, butter",
             "list:create",
             "LLM creates list → shopping_list saved as CSV"),
            ("add cheese to my shopping list",
             "list:add",
             "LLM reads existing list, applies delta, saves full updated list"),
            ("remove bread from my shopping list",
             "list:remove",
             "LLM reads existing list, removes item, saves updated list"),
            ("what's on my shopping list?",
             "list:recall",
             "Shortcut fires for list recall"),
        ],
    },

    # ── Memory: Delete (with guardrails) ─────────────────────────────────────
    "memory_delete": {
        "name": "Memory — Delete & Guardrails",
        "description": "Fact deletion with allowlist, high-stakes confirmation flow",
        "token_cost": "medium (~4K)",
        "scenarios": [
            ("please forget my car info",
             "delete:car",
             "Allowlisted key → deleted immediately, no confirmation needed"),
            ("what car do I drive?",
             "verify:car_gone",
             "Recall after delete → bot says no car info on record"),
            ("clear my shopping list",
             "delete:list_trigger",
             "High-stakes key → bot asks for confirmation instead of deleting"),
            ("yes",
             "delete:list_confirm",
             "Confirm → list actually deleted"),
        ],
    },

    # ── Memory: Reminders ─────────────────────────────────────────────────────
    "reminders": {
        "name": "Memory — Reminders",
        "description": "Schedule a reminder, list pending reminders",
        "token_cost": "low (~2K)",
        "scenarios": [
            ("remind me to call the dentist tomorrow at 10am",
             "reminder:create",
             "LLM creates reminder with ISO trigger time → stored in reminders table"),
            ("what reminders do I have?",
             "reminder:list",
             "LLM reads reminders_pending → displays formatted list"),
        ],
    },

    # ── Live Data: Weather (MCP → Open-Meteo, free, no key) ──────────────────
    "live_weather": {
        "name": "Live Data — Weather (MCP)",
        "description": "Weather queries routed to MCP /weather → Open-Meteo (free, no API key)",
        "token_cost": "low (~2K; data fetch is free via MCP)",
        "scenarios": [
            ("what's the weather forecast for Hyderabad today?",
             "weather:today",
             "MCP weather tool dispatched; uses city from facts or query"),
            ("will it rain tomorrow in Hyderabad?",
             "weather:tomorrow",
             "Forecast query → MCP returns 3-day forecast; LLM selects tomorrow"),
        ],
    },

    # ── Live Data: News (MCP → GNews/RSS) ────────────────────────────────────
    "live_news": {
        "name": "Live Data — News (MCP)",
        "description": "News queries routed to MCP /news → GNews or RSS fallback",
        "token_cost": "low (~2K; data fetch via MCP, GNews 100 req/day free)",
        "scenarios": [
            ("what's the latest news in India today?",
             "news:india",
             "MCP news tool called; GNews API or RSS fallback; result formatted"),
            ("any tech news today?",
             "news:tech",
             "Topic-specific news search; LLM formats headlines from MCP result"),
        ],
    },

    # ── Live Data: Stocks (MCP → yfinance) ───────────────────────────────────
    "live_stocks": {
        "name": "Live Data — Stocks (MCP)",
        "description": "Stock price queries routed to MCP /stocks → yfinance (free)",
        "token_cost": "low (~2K; data fetch via MCP, no API key needed)",
        "scenarios": [
            ("what's the Nifty 50 today?",
             "stocks:nifty",
             "MCP stocks tool; ^NSEI symbol; ~15min delayed data from Yahoo"),
            ("show me Reliance and TCS share prices",
             "stocks:specific",
             "MCP stocks tool with specific NSE symbols"),
        ],
    },

    # ── Live Data: Currency (MCP → Frankfurter) ───────────────────────────────
    "live_currency": {
        "name": "Live Data — Currency (MCP)",
        "description": "Exchange rate queries routed to MCP /currency → Frankfurter (free ECB)",
        "token_cost": "minimal (~1K; MCP cached 1hr)",
        "scenarios": [
            ("what's the USD to INR exchange rate?",
             "currency:usd_inr",
             "MCP currency tool; Frankfurter ECB data; no API key"),
        ],
    },

    # ── Zero-token: Time/date shortcuts ──────────────────────────────────────
    "time_shortcuts": {
        "name": "Zero-Token — Time & Date Shortcuts",
        "description": "Time/date queries answered from server clock — no LLM call at all",
        "token_cost": "ZERO (server clock only, no LLM involved)",
        "scenarios": [
            ("what time is it?",
             "time:current",
             "Shortcut fires: returns server clock time in APP_TIMEZONE"),
            ("what's today's date?",
             "date:today",
             "Shortcut fires: returns current date, no LLM token used"),
            ("what day is it today?",
             "date:day_name",
             "Shortcut fires: day name + date from server clock"),
        ],
    },

    # ── General Knowledge (no live data, LLM knowledge only) ─────────────────
    "general_knowledge": {
        "name": "General Knowledge",
        "description": "Factual questions answered from LLM training data (no tools)",
        "token_cost": "medium (~3K)",
        "scenarios": [
            ("what brand is Edifice watch?",
             "knowledge:brand",
             "Static brand fact → LLM answers from training (no tool dispatch)"),
            ("what's 245 multiplied by 67?",
             "knowledge:math",
             "Simple arithmetic → LLM answers directly"),
            ("explain what machine learning is in simple terms",
             "knowledge:explanation",
             "Conceptual question → LLM explanation, no facts or tools needed"),
        ],
    },

    # ── Robustness: Null-field resilience (the Red Tape bug) ─────────────────
    "null_field_resilience": {
        "name": "Robustness — Null Field Resilience",
        "description": "Opinion/discussion queries that previously crashed (question=null fix)",
        "token_cost": "low (~1K)",
        "scenarios": [
            ("what do you think about Red Tape running shoes?",
             "null_fix:opinion",
             "LLM returns question=null; OrchestratorResult must not crash"),
            ("do you think Python is better than JavaScript?",
             "null_fix:opinion2",
             "Another opinion query with likely null fields"),
        ],
    },

    # ── Robustness: Context switch ────────────────────────────────────────────
    "context_switch": {
        "name": "Robustness — Context Switch",
        "description": "Switching topics mid-conversation; memory still intact",
        "token_cost": "low (~2K)",
        "scenarios": [
            ("what's 3 squared?",
             "switch:math",
             "Non-personal query; no memory change expected"),
            ("what's my name?",
             "switch:recall_after",
             "Recall after topic switch; shortcut still works correctly"),
            ("explain photosynthesis briefly",
             "switch:knowledge",
             "Knowledge query; memory unaffected"),
        ],
    },

    # ── Robustness: Unknown information ──────────────────────────────────────
    "unknown_info": {
        "name": "Robustness — Unknown Information",
        "description": "Bot must honestly say it doesn't know rather than hallucinating",
        "token_cost": "low (~2K)",
        "scenarios": [
            ("what's my shoe size?",
             "unknown:shoe_size",
             "Never told → bot must say 'I don't have this info'"),
            ("what's my bank account number?",
             "unknown:bank",
             "Never told → honest response; must not hallucinate"),
        ],
    },

    # ── Privacy: Sensitive data handling ─────────────────────────────────────
    "privacy": {
        "name": "Privacy & Sensitive Data",
        "description": "SSNs and passwords must not be stored; bot should redirect",
        "token_cost": "low (~2K)",
        "scenarios": [
            ("my social security number is 123-45-6789",
             "privacy:ssn_write",
             "SSN should NOT be saved as a fact (LLM should decline)"),
            ("what's my social security number?",
             "privacy:ssn_recall",
             "Must not be recalled (either not stored, or correctly refused)"),
            ("I use 1Password as my password manager",
             "privacy:safe_info",
             "Non-sensitive tool name → can be saved as fact"),
        ],
    },

    # ── Comprehensive: Full profile dump ─────────────────────────────────────
    "profile_dump": {
        "name": "Comprehensive — Full Profile Recall",
        "description": "Ask bot to summarise everything it knows — tests context assembly",
        "token_cost": "high (~4K; long reply with all facts)",
        "scenarios": [
            ("tell me everything you remember about me — name, age, city, preferences, car, allergies",
             "dump:full_profile",
             "LLM assembles all facts into a coherent summary reply"),
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Token budget summary
# ─────────────────────────────────────────────────────────────────────────────

def _token_budget_summary() -> str:
    total_msgs = sum(len(g["scenarios"]) for g in FULL_SCENARIOS.values())
    est_tokens = total_msgs * 900
    return (
        f"{total_msgs} messages × ~900 tokens avg ≈ {est_tokens:,} tokens\n"
        f"   (Groq 70B free tier: 100K/day → this uses {est_tokens/1000:.0f}% of daily budget)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Webhook payload builder
# ─────────────────────────────────────────────────────────────────────────────

def _payload(message: str, phone: str, bot: str, session: str) -> dict:
    ts     = int(time.time())
    msg_id = f"true_{phone.split('@')[0]}_{ts}_{uuid.uuid4().hex[:8].upper()}"
    return {
        "id":      f"evt_{uuid.uuid4().hex}",
        "session": session,
        "event":   "message.any",
        "payload": {
            "id":        msg_id,
            "timestamp": ts,
            "from":      phone,
            "fromMe":    True,
            "source":    "app",
            "body":      message,
            "hasMedia":  False,
            "_data": {
                "key": {"remoteJid": phone, "fromMe": True, "id": msg_id},
                "messageTimestamp": ts,
                "pushName": "TestUser",
                "message":  {"conversation": message},
            },
        },
        "timestamp": ts * 1000,
        "me": {"id": bot, "pushName": "Shimmi"},
        "engine": "NOWEB",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tester class
# ─────────────────────────────────────────────────────────────────────────────

class Tester:
    def __init__(
        self,
        url: str,
        phone: str,
        bot: str,
        session: str,
        delay: float,
        timeout: int,
        quiet: bool,
    ):
        self.url     = url
        self.phone   = phone
        self.bot     = bot
        self.session = session
        self.delay   = delay
        self.timeout = timeout
        self.quiet   = quiet
        self.results: list = []

    def send(self, message: str, label: str = "", notes: str = "") -> bool:
        p = _payload(message, self.phone, self.bot, self.session)
        try:
            r  = requests.post(
                self.url, json=p,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            ok = r.status_code == 200
            if not self.quiet:
                icon = "✅" if ok else f"❌ {r.status_code}"
                preview = message[:65] + ("…" if len(message) > 65 else "")
                print(f"  {icon}  [{label:30s}]  {preview!r}")
                if notes and not ok:
                    print(f"         → expected: {notes}")
            self.results.append({
                "label": label, "message": message,
                "ok": ok, "status": r.status_code,
                "ts": datetime.now().isoformat(),
            })
            time.sleep(self.delay)
            return ok
        except requests.exceptions.ConnectionError:
            print(f"\n  ❌  Cannot connect to {self.url}")
            print(   "      Is the bot running?  systemctl status shimmi")
            self.results.append({
                "label": label, "message": message, "ok": False, "status": 0,
                "ts": datetime.now().isoformat(),
            })
            return False

    def summary(self) -> None:
        total = len(self.results)
        ok    = sum(1 for r in self.results if r["ok"])
        print(f"\n{'─'*68}")
        print(f"  {ok}/{total} webhook requests accepted (HTTP 200)")
        if ok < total:
            print("  Failed:")
            for r in self.results:
                if not r["ok"]:
                    print(f"    ✗  [{r['label']}]  HTTP {r['status']}  "
                          f"{r['message'][:50]}")
        print(f"{'─'*68}")
        print(
            "  NOTE: HTTP 200 = webhook ACCEPTED (message enqueued).\n"
            "  Check bot logs to verify LLM replies and memory writes."
        )

    def save(self, path: str) -> None:
        data = {
            "timestamp": datetime.now().isoformat(),
            "url": self.url,
            "total": len(self.results),
            "ok": sum(1 for r in self.results if r["ok"]),
            "results": self.results,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"  💾  Saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_offline(args) -> int:
    """TIER 0: Run all pytest tests — zero LLM quota."""
    print("🧪  TIER 0 — Offline tests (zero LLM quota)\n")
    root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(root / "tests/unit"),
            str(root / "tests/integration"),
            "-v", "--tb=short", "-q",
        ],
        cwd=str(root),
    )
    return result.returncode


def cmd_smoke(args) -> int:
    """TIER 1: 6-message smoke test — minimal quota."""
    print(f"🔬  TIER 1 — Smoke test  →  {args.url}")
    print(f"    6 messages · ~5K tokens · {DEFAULT_DELAY}s between requests\n")
    print("    What each message validates:")
    for msg, label, notes in SMOKE_SCENARIOS:
        print(f"    [{label:25s}]  {notes}")
    print()

    t = Tester(args.url, args.phone, args.bot, args.session,
               DEFAULT_DELAY, DEFAULT_TIMEOUT, args.quiet)
    for msg, label, notes in SMOKE_SCENARIOS:
        t.send(msg, label=label, notes=notes)

    t.summary()
    if args.output:
        t.save(args.output)
    return 0 if all(r["ok"] for r in t.results) else 1


def cmd_full(args) -> int:
    """TIER 2: Full scenario suite — costs real quota."""
    available = list(FULL_SCENARIOS.keys())

    # Resolve --groups aliases
    requested = args.groups or available
    if "memory" in requested:
        requested = [g for g in available if g.startswith("memory")] + \
                    [g for g in requested if g != "memory"]
    if "live" in requested:
        requested = [g for g in available if g.startswith("live")] + \
                    [g for g in requested if g != "live"]

    selected = [g for g in requested if g in FULL_SCENARIOS]
    invalid  = [g for g in requested if g not in FULL_SCENARIOS and g not in ("memory","live")]
    if invalid:
        print(f"  ❌  Unknown groups: {invalid}")
        print(f"     Run `shimmi_tester.py list` to see available groups.")
        return 1

    total_msgs = sum(len(FULL_SCENARIOS[g]["scenarios"]) for g in selected)
    est_tokens = total_msgs * 900

    print(f"⚠️   TIER 2 — Full test  →  {args.url}")
    print(f"    {total_msgs} messages · ~{est_tokens:,} tokens · groups: {selected}")
    if not args.yes:
        confirm = input("\n    This uses real quota. Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("    Aborted.")
            return 0
    print()

    t = Tester(args.url, args.phone, args.bot, args.session,
               args.delay, DEFAULT_TIMEOUT, args.quiet)

    for group_key in selected:
        g = FULL_SCENARIOS[group_key]
        cost = g.get("token_cost", "?")
        print(f"\n📋  {group_key}  —  {g['name']}  [{cost}]")
        print(f"    {g['description']}")
        for msg, label, notes in g["scenarios"]:
            t.send(msg, label=label, notes=notes)

    t.summary()
    if args.output:
        t.save(args.output)
    return 0 if all(r["ok"] for r in t.results) else 1


def cmd_list(_args) -> int:
    """List all scenario groups with message counts and costs."""
    print("\nScenario groups  (use: shimmi_tester.py full --groups <name>)\n")
    print(f"{'Group':<25}  {'Msgs':>4}  {'Cost':<22}  Description")
    print("─" * 95)
    for key, g in FULL_SCENARIOS.items():
        n = len(g["scenarios"])
        print(f"  {key:<23}  {n:>4}  {g.get('token_cost','?'):<22}  {g['description'][:55]}")
    print()
    print(f"  Aliases: --groups memory  (all memory_* groups)")
    print(f"           --groups live    (all live_* groups)")
    print(f"\n  Total: {_token_budget_summary()}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Shimmi Test Runner v4.0 — token-economy-aware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier summary:
  offline  Zero quota. Run on every code change.          → pytest unit + integration
  smoke    ~5K tokens. Run after every deployment.         → 6 critical-path messages
  full     ~40K tokens. Run once before a release.         → all 16 feature groups

Examples:
  python3 shimmi_tester.py offline
  python3 shimmi_tester.py smoke --url http://140.245.218.146:6000/webhook
  python3 shimmi_tester.py full
  python3 shimmi_tester.py full --groups memory live      (specific groups)
  python3 shimmi_tester.py full --groups memory_write memory_recall live_weather
  python3 shimmi_tester.py full --yes                     (skip confirmation)
  python3 shimmi_tester.py list
        """,
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("offline", help="Run pytest unit+integration tests (zero quota)")

    sp = sub.add_parser("smoke", help="6-message smoke test (~5K tokens)")
    sp.add_argument("--url",     default=DEFAULT_URL)
    sp.add_argument("--phone",   default=DEFAULT_PHONE)
    sp.add_argument("--bot",     default=DEFAULT_BOT)
    sp.add_argument("--session", default=DEFAULT_SESSION)
    sp.add_argument("--output",  default="",                  help="Save results to JSON")
    sp.add_argument("--quiet",   action="store_true")

    fp = sub.add_parser("full", help="Full scenario suite (costs real quota)")
    fp.add_argument("--url",     default=DEFAULT_URL)
    fp.add_argument("--phone",   default=DEFAULT_PHONE)
    fp.add_argument("--bot",     default=DEFAULT_BOT)
    fp.add_argument("--session", default=DEFAULT_SESSION)
    fp.add_argument("--groups",  nargs="*",              help="Run specific groups only")
    fp.add_argument("--delay",   type=float, default=float(DEFAULT_DELAY))
    fp.add_argument("--output",  default="test_results.json")
    fp.add_argument("--quiet",   action="store_true")
    fp.add_argument("--yes",     action="store_true",    help="Skip confirmation prompt")

    sub.add_parser("list", help="List all groups + token costs")

    args = p.parse_args()
    if not args.command:
        p.print_help()
        return 0

    dispatch = {
        "offline": cmd_offline,
        "smoke":   cmd_smoke,
        "full":    cmd_full,
        "list":    cmd_list,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
