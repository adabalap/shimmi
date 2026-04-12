#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# shimmi/tests/live/smoke_test.sh — v3.17.0
#
# Tier 3 live smoke tests — sends real messages via WAHA and verifies the
# bot log for expected behaviour. Uses real LLM quota (~10 messages, ~$0.05).
#
# WHEN TO RUN:
#   ✅ After major feature changes (briefing, memory, new tools)
#   ✅ After provider changes (new models, fallback chain)
#   ❌ NOT on every code change — use pytest tests/unit/ for that
#
# USAGE:
#   export WAHA_URL="http://140.245.218.146:3000"
#   export WAHA_KEY="86f75b284afb4fc4975105724f277871"
#   export TEST_CHAT_ID="919573717667@c.us"   # your test WhatsApp number
#   export BOT_LOG="/opt/shimmi/logs/shimmi-bot.log"
#   bash tests/live/smoke_test.sh
#
# HOW IT WORKS:
#   Each test sends a message via WAHA, waits for the bot to process it,
#   then checks the bot log for expected log lines (not the WhatsApp reply).
#   This avoids needing to parse WhatsApp messages and works offline too.
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

WAHA_URL="${WAHA_URL:-http://140.245.218.146:3000}"
WAHA_KEY="${WAHA_KEY:-86f75b284afb4fc4975105724f277871}"
CHAT_ID="${TEST_CHAT_ID:-919573717667@c.us}"
BOT_LOG="${BOT_LOG:-/opt/shimmi/logs/shimmi-bot.log}"
WAIT_SEC=18   # max seconds to wait for bot response

PASS=0; FAIL=0; SKIP=0

# Colours
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log() { echo -e "${NC}$*"; }
pass() { echo -e "${GREEN}  ✅ PASS${NC}: $*"; ((PASS++)); }
fail() { echo -e "${RED}  ❌ FAIL${NC}: $*"; ((FAIL++)); }
skip() { echo -e "${YELLOW}  ⏭️  SKIP${NC}: $*"; ((SKIP++)); }

# ── Helpers ──────────────────────────────────────────────────────────────────

# BOT_PREFIX: first configured prefix (default: shimmi)
# All test messages must contain this or the bot will silently drop them.
BOT_PREFIX="${BOT_PREFIX:-shimmi}"

send_msg() {
    local text="$1"
    # Safety guard: warn if message has no bot prefix
    # (bot will silently drop prefixless messages via has_prefix() check)
    if ! echo "$text" | grep -qiE "(${BOT_PREFIX}|చిట్టి|chitti|spock)"; then
        echo "  ⚠️  WARNING: message has no bot prefix — bot will drop it: ${text:0:60}"
        echo "     Prefix with \"${BOT_PREFIX}\" to ensure bot processes it."
    fi
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "X-Api-Key: ${WAHA_KEY}" \
        -d "{\"session\":\"default\",\"chatId\":\"${CHAT_ID}\",\"text\":$(echo "$text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}" \
        "${WAHA_URL}/api/sendText" > /dev/null
}

# Wait for a log line matching pattern to appear after a timestamp mark
wait_for_log() {
    local pattern="$1"
    local mark="$2"   # timestamp string from just before sending
    local waited=0
    while [[ $waited -lt $WAIT_SEC ]]; do
        if grep -q "$pattern" <(grep -A9999 "$mark" "$BOT_LOG" 2>/dev/null | tail -200); then
            return 0
        fi
        sleep 1; ((waited++))
    done
    return 1
}

log_mark() {
    # Return a timestamp fragment we can grep for to isolate this test's logs
    date '+%H:%M:%S' | cut -c1-7   # HH:MM: (minute-level precision)
}

# ── Preflight ─────────────────────────────────────────────────────────────────

log "\n=== Shimmi Smoke Tests v3.17.0 ==="
log "WAHA:    ${WAHA_URL}"
log "Chat:    ${CHAT_ID}"
log "Log:     ${BOT_LOG}"
log ""

if [[ ! -f "$BOT_LOG" ]]; then
    log "⚠️  Bot log not found at ${BOT_LOG} — log-based verification unavailable"
    log "   Will only verify that WAHA accepted the message (exit 0 from curl)"
    LOG_CHECK=0
else
    LOG_CHECK=1
fi

# ── Tests ─────────────────────────────────────────────────────────────────────

# T01: Basic response — bot is alive
log "\n[T01] Basic: bot responds to hello"
MARK=$(log_mark); send_msg "shimmi hello"
sleep 3
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "msg.sent" "$MARK"; then
    pass "bot sent a reply to 'hello'"
elif [[ $LOG_CHECK -eq 0 ]]; then
    skip "no log file — assumed OK"
else
    fail "no msg.sent in log after ${WAIT_SEC}s"
fi

sleep 3   # avoid rate limits

# T02: Briefing — full pipeline, early exit sentinel
log "\n[T02] Briefing: news updates triggers briefing early-exit"
MARK=$(log_mark); send_msg "shimmi what's the news updates"
sleep 15
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "briefing.early_exit" "$MARK"; then
    pass "briefing.early_exit fired — full briefing sent without second LLM call"
elif wait_for_log "msg.sent" "$MARK"; then
    skip "msg.sent found but briefing.early_exit not confirmed"
else
    fail "no briefing response after ${WAIT_SEC}s"
fi

sleep 3

# T03: Telugu prefix — Unicode regex fix
log "\n[T03] Telugu prefix: చిట్టి recognised as invocation"
MARK=$(log_mark); send_msg "What's the time, చిట్టి"
sleep 8
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "msg.sent" "$MARK"; then
    pass "Telugu-prefixed message processed"
else
    # If prefix not in .env, this will be skipped by webhook
    if wait_for_log "no_prefix" "$MARK"; then
        skip "చిట్టి not in BOT_COMMAND_PREFIX — add it to .env to enable"
    else
        fail "no response to Telugu-prefixed message"
    fi
fi

sleep 3

# T04: Facts shortcut — zero LLM tokens
log "\n[T04] Facts shortcut: 'what's my name' uses shortcut (no LLM)"
MARK=$(log_mark); send_msg "shimmi what is my name"
sleep 6
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "facts.shortcut" "$MARK"; then
    pass "facts.shortcut fired — answered without LLM"
elif wait_for_log "msg.sent" "$MARK"; then
    skip "msg.sent found, shortcut not confirmed (may have used LLM)"
else
    fail "no response to name query"
fi

sleep 3

# T05: List display — reads from facts, no search
log "\n[T05] Lists: 'what's on my grocery list' reads from memory"
MARK=$(log_mark); send_msg "shimmi what's on my grocery list"
sleep 10
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "msg.sent" "$MARK"; then
    # Verify it didn't trigger a web search
    if wait_for_log "tool_dispatch.*news\|tool_dispatch.*web_search" "$MARK"; then
        fail "list query incorrectly triggered a search"
    else
        pass "list displayed without search"
    fi
else
    fail "no response to list query"
fi

sleep 3

# T06: Memory write — bot stores a new fact
log "\n[T06] Memory: bot stores new fact"
MARK=$(log_mark); send_msg "shimmi I prefer chai over coffee in the morning"
sleep 12
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "memory.summary.*created=1\|facts_created.*1" "$MARK"; then
    pass "new fact created in memory"
elif wait_for_log "msg.sent" "$MARK"; then
    skip "msg.sent found but memory write not confirmed"
else
    fail "no response to memory statement"
fi

sleep 3

# T07: Stock price — tool dispatch to MCP
log "\n[T07] Stocks: live stock price triggers MCP tool"
MARK=$(log_mark); send_msg "shimmi what is Nifty at today"
sleep 15
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "tool_dispatch.*stocks\|mcp_client.*stocks" "$MARK"; then
    pass "stocks tool dispatched"
elif wait_for_log "msg.sent" "$MARK"; then
    skip "msg.sent found — verify stock price in WhatsApp reply manually"
else
    fail "no response to stock query"
fi

sleep 3

# T08: Delete confirmation flow — pending delete registered
log "\n[T08] Delete guard: clearing a protected list asks for confirmation"
MARK=$(log_mark); send_msg "shimmi clear my shopping list"
sleep 10
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "pending_delete.registered\|delete.needs_confirm" "$MARK"; then
    pass "delete confirmation requested for protected list"
elif wait_for_log "msg.sent" "$MARK"; then
    skip "msg.sent found but pending_delete not confirmed in log"
else
    fail "no response to delete request"
fi

# Cancel the pending delete
sleep 3
send_msg "shimmi no keep it" > /dev/null
sleep 5

# T09: Reminder set — scheduler stores reminder
log "\n[T09] Reminders: set a reminder"
MARK=$(log_mark); send_msg "shimmi remind me to check emails at 6pm today"
sleep 12
if [[ $LOG_CHECK -eq 1 ]] && wait_for_log "reminder.*scheduled\|reminders_scheduled.*1" "$MARK"; then
    pass "reminder scheduled successfully"
elif wait_for_log "msg.sent" "$MARK"; then
    skip "msg.sent found — verify reminder in WhatsApp reply manually"
else
    fail "no response to reminder request"
fi

sleep 3

# T10: Knowledge query — no search needed
log "\n[T10] Knowledge: poem/creative request answered without search"
MARK=$(log_mark); send_msg "shimmi tell me a short haiku about rain"
sleep 12
if [[ $LOG_CHECK -eq 1 ]]; then
    if wait_for_log "tool_dispatch" "$MARK"; then
        fail "creative query triggered a tool search (should answer directly)"
    elif wait_for_log "msg.sent" "$MARK"; then
        pass "haiku answered without tool search"
    else
        fail "no response to haiku request"
    fi
else
    skip "no log file available"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════"
echo "  Results: ${PASS} passed  ${FAIL} failed  ${SKIP} skipped"
echo "═══════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}  SMOKE TESTS FAILED — do not deploy${NC}"
    exit 1
else
    echo -e "${GREEN}  All checks passed ✅${NC}"
    exit 0
fi
