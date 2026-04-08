# SHIMMI BOT V6 - COMPLETE PRODUCTION PACKAGE
## Critical Bug Fixes + All Enhancements

## 🔴 CRITICAL FIXES INCLUDED

### 1. Threading Error Fixed
**Error:** `RuntimeError: Already borrowed` in sentence-transformers
**Fix:** Added thread-safe locking to embedding function

### 2. Multi-Provider System
**Before:** Only Groq with basic failover
**After:** Groq, Gemini, Claude, OpenAI with intelligent rotation

### 3. Memory Growth Managed
**Before:** Unlimited fact accumulation
**After:** Importance scoring, auto-cleanup, archiving

## 📦 WHAT'S IN THIS PACKAGE

```
shimmi_v6_complete/
├── DEPLOYMENT_INSTRUCTIONS.md  (this file)
├── CHANGELOG.md                 (what changed)
├── requirements.txt             (dependencies)
├── .env.template                (configuration template)
├── app/
│   ├── __init__.py
│   ├── main.py                 ✅ UPDATED - ambient memory integrated
│   ├── config.py               ✅ NEW - multi-provider support
│   ├── database.py             ✅ FIXED - threading issue resolved
│   ├── multi_provider_llm.py   ✅ NEW - intelligent provider rotation
│   ├── structured_actions.py   ✅ NEW - lists/reminders/todos
│   ├── ambient_memory.py       ✅ NEW - passive observation
│   ├── fact_mining.py          ✅ NEW - auto fact extraction
│   ├── agent_engine.py         ✅ UPDATED - uses multi-provider
│   ├── prompts.py              ✅ UPDATED - enhanced prompts
│   ├── rate_limit_manager.py   ✅ KEEP - your implementation
│   ├── utils.py                ✅ KEEP - no changes needed
│   ├── retry.py                ✅ KEEP - no changes needed
│   ├── waha_provider.py        ✅ KEEP - no changes needed
│   └── logging_setup.py        ✅ KEEP - no changes needed
└── scripts/
    └── migrate_memory.py       ✅ NEW - database migration tool

## 🚀 QUICK DEPLOYMENT

### Step 1: Backup Current Installation
```bash
cd /opt
tar czf shimmi_backup_$(date +%Y%m%d_%H%M).tar.gz shimmi/
```

### Step 2: Stop Bot
```bash
systemctl stop shimmi-bot
```

### Step 3: Apply Updates
```bash
cd /opt/shimmi

# Extract the package
unzip shimmi_v6_complete.zip
cd shimmi_v6_complete

# Copy NEW files
cp app/config.py ../app/
cp app/multi_provider_llm.py ../app/
cp app/structured_actions.py ../app/
cp app/ambient_memory.py ../app/
cp app/fact_mining.py ../app/

# Copy UPDATED files  
cp app/main.py ../app/
cp app/database.py ../app/
cp app/agent_engine.py ../app/
cp app/prompts.py ../app/

# Keep YOUR existing files (don't overwrite):
# - app/rate_limit_manager.py
# - app/utils.py
# - app/retry.py
# - app/waha_provider.py
# - app/logging_setup.py
```

### Step 4: Update Configuration
```bash
# Append new settings to .env
cat >> /opt/shimmi/.env << 'ENVEOF'

# Multi-Provider LLM
GROQ_ENABLED=1
GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant,mixtral-8x7b-32768
GROQ_DAILY_LIMIT=100000

GEMINI_ENABLED=1
GEMINI_MODELS=gemini-2.0-flash-exp,gemini-1.5-flash,gemini-1.5-flash-8b
GEMINI_DAILY_LIMIT=1500000

CLAUDE_ENABLED=0
CLAUDE_API_KEY=
CLAUDE_MODELS=claude-3-5-sonnet-20241022,claude-3-5-haiku-20241022

OPENAI_ENABLED=0
OPENAI_API_KEY=
OPENAI_MODELS=gpt-4o-mini,gpt-3.5-turbo

LLM_PROVIDER_PRIORITY=groq,gemini,claude,openai

# Features
ACTIONS_ENABLED=1
MEMORY_CLEANUP_ENABLED=1
MEMORY_CLEANUP_INTERVAL_HOURS=24
FACT_MINING_ENABLED=1
FACT_MINING_INTERVAL_HOURS=24

# Keep all your existing settings!
ENVEOF
```

### Step 5: Install Dependencies
```bash
cd /opt/shimmi
pip install anthropic openai --break-system-packages
```

### Step 6: Test Configuration
```bash
python3 << 'PYTEST'
from app.config import settings
print(f"Enabled providers: {settings.get_enabled_providers()}")
print(f"Groq models: {settings.get_provider_config('groq')['models']}")
print(f"Gemini models: {settings.get_provider_config('gemini')['models']}")
PYTEST
```

### Step 7: Start Bot
```bash
systemctl start shimmi-bot
```

### Step 8: Monitor Logs
```bash
tail -f /opt/shimmi/shimmi-bot.log

# Watch for:
# ✅ "provider=groq model=llama-3.3-70b-versatile"
# ✅ "provider=groq model=llama-3.1-8b-instant"  
# ✅ "provider=groq model=mixtral-8x7b-32768"
# ✅ No "RuntimeError: Already borrowed"
```

## ✅ VERIFICATION TESTS

### Test 1: Threading Fix
```bash
# Send multiple messages quickly in group
# Should NOT see "RuntimeError: Already borrowed"
```

### Test 2: Model Rotation
```bash
# Send 5 messages
# Check logs - should see different models being used
tail -50 /opt/shimmi/shimmi-bot.log | grep "provider="
```

### Test 3: Provider Failover
```bash
# Test will happen automatically when Groq rate limit hits
# Bot should seamlessly switch to Gemini
```

### Test 4: Structured Actions
```
User: "Spock, create a shopping list"
Bot: "Created your shopping list"

User: "Add milk and bread"  
Bot: "Added 2 items"

User: "Show my lists"
Bot: [displays list]
```

## 🔧 ROLLBACK IF NEEDED

```bash
systemctl stop shimmi-bot
cd /opt
rm -rf shimmi/
tar xzf shimmi_backup_YYYYMMDD_HHMM.tar.gz
systemctl start shimmi-bot
```

## 📊 WHAT CHANGED

See CHANGELOG.md for detailed list of changes.

## 🐛 TROUBLESHOOTING

### Issue: Bot not starting
```bash
# Check syntax errors
python3 -m py_compile /opt/shimmi/app/*.py

# Check logs
journalctl -u shimmi-bot -n 50
```

### Issue: "Already borrowed" error persists
```bash
# Verify database.py was updated
grep "embed_lock" /opt/shimmi/app/database.py
# Should see: self.embed_lock = asyncio.Lock()
```

### Issue: No model rotation
```bash
# Check .env formatting (no spaces in model list)
# Correct: model1,model2,model3
# Wrong: model1, model2, model3
```

## 📞 POST-DEPLOYMENT

1. **Monitor for 24 hours** - Watch logs for any errors
2. **Check memory growth** - Run: `du -sh /opt/shimmi/data`
3. **Verify providers** - Should see both Groq and Gemini in logs
4. **Test edge cases** - Empty messages, emojis, long messages

## 🎉 SUCCESS CRITERIA

✅ No "Already borrowed" errors
✅ Multiple models rotating in logs
✅ Bot responds even when Groq is rate-limited
✅ Memory stays under control (<100MB growth/day)
✅ Structured actions work (lists, reminders)

Your bot is now **production-grade enterprise quality**! 🚀
