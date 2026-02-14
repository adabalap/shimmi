# Shimmi / Spock — Final Build

## What this build fixes
- **fromMe**: allowed (when `ALLOW_FROMME=1`) and responds to your own messages, with strong loop protection.
- **Strict allowlist**: ONLY chats in `ALLOWED_GROUP_JIDS` are processed **and** logged **and** persisted.
- **WhatsApp-friendly output**: no markdown tables; answers are short, bulleted, and readable on WhatsApp.
- **Generic memory updates**: the LLM can propose **multiple** deterministic `key/value` facts (no pattern-specific code).
- **Live Search**: for weather/news/stocks/movies etc., uses Groq `web_search` and enriches query with stored facts/preferences.
- **Pydantic typing crash fixed**: avoids newer typing annotations that caused runtime errors.

## Run
```bash
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/verify_integrity.sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 6000
```

## Notes
- Add your DM chat id (often `...@lid`) to `ALLOWED_GROUP_JIDS` if you want fromMe testing in that chat.
