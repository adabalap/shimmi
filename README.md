# Shimmi / Spock — Production Grounded v5

## What v5 fixes (based on your v4 logs)
- Facts were saved later, but **planner still asked the wrong clarifying question** first (units instead of location).
- Bot answered a personal fact (“I live in…”) even when **facts.loaded** did not contain those keys.

## v5 upgrades
- Planner contract includes `requires_locale` boolean.
- Code enforces: if `requires_locale=true` but locale facts are missing, it **must ask for location** (never guess).
- System prompt clarifies: use FACTS as source-of-truth for user personal facts; do not claim personal facts unless present in FACTS.
- Keeps v4 fixes: canonical identity key, database module references, allowlist/echo/fromMe.

## Run
```bash
cd /opt/shimmi
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/verify_integrity.sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 6000
```
