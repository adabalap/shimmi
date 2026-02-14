# Shimmi / Spock — Refactor v3

## Changes vs v2
- Fixes **fromMe requests being ignored** (no longer blocks fromMe requests just because they contain prefixes).
- Keeps loop safety via **echo detection** (outbound id + outbound hash).
- Keeps strict allowlist for ALL chats, and logs only for allowlisted chats.
- Adds explicit empty-text ignore reason (prevents misleading no_prefix on blank payloads).

## Run
```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q .
python -m uvicorn app.main:app --host 0.0.0.0 --port 6000
```
