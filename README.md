# Shimmi / Spock — Production Grounded v4

## Critical fix vs v3
The logs showed `facts.loaded count=0` even after `memory.verified count=3`.
Root cause: `from app.database import sqlite_store` imports a *stale* name binding (stays None),
so the running code never used the initialized SQLite store for read/write.

v4 fixes this by importing the database module and referencing `database.sqlite_store` directly.

## Also included
- Canonical user id for memory keys: treats `...@c.us` and `...@lid` as the same identity.
- Facts are always retrieved and passed into planner + live search.
- WhatsApp-friendly formatting and invocation stripping remain.

## Run
```bash
cd /opt/shimmi
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/verify_integrity.sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 6000
```
