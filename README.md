# Shimmi / Spock — Production Grounded v3

## What v3 fixes (based on your attached logs)
- **Memory not being persisted** after you provided Hyderabad/India/postal code.
- **Live search ignoring locale facts**, leading to U.S./°F outputs.
- **Invocation stripping artifacts** like `Ok, .` and `..., ?`.

## Design alignment
Implements the intended pipeline:
1) Ingest (allowlist / echo / prefix gating)
2) Extract deterministic facts (generic)
3) Verify facts strictly against user text
4) Persist facts (SQLite)
5) Retrieve facts (SQLite) + context (Chroma)
6) Plan action (answer vs live_search vs ask_facts)
7) Live search uses facts to enforce locale units/currency

## Run
```bash
cd /opt/shimmi
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/verify_integrity.sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 6000
```
