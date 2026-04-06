# Shimmi Test Runner
# ──────────────────
# make test         → TIER 0: all offline tests, zero quota (run on every change)
# make smoke        → TIER 1: 6-message live smoke test (~5K tokens)
# make full         → TIER 2: all 16 feature groups (~40K tokens, pre-release only)
# make list         → show full scenario groups + token costs

PYTHON  := python3
PYTEST  := $(PYTHON) -m pytest
TESTER  := $(PYTHON) tests/shimmi_tester.py
URL     ?= http://localhost:6000/webhook

.PHONY: test smoke full list

## ── TIER 0: Offline — zero LLM quota ────────────────────────────────────────
test:
	@echo "🧪  TIER 0 — Offline tests (zero LLM tokens)\n"
	$(PYTEST) tests/unit/ tests/integration/ -v --tb=short -q

## ── TIER 1: Smoke — ~5K tokens, run after every deploy ──────────────────────
smoke:
	$(TESTER) smoke --url $(URL)

## ── TIER 2: Full — ~40K tokens, run once per release ─────────────────────────
full:
	$(TESTER) full --url $(URL)

## ── List all full-suite groups + token costs ─────────────────────────────────
list:
	$(TESTER) list
