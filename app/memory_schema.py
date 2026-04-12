"""
memory_schema.py — Single source of truth for Shimmi memory keys.

All canonical key names, aliases, deletion rules, and prompt filtering
are defined here and imported by database.py, agent_engine.py, and
prompts.py.  Previously these were scattered across three files with
duplicate definitions that drifted over time.

To add a new fact key:
  1. Add it to CANONICAL_KEYS (drives LLM prompts).
  2. Add any spelling variants to KEY_ALIASES.
  3. If it should be deletable by the agent, add to DELETABLE_KEYS.
  4. If deletion requires confirmation, add to CONFIRM_BEFORE_DELETE.
  5. If it should be excluded from the LLM prompt window, add to PROMPT_SKIP_KEYS.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List

# ---------------------------------------------------------------------------
# Canonical key registry
# ---------------------------------------------------------------------------
# The authoritative list of fact keys the LLM should use.
# Grouped for readability; order matters for prompt generation.
# ---------------------------------------------------------------------------

CANONICAL_KEYS: List[str] = [
    # Identity
    "name", "age",
    # Location
    "city", "country", "postal_code",
    # Work / education
    "occupation", "company", "education",
    # Preferences
    "favorite_color", "favorite_drink", "favorite_food",
    "favorite_cuisine", "favorite_trail",
    "dietary_restriction", "allergies",
    # Fitness / goals
    "fitness_goals", "personal_goals", "career_goals",
    # Travel
    "travel_plans", "travel_companion",
    # Transport / pets
    "car", "bike", "pets",
    # Books / language
    "recent_book", "preferred_language",
    # Lists
    "shopping_list", "grocery_list", "todo_list",
    # Finance
    "portfolio_stocks",        # flat ticker list  e.g. "PAYTM.NS, INFY.NS"
    "portfolio_holdings",      # JSON: [{"symbol":"PAYTM.NS","qty":100,"avg_price":1000}, ...]
    # Other personal
    "interests", "hobbies", "motivational_quote", "reminder_notes",
    # Bot interaction
    "bot_nickname",      # what the user calls Shimmi (e.g. "Chitti", "Buddy")
]

# Keys the LLM extractor should also recognise (narrower set for extraction prompt)
EXTRACTOR_KEYS: List[str] = CANONICAL_KEYS + [
    "reading_list",  # alias for recent_book in some prompts
]

# ---------------------------------------------------------------------------
# Key aliases — maps variant spellings → canonical key
# ---------------------------------------------------------------------------
# Normalised at DB write time by normalize_key().
# Add new entries here; never in database.py or agent_engine.py.
# ---------------------------------------------------------------------------

KEY_ALIASES: Dict[str, str] = {
    # ── name ──────────────────────────────────────────────────────────────
    "username": "name", "first_name": "name", "full_name": "name",
    "user_name": "name", "display_name": "name",

    # ── location ──────────────────────────────────────────────────────────
    "user_city": "city", "user_location": "city",
    "location": "city", "hometown": "city", "current_city": "city",
    "user_country": "country",
    "zip": "postal_code", "zipcode": "postal_code", "pin": "postal_code",
    "pincode": "postal_code",

    # ── color ─────────────────────────────────────────────────────────────
    "colour": "favorite_color", "favorite_colour": "favorite_color",
    "favourite_color": "favorite_color", "favourite_colour": "favorite_color",
    "preferred_color": "favorite_color", "preferred_colour": "favorite_color",

    # ── drink ─────────────────────────────────────────────────────────────
    "user_favorite_drink": "favorite_drink", "preferred_drink": "favorite_drink",
    "user_drink": "favorite_drink", "drink": "favorite_drink",
    "favourite_drink": "favorite_drink", "fav_drink": "favorite_drink",
    "coffee_order": "favorite_drink",

    # ── food ──────────────────────────────────────────────────────────────
    "favourite_food": "favorite_food", "fav_food": "favorite_food",
    "preferred_food": "favorite_food",
    "favourite_cuisine": "favorite_cuisine", "fav_cuisine": "favorite_cuisine",
    "preferred_cuisine": "favorite_cuisine",

    # ── interests / hobbies ───────────────────────────────────────────────
    "user_interests": "interests", "user_interest": "interests",
    "interest": "interests", "passion": "interests", "passions": "interests",
    "technical_interests": "interests",
    "user_hobby": "hobbies", "user_hobbies": "hobbies",
    "hobby": "hobbies",

    # ── occupation / work ─────────────────────────────────────────────────
    "user_occupation": "occupation", "user_job": "occupation",
    "job": "occupation", "profession": "occupation", "role": "occupation",
    "job_title": "occupation", "current_job_title": "occupation",
    "work": "occupation",
    "employer": "company", "current_company": "company",
    "workplace": "company", "work_place": "company",
    "work_experience": "occupation",

    # ── education ─────────────────────────────────────────────────────────
    "educational_background": "education",
    "degree_background": "education",
    "school": "education", "college": "education",

    # ── fitness / health ──────────────────────────────────────────────────
    "fitness_goal": "fitness_goals", "fitness_target": "fitness_goals",
    "health_goal": "fitness_goals", "health_goals": "fitness_goals",
    "marathon_goal": "fitness_goals",

    # ── travel ────────────────────────────────────────────────────────────
    "travel_plan": "travel_plans", "next_trip": "travel_plans",
    "upcoming_trip": "travel_plans",

    # ── pets ──────────────────────────────────────────────────────────────
    "pet": "pets", "pet_name": "pets",

    # ── vehicle ───────────────────────────────────────────────────────────
    "vehicle": "car",

    # ── books / reading ───────────────────────────────────────────────────
    "book": "recent_book", "books": "recent_book",
    "books_read": "recent_book", "current_book": "recent_book",
    "reading": "recent_book", "last_book": "recent_book",
    "reading_list": "recent_book",

    # ── finance / portfolio ──────────────────────────────────────────────
    "portfolio": "portfolio_stocks", "my_stocks": "portfolio_stocks",
    "watchlist": "portfolio_stocks",
    "stock_portfolio": "portfolio_stocks",
    "holdings": "portfolio_holdings", "my_holdings": "portfolio_holdings",
    "portfolio_details": "portfolio_holdings",

    # ── lists ─────────────────────────────────────────────────────────────
    "grocery": "grocery_list", "groceries": "grocery_list",
    "shopping": "shopping_list",
    "todo": "todo_list", "todos": "todo_list", "task": "todo_list",
    "bot_name": "bot_nickname", "nickname": "bot_nickname",
    "call_me": "bot_nickname", "shimmi_name": "bot_nickname",

    # ── language ──────────────────────────────────────────────────────────
    "user_language": "preferred_language", "language": "preferred_language",
    "lang": "preferred_language",
    "favorite_language": "preferred_language",

    # ── age ───────────────────────────────────────────────────────────────
    "user_age": "age",

    # ── career / goals ────────────────────────────────────────────────────
    "career_goal": "career_goals", "career_aspiration": "career_goals",
    "career_aspirations": "career_goals",
    "goal": "personal_goals", "life_goal": "personal_goals",
}

# ---------------------------------------------------------------------------
# Deletion guardrails
# ---------------------------------------------------------------------------

# Only these keys may be deleted by the agent.
DELETABLE_KEYS: FrozenSet[str] = frozenset({
    "name", "age",
    "city", "country", "postal_code",
    "occupation",
    "favorite_drink", "favorite_food", "favorite_cuisine",
    "favorite_color", "favorite_trail",
    "hobbies", "interests",
    "dietary_restriction", "allergies",
    "car", "bike", "vehicle",
    "pets",
    "shopping_list", "grocery_list", "todo_list",
    "motivational_quote", "preferred_language",
})

# Subset requiring explicit user confirmation before deletion fires.
CONFIRM_BEFORE_DELETE: FrozenSet[str] = frozenset({
    "shopping_list",
    "grocery_list",
    "todo_list",
})

# Keys that are structurally protected — never deletable.
PROTECTED_KEYS: FrozenSet[str] = frozenset({
    "whatsapp_id", "chat_id",
})

# Keys that the consolidation LLM must NEVER absorb into another key or
# use as a canonical target. These are user-curated lists with distinct
# semantics — merging them into e.g. recent_book would destroy their value.
# Evidence: consolidate.merged canonical=recent_book absorbed=['shopping_list']
#           → shopping list overwritten with book title. Bug confirmed in logs.
CONSOLIDATION_PROTECTED: FrozenSet[str] = frozenset({
    "shopping_list",
    "grocery_list",
    "todo_list",
    "lists",
    "reminders",
    "portfolio_stocks",
    "portfolio_holdings",
    "read_books",          # book list — distinct from recent_book (single title)
    "social_security_number",  # should never be touched by consolidation
    "allergies",           # medical — must not be merged with food preferences
})

# ---------------------------------------------------------------------------
# Prompt filtering
# ---------------------------------------------------------------------------

# Keys excluded from the orchestrator prompt window — they waste tokens or
# are a security risk. Still stored in DB for audit/consolidation.
PROMPT_SKIP_KEYS: FrozenSet[str] = frozenset({
    # Ephemeral — expire within the same session, stale on next conversation
    "recent_query",
    "recent_search",
    "conversation_since_morning",
    "arrival_time",
    "destination",
    "next_meeting_team",
    "next_meeting_time",
    "next_trip_start_date",
    # Verbose text blocks — waste 200-800 tokens on every prompt
    "last_summary",
    "recent_article_details",
    "favorite_news_source_details",
    # Structural noise — duplicate/vague/meaningless
    "greeting",                     # 'Hi there' — LLM never uses this
    "lists",                        # duplicate of shopping_list + grocery_list
    "favorite_video",               # always vague ('awesome video', etc.)
    "recent_article",               # URL only, not a personal fact
    "work_experience",              # superseded by occupation + company
    # Security — never shown in prompts (also blocked from storage entirely)
    "social_security_number",
    "bank_account",
    "credit_card",
    "password",
    "pin",
})

# ---------------------------------------------------------------------------
# NEVER_STORE_KEYS — hard blocklist at DB write time
# These keys are REJECTED by upsert_fact regardless of source.
# Use for anything that should never be persisted: PII, secrets, session noise.
# ---------------------------------------------------------------------------
NEVER_STORE_KEYS: FrozenSet[str] = frozenset({
    # Sensitive PII — must never be stored
    "social_security_number", "ssn",
    "bank_account", "bank_account_number",
    "credit_card", "credit_card_number",
    "password", "passphrase",
    "pin", "otp",
    "passport_number",
    "aadhaar", "aadhaar_number", "pan", "pan_number",
    # Per-stock noise keys — portfolio_holdings JSON already has this data
    # These get created when LLM stores individual stock facts instead of the JSON
    # Pattern: portfolio_stocks_<name>, portfolio_purchase_price_<name>, 
    #          stock_<name>_price, stock_<name>_quantity, favorite_stock
    # We can't block dynamic keys in a frozenset, but we can block the known ones.
    # The extractor prompt instructs the LLM to use portfolio_holdings instead.
    "favorite_stock",             # meaningless extracted artifact
    "stock_paytm_price",          # per-stock noise — in portfolio_holdings
    "stock_paytm_quantity",
    "stock_acmesolar_price",
    "stock_acmesolar_quantity",
    "portfolio_purchase_price_paytm",
    "portfolio_purchase_price_acmesolar",
    "portfolio_stocks_paytm",
    "portfolio_stocks_acmesolar",
    # Session-ephemeral noise — no value in persisting
    "greeting",
    "arrival_time",
    "destination",
    "next_meeting_team",
    "next_meeting_time",
    "next_trip_start_date",
    "recent_query",
    "recent_search",
    "conversation_since_morning",
    "lists",                        # duplicate of shopping_list / grocery_list
    "favorite_video",               # always vague
})

# Junk placeholder values — filtered at DB read time and prompt build time.
JUNK_VALUES: FrozenSet[str] = frozenset({
    "unknown", "none", "null", "n/a", "na",
    "not set", "not specified", "undefined",
    "empty", "no data", "", "false", "true",
})

# ---------------------------------------------------------------------------
# Prompt helpers — called by prompts.py to avoid hardcoded key lists
# ---------------------------------------------------------------------------

def canonical_keys_str() -> str:
    """Comma-separated canonical key list for injection into LLM prompts."""
    return ", ".join(CANONICAL_KEYS)


def deletable_keys_str() -> str:
    """Sorted comma-separated deletable key list for injection into prompts."""
    return ", ".join(sorted(DELETABLE_KEYS))


def high_stakes_keys_str() -> str:
    """Comma-separated high-stakes (confirm-before-delete) keys."""
    return ", ".join(sorted(CONFIRM_BEFORE_DELETE))
