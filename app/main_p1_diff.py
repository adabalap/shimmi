"""
main.py — Phase 1 change summary
=================================

Only ONE section of main.py changes in Phase 1: the memory-save loop
(~lines 450–500 in the Phase 0 version).

Replace the inner loop body:

    for mu in result.memory_updates:
        try:
            status = await database.sqlite_store.upsert_fact(
                sender_key, mu.key, mu.value,
            )
            ...

with the version below that also handles delete updates (P1-FEAT-2).

All other code in main.py is identical to Phase 0.
"""

# ─────────────────────────────────────────────────────────────────────────────
# REPLACE: memory-save loop inside the chat worker (main.py ~line 450)
# ─────────────────────────────────────────────────────────────────────────────
#
# OLD (Phase 0):
#
#   for mu in result.memory_updates:
#       try:
#           status = await database.sqlite_store.upsert_fact(
#               sender_key, mu.key, mu.value,
#           )
#           if status == "created":
#               created += 1
#               logger.info("🧠 memory.new  ...")
#           elif status == "updated":
#               updated += 1
#               logger.info("🧠 memory.updated  ...")
#           saved += 1
#       except Exception as exc:
#           ...
#
# NEW (Phase 1) — P1-FEAT-2: route delete vs upsert:

NEW_MEMORY_SAVE_LOOP = '''
                for mu in result.memory_updates:
                    try:
                        if mu.is_delete:
                            # P1-FEAT-2: hard-delete the fact
                            deleted_ok = await database.sqlite_store.delete_fact(
                                sender_key, mu.key,
                            )
                            if deleted_ok:
                                logger.info(
                                    "🗑️  memory.deleted  sender=%s  key=%s",
                                    sender_key, mu.key,
                                )
                            saved += 1
                        else:
                            status = await database.sqlite_store.upsert_fact(
                                sender_key, mu.key, mu.value,
                            )
                            if status == "created":
                                created += 1
                                logger.info(
                                    "🧠 memory.new      sender=%s  key=%s  value=%r",
                                    sender_key, mu.key, mu.value,
                                )
                            elif status == "updated":
                                updated += 1
                                logger.info(
                                    "🧠 memory.updated  sender=%s  key=%s  value=%r",
                                    sender_key, mu.key, mu.value,
                                )
                            saved += 1
                    except Exception as exc:
                        err_msg = f"{type(exc).__name__}: {exc}"
                        save_errors.append(f"{mu.key}: {err_msg}")
                        logger.error(
                            "🧠 memory.save_failed  sender=%s  key=%s  err=%s",
                            sender_key, mu.key, exc,
                        )
'''

# ─────────────────────────────────────────────────────────────────────────────
# Also update the import at the top of main.py:
#
# OLD:
#   from .agent_engine import (
#       init_llm, close_llm, run_agent, extract_reply_memory,
#       VALID_GROQ_PREFIXES, VALID_GEMINI_PREFIXES,
#       _is_reminder_duplicate, MODEL_CIRCUIT, PROVIDER_CIRCUIT, _TOKEN_BUDGET,
#   )
#
# NEW: add _gemini_rpm_limiter to the import list so the /health endpoint can
#      report current Gemini RPM usage:
#
#   from .agent_engine import (
#       init_llm, close_llm, run_agent, extract_reply_memory,
#       VALID_GROQ_PREFIXES, VALID_GEMINI_PREFIXES,
#       _is_reminder_duplicate, MODEL_CIRCUIT, PROVIDER_CIRCUIT, _TOKEN_BUDGET,
#       _gemini_rpm_limiter,   # P1-FEAT-3
#   )
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Optional: update /health endpoint to show Gemini RPM usage
#
# In the health() endpoint dict, add:
#   "gemini_rpm_current": _gemini_rpm_limiter.current_rpm(),
#   "gemini_rpm_limit":   _gemini_rpm_limiter.limit,
# ─────────────────────────────────────────────────────────────────────────────
