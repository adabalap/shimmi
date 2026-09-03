"""
tests/unit/test_metrics.py — Shimmi v3.17.5

Zero-quota unit tests for the Prometheus metrics module (`app/metrics.py`)
backing the `/metrics` endpoint.

Coverage:
  ① Counter increment / label handling
  ② Declared-but-unincremented counters still render (zero is a real signal)
  ③ Prometheus text exposition format — HELP/TYPE/sample shape
  ④ Escaping of help text and label values
  ⑤ Gauge families supplied by the caller at scrape time
  ⑥ Cardinality guard — no chat_id/sender labels leak into the scrape

`app.metrics` imports nothing from the rest of the app, so these tests run
without fastapi/groq/chromadb present.
"""
from __future__ import annotations

import pytest

from app import metrics


@pytest.fixture(autouse=True)
def clean_counters():
    """Each test starts from zeroed counters."""
    metrics.reset()
    yield
    metrics.reset()


# ─────────────────────────────────────────────────────────────────────────────
# ① Counter increment / labels
# ─────────────────────────────────────────────────────────────────────────────

class TestCounters:

    def test_inc_starts_at_zero_and_accumulates(self):
        assert metrics.get("shimmi_messages_received_total") == 0
        metrics.inc("shimmi_messages_received_total")
        metrics.inc("shimmi_messages_received_total")
        assert metrics.get("shimmi_messages_received_total") == 2

    def test_labels_are_tracked_as_separate_series(self):
        metrics.inc("shimmi_messages_skipped_total", reason="echo")
        metrics.inc("shimmi_messages_skipped_total", reason="echo")
        metrics.inc("shimmi_messages_skipped_total", reason="debounced")

        assert metrics.get("shimmi_messages_skipped_total", reason="echo") == 2
        assert metrics.get("shimmi_messages_skipped_total", reason="debounced") == 1
        # Unlabelled series is distinct from any labelled one.
        assert metrics.get("shimmi_messages_skipped_total") == 0

    def test_label_kwarg_order_collapses_to_one_series(self):
        metrics.inc("custom_total", a="1", b="2")
        metrics.inc("custom_total", b="2", a="1")
        assert metrics.get("custom_total", a="1", b="2") == 2

    def test_inc_accepts_a_custom_step(self):
        metrics.inc("custom_total", value=5)
        assert metrics.get("custom_total") == 5


# ─────────────────────────────────────────────────────────────────────────────
# ② + ③ Exposition format
# ─────────────────────────────────────────────────────────────────────────────

class TestRender:

    def test_declared_counter_renders_even_at_zero(self):
        # A counter that never fired must still appear — a missing series is
        # indistinguishable from "nothing went wrong" in an alert rule.
        out = metrics.render()
        assert "# TYPE shimmi_rate_limit_replies_total counter" in out
        assert "shimmi_rate_limit_replies_total 0" in out

    def test_help_and_type_emitted_once_per_family(self):
        metrics.inc("shimmi_messages_skipped_total", reason="echo")
        metrics.inc("shimmi_messages_skipped_total", reason="no_prefix")
        out = metrics.render()

        assert out.count("# TYPE shimmi_messages_skipped_total counter") == 1
        assert out.count("# HELP shimmi_messages_skipped_total") == 1
        assert 'shimmi_messages_skipped_total{reason="echo"} 1' in out
        assert 'shimmi_messages_skipped_total{reason="no_prefix"} 1' in out

    def test_output_ends_with_newline(self):
        # Prometheus rejects a body whose last line is unterminated.
        assert metrics.render().endswith("\n")

    def test_whole_numbers_render_without_decimal_point(self):
        metrics.inc("custom_total", value=3)
        assert "custom_total 3\n" in metrics.render()

    def test_fractional_values_survive(self):
        family = metrics.MetricFamily(
            "shimmi_token_budget_fraction", "help", "gauge",
            [({"provider": "groq_8b"}, 0.25)],
        )
        out = metrics.render([family])
        assert 'shimmi_token_budget_fraction{provider="groq_8b"} 0.25' in out


# ─────────────────────────────────────────────────────────────────────────────
# ④ Escaping
# ─────────────────────────────────────────────────────────────────────────────

class TestEscaping:

    def test_label_value_quotes_and_backslashes_are_escaped(self):
        metrics.inc("custom_total", reason='say "hi"\\now')
        out = metrics.render()
        # The raw quote must not terminate the label early.
        assert r'reason="say \"hi\"\\now"' in out

    def test_newlines_in_label_values_are_escaped(self):
        metrics.inc("custom_total", reason="a\nb")
        out = metrics.render()
        assert r'reason="a\nb"' in out
        # One sample per line — the embedded newline must not split the record.
        sample_lines = [ln for ln in out.splitlines() if ln.startswith("custom_total")]
        assert len(sample_lines) == 1

    def test_newlines_in_help_are_escaped(self):
        family = metrics.MetricFamily("g_metric", "line one\nline two", "gauge", [({}, 1)])
        out = metrics.render([family])
        assert "# HELP g_metric line one\\nline two" in out
        assert len([ln for ln in out.splitlines() if ln.startswith("# HELP g_metric")]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Gauges
# ─────────────────────────────────────────────────────────────────────────────

class TestGauges:

    def test_gauge_family_renders_with_gauge_type(self):
        family = metrics.MetricFamily(
            "shimmi_active_workers", "Workers alive.", "gauge", [({}, 3)],
        )
        out = metrics.render([family])
        assert "# HELP shimmi_active_workers Workers alive." in out
        assert "# TYPE shimmi_active_workers gauge" in out
        assert "shimmi_active_workers 3" in out

    def test_gauge_with_no_samples_still_declares_type(self):
        # e.g. no circuit breakers have tripped yet
        family = metrics.MetricFamily(
            "shimmi_model_circuit_tripped", "Tripped circuits.", "gauge", [],
        )
        out = metrics.render([family])
        assert "# TYPE shimmi_model_circuit_tripped gauge" in out

    def test_counters_and_gauges_coexist(self):
        metrics.inc("shimmi_messages_enqueued_total")
        family = metrics.MetricFamily("shimmi_active_workers", "Workers.", "gauge", [({}, 1)])
        out = metrics.render([family])
        assert "shimmi_messages_enqueued_total 1" in out
        assert "shimmi_active_workers 1" in out


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ Cardinality guard
# ─────────────────────────────────────────────────────────────────────────────

class TestCardinality:

    def test_no_unbounded_labels_in_declared_metrics(self):
        """
        Guards the rule documented in app/metrics.py: chat_id, sender and
        event_id are unbounded and must never become label names, or a
        Prometheus scraping this will blow up on series count.
        """
        forbidden = ("chat_id=", "sender=", "sender_key=", "event_id=", "whatsapp_id=")
        out = metrics.render()
        for token in forbidden:
            assert token not in out, f"unbounded label {token!r} leaked into scrape"
