"""
metrics.py — Shimmi v3.17.5

Dependency-free Prometheus metrics for the `/metrics` endpoint (roadmap item).

Deliberately stdlib-only: `prometheus_client` would be a fine choice, but this
project already carries a heavy install (chromadb + sentence-transformers →
PyTorch), and the exposition format needed here is a few dozen lines. Nothing
in this module imports from the rest of the app, so it stays trivially
testable in isolation.

Two kinds of metric:

  Counters — monotonic, incremented at the call site via `inc()`. State lives
             in this module for the lifetime of the process.

  Gauges   — point-in-time views of live state (worker count, queue depth,
             circuit breakers, token budget). These are NOT stored here; the
             caller collects them at scrape time and passes them to `render()`
             as `MetricFamily` values. That keeps this module free of
             back-references into app state.

Cardinality rule: never label a metric with a chat_id, sender, or event_id.
Those are unbounded and will blow up any Prometheus that scrapes this. Model
and provider names are bounded by config, so they are safe.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterable, List, NamedTuple, Sequence, Tuple

# Prometheus text exposition format, as served in the Content-Type header.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# label set → value, keyed by metric name. Labels are stored as a sorted tuple
# of (key, value) pairs so the same labels in a different kwarg order collapse
# onto one series.
_LabelKey = Tuple[Tuple[str, str], ...]
_COUNTERS: Dict[str, Dict[_LabelKey, float]] = {}
_HELP: Dict[str, str] = {}
# Label sets seeded by declare(), so reset() can restore the freshly-declared
# state rather than dropping the zero series a scrape is expected to show.
_SEEDED: Dict[str, List[_LabelKey]] = {}

# Counters are incremented from the asyncio loop, but also from executor
# threads (database callbacks) — cheap lock keeps increments from racing.
_LOCK = threading.Lock()


class MetricFamily(NamedTuple):
    """One metric family and all of its samples, ready to render."""
    name:    str
    help:    str
    type:    str                                    # "counter" | "gauge"
    samples: Sequence[Tuple[Dict[str, str], float]]  # (labels, value)


# ─────────────────────────────────────────────────────────────────────────────
# Counter API
# ─────────────────────────────────────────────────────────────────────────────

def declare(
    name: str,
    help_text: str,
    initial_labels: Sequence[Dict[str, str]] = ({},),
) -> None:
    """
    Register a counter and seed its series at zero.

    Declaring up front matters: a counter that has never been incremented would
    otherwise be missing from the scrape entirely, and "no samples" is
    indistinguishable from "nothing went wrong" in a dashboard or alert rule.

    `initial_labels` is the label sets to seed. It defaults to a single
    unlabelled series; for a labelled counter pass every value the app can
    emit, so a dashboard shows a real zero from process start rather than a
    gap until the first occurrence. Pass `()` to seed nothing when the label
    values aren't enumerable up front.
    """
    with _LOCK:
        _HELP[name] = help_text
        series = _COUNTERS.setdefault(name, {})
        seeded = _SEEDED.setdefault(name, [])
        for labels in initial_labels:
            key: _LabelKey = tuple(sorted((k, str(v)) for k, v in labels.items()))
            series.setdefault(key, 0.0)
            if key not in seeded:
                seeded.append(key)


def inc(name: str, value: float = 1.0, **labels: str) -> None:
    """Increment a counter. Unknown names are created on the fly."""
    key: _LabelKey = tuple(sorted((k, str(v)) for k, v in labels.items()))
    with _LOCK:
        series = _COUNTERS.setdefault(name, {})
        series[key] = series.get(key, 0.0) + value


def get(name: str, **labels: str) -> float:
    """Read a single counter value. Intended for tests."""
    key: _LabelKey = tuple(sorted((k, str(v)) for k, v in labels.items()))
    with _LOCK:
        return _COUNTERS.get(name, {}).get(key, 0.0)


def reset() -> None:
    """
    Return every counter to its freshly-declared state — seeded zero series
    are restored, everything observed at runtime is dropped. Intended for tests.
    """
    with _LOCK:
        for name in _COUNTERS:
            _COUNTERS[name] = {key: 0.0 for key in _SEEDED.get(name, ())}


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — Prometheus text exposition format v0.0.4
# ─────────────────────────────────────────────────────────────────────────────

def _escape_help(text: str) -> str:
    return text.replace("\\", r"\\").replace("\n", r"\n")


def _escape_label_value(value: str) -> str:
    return (
        value.replace("\\", r"\\")
        .replace('"', r"\"")
        .replace("\n", r"\n")
    )


def _format_value(value: float) -> str:
    # Render whole numbers without a trailing ".0" for readability; Prometheus
    # parses both forms identically.
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _render_sample(name: str, labels: Iterable[Tuple[str, str]], value: float) -> str:
    label_pairs = [
        f'{k}="{_escape_label_value(v)}"' for k, v in labels
    ]
    label_part = "{" + ",".join(label_pairs) + "}" if label_pairs else ""
    return f"{name}{label_part} {_format_value(value)}"


def _render_family(family: MetricFamily) -> List[str]:
    lines = [
        f"# HELP {family.name} {_escape_help(family.help)}",
        f"# TYPE {family.name} {family.type}",
    ]
    for labels, value in family.samples:
        lines.append(
            _render_sample(family.name, sorted(labels.items()), value)
        )
    return lines


def render(gauges: Sequence[MetricFamily] = ()) -> str:
    """
    Render all counters plus the supplied gauge families as one scrape body.

    `gauges` are collected by the caller at scrape time — see
    `app.main._collect_gauges()`.
    """
    lines: List[str] = []

    with _LOCK:
        counter_families = [
            MetricFamily(
                name=name,
                help=_HELP.get(name, name),
                type="counter",
                samples=[(dict(labels), value) for labels, value in sorted(series.items())],
            )
            for name, series in sorted(_COUNTERS.items())
        ]

    for family in counter_families:
        lines.extend(_render_family(family))
    for family in gauges:
        lines.extend(_render_family(family))

    # Exposition format requires a trailing newline.
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Declarations — every counter the app increments, so each renders at zero
# ─────────────────────────────────────────────────────────────────────────────

def _label_values(label: str, *values: str) -> List[Dict[str, str]]:
    return [{label: v} for v in values]


declare("shimmi_webhook_auth_failures_total",
        "Webhook requests rejected for a bad or missing HMAC signature.")
declare("shimmi_webhook_invalid_payload_total",
        "Webhook requests rejected because the body was not valid JSON.")
declare("shimmi_messages_received_total",
        "Inbound messages accepted from the webhook before filtering.")
declare("shimmi_messages_skipped_total",
        "Inbound messages dropped by a filter, by reason.",
        _label_values("reason",
                      "allowlist", "empty", "echo", "duplicate",
                      "from_me", "no_prefix", "debounced"))
declare("shimmi_messages_enqueued_total",
        "Messages handed to a per-chat worker queue.")
declare("shimmi_messages_dropped_total",
        "Messages dropped after passing filters, by reason (e.g. queue timeout).",
        _label_values("reason", "queue_timeout"))
declare("shimmi_messages_processed_total",
        "Messages fully processed by a worker, by outcome.",
        _label_values("outcome", "ok", "error"))
declare("shimmi_replies_sent_total",
        "Reply messages successfully handed to WAHA.")
declare("shimmi_rate_limit_replies_total",
        "Times the user got the 'at capacity' reply because all providers were rate-limited.")
declare("shimmi_memory_facts_total",
        "User fact writes, by operation (created/updated/unchanged/deleted).",
        _label_values("op", "created", "updated", "unchanged", "deleted"))
declare("shimmi_reminders_total",
        "Reminder deliveries attempted by the scheduler, by outcome.",
        _label_values("outcome", "sent", "retry", "failed", "stale", "bad_trigger"))
