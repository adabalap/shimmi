#!/usr/bin/env python3
"""Spock Bot Test Script v3.0

What v3 adds (over v2):
- Step-level expectations (reply_contains, reply_not_contains, etc.)
- Captures the *actual outbound WhatsApp message text* using WAHA API (recommended)
  or via a webhook capture listener (optional)
- Optional SQLite + Chroma DB validations with pluggable assertions
- Backward compatible with v2 scenario JSON format (phases with 'messages' / 'subsections')

Usage examples:
  python3 spock_bot_tester_v3.py --config config.json --data test_scenarios_v3.json --phases phase_1_memory_basics
  python3 spock_bot_tester_v3.py --config config.json --data test_scenarios_v3.json --quick

Notes:
- WAHA API capture mode uses GET /api/{session}/chats/overview?ids=<chatId>
  to fetch lastMessage, then correlates to the test message by timestamp and fromMe.
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path

import requests

try:
    import sqlite3
except Exception:
    sqlite3 = None

# -----------------------------
# Helpers
# -----------------------------

def now_iso():
    return datetime.now().isoformat()


def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


# -----------------------------
# WAHA outbound capture
# -----------------------------

class WahaApiCapture:
    """Capture outbound bot replies using WAHA API.

    Uses chats/overview endpoint to get lastMessage for the test chat.
    Docs mention filtering by ids and that lastMessage includes body/id/fromMe/timestamp. 
    """

    def __init__(self, base_url, api_key, session, chat_id, poll_timeout=40, poll_interval=1):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = session
        self.chat_id = chat_id
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval

    def _headers(self):
        h = {"Accept": "application/json"}
        if self.api_key:
            # WAHA uses X-Api-Key header
            h["X-Api-Key"] = self.api_key
        return h

    def fetch_last_message(self):
        # GET /api/{session}/chats/overview?ids=<chat_id>
        url = f"{self.base_url}/api/{self.session}/chats/overview"
        params = {"ids": self.chat_id, "limit": 1, "offset": 0}
        r = requests.get(url, headers=self._headers(), params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("lastMessage")
        return None

    def wait_for_outbound(self, after_timestamp, last_seen_message_id=None):
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            try:
                msg = self.fetch_last_message()
                if msg:
                    msg_id = msg.get("id")
                    ts = msg.get("timestamp")
                    from_me = msg.get("fromMe")
                    body = msg.get("body")
                    # Correlate: new message, fromMe true, timestamp >= after_timestamp
                    if msg_id and msg_id != last_seen_message_id and normalize_bool(from_me) and ts and ts >= after_timestamp:
                        return {
                            "id": msg_id,
                            "timestamp": ts,
                            "fromMe": from_me,
                            "body": body,
                            "raw": msg,
                        }
            except Exception:
                pass
            time.sleep(self.poll_interval)
        return None


# -----------------------------
# DB Validation (optional)
# -----------------------------

class SqliteValidator:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.baseline_counts = {}

    def open(self):
        if sqlite3 is None:
            raise RuntimeError("sqlite3 not available")
        if not Path(self.db_path).exists():
            raise FileNotFoundError(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def list_tables(self):
        cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [r[0] for r in cur.fetchall()]

    def table_count(self, table):
        cur = self.conn.execute(f"SELECT COUNT(*) as c FROM {table}")
        return int(cur.fetchone()[0])

    def snapshot_counts(self, tables):
        for t in tables:
            try:
                self.baseline_counts[t] = self.table_count(t)
            except Exception:
                # ignore missing tables
                pass

    def delta_rows(self, table):
        before = self.baseline_counts.get(table)
        if before is None:
            return None
        after = self.table_count(table)
        return after - before

    def run_sql(self, query):
        cur = self.conn.execute(query)
        rows = cur.fetchall()
        return rows

    def pii_scan(self, patterns, max_hits=None, tables=None):
        patterns = patterns or []
        if not patterns:
            return {"hits": 0, "details": []}
        compiled = [re.compile(p) for p in patterns]
        details = []
        hits = 0

        if tables is None:
            tables = self.list_tables()

        for t in tables:
            try:
                cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({t});").fetchall()]
                for c in cols:
                    # only scan text-like columns optimistically
                    try:
                        cur = self.conn.execute(f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT 500;")
                        for (val,) in cur.fetchall():
                            if val is None:
                                continue
                            s = str(val)
                            for rx in compiled:
                                if rx.search(s):
                                    hits += 1
                                    details.append({"table": t, "column": c, "sample": s[:120]})
                                    if max_hits is not None and hits >= max_hits:
                                        return {"hits": hits, "details": details}
                    except Exception:
                        continue
            except Exception:
                continue
        return {"hits": hits, "details": details}


class ChromaValidator:
    def __init__(self, persist_dir):
        self.persist_dir = persist_dir
        self.client = None

    def open(self):
        try:
            import chromadb
        except Exception as e:
            raise RuntimeError(f"chromadb not installed: {e}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        return self

    def list_collections(self):
        cols = self.client.list_collections()
        # chromadb returns Collection objects
        return [c.name for c in cols]

    def get_collection(self, name):
        return self.client.get_collection(name)

    def count(self, name):
        return self.get_collection(name).count()

    def query(self, name, query_text, top_k=3):
        col = self.get_collection(name)
        res = col.query(query_texts=[query_text], n_results=top_k)
        return res

    def scan_patterns(self, name, patterns, max_hits=None):
        # Best-effort: fetch a small sample of docs
        col = self.get_collection(name)
        compiled = [re.compile(p) for p in (patterns or [])]
        hits = 0
        details = []
        try:
            sample = col.get(limit=200)
            docs = sample.get("documents") or []
            # documents may be list[str]
            for d in docs:
                if d is None:
                    continue
                s = str(d)
                for rx in compiled:
                    if rx.search(s):
                        hits += 1
                        details.append({"sample": s[:160]})
                        if max_hits is not None and hits >= max_hits:
                            return {"hits": hits, "details": details}
        except Exception as e:
            return {"hits": None, "error": str(e), "details": []}
        return {"hits": hits, "details": details}


# -----------------------------
# Core Tester
# -----------------------------

DEFAULT_CONFIG = {
    "webhook_url": "http://127.0.0.1:6000/webhook",
    "user_phone": "4930656034916@lid",
    "user_name": "Sarah TestUser",
    "bot_phone": "919573717667@c.us",
    "session": "default",
    "delay": 1,
    "message_delay": 3,
    "timeout": 30,
    "capture": {"mode": "none"},
    "db_validation": {"enable": False},
    "simulation": {"incoming_fromMe": False},
}


class SpockTesterV3:
    def __init__(self, config, scenarios):
        self.config = config
        self.scenarios = scenarios
        self.test_results = []
        self.message_count = 0
        self.last_outbound_id = None

        self.capture = None
        cap = config.get("capture", {}) or {}
        if cap.get("mode") == "waha_api":
            self.capture = WahaApiCapture(
                base_url=cap.get("waha_base_url", "").strip(),
                api_key=cap.get("waha_api_key", "").strip(),
                session=config.get("session", "default"),
                chat_id=cap.get("chat_id") or config.get("user_phone"),
                poll_timeout=cap.get("poll_timeout_sec", 40),
                poll_interval=cap.get("poll_interval_sec", 1),
            )

        self.sqlite_validator = None
        self.chroma_validator = None
        self.db_enabled = bool((config.get("db_validation", {}) or {}).get("enable"))
        if self.db_enabled:
            db = config.get("db_validation", {})
            if db.get("sqlite_db_path"):
                self.sqlite_validator = SqliteValidator(db.get("sqlite_db_path"))
            if db.get("chroma_persist_dir"):
                self.chroma_validator = ChromaValidator(db.get("chroma_persist_dir"))

    def generate_message_id(self):
        timestamp = int(time.time())
        unique = str(uuid.uuid4())[:8].upper()
        return f"true_{self.config['user_phone'].split('@')[0]}_{timestamp}_{unique}"

    def create_payload(self, message_body):
        timestamp = int(time.time())
        message_id = self.generate_message_id()
        incoming_fromMe = normalize_bool(self.config.get("simulation", {}).get("incoming_fromMe"), False)

        payload = {
            "id": f"evt_{uuid.uuid4().hex}",
            "session": self.config["session"],
            "event": "message.any",
            "payload": {
                "id": message_id,
                "timestamp": timestamp,
                "from": self.config["user_phone"],
                "fromMe": incoming_fromMe,
                "source": "app",
                "body": message_body,
                "hasMedia": False,
                "media": None,
                "ack": 1,
                "ackName": "SERVER",
                "location": None,
                "vCards": None,
                "replyTo": None,
                "_data": {
                    "key": {
                        "remoteJid": self.config["user_phone"],
                        "fromMe": incoming_fromMe,
                        "id": message_id,
                    },
                    "messageTimestamp": timestamp,
                    "pushName": self.config["user_name"],
                    "broadcast": False,
                    "status": 2,
                    "message": {"conversation": message_body},
                },
            },
            "timestamp": timestamp * 1000,
            "metadata": {},
            "me": {
                "id": self.config["bot_phone"],
                "pushName": "Spock Bot",
                "lid": self.config["user_phone"],
            },
            "engine": "NOWEB",
            "environment": {
                "version": "2025.11.3",
                "engine": "NOWEB",
                "tier": "CORE",
                "browser": None,
            },
        }
        return payload

    def _evaluate_expectations(self, outbound_text, expect):
        """Evaluate text expectations and return (passed:bool, details:list)."""
        expect = expect or {}
        details = []
        passed = True

        if expect.get("reply_any"):
            if not outbound_text:
                passed = False
                details.append("Expected a reply, got none")

        contains = expect.get("reply_contains") or []
        for s in contains:
            if outbound_text is None or s.lower() not in outbound_text.lower():
                passed = False
                details.append(f"Missing expected token: {s}")

        contains_any = expect.get("reply_contains_any") or []
        if contains_any:
            if outbound_text is None or not any(s.lower() in outbound_text.lower() for s in contains_any):
                passed = False
                details.append(f"Expected at least one of tokens: {contains_any}")

        not_contains = expect.get("reply_not_contains") or []
        for s in not_contains:
            if outbound_text and s.lower() in outbound_text.lower():
                passed = False
                details.append(f"Reply contains forbidden token: {s}")

        return passed, details

    def _run_db_assertions(self, expect):
        if not self.db_enabled:
            return {"enabled": False}

        out = {"enabled": True, "sqlite": [], "chroma": []}
        db_expect = (expect or {}).get("db") or {}

        # SQLite
        if self.sqlite_validator:
            try:
                if self.sqlite_validator.conn is None:
                    self.sqlite_validator.open()
                tables = self.sqlite_validator.list_tables()
                out["sqlite"].append({"id": "sqlite_tables", "tables": tables})
            except Exception as e:
                out["sqlite"].append({"id": "sqlite_error", "error": str(e)})

            for a in db_expect.get("sqlite", []) or []:
                aid = a.get("id")
                atype = a.get("type")
                try:
                    if atype == "health":
                        min_tables = a.get("min_tables", 1)
                        ok = len(self.sqlite_validator.list_tables()) >= min_tables
                        out["sqlite"].append({"id": aid, "type": atype, "pass": ok, "min_tables": min_tables})
                    elif atype == "delta_rows":
                        table = a.get("table")
                        min_inc = a.get("min_increase", 1)
                        delta = self.sqlite_validator.delta_rows(table)
                        ok = (delta is not None and delta >= min_inc)
                        out["sqlite"].append({"id": aid, "type": atype, "table": table, "delta": delta, "pass": ok})
                    elif atype == "sql":
                        query = a.get("query")
                        rows = self.sqlite_validator.run_sql(query)
                        exp = a.get("expect", {})
                        ok = True
                        if "min_rows" in exp:
                            ok = ok and (len(rows) >= exp["min_rows"])
                        if "max_rows" in exp:
                            ok = ok and (len(rows) <= exp["max_rows"])
                        out["sqlite"].append({"id": aid, "type": atype, "rows": len(rows), "pass": ok, "query": query})
                    elif atype == "pii_scan":
                        patterns = a.get("patterns") or []
                        max_hits = a.get("max_hits")
                        scan = self.sqlite_validator.pii_scan(patterns, max_hits=max_hits)
                        # If max_hits specified, pass if hits <= max_hits
                        ok = True
                        if max_hits is not None and scan.get("hits") is not None:
                            ok = scan["hits"] <= max_hits
                        out["sqlite"].append({"id": aid, "type": atype, "pass": ok, **scan})
                    else:
                        out["sqlite"].append({"id": aid, "type": atype, "error": "Unknown assertion type"})
                except Exception as e:
                    out["sqlite"].append({"id": aid, "type": atype, "error": str(e)})

        # Chroma
        if self.chroma_validator:
            try:
                if self.chroma_validator.client is None:
                    self.chroma_validator.open()
                cols = self.chroma_validator.list_collections()
                out["chroma"].append({"id": "chroma_collections", "collections": cols})
            except Exception as e:
                out["chroma"].append({"id": "chroma_error", "error": str(e)})

            for a in db_expect.get("chroma", []) or []:
                aid = a.get("id")
                atype = a.get("type")
                try:
                    if atype == "health":
                        min_cols = a.get("min_collections", 1)
                        ok = len(self.chroma_validator.list_collections()) >= min_cols
                        out["chroma"].append({"id": aid, "type": atype, "pass": ok, "min_collections": min_cols})
                    elif atype == "delta_count":
                        # For delta_count we require baseline; implement only if snapshot exists
                        col = a.get("collection")
                        min_inc = a.get("min_increase", 1)
                        # baseline stored on object
                        baseline = getattr(self, "_chroma_baseline", {}).get(col)
                        current = self.chroma_validator.count(col)
                        delta = None if baseline is None else current - baseline
                        ok = (delta is not None and delta >= min_inc)
                        out["chroma"].append({"id": aid, "type": atype, "collection": col, "delta": delta, "current": current, "pass": ok})
                    elif atype == "query":
                        col = a.get("collection")
                        q = a.get("query_text")
                        top_k = a.get("top_k", 3)
                        res = self.chroma_validator.query(col, q, top_k=top_k)
                        docs = (res.get("documents") or [[]])[0]
                        expect_contains = a.get("expect_top_contains") or []
                        joined = "\n".join([str(d) for d in docs])
                        ok = all(s.lower() in joined.lower() for s in expect_contains)
                        out["chroma"].append({"id": aid, "type": atype, "pass": ok, "top_docs": docs, "query": q})
                    elif atype == "scan":
                        col = a.get("collection")
                        patterns = a.get("patterns") or []
                        max_hits = a.get("max_hits")
                        scan = self.chroma_validator.scan_patterns(col, patterns, max_hits=max_hits)
                        ok = True
                        if max_hits is not None and scan.get("hits") is not None:
                            ok = scan["hits"] <= max_hits
                        out["chroma"].append({"id": aid, "type": atype, "pass": ok, **scan})
                    else:
                        out["chroma"].append({"id": aid, "type": atype, "error": "Unknown assertion type"})
                except Exception as e:
                    out["chroma"].append({"id": aid, "type": atype, "error": str(e)})

        return out

    def _snapshot_db(self):
        if not self.db_enabled:
            return
        # sqlite baseline
        if self.sqlite_validator:
            try:
                if self.sqlite_validator.conn is None:
                    self.sqlite_validator.open()
                tables = self.config.get("db_validation", {}).get("sqlite_tables_hint")
                if not tables:
                    tables = self.sqlite_validator.list_tables()
                self.sqlite_validator.snapshot_counts(tables)
            except Exception:
                pass

        # chroma baseline
        if self.chroma_validator:
            try:
                if self.chroma_validator.client is None:
                    self.chroma_validator.open()
                hints = self.config.get("db_validation", {}).get("chroma_collections_hint") or []
                baseline = {}
                for c in hints:
                    try:
                        baseline[c] = self.chroma_validator.count(c)
                    except Exception:
                        continue
                self._chroma_baseline = baseline
            except Exception:
                self._chroma_baseline = {}

    def send_step(self, step, phase_name=""):
        """Send one step (message) and capture response + validate expectations."""
        # Handle control steps (e.g., switch user)
        if step.get("control", {}).get("set_config"):
            newc = step["control"]["set_config"]
            self.config.update(newc)
            # Update capture chat_id too if present
            if self.capture:
                if newc.get("session"):
                    self.capture.session = newc["session"]
                if newc.get("user_phone"):
                    # chat id may differ from lid; allow explicit in config.capture.chat_id
                    self.capture.chat_id = (self.config.get("capture", {}) or {}).get("chat_id") or newc["user_phone"]
            return {
                "type": "control",
                "phase": phase_name,
                "step_id": step.get("id"),
                "timestamp": now_iso(),
                "applied": newc,
            }

        message = step.get("user_message")
        expect = step.get("expect") or {}

        self.message_count += 1
        payload = self.create_payload(message)
        inbound_ts = payload["payload"]["timestamp"]

        # snapshot db counts before message (for delta assertions)
        self._snapshot_db()

        # send to bot webhook
        try:
            r = requests.post(
                self.config["webhook_url"],
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.get("timeout", 30),
            )
            status_code = r.status_code
            raw_response = r.text
        except Exception as e:
            status_code = 0
            raw_response = str(e)

        # capture outbound reply (actual WhatsApp message)
        outbound = None
        if self.capture:
            # allow the bot to send its message
            time.sleep(max(0.5, float(self.config.get("delay", 0))))
            outbound = self.capture.wait_for_outbound(after_timestamp=inbound_ts, last_seen_message_id=self.last_outbound_id)
            if outbound and outbound.get("id"):
                self.last_outbound_id = outbound["id"]

        outbound_text = outbound.get("body") if outbound else None
        pass_text, text_details = self._evaluate_expectations(outbound_text, expect)

        db_report = self._run_db_assertions(expect)
        # Determine overall pass: webhook success + expectation pass + db pass where available
        overall = (status_code == 200) and pass_text
        # incorporate db assertion passes if enabled
        if db_report.get("enabled"):
            for grp in (db_report.get("sqlite") or []) + (db_report.get("chroma") or []):
                if isinstance(grp, dict) and grp.get("pass") is False:
                    overall = False

        # inter-message delay (rate limiting)
        total_delay = float(self.config.get("delay", 0)) + float(self.config.get("message_delay", 0))
        if total_delay > 0:
            time.sleep(total_delay)

        return {
            "type": "step",
            "message_num": self.message_count,
            "phase": phase_name,
            "step_id": step.get("id"),
            "full_message": message,
            "inbound": {
                "id": payload["payload"]["id"],
                "timestamp": inbound_ts,
            },
            "webhook": {
                "status_code": status_code,
                "response": raw_response[:500] if raw_response else "",
                "full_response": raw_response,
            },
            "outbound": outbound,
            "expectation": {
                "pass": pass_text,
                "details": text_details,
                "expect": expect,
            },
            "db": db_report,
            "overall_pass": overall,
            "timestamp": now_iso(),
        }

    # -------------------------
    # Phase runners
    # -------------------------

    def run_phase(self, key):
        phase = self.scenarios.get(key)
        if not phase:
            raise KeyError(f"Phase not found: {key}")

        phase_name = phase.get("name") or key

        # v3 phases with steps
        if "steps" in phase:
            for step in phase["steps"]:
                res = self.send_step(step, phase_name=phase_name)
                self.test_results.append(res)
            return

        # v2 phases with subsections
        if "subsections" in phase:
            for subk, subsection in phase["subsections"].items():
                subname = subsection.get("name", subk)
                for msg in subsection.get("messages", []):
                    res = self.send_step({"id": f"{subk}_{uuid.uuid4().hex[:6]}", "user_message": msg, "expect": {"reply_any": True}}, phase_name=f"{phase_name} - {subname}")
                    self.test_results.append(res)
            return

        # v2 phases with messages list
        if "messages" in phase:
            for msg in phase["messages"]:
                res = self.send_step({"id": uuid.uuid4().hex[:8], "user_message": msg, "expect": {"reply_any": True}}, phase_name=phase_name)
                self.test_results.append(res)
            return

        raise ValueError(f"Unsupported phase structure: {key}")

    def run_phases(self, keys):
        for k in keys:
            self.run_phase(k)

    def list_phases(self):
        out = []
        for k, v in self.scenarios.items():
            if k == "meta":
                continue
            name = v.get("name", k)
            if "steps" in v:
                count = len(v["steps"])
            elif "subsections" in v:
                count = sum(len(s.get("messages", [])) for s in v["subsections"].values())
            else:
                count = len(v.get("messages", []))
            out.append((k, name, count))
        return out

    def summary(self):
        steps = [r for r in self.test_results if r.get("type") == "step"]
        total = len(steps)
        passed = sum(1 for r in steps if r.get("overall_pass"))
        failed = total - passed
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": (passed/total*100.0 if total else 0.0)}

    def save_results(self, filename):
        out = {
            "timestamp": now_iso(),
            "config": self.config,
            "summary": self.summary(),
            "results": self.test_results,
        }
        Path(filename).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return filename


def load_config(path):
    cfg = DEFAULT_CONFIG.copy()
    if path and Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    # nested merges
    for k in ("capture", "db_validation", "simulation"):
        merged = DEFAULT_CONFIG.get(k, {}).copy()
        merged.update(cfg.get(k, {}) or {})
        cfg[k] = merged
    return cfg


def load_scenarios(path):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description="Spock Bot Test Suite v3.0")
    p.add_argument('--config', default=None, help='Config file (JSON)')
    p.add_argument('--data', default='test_scenarios_v3.json', help='Scenario file (JSON)')
    p.add_argument('--phases', nargs='+', help='Phases to run')
    p.add_argument('--quick', action='store_true', help='Run quick (phase_0_health + phase_1_memory_basics)')
    p.add_argument('--list', action='store_true', help='List phases and exit')
    p.add_argument('--output', default='test_results_v3.json', help='Output results JSON')

    args = p.parse_args()

    cfg = load_config(args.config)
    scenarios = load_scenarios(args.data)
    tester = SpockTesterV3(cfg, scenarios)

    if args.list:
        print("AVAILABLE PHASES")
        for k, name, count in tester.list_phases():
            print(f"- {k}: {name} ({count} steps)")
        return

    if args.quick:
        run = ["phase_0_health", "phase_1_memory_basics"]
    elif args.phases:
        run = args.phases
    else:
        # default: run only v3 phases (skip legacy_*)
        run = [k for k in scenarios.keys() if k not in ("meta",) and not k.startswith("legacy_")]

    print(f"Running phases: {', '.join(run)}")
    tester.run_phases(run)

    s = tester.summary()
    print("\nSUMMARY")
    print(json.dumps(s, indent=2))

    out = tester.save_results(args.output)
    print(f"\nSaved results: {out}")


if __name__ == '__main__':
    main()
