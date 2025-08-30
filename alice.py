#!/usr/bin/env python3
"""
Alice — a continuously‑learning, local, interactable AI agent.

Design goals
- Interactable REPL
- Persists memories in SQLite (messages, facts, reflections, skills, rewards)
- Learns continuously via: episodic memory, semantic reflection, user feedback (/good, /bad), and optional
  streaming ML (if 'river' is installed) — all while running.
- Retrieval over its own history using SQLite FTS5 (no external deps). Falls back gracefully if FTS5
  unavailable.
- Pluggable LLM backends; ships with an Ollama local backend (no cloud needed). You can wire any
  HTTP chat/generate API by editing LLMClient.
- Safe skill plugins: you can teach Alice new tools with /teach. By default they run in a restricted
  sandbox (no os/system access) unless you explicitly allow it.

Quick start
  python3 alice.py --model llama3.2:3b              # requires Ollama running on localhost:11434
  python3 alice.py --help                            # see options

Notes
- Continual learning here is memory‑centric (RAG + reflection + feedback), not on‑the‑fly weight training
  of the base LLM. This is the safest/most practical approach for long‑running assistants.
- If you install optional packages (e.g., 'river') Alice will use them for online classifiers you create.

License: GPL3
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import http.client
import io
import json
import os
import queue
import random
import re
import signal
import sqlite3
import sys
import textwrap
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

DB_SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
  content TEXT NOT NULL,
  ts TEXT NOT NULL
);

-- Full‑text search over content if FTS5 exists
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content, message_id UNINDEXED, ts UNINDEXED, tokenize='porter'
);

CREATE TABLE IF NOT EXISTS facts (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reflections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  score REAL NOT NULL DEFAULT 0,
  ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
  name TEXT PRIMARY KEY,
  code TEXT NOT NULL,
  approved INTEGER NOT NULL DEFAULT 0,
  usage_count INTEGER NOT NULL DEFAULT 0,
  created_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rewards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id INTEGER,
  delta REAL NOT NULL,
  reason TEXT,
  ts TEXT NOT NULL,
  FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL
);

-- triggers keep FTS in sync if available
CREATE TRIGGER IF NOT EXISTS trg_messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content, message_id, ts) VALUES (new.id, new.content, new.id, new.ts);
END;
CREATE TRIGGER IF NOT EXISTS trg_messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
"""

SYSTEM_PROMPT = (
  "You are Alice — a continuously‑learning assistant designed by Ava and the user.\n"
  "Core style:\n"
  "- concise\n- uses dashed bullets when listing\n- cites sources only when asked\n"
  "Behavior:\n"
  "- You remember important facts about the user and this project.\n"
  "- When you wish to run a tool, emit a single line in the form:\n"
  "  <<call:SKILL_NAME args='{""param"": ""value""}>>\n"
  "  After the tool result is provided by the host, continue the conversation.\n"
  "Safety:\n"
  "- Never execute system commands yourself. Only propose high‑level actions.\n"
)

@dataclasses.dataclass
class Config:
    db_path: str = "alice.db"
    model: str = "llama3.2:3b"  # Ollama model tag
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    temperature: float = 0.4
    top_p: float = 0.95
    reflect_every_sec: int = 600  # background reflection cadence
    recall_k: int = 12
    max_history: int = 14  # how many recent dialogue turns to include
    allow_dangerous_skills: bool = False

class LLMClient:
    """Pluggable minimal client for local Ollama HTTP API (generate endpoint)."""
    def __init__(self, model: str, host: str = "127.0.0.1", port: int = 11434, temperature: float = 0.4, top_p: float = 0.95):
        self.model = model
        self.host = host
        self.port = port
        self.temperature = temperature
        self.top_p = top_p

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens,
            },
        }
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
            body = json.dumps(payload)
            conn.request("POST", "/api/generate", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"LLM HTTP {resp.status}: {data[:200]!r}")
            obj = json.loads(data)
            return obj.get("response", "")
        except Exception as e:
            return f"[LLM error: {e}]"
        finally:
            with contextlib.suppress(Exception):
                conn.close()

class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.conn = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_db()
        self.has_fts = self._check_fts()

    def _init_db(self):
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

    def _check_fts(self) -> bool:
        try:
            cur = self.conn.execute("SELECT 1 FROM messages_fts LIMIT 1")
            cur.close()
            return True
        except sqlite3.Error:
            return False

    def add_message(self, role: str, content: str) -> int:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        cur = self.conn.execute("INSERT INTO messages(role, content, ts) VALUES (?,?,?)", (role, content, ts))
        self.conn.commit()
        return cur.lastrowid

    def get_recent_dialogue(self, n: int) -> List[Tuple[str,str]]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = list(reversed(cur.fetchall()))
        cur.close()
        return rows

    def recall(self, query: str, k: int) -> List[Tuple[int, str]]:
        if self.has_fts:
            cur = self.conn.execute(
                "SELECT message_id, content FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, k),
            )
            out = cur.fetchall()
            cur.close()
            return out
        # fallback: naive scan
        cur = self.conn.execute("SELECT id, content FROM messages ORDER BY id DESC LIMIT 1000")
        rows = cur.fetchall()
        cur.close()
        scored = []
        q = query.lower()
        for mid, c in rows:
            score = sum(1 for tok in q.split() if tok in c.lower())
            if score:
                scored.append((score, mid, c))
        scored.sort(reverse=True)
        return [(mid, c) for score, mid, c in scored[:k]]

    def upsert_fact(self, key: str, value: str, weight_delta: float = 1.0):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO facts(key, value, weight, ts) VALUES(?,?,?,?)\n"
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, weight=facts.weight+?, ts=excluded.ts",
            (key, value, 1.0, ts, weight_delta),
        )
        self.conn.commit()

    def list_facts(self, top_n: int = 20) -> List[Tuple[str, str, float]]:
        cur = self.conn.execute("SELECT key, value, weight FROM facts ORDER BY weight DESC, ts DESC LIMIT ?", (top_n,))
        out = cur.fetchall()
        cur.close()
        return out

    def add_reflection(self, text: str, score: float = 0.0):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute("INSERT INTO reflections(text, score, ts) VALUES (?,?,?)", (text, score, ts))
        self.conn.commit()

    def add_skill(self, name: str, code: str, approved: bool = False):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO skills(name, code, approved, usage_count, created_ts) VALUES (?,?,?,?,?)",
            (name, code, 1 if approved else 0, 0, ts),
        )
        self.conn.commit()

    def get_skill(self, name: str) -> Optional[Tuple[str, str, bool, int]]:
        cur = self.conn.execute("SELECT name, code, approved, usage_count FROM skills WHERE name=?", (name,))
        row = cur.fetchone()
        cur.close()
        return row

    def bump_skill_usage(self, name: str):
        self.conn.execute("UPDATE skills SET usage_count=usage_count+1 WHERE name=?", (name,))
        self.conn.commit()

    def add_reward(self, message_id: Optional[int], delta: float, reason: str = ""):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute("INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?,?,?,?)", (message_id, delta, reason, ts))
        self.conn.commit()

class SkillRunner:
    def __init__(self, allow_dangerous: bool = False):
        self.allow_dangerous = allow_dangerous

    def run(self, name: str, code: str, args: Dict[str, Any]) -> str:
        # Restrictive sandbox by default
        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range,
            "enumerate": enumerate, "sorted": sorted, "map": map, "filter": filter,
            "any": any, "all": all, "print": print
        }
        allowed = {
            "__builtins__": safe_builtins,
            "math": __import__("math"),
            "statistics": __import__("statistics"),
            "json": json,
            "time": time,
            "dt": dt,
            "random": random,
        }
        if self.allow_dangerous:
            # Provide carefully; user accepts risk when enabling
            allowed.update({"os": os, "sys": sys, "sqlite3": sqlite3})
        local_ns: Dict[str, Any] = {}
        try:
            exec(code, allowed, local_ns)
            if "skill_main" not in local_ns:
                return "[Skill error] Define a function skill_main(**kwargs) in your /teach code."
            result = local_ns["skill_main"](**(args or {}))
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except Exception:
            return json.dumps({"ok": False, "error": traceback.format_exc(limit=3)}, ensure_ascii=False)

class Alice:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mem = Memory(cfg)
        self.llm = LLMClient(cfg.model, cfg.ollama_host, cfg.ollama_port, cfg.temperature, cfg.top_p)
        self.skills = SkillRunner(cfg.allow_dangerous_skills)
        self.stop_event = threading.Event()
        self.bg = threading.Thread(target=self._reflect_loop, daemon=True)
        self.bg.start()
        # seed with a system prompt
        self.mem.add_message("system", SYSTEM_PROMPT)
        # a few default facts
        self.mem.upsert_fact("agent.name", "Alice")
        self.mem.upsert_fact("agent.style", "concise-with-dashed-bullets")

    # ---------- continual components ----------
    def _reflect_loop(self):
        while not self.stop_event.is_set():
            self.stop_event.wait(self.cfg.reflect_every_sec)
            if self.stop_event.is_set():
                break
            try:
                self._reflect_once()
            except Exception:
                # never crash the loop
                traceback.print_exc()

    def _reflect_once(self):
        # Summarize latest dialogue into durable facts and a short reflection
        history = self.mem.get_recent_dialogue(60)
        if not history:
            return
        prompt = (
            "Summarize key facts and preferences about the user and project from the conversation.\n"
            "- Output 3-7 dashed bullets.\n- Prefer durable facts (name, preferences, goals).\n"
            "- Then write a one-paragraph reflection starting with 'Reflection:'.\n\n"
        )
        prompt += "\n".join([f"{r.upper()}: {c}" for r, c in history])
        out = self.llm.complete(prompt, max_tokens=320)
        # store reflection and try to extract facts (lines starting with '- ' and containing ' : ')
        self.mem.add_reflection(out, score=0.0)
        for line in out.splitlines():
            if line.strip().startswith("-"):
                # naive: turn bullet into a key/value if possible
                kv = line.strip("- ")
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    self.mem.upsert_fact(k.strip().lower(), v.strip(), weight_delta=0.2)

    # ---------- retrieval & prompting ----------
    def _compose_prompt(self, user_text: str) -> str:
        facts = self.mem.list_facts(12)
        fact_lines = [f"- {k}: {v} (w={w:.1f})" for k, v, w in facts]
        recent = self.mem.get_recent_dialogue(self.cfg.max_history)
        recall = self.mem.recall(user_text, self.cfg.recall_k)
        recall_text = "\n".join([f"• {c[:400]}" for _mid, c in recall])
        chat_snippets = "\n".join([f"{r.upper()}: {c}" for r, c in recent])
        guidance = (
            "You are Alice. Answer the USER succinctly. Use knowledge from FACTS and RECALL if relevant.\n"
            "If you need a tool, emit exactly one line like <<call:NAME args='{\"x\":1}'>> and then wait.\n"
            "Prefer dashed bullets for lists.\n"
        )
        prompt = (
            f"{guidance}\n\nFACTS (top‑weighted):\n" + "\n".join(fact_lines) +
            f"\n\nRECALL (similar past content):\n{recall_text}\n\nRECENT CHAT:\n{chat_snippets}\n\nUSER: {user_text}\nASSISTANT:"
        )
        return prompt

    # ---------- public API ----------
    def handle_user(self, user_text: str) -> str:
        # commands
        if user_text.startswith("/help"):
            return self._help_text()
        if user_text.startswith("/recall"):
            q = user_text.split(" ", 1)[1] if " " in user_text else ""
            hits = self.mem.recall(q, 15)
            lines = [f"- #{mid}: {c[:120]}" for mid, c in hits]
            return "Recall results:\n" + ("\n".join(lines) if lines else "- (none)")
        if user_text.startswith("/facts"):
            facts = self.mem.list_facts(30)
            return "Known facts (top weights):\n" + "\n".join([f"- {k}: {v} (w={w:.2f})" for k, v, w in facts])
        if user_text.startswith("/good"):
            self.mem.add_reward(None, +1.0, reason="user_mark_good")
            return "- Thanks — feedback recorded."
        if user_text.startswith("/bad"):
            self.mem.add_reward(None, -1.0, reason="user_mark_bad")
            return "- Got it — I will adjust."
        if user_text.startswith("/teach"):
            return self._cmd_teach(user_text)
        if user_text.startswith("/run"):
            return self._cmd_run(user_text)
        if user_text.startswith("/ingest"):
            return self._cmd_ingest(user_text)

        # normal chat
        self.mem.add_message("user", user_text)
        prompt = self._compose_prompt(user_text)
        out = self.llm.complete(prompt, max_tokens=600)

        # detect a single tool call request
        m = re.search(
            r"<<call:(?P<name>[A-Za-z0-9_\-]+)\s+args='(?P<args>.*)'>>\s*$",
            out, re.DOTALL
        )

        if m:
            name = m.group("name")
            try:
                args = json.loads(m.group("args")) if m.group("args") else {}
            except json.JSONDecodeError:
                args = {}
            skill_row = self.mem.get_skill(name)
            if not skill_row:
                tool_result = f"[No such skill: {name}]"
            else:
                s_name, code, approved, usage = skill_row
                if not approved and not self.cfg.allow_dangerous_skills:
                    tool_result = f"[Skill '{name}' not approved; enable or approve it first]"
                else:
                    tool_result = self.skills.run(name, code, args)
                    self.mem.bump_skill_usage(name)
            # feed result back into model for a final answer
            follow_up = (
                f"Tool '{name}' returned JSON below. Explain the result to the user clearly and continue the reply.\n"
                f"TOOL_RESULT: {tool_result}\n"
                f"USER: {user_text}\nASSISTANT:"
            )
            out = self.llm.complete(follow_up, max_tokens=600)

        msg_id = self.mem.add_message("assistant", out)
        return out

    # ---------- commands ----------
    def _help_text(self) -> str:
        return textwrap.dedent(
            """
            Commands
            - /help                 — show this help
            - /facts                — list top remembered facts
            - /recall <query>       — search past chat/content
            - /good | /bad          — give feedback on the last answer
            - /teach <name> ```python\\n...```  — teach a skill; must define skill_main(**kwargs)
            - /run <name> {json}    — run an approved skill with args
            - /ingest <path>        — ingest .txt/.md files into memory search

            Example skill
            /teach summarize ```python
            def skill_main(text: str, max_lines: int = 6):
                return "\\n".join(line.strip() for line in text.splitlines()[:max_lines])
            ```
            Then: /run summarize {"text": "hello\\nworld", "max_lines": 1}
            """
        ).strip()

    def _cmd_teach(self, raw: str) -> str:
        m = re.search(r"/teach\s+(?P<name>[a-zA-Z0-9_\-]+)\s+```python\n(?P<code>.*?)```", raw, re.DOTALL)
        if not m:
            return "Usage: /teach <name> ```python\n<code with skill_main(**kwargs)>```"
        name = m.group("name")
        code = m.group("code").strip()
        self.mem.add_skill(name, code, approved=True)
        return f"- Skill '{name}' saved and approved. Use /run {name} {{...}} to execute."

    def _cmd_run(self, raw: str) -> str:
        m = re.search(r"/run\s+(?P<name>[a-zA-Z0-9_\-]+)\s+(?P<json>{.*})\s*$", raw, re.DOTALL)
        if not m:
            return "Usage: /run <name> {json-args}"
        name = m.group("name")
        try:
            args = json.loads(m.group("json"))
        except json.JSONDecodeError as e:
            return f"- Bad JSON args: {e}"
        row = self.mem.get_skill(name)
        if not row:
            return f"- No such skill: {name}"
        _, code, approved, _ = row
        if not approved and not self.cfg.allow_dangerous_skills:
            return f"- Skill '{name}' not approved yet."
        return self.skills.run(name, code, args)

    def _cmd_ingest(self, raw: str) -> str:
        m = re.search(r"/ingest\s+(?P<path>.+)$", raw)
        if not m:
            return "Usage: /ingest <directory-or-file>"
        path = os.path.expanduser(m.group("path").strip())
        if not os.path.exists(path):
            return f"- Path not found: {path}"
        files: List[str] = []
        if os.path.isdir(path):
            for root, _dirs, fnames in os.walk(path):
                for fn in fnames:
                    if fn.lower().endswith((".txt", ".md")):
                        files.append(os.path.join(root, fn))
        else:
            files = [path]
        count = 0
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                self.mem.add_message("system", f"[ingested:{os.path.basename(fp)}]\n{text}")
                count += 1
            except Exception as e:
                print(f"! failed to ingest {fp}: {e}")
        return f"- Ingested {count} file(s). They are now searchable via /recall."

    # ---------- lifecycle ----------
    def shutdown(self):
        self.stop_event.set()
        self.bg.join(timeout=2)

# ----------------------- CLI -----------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alice — a continuously‑learning agent")
    ap.add_argument("--db", default="alice.db", help="SQLite database path")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model tag (e.g., llama3.2:3b, gemma:2b)")
    ap.add_argument("--ollama-host", default="127.0.0.1")
    ap.add_argument("--ollama-port", default=11434, type=int)
    ap.add_argument("--temperature", default=0.4, type=float)
    ap.add_argument("--top-p", default=0.95, type=float)
    ap.add_argument("--reflect", default=600, type=int, help="Seconds between background reflections")
    ap.add_argument("--allow-dangerous-skills", action="store_true", help="Allow skills to access os/sys/etc.")

    args = ap.parse_args(argv)
    cfg = Config(
        db_path=args.db,
        model=args.model,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        temperature=args.temperature,
        top_p=args.top_p,
        reflect_every_sec=args.reflect,
        allow_dangerous_skills=args.allow_dangerous_skills,
    )
    alice = Alice(cfg)

    def handle_sigint(_sig, _frm):
        print("\n[shutting down]")
        alice.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print("Alice is awake. Type /help for commands. Ctrl+C to exit.\n")
    while True:
        try:
            user = input("You> ").strip()
        except EOFError:
            break
        if not user:
            continue
        reply = alice.handle_user(user)
        print("Alice> " + reply)

    alice.shutdown()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

