#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---
# Alice p0.1.5 — NOAA weather, web access toggle, package install, pentest (detection-only), performance tweaks
# New in p0.1.5:
# - Pentest mode, allowlist, and detection-only built-in tools (nmap/nikto/zap/nuclei/msf_check/wifi)
# - /pentest controls and 'doctor' diagnostics
# - Safety first: exploitation disabled; Metasploit restricted to 'check' only
# - NOAA: /weather <lat,lon> [hourly] via api.weather.gov
# - All features remain model-agnostic (Ollama by default)
# ---

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import http.client
import json
import math
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
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional deps
try:
    import requests
except Exception:
    requests = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# ----------------------- DB SCHEMA -----------------------
DB_SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
  content TEXT NOT NULL,
  ts TEXT NOT NULL
);

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

CREATE TABLE IF NOT EXISTS embeddings (
  message_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec_json TEXT NOT NULL,
  ts TEXT NOT NULL,
  FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE TRIGGER IF NOT EXISTS trg_messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content, message_id, ts) VALUES (new.id, new.content, new.id, new.ts);
END;
CREATE TRIGGER IF NOT EXISTS trg_messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
"""

# ----------------------- PROMPTS -----------------------
BASE_PROMPT = (
    "You are Alice — a continuously-learning AI assistant.\n"
    "Style:\n"
    "- concise; prefer dashed bullets for lists\n"
    "- cite sources only if explicitly asked\n"
    "Memory:\n"
    "- you remember important facts and files the user ingests\n"
)
TOOLS_PROMPT = (
    "Tools:\n"
    "- Built-ins: calc(expr), recall(query), set_fact(key,value), get_facts()\n"
    "- browse(url): fetch content from a webpage (only if web access is enabled)\n"
    "- noaa_weather(lat, lon, hourly=False): get National Weather Service forecast for coordinates\n"
    "- Pentest (detection-only by default; target must be authorized):\n"
    "- nmap_scan(target, options) — enumerate ports/services (requires nmap)\n"
    "- nikto_scan(url) — web server checks (requires nikto)\n"
    "- zap_scan(url) — OWASP ZAP active scan via API (ZAP must be running)\n"
    "- nuclei_scan(target) — run nuclei templates against target (requires nuclei)\n"
    "- msf_check(module, rhost, rport, opts) — run Metasploit 'check' only (RPC)\n"
    "- wifi_scan(interface=None) — list nearby Wi‑Fi networks (nmcli or airodump-ng)\n"
    "- User-taught skills may exist; only call tools you are told exist.\n"
    "To use a tool, emit exactly one line:\n"
    "<<call:NAME args='{\"param\": \"value\"}'>>\n"
    "Host will reply with TOOL_RESULT; then continue.\n"
)
SAFETY_PROMPT = (
    "Safety:\n"
    "- never execute OS commands; use only provided tools\n"
    "- do not claim to have run code unless host confirms\n"
)
THOUGHT_RE = re.compile(r"^\s*(Thought|Chain|Internal):", re.IGNORECASE)

# ----------------------- CONFIG -----------------------
@dataclasses.dataclass
class Config:
    db_path: str = "alice.db"
    model: str = "gpt-oss:20b"
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    temperature: float = 0.4
    top_p: float = 0.95
    top_k: int = 40
    num_ctx: int = 4096
    reflect_every_sec: int = 600
    recall_k: int = 12
    max_history: int = 14
    allow_dangerous_skills: bool = False
    pentest_enabled: bool = False
    msfrpc_host: str = "127.0.0.1"
    msfrpc_port: int = 55552
    msfrpc_user: str = "msf"
    msfrpc_pass: str = ""
    zap_api: str = "http://127.0.0.1:8080"
    zap_api_key: str = ""
    nmap_binary: str = "nmap"
    nikto_binary: str = "nikto"
    nuclei_binary: str = "nuclei"
    sqlmap_binary: str = "sqlmap"
    nmcli_binary: str = "nmcli"
    airodump_binary: str = "airodump-ng"
    embed_enable: bool = True
    embed_model: str = "nomic-embed-text"
    train_mode: bool = False
    train_interval_sec: int = 3600
    train_dir: str = "train"
    darkvision_interval_sec: int = 900
    autonomous_tools: bool = False
    web_access: bool = False
    contact_email: str = ""

# ----------------------- OLLAMA CLIENT -----------------------
class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _http(self, timeout: int = 180) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.cfg.ollama_host, self.cfg.ollama_port, timeout=timeout)

    def _request(self, method: str, path: str, payload: Optional[dict] = None, timeout: int = 180) -> Tuple[int, bytes]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn = self._http(timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, data
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def complete(self, prompt: str, *, system: Optional[str] = None, max_tokens: int = 700) -> str:
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "top_k": self.cfg.top_k,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        delays = [0.0, 0.6, 1.2]
        last_err = None
        for d in delays:
            if d:
                time.sleep(d)
            try:
                status, data = self._request("POST", "/api/generate", payload, timeout=240)
                if status >= 400:
                    last_err = f"HTTP {status}: {data[:200]!r}"
                    continue
                obj = json.loads(data)
                return obj.get("response", "")
            except Exception as e:
                last_err = str(e)
                continue
        return f"[LLM error: {last_err}]"

    def embed(self, text: str) -> Optional[List[float]]:
        if not text.strip():
            return None
        try:
            status, data = self._request("POST", "/api/embeddings", {"model": self.cfg.embed_model, "prompt": text}, timeout=120)
            if status >= 400:
                return None
            obj = json.loads(data)
            emb = obj.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                return [float(x) for x in emb]
            return None
        except Exception:
            return None

    def list_models(self) -> List[str]:
        try:
            conn = self._http(timeout=30)
            conn.request("GET", "/api/tags")
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
            if resp.status >= 400:
                return []
            obj = json.loads(data)
            names = []
            for m in obj.get("models", []):
                tag = m.get("model") or m.get("name")
                if tag:
                    names.append(tag)
            return sorted(set(names))
        except Exception:
            return []

    def show_model(self, name: Optional[str] = None) -> Dict[str, Any]:
        try:
            status, data = self._request("POST", "/api/show", {"name": name or self.cfg.model}, timeout=30)
            if status >= 400:
                return {}
            return json.loads(data)
        except Exception:
            return {}

# ----------------------- MEMORY -----------------------
class Memory:
    def __init__(self, cfg: Config, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm
        self.conn = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_db()
        self.has_fts = self._check_fts()
        self.index_q: "queue.Queue[Tuple[int,str]]" = queue.Queue()
        self.indexer = threading.Thread(target=self._index_loop, daemon=True)
        self.indexer.start()

    def _init_db(self):
        self.conn.executescript(DB_SCHEMA)
        self.conn.commit()

    def _check_fts(self) -> bool:
        try:
            self.conn.execute("SELECT 1 FROM messages_fts LIMIT 1").close()
            return True
        except sqlite3.Error:
            return False

    def add_message(self, role: str, content: str, index: bool = True) -> int:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        cur = self.conn.execute("INSERT INTO messages(role, content, ts) VALUES (?,?,?)", (role, content, ts))
        self.conn.commit()
        mid = cur.lastrowid
        if index and self.cfg.embed_enable:
            with contextlib.suppress(Exception):
                self.index_q.put((mid, content))
        return mid

    def get_recent_dialogue(self, n_pairs: int) -> List[Tuple[str, str]]:
        cur = self.conn.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n_pairs,))
        rows = list(reversed(cur.fetchall()))
        cur.close()
        return rows

    def recall(self, query: str, k: int) -> List[Tuple[int, str]]:
        got: Dict[int, Tuple[float, str]] = {}
        # Semantic vector search (brute-force)
        if self.cfg.embed_enable:
            rows = self.conn.execute("SELECT message_id, vec_json FROM embeddings").fetchall()
            if rows:
                qvec = self.llm.embed(query) or []
                if qvec:
                    def cos(a: List[float], b: List[float]) -> float:
                        dot = na = nb = 0.0
                        for x, y in zip(a, b):
                            dot += x*y; na += x*x; nb += y*y
                        return dot / math.sqrt(na*nb) if na and nb else 0.0
                    for mid, vec_json in rows:
                        try:
                            v = json.loads(vec_json); s = cos(qvec, v)
                        except Exception:
                            s = 0.0
                        if s > 0:
                            row = self.conn.execute("SELECT content FROM messages WHERE id=?", (mid,)).fetchone()
                            if row:
                                got[int(mid)] = (max(s, got.get(int(mid), (0.0, ""))[0]), row[0])
        # Full-text search
        if self.has_fts:
            try:
                tokens = re.findall(r"[A-Za-z0-9_]{2,}", query)
                if tokens:
                    fts_q = " OR ".join(tokens)
                    for mid, content in self.conn.execute(
                        "SELECT message_id, content FROM messages_fts WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?",
                        (fts_q, k),
                    ):
                        got.setdefault(int(mid), (0.001, content))
            except sqlite3.OperationalError:
                pass
        # Fallback naive
        if not got:
            rows = self.conn.execute("SELECT id, content FROM messages ORDER BY id DESC LIMIT 1000").fetchall()
            ql = query.lower()
            for mid, content in rows:
                score = sum(1 for tok in ql.split() if tok in content.lower())
                if score:
                    got[int(mid)] = (float(score), content)
        ranked = sorted(got.items(), key=lambda kv: (-kv[1][0], -kv[0]))
        return [(mid, content) for mid, (_s, content) in ranked[:k]]

    def list_facts(self, n: int) -> List[Tuple[str, str, float]]:
        cur = self.conn.execute("SELECT key, value, weight FROM facts ORDER BY weight DESC LIMIT ?", (n,))
        out = cur.fetchall()
        cur.close()
        return out

    def upsert_fact(self, key: str, value: str, weight_delta: float = 0.0):
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        cur = self.conn.execute("SELECT key, weight FROM facts WHERE key=?", (key,))
        row = cur.fetchone()
        cur.close()
        if row:
            new_w = float(row[1]) + weight_delta
            self.conn.execute("UPDATE facts SET value=?, weight=?, ts=? WHERE key=?", (value, new_w, now, key))
        else:
            self.conn.execute("INSERT OR REPLACE INTO facts(key, value, weight, ts) VALUES (?,?,?,?)",
                              (key, value, 1.0 + weight_delta, now))
        self.conn.commit()

    def add_reflection(self, text: str, score: float):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute("INSERT INTO reflections(text, score, ts) VALUES (?,?,?)", (text, score, ts))
        self.conn.commit()

    def get_skill(self, name: str) -> Optional[Tuple[str, str, int, int]]:
        row = self.conn.execute("SELECT name, code, approved, usage_count FROM skills WHERE name=?", (name,)).fetchone()
        return row

    def add_skill(self, name: str, code: str, approved: bool = False):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO skills(name, code, approved, usage_count, created_ts) VALUES (?,?,?,?,?)",
            (name, code, 1 if approved else 0, 0, ts),
        )
        self.conn.commit()

    def increment_skill_usage(self, name: str):
        self.conn.execute("UPDATE skills SET usage_count=usage_count+1 WHERE name=?", (name,))
        self.conn.commit()

    def add_reward(self, message_id: Optional[int], delta: float, reason: str):
        if message_id is None:
            return
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute("INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?,?,?,?)",
                          (message_id, delta, reason, ts))
        self.conn.commit()

    def shutdown(self):
        with contextlib.suppress(Exception):
            self.index_q.put((None, ""))
        with contextlib.suppress(Exception):
            self.indexer.join(timeout=2)
        with contextlib.suppress(Exception):
            self.conn.close()

    def _index_loop(self):
        while True:
            try:
                message_id, content = self.index_q.get()
                if message_id is None:
                    break
                if not self.cfg.embed_enable:
                    continue
                vec = self.llm.embed(content)
                if not vec:
                    continue
                ts = dt.datetime.now(dt.timezone.utc).isoformat()
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings(message_id, model, dim, vec_json, ts) VALUES (?,?,?,?,?)",
                    (message_id, self.cfg.embed_model, len(vec), json.dumps(vec), ts),
                )
                self.conn.commit()
            except Exception:
                traceback.print_exc()

# ----------------------- SKILLS SANDBOX -----------------------
SAFE_MODULES = {"math": math, "json": json, "re": re, "random": random, "datetime": dt}
def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in SAFE_MODULES:
        return SAFE_MODULES[name]
    raise ImportError(f"Import of '{name}' is not allowed")

class SkillRunner:
    def __init__(self, allow_dangerous: bool = False):
        self.allow_dangerous = allow_dangerous

    def run(self, name: str, code: str, args: Dict[str, Any]) -> str:
        try:
            if not self.allow_dangerous:
                if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
                    return json.dumps({"ok": False, "error": "Skill blocked by safety filter"}, ensure_ascii=False)
                safe_builtins = {
                    "__import__": _safe_import,
                    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
                    "range": range, "enumerate": enumerate, "sorted": sorted,
                    "map": map, "filter": filter, "any": any, "all": all, "print": print
                }
                env = {"__builtins__": safe_builtins, **SAFE_MODULES}
            else:
                env = {"__builtins__": __builtins__, **SAFE_MODULES}
            local_ns: Dict[str, Any] = {}
            compile(code, "<skill>", "exec")
            exec(code, env, local_ns)
            if "skill_main" not in local_ns:
                return json.dumps({"ok": False, "error": "Define skill_main(**kwargs)"}, ensure_ascii=False)
            result = local_ns["skill_main"](**(args or {}))
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except Exception:
            return json.dumps({"ok": False, "error": traceback.format_exc(limit=2)}, ensure_ascii=False)

# ----------------------- TRAINER / DARKVISION -----------------------
class Trainer:
    def __init__(self, cfg: Config, mem: Memory):
        self.cfg = cfg
        self.mem = mem
        self.stop = threading.Event()
        os.makedirs(self.cfg.train_dir, exist_ok=True)
        self.ds_path = os.path.join(self.cfg.train_dir, "alice_sft.jsonl")
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.stop.set()
        if self.thread:
            with contextlib.suppress(Exception):
                self.thread.join(timeout=2)

    def _loop(self):
        while not self.stop.wait(self.cfg.train_interval_sec):
            try:
                self._snapshot_dataset()
            except Exception:
                traceback.print_exc()

    def snapshot_now(self) -> bool:
        try:
            self._snapshot_dataset()
            return True
        except Exception:
            traceback.print_exc()
            return False

    def lora_now(self) -> bool:
        # Placeholder; user can wire actual HF training externally
        return False

    def _snapshot_dataset(self):
        dialog = self.mem.get_recent_dialogue(400)
        pairs: List[Tuple[str, str]] = []
        last_user: Optional[str] = None
        for role, content in dialog:
            if role == "user":
                last_user = content
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, content))
                last_user = None
        lines = [json.dumps({"prompt": u, "response": a}, ensure_ascii=False) for u, a in pairs[-200:]]
        if not lines:
            return
        with open(self.ds_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

class Darkvision:
    def __init__(self, cfg: Config, llm: LLMClient, mem: Memory, skills: SkillRunner, trainer: Trainer):
        self.cfg = cfg
        self.llm = llm
        self.mem = mem
        self.skills = skills
        self.trainer = trainer
        self.active = False
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.log_path = os.path.join(self.cfg.train_dir, "darkvision.log")
        os.makedirs(self.cfg.train_dir, exist_ok=True)
        self.last_cycle_utc: Optional[str] = None
        self.skills_learned: int = 0
        self.last_skill_name: Optional[str] = None
        self._prev_train_mode: Optional[bool] = None

    def _log(self, msg: str):
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    def start(self):
        if self.active:
            return
        self._log("Darkvision starting")
        self.active = True
        self.stop.clear()
        self._prev_train_mode = self.cfg.train_mode
        self.cfg.train_mode = True
        self.trainer.start()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop_now(self):
        if not self.active:
            return
        self._log("Darkvision stopping")
        self.active = False
        self.stop.set()
        if self.thread:
            with contextlib.suppress(Exception):
                self.thread.join(timeout=3)
        if self._prev_train_mode is not None:
            self.cfg.train_mode = self._prev_train_mode

    def status(self) -> Dict[str, Any]:
        size = 0
        try:
            p = os.path.join(self.cfg.train_dir, "alice_sft.jsonl")
            size = os.path.getsize(p) if os.path.exists(p) else 0
        except Exception:
            size = 0
        return {
            "active": self.active,
            "last_cycle_utc": self.last_cycle_utc,
            "dataset_bytes": size,
            "skills_learned": self.skills_learned,
            "last_skill_name": self.last_skill_name,
            "log_path": self.log_path,
        }

    def _loop(self):
        while not self.stop.wait(self.cfg.darkvision_interval_sec):
            try:
                self._cycle()
            except Exception:
                traceback.print_exc()
                self._log("Cycle error; continuing")

    def _cycle(self):
        snap_ok = self.trainer.snapshot_now()
        if snap_ok:
            self._log("Snapshot: updated dataset from recent dialog")
        # propose skill (safe)
        new_skill = self._propose_skill()
        if new_skill:
            name, code = new_skill
            try:
                safe_runner = SkillRunner(allow_dangerous=False)
                approve = True
                try:
                    test_output = safe_runner.run(name, code, {})
                    result_obj = json.loads(test_output)
                    if not result_obj.get("ok"):
                        approve = False
                        self._log(f"Skill test failed: {name} — {result_obj.get('error')}")
                except Exception as e:
                    approve = False
                    self._log(f"Skill test exception: {name} — {e}")
                self.mem.add_skill(name, code, approved=approve)
                if approve:
                    self.skills_learned += 1
                    self.last_skill_name = name
                    self._log(f"Skill learned: {name}")
                else:
                    self._log(f"Skill proposed (not auto-approved): {name}")
            except Exception as e:
                self._log(f"Failed to register skill {name}: {e}")
        self.last_cycle_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _propose_skill(self) -> Optional[Tuple[str, str]]:
        history = self.mem.get_recent_dialogue(30)
        convo = "\n".join(f"{r.upper()}: {c}" for r, c in history)
        prompt = textwrap.dedent(f"""
        You are designing one tiny reusable Python skill to help with tasks seen below.
        - Use ONLY Python stdlib; no OS, no network, no file I/O, no subprocess (safe environment).
        - The skill MUST define:  def skill_main(**kwargs): ...
        - Keep code short and robust. Include a docstring explaining inputs/outputs.
        - Return ONLY a Python code block, nothing else.

        Recent conversation:
        {convo}

        Output:
        ```python
        # skill: <short_name_in_snake_case>
        <code that defines skill_main(**kwargs)>
        ```
        """).strip()
        out = self.llm.complete(prompt, system="You are a careful, truthful skill generator.", max_tokens=800)
        m = re.search(r"```python\s*(?P<code>[\s\S]+?)\s*```", out)
        if not m:
            return None
        code = m.group("code").strip()
        name_match = re.search(r"^#\s*skill:\s*([a-zA-Z0-9_]{3,40})", code, re.MULTILINE)
        name = name_match.group(1) if name_match else f"skill_{abs(hash(code)) % 10000}"
        if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
            self._log(f"Rejected unsafe skill proposal: {name}")
            return None
        if "def skill_main" not in code:
            self._log("Rejected skill proposal (no skill_main)")
            return None
        try:
            compile(code, "<proposal>", "exec")
        except Exception as e:
            self._log(f"Rejected skill (syntax error): {e}")
            return None
        return name, code

# ----------------------- ALICE CORE -----------------------
class Alice:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.mem = Memory(cfg, self.llm)
        self.skills = SkillRunner(cfg.allow_dangerous_skills)
        self.trainer = Trainer(cfg, self.mem)
        self.darkvision = Darkvision(cfg, self.llm, self.mem, self.skills, self.trainer)
        self.stop_event = threading.Event()
        self.reflect_thread = threading.Thread(target=self._reflect_loop, daemon=True)
        self.reflect_thread.start()
        self.mem.add_message("system", BASE_PROMPT + SAFETY_PROMPT, index=False)
        self.mem.upsert_fact("agent.name", "Alice")
        self.mem.upsert_fact("agent.style", "concise-with-dashed-bullets")
        self.last_assistant_id: Optional[int] = None
        self.darkmode_enabled = bool(self.cfg.autonomous_tools)

    # ---------- reflection ----------
    def _reflect_loop(self):
        while not self.stop_event.wait(self.cfg.reflect_every_sec):
            if self.stop_event.is_set():
                break
            try:
                self._reflect_once()
            except Exception:
                traceback.print_exc()

    def _reflect_once(self):
        history = self.mem.get_recent_dialogue(60)
        if not history:
            return
        prompt = (
            "Summarize any durable facts or user preferences noted. Output 3-7 bullets 'key: value', "
            "then one paragraph starting 'Reflection:'.\n\n"
            + "\n".join([f"{r.upper()}: {c}" for r, c in history])
        )
        out = self.llm.complete(prompt, system="You are a careful, truthful summarizer.", max_tokens=320)
        self.mem.add_reflection(out, score=0.0)
        for line in out.splitlines():
            if line.strip().startswith("-"):
                kv = line.strip("- ").strip()
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    key = k.strip().lower()
                    val = v.strip()
                    if key:
                        self.mem.upsert_fact(key, val, weight_delta=0.2)

    # ---------- prompting ----------
    def _compose_prompt(self, user_text: str) -> str:
        facts = self.mem.list_facts(12)
        fact_lines = [f"- {k}: {v} (w={w:.1f})" for k, v, w in facts]
        recent = self.mem.get_recent_dialogue(self.cfg.max_history)
        recall = self.mem.recall(user_text, self.cfg.recall_k)
        recall_text = "\n".join([f"• {c[:400]}" for _mid, c in recall])
        chat_snips = "\n".join([f"{r.upper()}: {c}" for r, c in recent])
        guidance = "Answer succinctly. Use FACTS and RECALL if relevant. Prefer dashed bullets for lists.\n"
        return (
            f"{guidance}\n\nFACTS (top-weighted):\n" + ("\n".join(fact_lines) if fact_lines else "(none)") +
            f"\n\nRECALL (similar topics):\n" + (recall_text or "(none)") +
            f"\n\nRECENT CHAT:\n{chat_snips}\n\nUSER: {user_text}\nASSISTANT:"
        )

    def _system_for_chat(self) -> str:
        return BASE_PROMPT + (TOOLS_PROMPT if self.darkmode_enabled else "") + SAFETY_PROMPT

    # ---------- helpers ----------
    def _noaa_headers(self) -> Dict[str, str]:
        contact = self.cfg.contact_email or os.environ.get("ALICE_CONTACT_EMAIL") or os.environ.get("ALICE_CONTACT") or "contact@example.invalid"
        return {
            "User-Agent": f"AliceWeather/0.1 (+{contact})",
            "Accept": "application/geo+json, application/json;q=0.9"
        }

    # ---------- built-in tools ----------
    def _tool_calc(self, args: Dict[str, Any]) -> Dict[str, Any]:
        expr = str(args.get("expr", "")).strip()
        if not expr:
            return {"ok": False, "error": "missing 'expr'"}
        try:
            import ast, math as _m
            class SafeEval(ast.NodeVisitor):
                allowed = (
                    ast.Expression, ast.BinOp, ast.UnaryOp,
                    ast.Num, ast.Constant,
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                    ast.FloorDiv, ast.USub, ast.UAdd,
                    ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor,
                    ast.Call, ast.Name
                )
                def visit(self, node):
                    if not isinstance(node, self.allowed):
                        raise ValueError("disallowed expression")
                    return super().visit(node)
            tree = ast.parse(expr, mode="eval")
            SafeEval().visit(tree)
            val = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {
                "pi": _m.pi, "e": _m.e, "sin": _m.sin, "cos": _m.cos, "tan": _m.tan, "sqrt": _m.sqrt
            })
            return {"ok": True, "result": val}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tool_recall(self, args: Dict[str, Any]) -> Dict[str, Any]:
        q = str(args.get("query", "")).strip()
        k = int(args.get("k", 6))
        hits = self.mem.recall(q, k)
        return {"ok": True, "results": [{"id": mid, "snippet": c[:400]} for mid, c in hits]}

    def _tool_set_fact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        k = str(args.get("key", "")).strip().lower()
        v = str(args.get("value", "")).strip()
        if not k:
            return {"ok": False, "error": "missing 'key'"}
        self.mem.upsert_fact(k, v, weight_delta=1.0)
        return {"ok": True, "set": {"key": k, "value": v}}

    def _tool_get_facts(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        facts = self.mem.list_facts(20)
        return {"ok": True, "facts": [{"key": k, "value": v, "weight": w} for k, v, w in facts]}

    def _tool_browse(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.web_access:
            return {"ok": False, "error": "web access is disabled"}
        url = str(args.get("url", "")).strip()
        if not url:
            return {"ok": False, "error": "missing 'url'"}
        if not requests:
            return {"ok": False, "error": "requests library not installed"}
        try:
            if not url.lower().startswith(("http://", "https://")):
                return {"ok": False, "error": "invalid URL scheme"}
            headers = {"User-Agent": "AliceBot/0.1"}
            resp = requests.get(url, headers=headers, timeout=10)
            text = resp.text
            if BeautifulSoup:
                soup = BeautifulSoup(text, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                page_text = soup.get_text(separator="\n")
                snippet = page_text.strip()
                if len(snippet) > 1000:
                    snippet = snippet[:1000] + "..."
            else:
                snippet = text[:1000] + ("..." if len(text) > 1000 else "")
            return {"ok": True, "content": snippet}
        except Exception as e:
            return {"ok": False, "error": f"request failed: {e}"}

    def _tool_noaa_weather(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.web_access:
            return {"ok": False, "error": "web access is disabled"}
        if not requests:
            return {"ok": False, "error": "requests library not installed"}
        try:
            lat = float(args.get("lat", ""))
            lon = float(args.get("lon", ""))
        except Exception:
            return {"ok": False, "error": "provide numeric 'lat' and 'lon' (e.g., 38.8977, -77.0365)"}
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {"ok": False, "error": "lat must be [-90,90], lon must be [-180,180]"}
        hourly = bool(args.get("hourly", False))
        try:
            p_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
            h = self._noaa_headers()
            p_resp = requests.get(p_url, headers=h, timeout=10)
            if p_resp.status_code >= 400:
                return {"ok": False, "error": f"points lookup failed ({p_resp.status_code})"}
            pdata = p_resp.json()
            props = pdata.get("properties") or {}
            rel = (props.get("relativeLocation") or {}).get("properties") or {}
            location = ", ".join(filter(None, [rel.get("city"), rel.get("state")])) or f"{lat:.4f},{lon:.4f}"
            forecast_url = props.get("forecastHourly" if hourly else "forecast")
            if not forecast_url:
                return {"ok": False, "error": "forecast URL not available for this location"}
            f_resp = requests.get(forecast_url, headers=h, timeout=10)
            if f_resp.status_code >= 400:
                return {"ok": False, "error": f"forecast fetch failed ({f_resp.status_code})"}
            fdata = f_resp.json()
            periods = (fdata.get("properties") or {}).get("periods") or []
            take = 6 if hourly else 4
            out_periods = []
            for p in periods[:take]:
                out_periods.append({
                    "name": p.get("name"),
                    "startTime": p.get("startTime"),
                    "endTime": p.get("endTime"),
                    "temperature": p.get("temperature"),
                    "unit": p.get("temperatureUnit"),
                    "wind": f"{p.get('windSpeed','')} {p.get('windDirection','')}".strip(),
                    "shortForecast": p.get("shortForecast"),
                    "detailedForecast": p.get("detailedForecast", "")[:240] + ("..." if p.get("detailedForecast") and len(p.get("detailedForecast")) > 240 else "")
                })
            return {"ok": True, "location": location, "hourly": hourly, "periods": out_periods}
        except Exception as e:
            return {"ok": False, "error": f"noaa error: {e}"}

    def _run_tool(self, name: str, args: Dict[str, Any]) -> str:
        builtins = {
            "calc": self._tool_calc,
            "recall": self._tool_recall,
            "set_fact": self._tool_set_fact,
            "get_facts": self._tool_get_facts,
            "browse": self._tool_browse,
            "noaa_weather": self._tool_noaa_weather,
            "nmap_scan": self._tool_nmap_scan,
            "nikto_scan": self._tool_nikto_scan,
            "zap_scan": self._tool_zap_scan,
            "nuclei_scan": self._tool_nuclei_scan,
            "msf_check": self._tool_msf_check,
            "wifi_scan": self._tool_wifi_scan,
        }
        if name in builtins:
            try:
                res = builtins[name](args)
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            return json.dumps(res, ensure_ascii=False)
        row = self.mem.get_skill(name)
        if not row:
            return json.dumps({"ok": False, "error": f"no such tool/skill: {name}"}, ensure_ascii=False)
        skill_name, code, approved, usage_count = row
        if not approved and not self.cfg.allow_dangerous_skills:
            return json.dumps({"ok": False, "error": f"skill '{name}' not approved"}, ensure_ascii=False)
        result_json = self.skills.run(skill_name, code, args)
        self.mem.increment_skill_usage(skill_name)
        return result_json

    # ---------- pentest helpers ----------
    def _pentest_allowed(self, target: str) -> bool:
        target = (target or "").strip().lower()
        if not target:
            return False
        cur = self.mem.conn.execute("SELECT key FROM facts WHERE key=?", (f"pentest.allow:{target}",))
        row = cur.fetchone(); cur.close()
        return bool(self.cfg.pentest_enabled and row)

    def _pentest_require(self, target: str) -> Optional[str]:
        if not self.cfg.pentest_enabled:
            return "- Pentest is disabled. Use /pentest on to enable."
        if not self._pentest_allowed(target):
            return f"- Target '{target}' not authorized. Use /pentest allow {target} to authorize."
        return None

    def _which(self, name: str) -> Optional[str]:
        return shutil.which(name)

    # ---------- built-in pentest tools ----------
    def _tool_nmap_scan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        target = str(args.get("target","")).strip()
        opts = str(args.get("options","-sV -T4")).strip()
        err = self._pentest_require(target)
        if err:
            return {"ok": False, "error": err}
        binp = self.cfg.nmap_binary or "nmap"
        if not shutil.which(binp):
            return {"ok": False, "error": "nmap not found"}
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
                xmlfile = tmp.name
            cmd = [binp] + opts.split() + ["-oX", xmlfile, target]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            txt = cp.stdout + "\n" + cp.stderr
            data = {"stdout": txt.strip(), "xml": None}
            try:
                x = open(xmlfile,"r",encoding="utf-8",errors="ignore").read()
                data["xml"] = x
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    os.remove(xmlfile)
            return {"ok": True, "result": data}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tool_nikto_scan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        url = str(args.get("url","")).strip()
        if not url:
            return {"ok": False, "error": "missing 'url'"}
        host = url.split("://")[-1].split("/")[0].lower()
        err = self._pentest_require(host)
        if err:
            return {"ok": False, "error": err}
        binp = self.cfg.nikto_binary or "nikto"
        if not shutil.which(binp):
            return {"ok": False, "error": "nikto not found"}
        try:
            import subprocess
            cmd = [binp, "-host", url, "-ask", "no", "-maxtime", "600"]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            out = cp.stdout or cp.stderr
            return {"ok": True, "result": out.strip()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tool_zap_scan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not requests:
            return {"ok": False, "error": "requests not installed"}
        base = (self.cfg.zap_api or "http://127.0.0.1:8080").rstrip("/")
        key = self.cfg.zap_api_key or os.environ.get("ZAP_API_KEY","")
        url = str(args.get("url","")).strip()
        if not url:
            return {"ok": False, "error": "missing 'url'"}
        host = url.split("://")[-1].split("/")[0].lower()
        err = self._pentest_require(host)
        if err:
            return {"ok": False, "error": err}
        try:
            # Kick off an active scan
            r = requests.get(f"{base}/JSON/ascan/action/scan/", params={"apikey": key, "url": url, "recurse": "true"}, timeout=10)
            if r.status_code >= 400:
                return {"ok": False, "error": f"ZAP scan start failed ({r.status_code})"}
            # fetch alerts summary
            a = requests.get(f"{base}/JSON/core/view/alertsSummary/", params={"apikey": key, "baseurl": url}, timeout=10)
            return {"ok": True, "result": a.json() if a.ok else {"status": "scan started"}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tool_nuclei_scan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        target = str(args.get("target","")).strip()
        err = self._pentest_require(target)
        if err:
            return {"ok": False, "error": err}
        binp = self.cfg.nuclei_binary or "nuclei"
        if not shutil.which(binp):
            return {"ok": False, "error": "nuclei not found"}
        try:
            import subprocess
            cmd = [binp, "-u", target, "-silent", "-timeout", "10"]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            return {"ok": True, "result": cp.stdout.strip() or cp.stderr.strip()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tool_msf_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Stub: require msfrpc setup externally; we only allow 'check'
        return {"ok": False, "error": "msf_check not configured in this build"}

    def _tool_wifi_scan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        binp = self.cfg.nmcli_binary or shutil.which("nmcli")
        if not binp:
            return {"ok": False, "error": "nmcli not found"}
        try:
            import subprocess
            cp = subprocess.run([binp, "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi"], capture_output=True, text=True, timeout=60)
            lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
            nets = []
            for ln in lines:
                parts = ln.split(":")
                ssid = parts[0] if parts else ""
                signal = parts[1] if len(parts)>1 else ""
                sec = ":".join(parts[2:]) if len(parts)>2 else ""
                nets.append({"ssid": ssid, "signal": signal, "security": sec})
            return {"ok": True, "networks": nets}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------- commands ----------
    def _help_text(self) -> str:
        return textwrap.dedent("""\
        Commands:
        - /help                  — show this help
        - /facts                 — list top remembered facts
        - /recall <query>        — search past chat/content
        - /good | /bad           — reward/punish last answer
        - /teach <name> ```python
          ...```                 — add a skill (must define skill_main(**kwargs))
        - /run <name> {json}     — run a built-in tool or approved skill
        - /ingest <path>         — ingest .txt/.md file(s) into memory
        - /train status|now      — snapshot dataset
        - /darkvision start|status|stop  — autonomous self-improvement loop
        - /darkmode on|off|status — enable autonomous tool use + terse style
        - /model list|info|set <tag>     — list/info/set LLM model
        - /embed on|off|model <name>     — control semantic memory/embedding model
        - /web on|off            — enable or disable internet access (web browsing)
        - /install <package>     — install a Python package (pip)
        - /weather <lat,lon> [hourly] — NOAA forecast shortcut
        - /pentest on|off|status|allow <target>|revoke <target>|list|doctor — pentest controls & diagnostics""").strip()

    def _cmd_teach(self, raw: str) -> str:
        m = re.search(r"/teach\s+(?P<name>[A-Za-z0-9_\-]+)\s+```python\n(?P<code>.*?)```", raw, re.DOTALL)
        if not m:
            return "Usage: /teach <name> ```python\n<code with skill_main(**kwargs)>```"
        name = m.group("name")
        code = m.group("code").strip()
        if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
            return "- Rejected for safety: found disallowed APIs."
        try:
            compile(code, "<user-skill>", "exec")
        except Exception as e:
            return f"- Syntax error: {e}"
        self.mem.add_skill(name, code, approved=True)
        return f"- Skill '{name}' saved and approved. Use /run {name} {{...}} to execute."

    def _cmd_run(self, raw: str) -> str:
        m = re.search(r"/run\s+(?P<name>[A-Za-z0-9_\-]+)\s+(?P<json>{.*})\s*$", raw, re.IGNORECASE | re.DOTALL)
        if not m:
            return "Usage: /run <name> {json-args}"
        name = m.group("name")
        try:
            args = json.loads(m.group("json"))
        except json.JSONDecodeError as e:
            return f"- Bad JSON args: {e}"
        if name in ("calc", "recall", "set_fact", "get_facts", "browse", "noaa_weather",
                    "nmap_scan", "nikto_scan", "zap_scan", "nuclei_scan", "msf_check", "wifi_scan"):
            return self._run_tool(name, args)
        row = self.mem.get_skill(name)
        if not row:
            return f"- No such skill: {name}"
        _skill, code, approved, _usage = row
        if not approved and not self.cfg.allow_dangerous_skills:
            return f"- Skill '{name}' not approved."
        return self.skills.run(name, code, args)

    def _cmd_ingest(self, raw: str) -> str:
        m = re.search(r"/ingest\s+(?P<path>.+)$", raw)
        if not m:
            return "Usage: /ingest <file-or-directory>"
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
        return f"- Ingested {count} file(s). Use /recall to search their content."

    def _cmd_train(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) >= 2 and parts[1] == "status":
            size = 0
            try:
                p = os.path.join(self.cfg.train_dir, "alice_sft.jsonl")
                size = os.path.getsize(p) if os.path.exists(p) else 0
            except Exception:
                size = 0
            return f"- train_mode={'on' if self.cfg.train_mode else 'off'}, dataset_bytes={size}"
        if len(parts) >= 2 and parts[1] == "now":
            snap = self.trainer.snapshot_now()
            if not snap:
                return "- Train attempt completed. (No data to train yet.)"
            return "- Train attempt completed. Dataset saved."
        return "Usage: /train status | /train now"

    def _cmd_darkvision(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /darkvision start | /darkvision status | /darkvision stop"
        action = parts[1].lower()
        if action in ("start", "on"):
            self.darkvision.start()
            return "- Darkvision: ON (self-training + safe self-skill-learning started)"
        if action in ("stop", "off"):
            self.darkvision.stop_now()
            return "- Darkvision: OFF (self-improvement halted)"
        if action == "status":
            st = self.darkvision.status()
            lines = [
                f"- darkvision={'on' if st['active'] else 'off'}",
                f"- last_cycle_utc={st['last_cycle_utc'] or '(n/a)'}",
                f"- dataset_bytes={st['dataset_bytes']}",
                f"- skills_learned={st['skills_learned']}",
                f"- last_skill={st['last_skill_name'] or '(none)'}",
                f"- log={st['log_path']}",
            ]
            return "\n".join(lines)
        return "Usage: /darkvision start | /darkvision status | /darkvision stop"

    def _cmd_darkmode(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /darkmode on | off | status"
        action = parts[1].lower()
        if action in ("on", "start"):
            self.darkmode_enabled = True
            self.cfg.autonomous_tools = True
            return "- Darkmode: ON (autonomous tool use enabled; terse style)"
        if action in ("off", "stop"):
            self.darkmode_enabled = False
            self.cfg.autonomous_tools = False
            return "- Darkmode: OFF (tools only when explicitly /run)"
        if action == "status":
            return f"- darkmode={'on' if self.darkmode_enabled else 'off'}"
        return "Usage: /darkmode on | off | status"

    def _cmd_model(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /model list | info | set <tag>"
        sub = parts[1].lower()
        if sub == "list":
            models = self.llm.list_models()
            if not models:
                return "- no models found (is ollama running and models pulled?)"
            return "- Installed models:\n" + "\n".join(f"- {m}" for m in models)
        if sub == "info":
            info = self.llm.show_model() or {}
            det = info.get("details", {})
            fam = det.get("family") or det.get("families") or "?"
            return "\n".join([
                f"- model={self.cfg.model}",
                f"- context_length={det.get('context_length','?')}",
                f"- family={fam}",
                f"- parameter_size={det.get('parameter_size','?')}",
            ])
        if sub == "set" and len(parts) >= 3:
            tag = " ".join(parts[2:]).strip()
            old = self.cfg.model
            self.cfg.model = tag
            info = self.llm.show_model(tag) or {}
            det = info.get("details", {})
            ctx_len = det.get("context_length")
            if isinstance(ctx_len, int) and ctx_len > 0:
                self.cfg.num_ctx = min(max(1024, ctx_len), 32768)
            return f"- model switched: {old} → {self.cfg.model} (num_ctx={self.cfg.num_ctx})"
        return "Usage: /model list | info | set <tag>"

    def _cmd_embed(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /embed on|off | /embed model <name>"
        sub = parts[1].lower()
        if sub == "on":
            self.cfg.embed_enable = True
            return "- embeddings: ON"
        if sub == "off":
            self.cfg.embed_enable = False
            return "- embeddings: OFF"
        if sub == "model" and len(parts) >= 3:
            name = " ".join(parts[2:]).strip()
            old = self.cfg.embed_model
            self.cfg.embed_model = name
            probe = self.llm.embed("test 123")
            ok = "ok" if probe else "not-ready"
            return f"- embed model: {old} → {name} ({ok})"
        return "Usage: /embed on|off | /embed model <name>"

    def _cmd_web(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /web on | off"
        action = parts[1].lower()
        if action in ("on", "enable", "start"):
            self.cfg.web_access = True
            return "- Web access: ENABLED (Alice can browse the internet & reach NOAA)"
        if action in ("off", "disable", "stop"):
            self.cfg.web_access = False
            return "- Web access: DISABLED"
        return "Usage: /web on | off"

    def _cmd_install(self, raw: str) -> str:
        parts = raw.strip().split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /install <package_name>"
        pkg = parts[1].strip()
        if not pkg:
            return "Usage: /install <package_name>"
        try:
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True, text=True)
            if result.returncode == 0:
                return f"- Package '{pkg}' installed successfully."
            else:
                err = result.stderr.strip() or result.stdout.strip()
                return f"- Failed to install '{pkg}': {err}"
        except Exception as e:
            return f"- Error running pip: {e}"

    def _cmd_weather(self, raw: str) -> str:
        m = re.search(r"/weather\s+(?P<lat>-?\d+(\.\d+)?)\s*,\s*(?P<lon>-?\d+(\.\d+)?)\s*(?P<hourly>hourly)?", raw, re.IGNORECASE)
        if not m:
            return "Usage: /weather <lat,lon> [hourly] (e.g., /weather 38.8977,-77.0365 hourly)"
        lat = float(m.group("lat")); lon = float(m.group("lon")); hourly = bool(m.group("hourly"))
        result = json.loads(self._run_tool("noaa_weather", {"lat": lat, "lon": lon, "hourly": hourly}))
        if not result.get("ok"):
            return f"- NOAA error: {result.get('error')}"
        loc = result.get("location", f"{lat:.4f},{lon:.4f}")
        lines = [f"- location: {loc}", f"- mode: {'hourly' if hourly else 'periods'}"]
        for p in result.get("periods", []):
            lines.append(f"- {p.get('name')}: {p.get('shortForecast')} — {p.get('temperature')}{p.get('unit','')} | wind {p.get('wind')}")
        return "\n".join(lines) if lines else "- (no forecast data)"

    # ---------- main chat ----------
    def handle_user(self, user_text: str) -> str:
        if user_text.startswith("/help"):
            return self._help_text()
        if user_text.startswith("/pentest"):
            return self._cmd_pentest(user_text)
        if user_text.startswith("/facts"):
            facts = self.mem.list_facts(30)
            return "Known facts (top weights):\n" + ("\n".join([f"- {k}: {v} (w={w:.2f})" for k, v, w in facts]) if facts else "- (none)")
        if user_text.startswith("/recall"):
            q = user_text[len("/recall"):].strip() or "*"
            hits = self.mem.recall(q, self.cfg.recall_k)
            return "\n".join([f"- {mid}: {snip[:160]}" for mid, snip in hits]) or "- (no matches)"
        if user_text.startswith("/good"):
            self.mem.add_reward(self.last_assistant_id, +1.0, reason="user_mark_good")
            return "- thanks — feedback recorded"
        if user_text.startswith("/bad"):
            self.mem.add_reward(self.last_assistant_id, -1.0, reason="user_mark_bad")
            return "- got it — I’ll adjust"
        if user_text.startswith("/teach"):
            return self._cmd_teach(user_text)
        if user_text.startswith("/run"):
            return self._cmd_run(user_text)
        if user_text.startswith("/ingest"):
            return self._cmd_ingest(user_text)
        if user_text.startswith("/train"):
            return self._cmd_train(user_text)
        if user_text.startswith("/darkvision"):
            return self._cmd_darkvision(user_text)
        if user_text.startswith("/darkmode"):
            return self._cmd_darkmode(user_text)
        if user_text.startswith("/model"):
            return self._cmd_model(user_text)
        if user_text.startswith("/embed"):
            return self._cmd_embed(user_text)
        if user_text.startswith("/web"):
            return self._cmd_web(user_text)
        if user_text.startswith("/install"):
            return self._cmd_install(user_text)
        if user_text.startswith("/weather"):
            return self._cmd_weather(user_text)

        _t = user_text.strip()
        if _t.lower().startswith('say') or _t.lower().startswith('/say'):
            parts = _t.split(' ', 1)
            if len(parts) > 1:
                echo = parts[1].strip()
                if (echo.startswith('"') and echo.endswith('"')) or (echo.startswith("'") and echo.endswith("'")):
                    echo = echo[1:-1]
                return echo

        self.mem.add_message("user", user_text)
        prompt = self._compose_prompt(user_text)
        system = self._system_for_chat()

        resp = self.llm.complete(prompt, system=system, max_tokens=700)

        if self.darkmode_enabled:
            steps = 0
            while True:
                call = self._maybe_tool_call(resp)
                if not call or steps >= 5:
                    break
                steps += 1
                name, args = call
                tool_json = self._run_tool(name, args)
                follow = (
                    f"Tool '{name}' returned JSON below. Use it to continue solving the user's request.\n"
                    f"TOOL_RESULT: {tool_json}\n"
                    f"USER: {user_text}\nASSISTANT:"
                )
                resp = self.llm.complete(follow, system=system, max_tokens=700)

        final = "\n".join(ln for ln in resp.splitlines() if not THOUGHT_RE.match(ln)).strip()
        self.last_assistant_id = self.mem.add_message("assistant", final)
        return final

    # ---------- lifecycle ----------
    def shutdown(self):
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.reflect_thread.join(timeout=2)
        self.darkvision.stop_now()
        self.trainer.shutdown()
        self.mem.shutdown()

    # ---------- tool-call parsing ----------
    def _maybe_tool_call(self, out: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.search(r"<<call:(?P<name>[A-Za-z0-9_\-]+)\s+args='(?P<args>.*)'>>\s*$", out, re.DOTALL)
        if not m:
            return None
        name = m.group("name")
        try:
            args = json.loads(m.group("args")) if m.group("args") else {}
        except json.JSONDecodeError:
            args = {}
        return name, args

    # ---------- /pentest controls ----------
    def _cmd_pentest(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /pentest on|off|status|allow <target>|revoke <target>|list|doctor"
        sub = parts[1].lower()
        if sub in ("on","enable","start"):
            self.cfg.pentest_enabled = True
            return "- Pentest: ON (detection-only tools available for allowlisted targets)"
        if sub in ("off","disable","stop"):
            self.cfg.pentest_enabled = False
            return "- Pentest: OFF"
        if sub == "status":
            # show allowlist facts
            cur = self.mem.conn.execute("SELECT key FROM facts WHERE key LIKE 'pentest.allow:%' ORDER BY key")
            allow = [k.split(":",1)[1] for (k,) in cur.fetchall()]
            return "\n".join([
                f"- pentest={'on' if self.cfg.pentest_enabled else 'off'}",
                f"- allowlist: {', '.join(allow) if allow else '(empty)'}"
            ])
        if sub == "allow" and len(parts)>=3:
            tgt = parts[2].strip().lower()
            self.mem.upsert_fact(f"pentest.allow:{tgt}", "1", weight_delta=0.5)
            return f"- allowlisted: {tgt}"
        if sub == "revoke" and len(parts)>=3:
            tgt = parts[2].strip().lower()
            with contextlib.suppress(Exception):
                self.mem.conn.execute("DELETE FROM facts WHERE key=?", (f"pentest.allow:{tgt}",)); self.mem.conn.commit()
            return f"- revoked: {tgt}"
        if sub == "list":
            cur = self.mem.conn.execute("SELECT key FROM facts WHERE key LIKE 'pentest.allow:%' ORDER BY key")
            allow = [k.split(":",1)[1] for (k,) in cur.fetchall()]
            return "- allowlist:\n" + "\n".join(f"- {a}" for a in allow) if allow else "- allowlist empty"
        if sub == "doctor":
            bins = { "nmap": self._which(self.cfg.nmap_binary), "nikto": self._which(self.cfg.nikto_binary),
                     "nuclei": self._which(self.cfg.nuclei_binary), "nmcli": self._which(self.cfg.nmcli_binary) }
            return "- diagnostics:\n" + "\n".join(f"- {k}: {'ok' if v else 'missing'} ({v or ''})" for k,v in bins.items())
        return "Usage: /pentest on|off|status|allow <target>|revoke <target>|list|doctor"

# ----------------------- CLI -----------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alice p0.1.5 — continuously-learning local AI (NOAA weather enabled)")
    ap.add_argument("--db", default="alice.db")
    ap.add_argument("--model", default="gpt-oss:20b", help="Model tag (Ollama or other backend)")
    ap.add_argument("--ollama-host", default="127.0.0.1")
    ap.add_argument("--ollama-port", default=11434, type=int)
    ap.add_argument("--temperature", default=0.4, type=float)
    ap.add_argument("--top-p", default=0.95, type=float)
    ap.add_argument("--top-k", default=40, type=int)
    ap.add_argument("--num-ctx", default=4096, type=int)
    ap.add_argument("--reflect", default=600, type=int)
    ap.add_argument("--allow-dangerous-skills", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--train-mode", action="store_true")
    ap.add_argument("--train-interval", default=3600, type=int)
    ap.add_argument("--darkvision-interval", default=900, type=int)
    ap.add_argument("--autonomous-tools", action="store_true", help="Start with Darkmode on (autonomous tool use)")
    ap.add_argument("--enable-web", action="store_true", help="Start with web browsing enabled (internet access)")
    ap.add_argument("--contact-email", default="", help="Contact email for NOAA User-Agent header")
    args = ap.parse_args(argv)

    cfg = Config(
        db_path=args.db,
        model=args.model,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_ctx=args.num_ctx,
        reflect_every_sec=args.reflect,
        allow_dangerous_skills=args.allow_dangerous_skills,
        embed_enable=(not args.no_embed),
        embed_model=args.embed_model,
        train_mode=args.train_mode,
        train_interval_sec=args.train_interval,
        darkvision_interval_sec=args.darkvision_interval,
        autonomous_tools=args.autonomous_tools,
        web_access=args.enable_web,
        contact_email=args.contact_email.strip()
    )
    alice = Alice(cfg)

    def handle_sigint(_sig, _frame):
        print("\n[shutting down]")
        alice.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print("Alice p0.1.5 is awake. Type /help for commands. (Ctrl+C to exit.)")

    # lightweight CLI spinner
    class _BarSpinner:
        """Lightweight CLI bar spinner for long-running steps."""
        def __init__(self, label: str = "working", width: int = 12, interval: float = 0.08):
            self.label = label
            self.width = max(6, int(width))
            self.interval = max(0.03, float(interval))
            import threading
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._run, daemon=True)
        def _frame(self, i: int) -> str:
            span = self.width * 2 - 2
            pos = (i % span)
            if pos >= self.width:
                pos = span - pos
            cells = [" "] * self.width
            cells[pos] = "="
            return "[" + "".join(cells) + "]"
        def _run(self):
            import sys, time
            i = 0
            prefix = f"{self.label} " if self.label else ""
            try:
                while not self._stop.is_set():
                    sys.stdout.write("\r" + prefix + self._frame(i))
                    sys.stdout.flush()
                    time.sleep(self.interval)
                    i += 1
            finally:
                sys.stdout.write("\r" + " " * (len(prefix) + self.width + 2) + "\r")
                sys.stdout.flush()
        def start(self):
            self._t.start()
        def stop(self):
            import sys
            self._stop.set()
            self._t.join(timeout=1.0)
            sys.stdout.write("\r")
            sys.stdout.flush()

    while True:
        try:
            user = input("You> ").strip()
        except EOFError:
            break
        if not user:
            continue
        _spin = _BarSpinner("thinking")
        _spin.start()
        try:
            reply = alice.handle_user(user)
        finally:
            _spin.stop()
        print("Alice> " + (reply or ""))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
