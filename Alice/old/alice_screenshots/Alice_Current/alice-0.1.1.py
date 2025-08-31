#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---
# Alice 0.1.1 — unified main program (Darkmode built-in, model-agnostic)
# Highlights:
# - /darkmode on|off|status   → toggles autonomous tool use & terse style
# - /model list|info|set TAG  → live model switching, capability probing
# - /embed on|off|model NAME  → control semantic memory at runtime
# - safer calc via AST; robust Ollama retries; keeps darkvision, training, skills, ingest, recall
# - keeps concise style, dashed bullets, and thoughts stripping
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

-- Semantic embeddings table (optional; Ollama /api/embeddings)
CREATE TABLE IF NOT EXISTS embeddings (
  message_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec_json TEXT NOT NULL,
  ts TEXT NOT NULL,
  FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- Keep FTS in sync
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
    model: str = "gpt-oss:20b"          # default remains gpt-oss:20b, but live /model switching supported
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
    # embeddings
    embed_enable: bool = True
    embed_model: str = "nomic-embed-text"
    # training / LoRA
    train_mode: bool = False
    train_interval_sec: int = 3600
    train_dir: str = "train"
    # darkvision (autonomous self-improvement)
    darkvision_interval_sec: int = 900
    # darkmode/autonomous tool use toggle
    autonomous_tools: bool = False

# ----------------------- OLLAMA CLIENT -----------------------
class LLMClient:
    """Minimal pluggable client for Ollama with retries + capability probing."""
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

        # simple retry loop
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
            status, data = self._request(
                "POST", "/api/embeddings",
                {"model": self.cfg.embed_model, "prompt": text},
                timeout=120
            )
            if status >= 400:
                return None
            obj = json.loads(data)
            emb = obj.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                return [float(x) for x in emb]
            return None
        except Exception:
            return None

    # ---- model-agnostic helpers ----
    def list_models(self) -> List[str]:
        """Return installed model tags (best-effort)."""
        try:
            # /api/tags is GET in Ollama
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
                # prefer 'model' (tag) or 'name'
                tag = m.get("model") or m.get("name")
                if tag:
                    names.append(tag)
            return sorted(set(names))
        except Exception:
            return []

    def show_model(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Return /api/show details if available."""
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
        # semantic
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
                                got[int(mid)] = (s, row[0])
        # FTS
        if self.has_fts:
            try:
                q = " OR ".join(re.findall(r"[A-Za-z0-9_]{2,}", query))
                if q:
                    for mid, content in self.conn.execute(
                        "SELECT message_id, content FROM messages_fts WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?",
                        (q, k),
                    ):
                        got.setdefault(int(mid), (0.0, content))
            except sqlite3.OperationalError:
                pass
        # naive fallback
        if not got:
            rows = self.conn.execute("SELECT id, content FROM messages ORDER BY id DESC LIMIT 1000").fetchall()
            ql = query.lower()
            for mid, content in rows:
                score = sum(1 for tok in ql.split() if tok in content.lower())
                if score:
                    got[int(mid)] = (float(score), content)
        ranked = sorted(got.items(), key=lambda kv: (-kv[1][0], -kv[0]))
        return [(mid, content) for mid, (_s, content) in ranked[:k]]

    def upsert_fact(self, key: str, value: str, weight_delta: float = 1.0):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO facts(key, value, weight, ts) VALUES(?,?,?,?) "
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
        row = self.conn.execute("SELECT name, code, approved, usage_count FROM skills WHERE name=?", (name,)).fetchone()
        return row

    def bump_skill_usage(self, name: str):
        self.conn.execute("UPDATE skills SET usage_count=usage_count+1 WHERE name=?", (name,))
        self.conn.commit()

    def add_reward(self, message_id: Optional[int], delta: float, reason: str = ""):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute("INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?,?,?,?)", (message_id, delta, reason, ts))
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
SAFE_MODULES = {
    "math": math,
    "json": json,
    "re": re,
    "random": random,
    "datetime": dt,
}

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

# ----------------------- TRAINER (optional) -----------------------
class Trainer:
    """Background dataset builder + optional LoRA fine-tune (safe; no installs required)."""
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
                self._maybe_train_lora()
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
        try:
            self._maybe_train_lora()
            return True
        except Exception:
            traceback.print_exc()
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

    def _maybe_train_lora(self):
        if not self.cfg.train_mode:
            return
        base = os.environ.get("ALICE_LORA_BASE", "")
        if not base:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer as HFTrainer, TrainingArguments
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model
        except Exception:
            return
        try:
            ds = load_dataset("json", data_files=self.ds_path)
            if ds["train"].num_rows < 16:
                return
            tok = AutoTokenizer.from_pretrained(base, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(base)
            lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lora)
            def fmt(example):
                x, y = example["prompt"], example["response"]
                text = f"<s>System: You are Alice.\nUser: {x}\nAssistant: {y}</s>"
                ids = tok(text, truncation=True, max_length=1024)
                ids["labels"] = ids["input_ids"].copy()
                return ids
            ds_tok = ds.map(fmt, remove_columns=ds["train"].column_names)
            args = TrainingArguments(
                output_dir=os.path.join(self.cfg.train_dir, "lora_out"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=1e-4,
                logging_steps=10,
                save_steps=0,
                report_to=[],
            )
            trainer = HFTrainer(model=model, args=args, train_dataset=ds_tok["train"])
            trainer.train()
            model.save_pretrained(os.path.join(self.cfg.train_dir, "lora_adapter"))
        except Exception:
            pass

# ----------------------- DARKVISION -----------------------
class Darkvision:
    """Autonomous self-improvement loop: dataset snapshot, optional LoRA, propose safe skills."""
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
        # Stats
        self.last_cycle_utc: Optional[str] = None
        self.skills_learned: int = 0
        self.last_skill_name: Optional[str] = None
        self._prev_train_mode: Optional[bool] = None

    def _log(self, msg: str):
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

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
        if self.cfg.train_mode:
            if self.trainer.lora_now():
                self._log("Training: attempted LoRA fine-tune (if environment available)")
        new_skill = self._propose_skill()
        if new_skill:
            name, code = new_skill
            try:
                self.mem.add_skill(name, code, approved=True)
                self.skills_learned += 1
                self.last_skill_name = name
                self._log(f"Skill learned: {name}")
            except Exception as e:
                self._log(f"Failed to register skill {name}: {e}")
        self.last_cycle_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _propose_skill(self) -> Optional[Tuple[str, str]]:
        history = self.mem.get_recent_dialogue(30)
        convo = "\n".join(f"{r.upper()}: {c}" for r, c in history)
        prompt = textwrap.dedent(f"""
        You are designing one tiny reusable Python skill to help with tasks seen below.
        - Use ONLY Python stdlib; no OS, no network, no file I/O, no subprocess.
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
        out = self.llm.complete(prompt, system="You write safe, minimal Python.", max_tokens=800)
        m = re.search(r"```python\s*(?P<code>[\s\S]+?)\s*```", out)
        if not m:
            return None
        code = m.group("code").strip()
        name_match = re.search(r"^#\s*skill:\s*([a-zA-Z0-9_]{3,40})", code, re.MULTILINE)
        name = name_match.group(1) if name_match else f"skill_{abs(hash(code)) % 10_000}"
        if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
            self._log(f"Rejected unsafe skill proposal: {name}")
            return None
        if "def skill_main" not in code:
            self._log("Rejected skill (no skill_main)")
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
        # Seed prompt facts
        self.mem.add_message("system", BASE_PROMPT + SAFETY_PROMPT, index=False)
        self.mem.upsert_fact("agent.name", "Alice")
        self.mem.upsert_fact("agent.style", "concise-with-dashed-bullets")
        self.last_assistant_id: Optional[int] = None
        # Darkmode flag mirrors autonomous tools
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
            "Summarize durable facts/preferences noted. Output 3-7 bullets 'key: value', "
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
                    self.mem.upsert_fact(k.strip(), v.strip(), weight_delta=0.2)

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
            f"{guidance}\n\nFACTS (top-weighted):\n" + "\n".join(fact_lines) +
            f"\n\nRECALL (similar past content):\n{recall_text}\n\nRECENT CHAT:\n{chat_snips}\n\nUSER: {user_text}\nASSISTANT:"
        )

    def _system_for_chat(self) -> str:
        # In darkmode, include tool instructions so the model can call tools autonomously.
        return BASE_PROMPT + (TOOLS_PROMPT if self.darkmode_enabled else "") + SAFETY_PROMPT

    # ---------- built-in tools ----------
    def _tool_calc(self, args: Dict[str, Any]) -> Dict[str, Any]:
        expr = str(args.get("expr", "")).strip()
        if not expr:
            return {"ok": False, "error": "missing 'expr'"}
        # safer arithmetic via AST
        try:
            import ast
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
            val = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}},
                       {"pi": math.pi, "e": math.e, "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt})
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

    def _run_tool(self, name: str, args: Dict[str, Any]) -> str:
        # built-ins first
        builtins = {
            "calc": self._tool_calc,
            "recall": self._tool_recall,
            "set_fact": self._tool_set_fact,
            "get_facts": self._tool_get_facts,
        }
        if name in builtins:
            try:
                res = builtins[name](args)
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            return json.dumps(res, ensure_ascii=False)
        # then user skills
        row = self.mem.get_skill(name)
        if not row:
            return json.dumps({"ok": False, "error": f"no such tool/skill: {name}"}, ensure_ascii=False)
        _, code, approved, _ = row
        if not approved and not self.cfg.allow_dangerous_skills:
            return json.dumps({"ok": False, "error": f"skill '{name}' not approved"}, ensure_ascii=False)
        return self.skills.run(name, code, args)

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

    # ---------- commands ----------
    def _help_text(self) -> str:
        return textwrap.dedent("""\
        Commands
        - /help                  — show this help
        - /facts                 — list top remembered facts
        - /recall <query>        — search past chat/content
        - /good | /bad           — reward/punish last answer
        - /teach <name> ```python\n...``` — add a skill (must define skill_main(**kwargs))
        - /run <name> {json}     — run a built-in tool or approved skill
        - /ingest <path>         — ingest .txt/.md into memory
        - /train status|now      — snapshot dataset and optional LoRA
        - /darkvision start|status|stop  — autonomous self-improvement loop
        - /darkmode on|off|status — enable autonomous tool use + terse style
        - /model list|info|set <tag>     — list/info/set Ollama model
        - /embed on|off|model <name>     — control semantic memory/embedding model
        """).strip()

    def _cmd_teach(self, raw: str) -> str:
        m = re.search(r"/teach\s+(?P<name>[A-Za-z0-9_\-]+)\s+```python\n(?P<code>.*?)```", raw, re.DOTALL)
        if not m:
            return "Usage: /teach <name> ```python\n<code with skill_main(**kwargs)>```"
        name = m.group("name"); code = m.group("code").strip()
        if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
            return "- Rejected for safety: found disallowed APIs."
        try:
            compile(code, "<user-skill>", "exec")
        except Exception as e:
            return f"- Syntax error: {e}"
        self.mem.add_skill(name, code, approved=True)
        return f"- Skill '{name}' saved and approved. Use /run {name} {{...}} to execute."

    def _cmd_run(self, raw: str) -> str:
        m = re.search(r"/run\s+(?P<name>[A-Za-z0-9_\-]+)\s+(?P<json>{.*})\s*$", raw, re.DOTALL)
        if not m:
            return "Usage: /run <name> {json-args}"
        name = m.group("name")
        try:
            args = json.loads(m.group("json"))
        except json.JSONDecodeError as e:
            return f"- Bad JSON args: {e}"
        # allow direct built-ins via /run
        if name in ("calc", "recall", "set_fact", "get_facts"):
            return self._run_tool(name, args)
        # skill
        row = self.mem.get_skill(name)
        if not row:
            return f"- No such skill: {name}"
        _, code, approved, _ = row
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
        cmd = raw.strip().split()
        if len(cmd) >= 2 and cmd[1] == "status":
            size = 0
            try:
                p = os.path.join(self.cfg.train_dir, "alice_sft.jsonl")
                size = os.path.getsize(p) if os.path.exists(p) else 0
            except Exception:
                size = 0
            return f"- train_mode={'on' if self.cfg.train_mode else 'off'}, dataset_bytes={size}"
        if len(cmd) >= 2 and cmd[1] == "now":
            snap = self.trainer.snapshot_now()
            self.trainer.lora_now()
            if not snap:
                return "- Train attempt completed. (No data to train yet.)"
            if self.cfg.train_mode and os.environ.get("ALICE_LORA_BASE"):
                return "- Train attempt completed. Dataset saved, and a LoRA fine-tune was attempted."
            else:
                return "- Train attempt completed. Dataset saved (no fine-tune since training mode off or base model not set)."
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
            return "Usage: /darkmode on | /darkmode off | /darkmode status"
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
        return "Usage: /darkmode on | /darkmode off | /darkmode status"

    def _cmd_model(self, raw: str) -> str:
        parts = raw.strip().split()
        if len(parts) == 1:
            return "Usage: /model list | /model info | /model set <tag>"
        sub = parts[1].lower()
        if sub == "list":
            models = self.llm.list_models()
            if not models:
                return "- no models found (is ollama running and models pulled?)"
            return "- Installed models:\n" + "\n".join(f"- {m}" for m in models)
        if sub == "info":
            info = self.llm.show_model() or {}
            det = info.get("details", {})
            ctx = det.get("context_length") or det.get("parameter_size") or "?"
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
            # Probe context length if available; adopt if larger
            info = self.llm.show_model(tag) or {}
            det = info.get("details", {})
            ctx_len = det.get("context_length")
            if isinstance(ctx_len, int) and ctx_len > 0:
                self.cfg.num_ctx = min(max(1024, ctx_len), 32768)
            return f"- model switched: {old} → {self.cfg.model} (num_ctx={self.cfg.num_ctx})"
        return "Usage: /model list | /model info | /model set <tag>"

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
            # quick probe
            probe = self.llm.embed("test 123")
            ok = "ok" if probe else "not-ready"
            return f"- embed model: {old} → {name} ({ok})"
        return "Usage: /embed on|off | /embed model <name>"

    # ---------- main chat ----------
    def handle_user(self, user_text: str) -> str:
        # Slash commands
        if user_text.startswith("/help"):
            return self._help_text()
        if user_text.startswith("/facts"):
            facts = self.mem.list_facts(30)
            return "Known facts (top weights):\n" + ("\n".join([f"- {k}: {v} (w={w:.2f})" for k, v, w in facts]) if facts else "- (none)")
        if user_text.startswith("/recall"):
            q = user_text.split(" ", 1)[1] if " " in user_text else ""
            hits = self.mem.recall(q, self.cfg.recall_k)
            return "Recall results:\n" + ("\n".join([f"- #{mid}: {c[:120]}" for mid, c in hits]) if hits else "- (none)")
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

        # echo helper (say "...", say: ..., /say ...)
        _t = user_text.strip()
        if _t.lower().startswith('say') or _t.lower().startswith('/say'):
            parts = _t.split(' ', 1)
            if len(parts) > 1:
                echo = parts[1].strip()
                if (echo.startswith('"') and echo.endswith('"')) or (echo.startswith("'") and echo.endswith("'")):
                    echo = echo[1:-1]
                return echo

        # Normal chat
        self.mem.add_message("user", user_text)
        prompt = self._compose_prompt(user_text)
        system = self._system_for_chat()

        # Generate
        resp = self.llm.complete(prompt, system=system, max_tokens=700)

        # If darkmode (autonomous tools) — handle tool loop
        if self.darkmode_enabled:
            steps = 0
            while True:
                call = self._maybe_tool_call(resp)
                if not call or steps >= 3:
                    break
                steps += 1
                name, args = call
                tool_json = self._run_tool(name, args)
                follow = (
                    f"Tool '{name}' returned JSON below. Use it and continue.\n"
                    f"TOOL_RESULT: {tool_json}\n"
                    f"USER: {user_text}\nASSISTANT:"
                )
                resp = self.llm.complete(follow, system=system, max_tokens=700)

        # Strip any explicit thoughts
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

# ----------------------- CLI -----------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alice — unified, continuously-learning local AI (Darkmode built-in)")
    ap.add_argument("--db", default="alice.db")
    ap.add_argument("--model", default="gpt-oss:20b", help="Ollama model tag (any pulled model)")
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
    ap.add_argument("--autonomous-tools", action="store_true", help="Start with Darkmode on")
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
    )
    alice = Alice(cfg)

    def handle_sigint(_sig, _frm):
        print("\n[shutting down]")
        alice.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print("Alice is awake. Type /help for commands. (Ctrl+C to exit.)\n")
    while True:
        try:
            user = input("You> ").strip()
        except EOFError:
            break
        if not user:
            continue
        reply = alice.handle_user(user)
        # In Darkmode we still just print final; no streaming, no spinner
        print("Alice> " + (reply or ""))

    alice.shutdown()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
