#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alice 0.0.5 — a continuously-learning, local, interactable AI agent (non-autonomous by default)
New: 'darkvision' mode for autonomous self-training and self-skill-learning.

Key add-ons vs 0.0.4
- /darkvision start|status|stop to control a safe, background self-improvement loop
- Self-training reuses dataset snapshots + optional LoRA fine-tuning (if libs present)
- Self-skill-learning: the model proposes safe Python skills, which are compiled, validated, auto-approved, and logged
- Separate learning log at train/darkvision.log
- No shell access, no package installs, no OS interaction from generated skills
"""

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
    "- concise, friendly; use dashed bullets for lists.\n"
    "Memory:\n"
    "- You remember important facts and files the user ingests.\n"
)
SAFETY_PROMPT = (
    "Safety:\n"
    "- Never execute OS commands or interact with the operating system.\n"
    "- Use only provided tools; do not claim to have run code unless host confirms.\n"
)

THOUGHT_RE = re.compile(r"^\s*(Thought|Chain|Internal):", re.IGNORECASE)

# ----------------------- CONFIG -----------------------
@dataclasses.dataclass
class Config:
    db_path: str = "alice.db"
    model: str = "llama3.2:3b"
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
    embed_enable: bool = True
    embed_model: str = "nomic-embed-text"
    # Training / LoRA
    train_mode: bool = False
    train_interval_sec: int = 3600
    train_dir: str = "train"
    # Darkvision
    darkvision_interval_sec: int = 900   # every 15 min
    autonomous_tools: bool = False       # keep off by default

# ----------------------- OLLAMA CLIENT -----------------------
class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _http(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.cfg.ollama_host, self.cfg.ollama_port, timeout=180)

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 600) -> str:
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
        try:
            conn = self._http()
            conn.request("POST", "/api/generate", body=json.dumps(payload), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"LLM HTTP {resp.status}: {data[:200]!r}")
            return json.loads(data).get("response", "")
        except Exception as e:
            return f"[LLM error: {e}]"
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def embed(self, text: str) -> Optional[List[float]]:
        if not text.strip():
            return None
        try:
            conn = self._http()
            payload = {"model": self.cfg.embed_model, "prompt": text}
            conn.request("POST", "/api/embeddings", body=json.dumps(payload), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                return None
            obj = json.loads(data)
            emb = obj.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (float, int)):
                return [float(x) for x in emb]
            return None
        except Exception:
            return None
        finally:
            with contextlib.suppress(Exception):
                conn.close()

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
        # naive
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
                # deny-list quick scan
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
            # Compile to catch syntax errors without executing
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
    """Background dataset builder + optional LoRA fine-tune (safe, no installs)."""
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

    def snapshot_now(self):
        try:
            self._snapshot_dataset()
            return True
        except Exception:
            traceback.print_exc()
            return False

    def lora_now(self):
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

# ----------------------- DARKVISION MANAGER -----------------------
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
        # Ensure training is on & trainer running
        self._prev_train_mode = self.cfg.train_mode
        self.cfg.train_mode = True
        self.trainer.start()
        # Thread
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
        # Restore previous training mode
        if self._prev_train_mode is not None:
            self.cfg.train_mode = self._prev_train_mode
        # Note: we do not force-stop Trainer here; it will keep running if train_mode True

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
        # 1) Snapshot dataset (always)
        snap_ok = self.trainer.snapshot_now()
        if snap_ok:
            self._log("Snapshot: updated dataset from recent dialog")
        # 2) Attempt LoRA if configured
        lora_ok = False
        if self.cfg.train_mode:
            lora_ok = self.trainer.lora_now()
            if lora_ok:
                self._log("Training: attempted LoRA fine-tune (if environment available)")
        # 3) Propose a new safe skill (optional)
        new_skill = self._propose_skill()
        if new_skill:
            name, code = new_skill
            saved = self._register_skill(name, code)
            if saved:
                self.skills_learned += 1
                self.last_skill_name = name
                self._log(f"Skill learned: {name}")
        self.last_cycle_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _propose_skill(self) -> Optional[Tuple[str, str]]:
        # Collect some recent context
        history = self.mem.get_recent_dialogue(30)
        convo = "\n".join(f"{r.upper()}: {c}" for r, c in history)
        # Ask the model to propose exactly one minimal, safe Python skill
        prompt = textwrap.dedent(f"""
        You are designing one tiny reusable Python skill to help with tasks seen below.
        - Use ONLY Python standard library; no OS, no network, no file I/O, no subprocess.
        - The skill MUST define:  def skill_main(**kwargs): ...
        - Keep code short and robust. Include a docstring explaining inputs/outputs.
        - Return ONLY a Python code block, nothing else.

        Recent conversation (for inspiration):
        {convo}

        Now propose ONE useful skill that would help with similar tasks, for example:
        - text utilities (cleaning, formatting, extracting)
        - simple math helpers
        - parsing small structured strings
        - short transformations

        Output:
        ```python
        # skill: <short_name_in_snake_case>
        <code that defines skill_main(**kwargs)>
        ```
        """).strip()
        out = self.llm.complete(prompt, system="You are a careful Python assistant who writes safe minimal code.", max_tokens=800)
        # Extract code block
        m = re.search(r"```python\s*(?P<code>[\s\S]+?)\s*```", out)
        if not m:
            return None
        code = m.group("code").strip()
        # Pull a suggested name (header comment) or synthesize one
        name_match = re.search(r"^#\s*skill:\s*([a-zA-Z0-9_]{3,40})", code, re.MULTILINE)
        name = name_match.group(1) if name_match else f"skill_{abs(hash(code)) % 10_000}"
        # Safety checks
        if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
            self._log(f"Rejected unsafe skill proposal: {name}")
            return None
        if "def skill_main" not in code:
            self._log("Rejected skill (no skill_main)")
            return None
        # Try compiling only (no execution)
        try:
            compile(code, "<proposal>", "exec")
        except Exception as e:
            self._log(f"Rejected skill (syntax error): {e}")
            return None
        return name, code

    def _register_skill(self, name: str, code: str) -> bool:
        try:
            self.mem.add_skill(name, code, approved=True)
            return True
        except Exception as e:
            self._log(f"Failed to register skill {name}: {e}")
            return False

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
        # Seed system prompt
        self.mem.add_message("system", BASE_PROMPT + SAFETY_PROMPT, index=False)
        self.mem.upsert_fact("agent.name", "Alice")
        self.mem.upsert_fact("agent.style", "concise-with-dashed-bullets")
        self.last_assistant_id: Optional[int] = None

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
            "Summarize durable facts/preferences noted. Output 3-7 bullets 'key: value', then one paragraph starting 'Reflection:'.\n\n"
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

    # ---------- prompt ----------
    def _compose_prompt(self, user_text: str) -> str:
        facts = self.mem.list_facts(12)
        fact_lines = [f"- {k}: {v} (w={w:.1f})" for k, v, w in facts]
        recent = self.mem.get_recent_dialogue(self.cfg.max_history)
        recall = self.mem.recall(user_text, self.cfg.recall_k)
        recall_text = "\n".join([f"• {c[:400]}" for _mid, c in recall])
        chat_snippets = "\n".join([f"{r.upper()}: {c}" for r, c in recent])
        guidance = (
            "Answer succinctly. Use FACTS and RECALL if relevant. Prefer dashed bullets for lists.\n"
        )
        return (
            f"{guidance}\n\nFACTS (top-weighted):\n" + "\n".join(fact_lines) +
            f"\n\nRECALL (similar past content):\n{recall_text}\n\nRECENT CHAT:\n{chat_snippets}\n\nUSER: {user_text}\nASSISTANT:"
        )

    # ---------- commands & chat ----------
    def _help_text(self) -> str:
        return textwrap.dedent("""\
        **Hi, I’m Alice.** I learn from our conversations and from files you share.

        • **Chat naturally.** Ask questions; I’ll keep it concise.
        • **Teach me skills.** `/teach <name>` with a Python block defining `skill_main(**kwargs)`, then `/run <name> { ... }`.
        • **Add knowledge.** `/ingest <file-or-folder>` for .txt/.md; then find things with `/recall <query>`.
        • **Guide my learning.** `/good` or `/bad` tunes my responses.
        • **Self-train (optional).** `/darkvision start` turns on autonomous learning; `/darkvision status` checks progress; `/darkvision stop` halts it.

        Tip: Type `/help` anytime for this guide.
        """).strip()

    def _cmd_teach(self, raw: str) -> str:
        m = re.search(r"/teach\s+(?P<name>[A-Za-z0-9_\-]+)\s+```python\n(?P<code>.*?)```", raw, re.DOTALL)
        if not m:
            return "Usage: /teach <name> ```python\n<code defining skill_main(**kwargs)>```"
        name = m.group("name")
        code = m.group("code").strip()
        # safety scan
        if re.search(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|open|eval|exec|compile|ctypes|multiprocessing)\b", code):
            return "- Rejected for safety: found disallowed APIs."
        try:
            compile(code, "<user-skill>", "exec")
        except Exception as e:
            return f"- Syntax error: {e}"
        self.mem.add_skill(name, code, approved=True)
        return f"- Skill '{name}' saved and approved! Try `/run {name} {{...}}`."

    def _cmd_run(self, raw: str) -> str:
        m = re.search(r"/run\s+(?P<name>[A-Za-z0-9_\-]+)\s+(?P<json>{.*})\s*$", raw, re.DOTALL)
        if not m:
            return "Usage: /run <name> {json-args}"
        name = m.group("name")
        try:
            args = json.loads(m.group("json"))
        except json.JSONDecodeError as e:
            return f"- Bad JSON args: {e}"
        if name in ("calc", "recall", "set_fact", "get_facts"):
            return self._run_tool(name, args)
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
            return "- Darkvision: ON (self-training and self-skill-learning started)"
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

    def handle_user(self, user_text: str) -> str:
        # Slash commands
        if user_text.startswith("/help"):
            return self._help_text()
        if user_text.startswith("/facts"):
            facts = self.mem.list_facts(30)
            if not facts:
                return "No facts learned yet."
            return "Known facts (top-weighted):\n" + "\n".join([f"- {k}: {v} (w={w:.2f})" for k, v, w in facts])
        if user_text.startswith("/recall"):
            q = user_text.split(" ", 1)[1] if " " in user_text else ""
            hits = self.mem.recall(q, self.cfg.recall_k)
            return "Recall results:\n" + ("\n".join([f"- #{mid}: {c[:120]}" for mid, c in hits]) if hits else "- (none)")
        if user_text.startswith("/good"):
            self.mem.add_reward(self.last_assistant_id, +1.0, reason="user_mark_good")
            return "- Thanks — feedback recorded."
        if user_text.startswith("/bad"):
            self.mem.add_reward(self.last_assistant_id, -1.0, reason="user_mark_bad")
            return "- Got it — I’ll adjust."
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

        # Normal chat
        self.mem.add_message("user", user_text)
        prompt = self._compose_prompt(user_text)
        # Show spinner while thinking (classic | / - \)
        def _spinner(stop_event: threading.Event):
            syms = ['|', '/', '-', '\\']
            i = 0
            sys.stdout.write("Alice> " + syms[i]); sys.stdout.flush()
            i = (i + 1) % len(syms)
            while not stop_event.wait(0.1):
                sys.stdout.write("\b" + syms[i]); sys.stdout.flush()
                i = (i + 1) % len(syms)

        stop = threading.Event()
        t = threading.Thread(target=_spinner, args=(stop,), daemon=True)
        t.start()
        try:
            resp = self.llm.complete(prompt, system=(BASE_PROMPT + SAFETY_PROMPT), max_tokens=700)
        finally:
            stop.set(); t.join(); sys.stdout.write("\b \b"); sys.stdout.flush()

        # Strip internal thoughts
        final = "\n".join(line for line in resp.splitlines() if not THOUGHT_RE.match(line)).strip()
        self.last_assistant_id = self.mem.add_message("assistant", final)
        print(final)
        return ""

    # ---------- built-in tool runner ----------
    def _run_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "calc":
            expr = str(args.get("expr", ""))
            try:
                allowed_names = {"__builtins__": None, "math": math}
                result = eval(expr, allowed_names, {"math": math})
                return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
        if name == "recall":
            q = str(args.get("query", ""))
            k = int(args.get("k", 5))
            hits = self.mem.recall(q, k)
            return json.dumps({"ok": True, "results": [{"id": mid, "content": c[:200]} for mid, c in hits]}, ensure_ascii=False)
        if name == "set_fact":
            key = str(args.get("key", "")).strip()
            val = str(args.get("value", "")).strip()
            if not key:
                return json.dumps({"ok": False, "error": "missing key"}, ensure_ascii=False)
            self.mem.upsert_fact(key, val, weight_delta=1.0)
            return json.dumps({"ok": True, "set": {"key": key, "value": val}}, ensure_ascii=False)
        if name == "get_facts":
            facts = self.mem.list_facts(20)
            return json.dumps({"ok": True, "facts": [{"key": k, "value": v, "weight": w} for (k, v, w) in facts]}, ensure_ascii=False)
        # Skill
        row = self.mem.get_skill(name)
        if not row:
            return json.dumps({"ok": False, "error": f"No such tool/skill: {name}"}, ensure_ascii=False)
        _, code, approved, _ = row
        if not approved and not self.cfg.allow_dangerous_skills:
            return json.dumps({"ok": False, "error": f"Skill '{name}' not approved"}, ensure_ascii=False)
        return self.skills.run(name, code, args)

    def shutdown(self):
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.reflect_thread.join(timeout=2)
        self.darkvision.stop_now()
        self.trainer.shutdown()
        self.mem.shutdown()

# ----------------------- CLI -----------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alice — a continuously-learning local AI (with optional darkvision mode)")
    ap.add_argument("--db", default="alice.db")
    ap.add_argument("--model", default="llama3.2:3b")
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
    )
    alice = Alice(cfg)

    def handle_sigint(_sig, _frm):
        print("\n[shutting down]")
        alice.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    print("Alice is awake. Type /help for a quick guide. (Ctrl+C to exit.)\n")
    while True:
        try:
            user = input("You> ").strip()
        except EOFError:
            break
        if not user:
            continue

        # Run and show spinner while waiting
        result: Optional[str] = None
        done = threading.Event()

        def _work():
            nonlocal result
            result = alice.handle_user(user)
            done.set()

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        spinner = ['|', '/', '-', '\\']
        i = 0
        sys.stdout.write("Alice> "); sys.stdout.flush()
        while not done.wait(0.1):
            sys.stdout.write(spinner[i] + "\b"); sys.stdout.flush()
            i = (i + 1) % len(spinner)
        t.join()
        sys.stdout.write("\b")  # clear spinner char
        if result:
            print(result)

    alice.shutdown()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
