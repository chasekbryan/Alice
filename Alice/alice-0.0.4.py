#!/usr/bin/env python3
"""
Alice 0.0.4 — a continuously-learning, local, interactable AI agent.

Key features:
- Interactive CLI assistant with persistent SQLite memory (messages, facts, etc.).
- Learns new **skills** via `/teach` (safe, sandboxed Python plugins).
- Can ingest text files via `/ingest` and **recall** information with `/recall`.
- Supports optional semantic memory (embeddings) and on-demand fine-tuning.
- Non-autonomous by default (won't call tools unless instructed; can opt-in via flag).
"""

from __future__ import annotations
import argparse
import contextlib
import dataclasses
import datetime as dt
import http.client
import io
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

-- Full-text search over content if FTS5 exists
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

-- Semantic embeddings table (optional; used if embeddings enabled)
CREATE TABLE IF NOT EXISTS embeddings (
  message_id INTEGER PRIMARY KEY,     -- references messages.id
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec_json TEXT NOT NULL,             -- JSON array of floats
  ts TEXT NOT NULL,
  FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS trg_messages_ai
AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content, message_id, ts)
  VALUES (new.id, new.content, new.id, new.ts);
END;
CREATE TRIGGER IF NOT EXISTS trg_messages_ad
AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content)
  VALUES('delete', old.id, old.content);
END;
"""

# ----------------------- SYSTEM PROMPTS -----------------------
BASE_PROMPT = (
    "You are Alice — a continuously-learning AI assistant.\n"
    "Core style:\n"
    "- concise and friendly\n"
    "- use dashed bullets for lists\n"
    "- cite sources only if explicitly asked\n"
    "Memory:\n"
    "- You remember important facts about the user and the project.\n"
)
TOOLS_PROMPT = (
    "Tools:\n"
    "- Built-ins: calc(expr), recall(query), set_fact(key, value), get_facts()\n"
    "- User-taught skills may also be available.\n"
    "To use a tool, output a line like:\n"
    "<<call:NAME args='{\"param\": \"value\"}'>>\n"
    "The host will provide TOOL_RESULT, then you continue the answer.\n"
)
SAFETY_PROMPT = (
    "Safety:\n"
    "- Never execute OS commands or access the system beyond provided tools.\n"
    "- Only call tools that you have been told exist.\n"
)

# Regex to detect internal "thought" or chain-of-thought lines in model output
THOUGHT_RE = re.compile(r"^\s*(Thought|Chain|Internal):", re.IGNORECASE)

@dataclasses.dataclass
class Config:
    db_path: str = "alice.db"
    model: str = "llama3.2:3b"            # Ollama model name or tag
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    temperature: float = 0.4
    top_p: float = 0.95
    top_k: int = 40
    num_ctx: int = 4096                   # context tokens if model supports it
    reflect_every_sec: int = 600         # reflection interval
    recall_k: int = 12
    max_history: int = 14                # max dialogue turns to include
    allow_dangerous_skills: bool = False
    embed_enable: bool = True            # enable semantic embeddings for memory
    embed_model: str = "nomic-embed-text"
    train_mode: bool = False             # enable background LoRA training
    train_interval_sec: int = 3600
    train_dir: str = "train"
    autonomous: bool = False             # allow autonomous tool use (opt-in)

class LLMClient:
    """Client for Ollama API (generate and embeddings)."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _http_conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.cfg.ollama_host, self.cfg.ollama_port, timeout=180)

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "top_k": self.cfg.top_k,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": max_tokens
            }
        }
        if system:
            payload["system"] = system
        try:
            conn = self._http_conn()
            conn.request("POST", "/api/generate", body=json.dumps(payload),
                         headers={"Content-Type": "application/json"})
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

    def embed(self, text: str) -> Optional[List[float]]:
        if not text.strip():
            return None
        try:
            conn = self._http_conn()
            payload = {"model": self.cfg.embed_model, "prompt": text}
            conn.request("POST", "/api/embeddings", body=json.dumps(payload),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                # Embedding model might not be loaded; return None to fall back
                return None
            obj = json.loads(data)
            emb = obj.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                return [float(x) for x in emb]
            return None
        except Exception:
            return None
        finally:
            with contextlib.suppress(Exception):
                conn.close()

class Memory:
    def __init__(self, cfg: Config, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm
        self.conn = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_db()
        self.has_fts = self._check_fts()
        # Queue and thread for embedding new messages in background
        self.index_q: queue.Queue[Tuple[int, str]] = queue.Queue()
        self.indexer = threading.Thread(target=self._index_loop, daemon=True)
        self.indexer.start()

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

    def add_message(self, role: str, content: str, index: bool = True) -> int:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO messages(role, content, ts) VALUES (?, ?, ?)",
            (role, content, ts)
        )
        self.conn.commit()
        mid = cur.lastrowid
        # If enabled, queue this message for semantic embedding
        if index and self.cfg.embed_enable:
            with contextlib.suppress(Exception):
                self.index_q.put((mid, content))
        return mid

    def get_recent_dialogue(self, n: int) -> List[Tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = list(reversed(cur.fetchall()))
        cur.close()
        return rows

    def recall(self, query: str, k: int) -> List[Tuple[int, str]]:
        # Semantic recall first (if enabled and data indexed)
        results: List[Tuple[int, float, str]] = []
        if self.cfg.embed_enable:
            # Retrieve all stored embeddings
            cur = self.conn.execute("SELECT message_id, vec_json FROM embeddings")
            rows = cur.fetchall()
            cur.close()
            if rows:
                qvec = self.llm.embed(query) or []
                if qvec:
                    # Compute cosine similarity for each embedding
                    def cosine(a: List[float], b: List[float]) -> float:
                        dot = na = nb = 0.0
                        for x, y in zip(a, b):
                            dot += x * y
                            na += x * x
                            nb += y * y
                        return dot / math.sqrt(na * nb) if na and nb else 0.0
                    for mid, vec_json in rows:
                        try:
                            vec = json.loads(vec_json)
                            score = cosine(qvec, vec)
                        except Exception:
                            score = 0.0
                        if score > 0:
                            # retrieve content snippet
                            cur2 = self.conn.execute("SELECT content FROM messages WHERE id=?", (mid,))
                            row = cur2.fetchone()
                            cur2.close()
                            if row:
                                results.append((mid, score, row[0]))
                    results.sort(key=lambda x: x[1], reverse=True)
                    results = results[:k]
        # FTS or fallback keyword search
        found: Dict[int, Tuple[float, str]] = {}
        if self.has_fts:
            try:
                q = " OR ".join(re.findall(r"[A-Za-z0-9_]{2,}", query))
                if q:
                    cur = self.conn.execute(
                        "SELECT message_id, content FROM messages_fts "
                        "WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?",
                        (q, k)
                    )
                    for mid, content in cur.fetchall():
                        # use score 0 for FTS results (if semantic results exist, they outrank by score)
                        found[int(mid)] = (0.0, content)
                    cur.close()
            except sqlite3.OperationalError:
                pass
        if not found:
            # naive search through recent messages if FTS not available or no hits
            cur = self.conn.execute("SELECT id, content FROM messages ORDER BY id DESC LIMIT 1000")
            rows = cur.fetchall()
            cur.close()
            q_lower = query.lower()
            for mid, content in rows:
                match_score = sum(1 for tok in q_lower.split() if tok in content.lower())
                if match_score:
                    found[int(mid)] = (float(match_score), content)
        # Merge semantic and text results, preferring higher semantic scores
        for mid, score, content in results:
            found[mid] = (score, content)
        # Sort by score (semantic score first, break ties by recency via id)
        ranked = sorted(found.items(), key=lambda kv: (-kv[1][0], -kv[0]))
        # Return list of (message_id, content) for top k
        return [(mid, content) for mid, (_score, content) in ranked[:k]]

    def upsert_fact(self, key: str, value: str, weight_delta: float = 1.0):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO facts(key, value, weight, ts) VALUES (?, ?, ?, ?)\n"
            "ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, weight=facts.weight+?, ts=excluded.ts",
            (key, value, 1.0, ts, weight_delta)
        )
        self.conn.commit()

    def list_facts(self, top_n: int = 20) -> List[Tuple[str, str, float]]:
        cur = self.conn.execute(
            "SELECT key, value, weight FROM facts "
            "ORDER BY weight DESC, ts DESC LIMIT ?",
            (top_n,)
        )
        rows = cur.fetchall()
        cur.close()
        return rows

    def add_reflection(self, text: str, score: float = 0.0):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO reflections(text, score, ts) VALUES (?, ?, ?)",
            (text, score, ts)
        )
        self.conn.commit()

    def add_skill(self, name: str, code: str, approved: bool = False):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO skills(name, code, approved, usage_count, created_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, code, 1 if approved else 0, 0, ts)
        )
        self.conn.commit()

    def get_skill(self, name: str) -> Optional[Tuple[str, str, bool, int]]:
        cur = self.conn.execute(
            "SELECT name, code, approved, usage_count FROM skills WHERE name=?",
            (name,)
        )
        row = cur.fetchone()
        cur.close()
        return row

    def bump_skill_usage(self, name: str):
        self.conn.execute(
            "UPDATE skills SET usage_count = usage_count + 1 WHERE name=?",
            (name,)
        )
        self.conn.commit()

    def add_reward(self, message_id: Optional[int], delta: float, reason: str = ""):
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?, ?, ?, ?)",
            (message_id, delta, reason, ts)
        )
        self.conn.commit()

    def shutdown(self):
        # Signal the indexer thread to stop
        with contextlib.suppress(Exception):
            self.index_q.put((None, ""))  # sentinel to break loop
        with contextlib.suppress(Exception):
            self.indexer.join(timeout=2)

    def _index_loop(self):
        """Background thread: consume index queue and store embeddings."""
        while True:
            try:
                mid, content = self.index_q.get()
            except Exception:
                break
            # Sentinel check
            if mid is None:
                break
            try:
                if not self.cfg.embed_enable:
                    continue  # embedding disabled, skip
                vec = self.llm.embed(content)
                if not vec:
                    continue  # embedding failed or empty content
                ts = dt.datetime.now(dt.timezone.utc).isoformat()
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings(message_id, model, dim, vec_json, ts) VALUES (?, ?, ?, ?, ?)",
                    (mid, self.cfg.embed_model, len(vec), json.dumps(vec), ts)
                )
                self.conn.commit()
            except Exception as e:
                traceback.print_exc()
                # Continue loop even if one embedding failed
                continue

# ----------------------- SKILL RUNNER -----------------------
class SkillRunner:
    """Runs user-taught skills (in a sandboxed environment)."""
    def __init__(self, allow_dangerous: bool):
        self.allow_dangerous = allow_dangerous

    def run(self, name: str, code: str, args: Dict[str, Any]) -> str:
        # Execute skill code safely (restrict builtins if not dangerous)
        try:
            # Prepare a restricted exec environment
            safe_globals: Dict[str, Any] = {}
            if not self.allow_dangerous:
                # Remove dangerous builtins (this is a simple sandbox approach)
                safe_globals["__builtins__"] = {"print": print, "len": len, "range": range}
            else:
                safe_globals["__builtins__"] = __builtins__
            local_ns: Dict[str, Any] = {}
            exec(code, safe_globals, local_ns)
            if "skill_main" not in local_ns:
                return json.dumps({"ok": False, "error": "No skill_main() in skill code"}, ensure_ascii=False)
            result = local_ns["skill_main"](**(args or {}))
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except Exception:
            # Return the traceback of the skill error in JSON
            return json.dumps({"ok": False, "error": traceback.format_exc(limit=2)}, ensure_ascii=False)

# ----------------------- TRAINER (Optional) -----------------------
class Trainer:
    """Background trainer for dataset snapshot and optional LoRA fine-tuning."""
    def __init__(self, cfg: Config, mem: Memory):
        self.cfg = cfg
        self.mem = mem
        self.stop_event = threading.Event()
        # Paths
        os.makedirs(self.cfg.train_dir, exist_ok=True)
        self.dataset_path = os.path.join(self.cfg.train_dir, "alice_sft.jsonl")
        # Launch background training loop if enabled
        self.thread = threading.Thread(target=self._loop, daemon=True)
        if self.cfg.train_mode:
            self.thread.start()

    def shutdown(self):
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.thread.join(timeout=2)

    def _loop(self):
        """Periodically snapshot dataset and run LoRA training if conditions meet."""
        while not self.stop_event.wait(self.cfg.train_interval_sec):
            try:
                self._snapshot_dataset()
                self._maybe_train_lora()
            except Exception:
                traceback.print_exc()
                continue

    def _snapshot_dataset(self):
        # Gather recent user-assistant exchanges into dataset (last 200 pairs)
        dialogue = self.mem.get_recent_dialogue(400)
        pairs: List[Tuple[str, str]] = []
        last_user = None
        for role, content in dialogue:
            if role == "user":
                last_user = content
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, content))
                last_user = None
        if not pairs:
            return  # nothing to snapshot
        # Prepare JSONL lines
        lines = []
        for u, a in pairs[-200:]:
            lines.append(json.dumps({"prompt": u, "response": a}, ensure_ascii=False))
        payload = "\n".join(lines) + "\n"
        # Write (overwrite) the dataset file
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            f.write(payload)

    def _maybe_train_lora(self):
        if not self.cfg.train_mode:
            return
        base_model = os.environ.get("ALICE_LORA_BASE", "")
        if not base_model:
            return  # no base model specified for fine-tuning
        try:
            # Attempt a LoRA fine-tuning using HuggingFace and PEFT libraries if available
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer as HfTrainer, TrainingArguments
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model
        except ImportError:
            return  # missing required libs, skip
        # Load dataset from the JSONL file
        ds = load_dataset("json", data_files=self.dataset_path)
        if ds["train"].num_rows < 16:
            return  # not enough data to fine-tune
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                 bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        # Prepare dataset for training
        def format_example(rec):
            prompt = rec["prompt"]; response = rec["response"]
            text = f"<s>System: You are Alice.\nUser: {prompt}\nAssistant: {response}</s>"
            tokens = tokenizer(text, truncation=True, max_length=1024)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        tokenized_ds = ds["train"].map(format_example, remove_columns=ds["train"].column_names)
        args = TrainingArguments(output_dir=os.path.join(self.cfg.train_dir, "lora_out"),
                                 per_device_train_batch_size=1,
                                 gradient_accumulation_steps=4,
                                 num_train_epochs=1,
                                 learning_rate=1e-4,
                                 logging_steps=50,
                                 save_steps=0,
                                 report_to=[])
        trainer = HfTrainer(model=model, args=args, train_dataset=tokenized_ds)
        trainer.train()
        # Save the LoRA adapter
        model.save_pretrained(os.path.join(self.cfg.train_dir, "lora_adapter"))

# ----------------------- ALICE AGENT -----------------------
class Alice:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm = LLMClient(cfg)
        self.mem = Memory(cfg, self.llm)
        self.skills = SkillRunner(cfg.allow_dangerous_skills)
        self.trainer = Trainer(cfg, self.mem)
        # Thread for periodic reflection
        self.stop_event = threading.Event()
        self.reflect_thread = threading.Thread(target=self._reflect_loop, daemon=True)
        self.reflect_thread.start()
        # Initialize system prompt and some base facts
        if cfg.autonomous:
            system_prompt = BASE_PROMPT + TOOLS_PROMPT + SAFETY_PROMPT
        else:
            system_prompt = BASE_PROMPT + SAFETY_PROMPT
        self.mem.add_message("system", system_prompt, index=False)
        self.mem.upsert_fact("agent.name", "Alice")
        self.mem.upsert_fact("agent.style", "concise-with-dashed-bullets")
        self.last_assistant_id: Optional[int] = None

    def _reflect_loop(self):
        # Background reflection process: periodically summarize recent dialogue into facts
        while not self.stop_event.wait(self.cfg.reflect_every_sec):
            if self.stop_event.is_set():
                break
            try:
                self._reflect_once()
            except Exception as e:
                traceback.print_exc()

    def _reflect_once(self):
        # Summarize last N messages into facts and a reflection
        history = self.mem.get_recent_dialogue(60)
        if not history:
            return
        prompt = (
            "Summarize any durable facts or user preferences noted in the conversation.\n"
            "- Output 3-7 bullet points in the form 'key: value'.\n"
            "- Then write one paragraph starting with 'Reflection:' summarizing the situation.\n\n"
        )
        # Include recent conversation in prompt
        for role, content in history:
            prompt += f"{role.upper()}: {content}\n"
        summary = self.llm.complete(prompt, system="You are a careful and truthful summarizer.", max_tokens=300)
        # Save the reflection text and extract any fact bullets
        self.mem.add_reflection(summary, score=0.0)
        for line in summary.splitlines():
            if line.strip().startswith("-"):
                cleaned = line.lstrip("- ").strip()
                if ":" in cleaned:
                    key, val = cleaned.split(":", 1)
                    self.mem.upsert_fact(key.strip(), val.strip(), weight_delta=1.0)

    def handle_user(self, user_text: str) -> str:
        # Slash commands take precedence
        if user_text.startswith("/help"):
            return self._help_text()
        if user_text.startswith("/facts"):
            facts = self.mem.list_facts(30)
            if not facts:
                return "No facts learned yet."
            return "Known facts (top-weighted):\n" + "\n".join(f"- {k}: {v} (w={w:.2f})" for k, v, w in facts)
        if user_text.startswith("/recall"):
            query = user_text.split(" ", 1)[1] if " " in user_text else ""
            hits = self.mem.recall(query, self.cfg.recall_k)
            if not hits:
                return "No relevant memory found."
            return "Recall results:\n" + "\n".join(f"- #{mid}: {content[:100]}" for mid, content in hits)
        if user_text.startswith("/good"):
            # Positive feedback
            self.mem.add_reward(self.last_assistant_id, +1.0, reason="user_mark_good")
            return "- Thanks for the feedback! (marked as good)"
        if user_text.startswith("/bad"):
            # Negative feedback
            self.mem.add_reward(self.last_assistant_id, -1.0, reason="user_mark_bad")
            return "- Feedback noted. I'll adjust (marked as bad)."
        if user_text.startswith("/teach"):
            return self._cmd_teach(user_text)
        if user_text.startswith("/run"):
            return self._cmd_run(user_text)
        if user_text.startswith("/ingest"):
            return self._cmd_ingest(user_text)
        if user_text.startswith("/set "):
            # Quick fact setting: /set key value...
            try:
                _, rest = user_text.split(" ", 1)
                key, val = rest.split(" ", 1)
                self.mem.upsert_fact(key.strip(), val.strip(), weight_delta=1.0)
                return f"- Set {key.strip()} = {val.strip()}"
            except Exception:
                return "Usage: /set <key> <value>"
        if user_text.startswith("/train"):
            return self._cmd_train(user_text)

        # Normal conversation (non-command input)
        # Add user message to memory
        self.mem.add_message("user", user_text)
        # Compose prompt with limited history
        prompt = self._compose_prompt(user_text)
        # Call LLM to generate a response
        response = self.llm.complete(prompt, system=(BASE_PROMPT + SAFETY_PROMPT), max_tokens=700)
        # If autonomy is enabled, allow the model to call tools in steps
        if self.cfg.autonomous:
            steps = 0
            # Loop to handle up to a few tool calls in sequence
            while True:
                call = self._maybe_tool_call(response)
                if not call or steps >= 3:
                    break
                steps += 1
                tool_name, args = call
                tool_output = self._run_tool(tool_name, args)
                follow_up = (
                    f"Tool '{tool_name}' returned JSON below. Use it to continue the answer.\n"
                    f"TOOL_RESULT: {tool_output}\n"
                    f"USER: {user_text}\nASSISTANT:"
                )
                response = self.llm.complete(follow_up, system=(BASE_PROMPT + SAFETY_PROMPT), max_tokens=700)
        # Strip any internal 'Thought:' lines before returning
        final_answer = "\n".join(
            line for line in response.splitlines() if not THOUGHT_RE.match(line)
        ).strip()
        # Save assistant message and update last_assistant_id for feedback reference
        self.last_assistant_id = self.mem.add_message("assistant", final_answer)
        return final_answer

    def _compose_prompt(self, latest_user_input: str) -> str:
        """Construct the conversation prompt with recent history and current user input."""
        # Retrieve up to max_history turns of recent dialogue from memory
        history = self.mem.get_recent_dialogue(self.cfg.max_history * 2)
        # Build the prompt from history (excluding system prompt already in memory)
        convo = ""
        for role, content in history:
            if role == "system":
                continue  # system prompt is already accounted for separately
            convo += f"{role.capitalize()}: {content}\n"
        # Append the latest user input
        convo += f"User: {latest_user_input}\nAssistant:"
        return convo

    def _maybe_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """If the LLM output ends with a tool call directive, parse it."""
        m = re.search(r"<<call:(?P<name>[A-Za-z0-9_-]+)\s+args='(?P<args>.*)'>>\s*$", text, re.DOTALL)
        if not m:
            return None
        name = m.group("name")
        args_str = m.group("args")
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        return name, args

    def _run_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a built-in tool or user skill and return its output JSON."""
        # Built-in tools:
        if name == "calc":
            # Safe evaluation for simple math expressions
            expr = str(args.get("expr", ""))
            try:
                # Evaluate expression in a restricted math context
                allowed_names = {"__builtins__": None, "math": math}
                result = eval(expr, allowed_names, {"math": math})
                return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
        if name == "recall":
            query = str(args.get("query", ""))
            hits = self.mem.recall(query, args.get("k", 5))
            return json.dumps({"ok": True, "results": [
                {"id": mid, "content": content[:200]} for mid, content in hits
            ]}, ensure_ascii=False)
        if name == "set_fact":
            key = str(args.get("key", "")).strip()
            val = str(args.get("value", "")).strip()
            if not key:
                return json.dumps({"ok": False, "error": "missing key"}, ensure_ascii=False)
            self.mem.upsert_fact(key, val, weight_delta=1.0)
            return json.dumps({"ok": True, "set": {"key": key, "value": val}}, ensure_ascii=False)
        if name == "get_facts":
            facts = self.mem.list_facts(20)
            return json.dumps({"ok": True, "facts": [
                {"key": k, "value": v, "weight": w} for (k, v, w) in facts
            ]}, ensure_ascii=False)
        # User-taught skill:
        skill = self.mem.get_skill(name)
        if not skill:
            return json.dumps({"ok": False, "error": f"No such tool/skill: {name}"}, ensure_ascii=False)
        _, code, approved, _usage = skill
        if not approved and not self.cfg.allow_dangerous_skills:
            return json.dumps({"ok": False, "error": f"Skill '{name}' not approved"}, ensure_ascii=False)
        # Run the skill code
        return self.skills.run(name, code, args)

    def _help_text(self) -> str:
        # New help output: friendly brief guide
        return textwrap.dedent("""\
        **Hello!** I'm *Alice*, your continuously-learning AI assistant. Here’s how to use me:
        - **Just chat with me:** You can talk to me naturally, ask questions, or give instructions.
        - **Teach new skills:** Use `/teach <name>` with a Python code block to teach me a new skill. (I’ll save and remember it!)
        - **Run skills:** After teaching, use `/run <name> { ... }` to execute that skill with JSON arguments.
        - **Ingest knowledge:** Use `/ingest <file>` to add a text/markdown file to my memory. I can then **search** it with `/recall <keywords>`.
        - **Memory recall:** Use `/recall <query>` to search our past conversations and any ingested content for relevant info.
        - **Feedback:** Mark my last answer with `/good` or `/bad` to help me learn from it.

        *Tip:* Type `/help` anytime to see this guide again. Enjoy exploring with Alice!
        """).strip()

    def _cmd_teach(self, raw: str) -> str:
        # Expect format: /teach name ```python\n<code>```
        m = re.search(r"/teach\s+(?P<name>[A-Za-z0-9_\-]+)\s+```python\n(?P<code>.*?)```", raw, re.DOTALL)
        if not m:
            return "Usage: /teach <name> ```python\n<code defining skill_main(**kwargs)>```"
        name = m.group("name")
        code = m.group("code").strip()
        self.mem.add_skill(name, code, approved=True)
        return f"- Skill '{name}' saved and approved! Try `/run {name} {{...}}` to test it."

    def _cmd_run(self, raw: str) -> str:
        # Expect format: /run name {json}
        m = re.search(r"/run\s+(?P<name>[A-Za-z0-9_\-]+)\s+(?P<json>{.*})\s*$", raw, re.DOTALL)
        if not m:
            return "Usage: /run <name> {json-args}"
        name = m.group("name")
        try:
            args = json.loads(m.group("json"))
        except json.JSONDecodeError as e:
            return f"- Bad JSON args: {e}"
        # If calling a built-in tool directly via /run, handle it:
        if name in ("calc", "recall", "set_fact", "get_facts"):
            return self._run_tool(name, args)
        # Otherwise, look for a taught skill
        skill = self.mem.get_skill(name)
        if not skill:
            return f"- No such skill: {name}"
        _, code, approved, _ = skill
        if not approved and not self.cfg.allow_dangerous_skills:
            return f"- Skill '{name}' is not approved yet."
        return self.skills.run(name, code, args)

    def _cmd_ingest(self, raw: str) -> str:
        # Expect format: /ingest <path>
        m = re.search(r"/ingest\s+(?P<path>.+)$", raw)
        if not m:
            return "Usage: /ingest <file-or-directory-path>"
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
                # Store the file content as a system message (so it’s searchable via recall)
                self.mem.add_message("system", f"[ingested:{os.path.basename(fp)}]\n{text}")
                count += 1
            except Exception as e:
                print(f"! failed to ingest {fp}: {e}")
        return f"- Ingested {count} file(s). You can now /recall to search their content."

    def _cmd_train(self, raw: str) -> str:
        cmd = raw.strip().split()
        if len(cmd) >= 2 and cmd[1] == "status":
            # Show whether train_mode is on and dataset file size
            size = 0
            try:
                p = os.path.join(self.cfg.train_dir, "alice_sft.jsonl")
                size = os.path.getsize(p) if os.path.exists(p) else 0
            except Exception:
                size = 0
            status = "on" if self.cfg.train_mode else "off"
            return f"- train_mode={status}, dataset_bytes={size}"
        if len(cmd) >= 2 and cmd[1] == "now":
            try:
                self.trainer._snapshot_dataset()
                self.trainer._maybe_train_lora()
            except Exception as e:
                return f"- Train attempt failed: {e}"
            # Provide user guidance after training attempt
            ds_size = 0
            try:
                ds_size = os.path.getsize(self.trainer.dataset_path) if os.path.exists(self.trainer.dataset_path) else 0
            except Exception:
                ds_size = 0
            if ds_size == 0:
                return "- Train attempt completed. (No data to train yet — dataset is empty.)"
            if self.cfg.train_mode and os.environ.get("ALICE_LORA_BASE"):
                return "- Train attempt completed. Dataset saved, and a LoRA fine-tune was attempted."
            else:
                return "- Train attempt completed. Dataset saved (no fine-tune since training mode off or base model not set)."
        return "Usage: /train status | /train now"

    def shutdown(self):
        # Stop reflection and training threads and close DB
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.reflect_thread.join(timeout=2)
        self.trainer.shutdown()
        self.mem.shutdown()

# ----------------------- CLI ENTRY POINT -----------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alice — a continuously-learning AI agent")
    ap.add_argument("--db", default="alice.db", help="SQLite database file path")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model name or tag (e.g. llama3.2:3b)")
    ap.add_argument("--ollama-host", default="127.0.0.1", help="Ollama server hostname")
    ap.add_argument("--ollama-port", default=11434, type=int, help="Ollama server port")
    ap.add_argument("--temperature", default=0.4, type=float, help="LLM generation temperature")
    ap.add_argument("--top-p", default=0.95, type=float, help="LLM top-p sampling")
    ap.add_argument("--top-k", default=40, type=int, help="LLM top-k sampling")
    ap.add_argument("--num-ctx", default=4096, type=int, help="LLM context window size (tokens)")
    ap.add_argument("--reflect", default=600, type=int, help="Seconds between background reflections")
    ap.add_argument("--allow-dangerous-skills", action="store_true", help="Allow taught skills to use unrestricted builtins")
    ap.add_argument("--no-embed", action="store_true", help="Disable semantic embeddings for memory")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Embedding model for semantic memory")
    ap.add_argument("--train-mode", action="store_true", help="Enable background dataset logging and LoRA fine-tuning attempts")
    ap.add_argument("--train-interval", default=3600, type=int, help="Seconds between automatic training attempts")
    ap.add_argument("--autonomous-tools", action="store_true", help="Allow Alice to call tools autonomously in conversation")
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
        autonomous=args.autonomous_tools
    )
    alice = Alice(cfg)

    def handle_sigint(sig, frame):
        print("\n[shutting down]")
        alice.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    print("Alice is awake! Type /help for a quick guide. (Ctrl+C to exit.)\n")
    # Main REPL loop
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            break
        if not user_input:
            continue

        # Run the response generation in a separate thread to allow spinner
        result: Optional[str] = None
        done_event = threading.Event()

        def generate_response():
            nonlocal result
            result = alice.handle_user(user_input)
            done_event.set()

        t = threading.Thread(target=generate_response)
        t.daemon = True
        t.start()
        # Spinner loop
        spinner = ["|", "/", "-", "\\"]
        idx = 0
        sys.stdout.write("Alice> ")
        sys.stdout.flush()
        while not done_event.wait(0.1):
            # animate spinner
            sys.stdout.write(spinner[idx] + "\b")
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
        t.join()
        # Clear spinner character and print the result
        sys.stdout.write("\b")  # erase the last spinner char
        print(result if result is not None else "")
    # End of loop

    alice.shutdown()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
