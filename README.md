# Alice — A Continuously‑Learning Local Agent

![alice-in-wonderland-gif-2](https://github.com/user-attachments/assets/9999b0dd-e02a-428f-a20f-22225cef4121)



*A small, self‑hosted, interactable AI that learns*

---

## Highlights
- Local, private, and offline‑capable via a small LLM (default: Ollama + `llama3.2:3b`)
- True “always‑on” learning using memory‑centric methods: episodic logs, semantic facts, and background reflection
- Retrieval over its own history with SQLite FTS5 (no external vector DB required)
- Interactable REPL with friendly commands and human‑in‑the‑loop feedback
- Hot‑pluggable skills you can teach at runtime (`/teach`), sandboxed by default
- Lightweight single‑file app (`alice.py`) with a single SQLite database (`alice.db`)

---

## Philosophy
- Continual learning in a practical agent is best achieved with memory + reflection + retrieval rather than trying to fine‑tune model weights live.
- Human feedback is the fastest way to shape behavior; Alice records rewards with `/good` and `/bad`.
- Simplicity beats complexity. One file. One database. No exotic dependencies.

---

## Requirements
- Python 3.10+
- SQLite with FTS5 (bundled on most systems)
- Ollama running locally and a small model pulled (e.g., `llama3.2:3b`)

---

## Install
- Ensure Python is available on your path.
- Install and start **Ollama**.
- Pull a model in a terminal:
  ```bash
  ollama pull llama3.2:3b
  ```
- Place `alice.py` in a working directory where you are comfortable storing `alice.db`.

---

## Quickstart
```bash
python3 alice.py --model llama3.2:3b
```
- Alice will greet you in a REPL.
- Type `/help` to view commands.

Common flags:
- `--reflect 60` — reflect every 60 seconds while you shape behaviors
- `--db /path/to/alice.db` — choose a custom database path
- `--ollama-host 127.0.0.1 --ollama-port 11434` — point to a different host/port if needed
- `--allow-dangerous-skills` — permit skills to import `os/sys/sqlite3` (off by default)

---

## Commands
- `/help` — show help
- `/facts` — list top remembered facts (semantic memory)
- `/recall <query>` — search past chat and ingested content
- `/good` | `/bad` — record reward feedback
- `/teach <name> ```python\n...``` ` — register a new skill at runtime
- `/run <name> {json}` — execute a skill with arguments
- `/ingest <path>` — index `.txt` and `.md` files for retrieval

---

## Teaching Skills (Plugins)
**Example**
```text
/teach summarize ```python
def skill_main(text: str, max_lines: int = 6):
    return "\n".join(line.strip() for line in text.splitlines()[:max_lines])
```
/run summarize {"text": "line1\nline2\nline3", "max_lines": 2}
```

Security model for skills:
- Default sandbox exposes a tiny set of safe builtins plus `math`, `statistics`, `json`, `time`, `random`.
- No filesystem or OS access unless you launch Alice with `--allow-dangerous-skills`.

---

## Memory Model
- **Episodic memory** — every message is stored in `messages`, searchable via FTS.
- **Semantic memory (facts)** — distilled key–value preferences and traits in `facts`, with weights.
- **Reflections** — periodic summaries stored in `reflections` to keep context compact and durable.

You can view what Alice believes with:
```text
/facts
/recall <keywords>
```

---

## Continual Learning Loop
- While Alice runs, a background thread executes every `--reflect` seconds (default 600).
- It summarizes recent conversation into concise bullets and a one‑paragraph “Reflection”.
- New durable bullets are inserted or upweighted in `facts`.
- On every turn, Alice retrieves:
  - Top‑weighted facts
  - Similar prior content from FTS
  - Recent dialogue snippets

Result: answers that adapt as you talk and continue to improve over time.

---

## Retrieval & Ingestion
- `/ingest` accepts directories or single files with `.txt` or `.md`.
- Content is stored in the database and becomes searchable with `/recall`.

---

## Configuration
Run `python3 alice.py --help` for the full list. Key options:
- `--model` — model tag for Ollama (e.g., `llama3.2:3b`)
- `--reflect` — seconds between reflections
- `--recall-k` — number of similar messages to retrieve (code default is 12)
- `--max-history` — dialogue turns to include in prompts (code default is 14)
- `--temperature`, `--top-p` — generation settings

---

## Persistence & Files
- The SQLite database (`alice.db` by default) sits beside `alice.py` unless you pass `--db`.
- WAL mode is enabled for durability and concurrency.
- You can back up or move Alice simply by copying the database file.

---

## Troubleshooting
**Alice prints `[LLM error: Connection refused]`**
- Start Ollama and ensure the host/port matches Alice’s flags.

**FTS errors or no search hits**
- Your Python SQLite may lack FTS5. Alice will fall back to a simple scorer, but full‑text search is recommended.

**Timezone warnings**
- If you see `utcnow()` deprecation warnings, you can swap to `dt.datetime.now(dt.timezone.utc).isoformat()` in the few timestamp lines.

**Long outputs or slow replies**
- Use a smaller model, lower `--max-history`, or raise `--reflect` to reflect less often.

---

## Design Notes
- RAG‑style retrieval is used to ground responses in Alice’s own history.
- Reflections provide a compact, durable summary that turns ephemeral chat into reusable knowledge.
- Memory weights bias which facts are surfaced most often.
- Skills are pure functions (`skill_main(**kwargs)`) so you can compose tools safely.

---

## Roadmap (nice‑to‑haves)
- Optional vector store drop‑in for hybrid search while keeping FTS as a default
- Streaming online learners (if you install a library that supports incremental updates)
- Scheduled prompts and reminders
- GUI shell on top of the REPL

---

## FAQ
**Does Alice fine‑tune the model while running?**
- No. Alice learns by memory, retrieval, reflection, and your feedback. This is safer, faster, and reversible.

**Can Alice access the internet or my files?**
- Not by default. She only reads what you ingest or type. Skills can add capabilities.

**How do I wipe memory?**
- Delete `alice.db` or pass a new path with `--db`.

---

```
python3 alice_darkmode.py \
  --model llama3.2:3b \
  --embed-model nomic-embed-text \
  --train-mode \
  --reflect 600
```

---

## License
- **GNU General Public License v3.0 (GPL‑3.0)**
- You should have received a copy of the GNU General Public License along with this program. If not, see the official GNU GPL‑3.0 text.

**Note** — the source header in `alice.py` may still mention a different license in comments from early drafts. The project’s governing license is GPL‑3.0; update the file header to match.

---

## Acknowledgments
- Born to be a sturdy, private, always‑learning lab companion.

