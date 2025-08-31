# Alice — A Continuously‑Learning Local Agent

![alice-in-wonderland-gif-2](https://github.com/user-attachments/assets/9999b0dd-e02a-428f-a20f-22225cef4121)

 Alice (0.1.1)

Local, continuously‑learning AI with autonomous tools, skills sandbox, and model‑agnostic Ollama support — all packaged in a single `alice.py`.

- Status: this README is written for **Alice 0.1.1**
- Design goals: local first, durable memory, safe tool use, minimal dependencies, fast iteration

## What’s new in 0.1.1

- Unified main — Darkmode features are built into `alice.py` (no separate darkmode script)
- Model‑agnostic — works with **any** pulled Ollama model; switch at runtime with `/model set`
- `/darkmode` toggle — enables autonomous tool calls and a terse, tactical answer style
- Safer math — `/run calc` uses an AST safelist (no arbitrary `eval`)
- Smarter memory — embeddings can be turned on/off or re‑pointed with `/embed`
- Resilience — Ollama requests auto‑retry with short backoff; clearer errors and context‑length probing

## Quick start

- Prereqs
  - Python 3.9+ and SQLite (bundled with Python)
  - Ollama running locally and at least one model pulled
- Pull one or more models (examples)
  - `ollama pull gpt-oss:20b`
  - `ollama pull llama3.2:3b`
- Run Alice
  - `python3 alice.py --model gpt-oss:20b`
  - `python3 alice.py --model llama3.2:3b`
- First steps
  - Type `/help` for commands
  - Type `/darkmode on` to allow autonomous tool use
  - Type `/model list` to see all installed models

## Daily use patterns

- Regular chat — just type your question or instruction
- Darkmode
  - `/darkmode on` to enable autonomous tool calls and terse style
  - `/darkmode off` to require explicit tool use only
- Model switching
  - `/model list` to view installed models
  - `/model info` to inspect the current model’s basics (family, context length)
  - `/model set <tag>` to switch models live
- Memory & recall
  - `/facts` to see top remembered facts
  - `/recall <query>` to search past content (FTS + optional embeddings)
- Teach skills
  - `/teach my_tool ```python
<code defining skill_main(**kwargs)>```
  - `/run my_tool {"k": "v"}` to execute it
- Ingest notes & docs
  - `/ingest ./notes/` to ingest `.txt`/`.md` recursively
- Self‑training & auto‑improvement
  - `/train now` to snapshot a supervised dataset from recent dialog
  - `/darkvision start` to run periodic snapshots and propose safe micro‑skills

## Commands (cheatsheet)

- `/help` — show this help
- `/facts` — list top remembered facts
- `/recall <query>` — search past chat/content
- `/good` | `/bad` — reward/punish last answer
- `/teach <name> ```python ... ``` ` — add a Python skill with `skill_main(**kwargs)`
- `/run <name> {json}` — run a built‑in tool or an approved skill
- `/ingest <path>` — ingest `.txt`/`.md`
- `/train status|now` — snapshot dataset and optionally fine‑tune if enabled
- `/darkvision start|status|stop` — autonomous self‑improvement loop
- `/darkmode on|off|status` — toggle autonomous tool use + terse style
- `/model list|info|set <tag>` — list/info/set Ollama models
- `/embed on|off|model <name>` — enable/disable embeddings or set embed model

## Built‑in tools

- `calc(expr)` — safe arithmetic, common math functions
- `recall(query, k)` — hybrid search over history (FTS + optional semantic)
- `set_fact(key, value)` — persist a durable fact with weight
- `get_facts()` — return current top‑weighted facts

## Configuration & flags

- Core
  - `--db alice.db` — SQLite path
  - `--model gpt-oss:20b` — default model (use any pulled tag)
  - `--ollama-host 127.0.0.1` — Ollama host
  - `--ollama-port 11434` — Ollama port
  - `--num-ctx 4096` — prompt window (auto‑set when possible via `/model set`)
  - `--temperature 0.4 --top-p 0.95 --top-k 40` — decoding options
- Memory
  - `--no-embed` — disable semantic embeddings
  - `--embed-model nomic-embed-text` — embedding model tag
- Learning
  - `--train-mode` — allow background LoRA attempts when environment supports it
  - `--train-interval 3600` — seconds between training checks
  - `--darkvision-interval 900` — seconds between auto‑improvement cycles
- Mode
  - `--autonomous-tools` — start with Darkmode enabled
  - `--allow-dangerous-skills` — lift the skill sandbox restrictions (not recommended)

## Data layout

- `alice.db`
  - `messages` — full dialog log
  - `messages_fts` — FTS5 index for fast text search
  - `facts` — durable key/value memory with weights
  - `embeddings` — optional semantic vectors per message
  - `skills` — user‑taught functions with approval flag and usage stats
- `train/`
  - `alice_sft.jsonl` — rolling supervised dataset from recent dialog
  - `darkvision.log` — auto‑improvement cycle logs
  - `lora_adapter/` — optional adapter output if LoRA is run

## Skills API (sandboxed)

- Write skills in Python using **only** the standard library
- Define exactly `skill_main(**kwargs)` and return JSON‑serializable data
- Disallowed by default in skills: OS/network/file I/O, subprocess, `eval/exec`, unsafe imports
- Promote to “dangerous” mode only if you fully understand the risks

## Darkmode vs Normal mode

- Normal
  - No autonomous tool use
  - Tools only run when explicitly called via `/run` or when you instruct Alice to use them
- Darkmode
  - Autonomous tool calling enabled via the `<<call:NAME args='{}'>>` pattern
  - Terser style aimed at rapid, actionable answers

## Safety & limitations

- Alice never executes shell commands; it only runs built‑in tools or approved skills
- Embeddings and self‑training stay on the local machine
- Large models require significant RAM/VRAM on the Ollama side
- Termux‑specific patches are not required for 0.1.1 (general usage only)

## Troubleshooting

- Ollama connection refused
  - Ensure the Ollama service is running and the host/port match Alice flags
- “No models found” on `/model list`
  - Pull at least one model with `ollama pull <tag>`
- Timeouts or truncated replies
  - Try a smaller model or reduce `num_predict` indirectly by asking for more concise output
  - Check that the selected model fits the available memory
- Embeddings don’t appear
  - Use `/embed on` and make sure the chosen embed model is installed
- Dataset not growing
  - Use `/train now` after a conversation with user/assistant pairs to snapshot data

## FAQ

- Does Alice run offline?
  - Yes for memory, skills, and logic; Ollama must be reachable locally for model inference
- Which models are best?
  - Start small to validate, then move up to larger models if your hardware permits
- Can Alice switch models mid‑session?
  - Yes, `/model set <tag>` switches live and probes context length where available
- How do I disable autonomous behavior?
  - `/darkmode off` or avoid the `--autonomous-tools` flag at launch

---

This README targets **Alice 0.1.1**.

## License
- **GNU General Public License v3.0 (GPL‑3.0)**
- You should have received a copy of the GNU General Public License along with this program. If not, see the official GNU GPL‑3.0 text.

**Note** — the source header in `alice.py` may still mention a different license in comments from early drafts. The project’s governing license is GPL‑3.0; update the file header to match.

---

## Acknowledgments
- Born to be a sturdy, private, always‑learning lab companion.

