# Alice p0.1.5 — Command & Control (alice-cnc.md)

- build: 0.1.5-fixed6
- date: 2025-08-31
- scope: operator-facing update sheet for running, controlling, and auditing Alice

## summary

# Alice 0.1.5-fixed6 — Updates

- version
  - bump: 0.1.5 → 0.1.5-fixed6
  - rationale: repair truncated/duplicated fragments and deliver a clean, runnable build

- core stability
  - rebuilt the chat router `handle_user()` to fix indentation and cut-off lines
  - consolidated duplicate argparse + CLI/spinner blocks into a single implementation
  - normalized version strings and help text across the app
  - added graceful SIGINT shutdown (persists memory; stops background threads)

- UX / CLI
  - new lightweight `_BarSpinner` with proper `flush()` and clean erase on stop
  - consistent “Alice p0.1.5 is awake. Type /help …” startup banner
  - added `/say "text"` convenience echo (handles quoted strings)

- model control (Ollama / model-agnostic)
  - `/model list` — enumerate installed models via backend
  - `/model info` — show context size, family, parameters
  - `/model set <tag>` — switch model and auto-tune `num_ctx` when known

- memory & recall
  - SQLite schema maintained; FTS5 virtual table kept
  - hybrid recall: embeddings (if enabled) + FTS + naive fallback
  - background embedding indexer thread with WAL + busy_timeout for stability
  - `/facts` shows top-weighted facts; `/recall <query>` returns best matches

- embeddings
  - toggle: `/embed on|off`
  - model: `/embed model <name>`
  - uses `/api/embeddings` when available; gracefully degrades if not

- safe skills sandbox
  - `/teach <name> ```python ... ````
    - requires `def skill_main(**kwargs): ...`
    - auto-approves but runs inside a restricted stdlib-only sandbox
  - `/run <name> {json}` executes built-ins or approved skills
  - blocked by default in skills: `os, sys, subprocess, socket, shutil, pathlib, requests, urllib, open, eval, exec, compile, ctypes, multiprocessing`
  - optional override at launch: `--allow-dangerous-skills` (not recommended)

- web access & NOAA weather
  - global toggle: `/web on|off` (off by default)
  - simple fetch tool via `/run browse {"url":"https://..."}` with HTML-to-text extraction
  - NOAA shortcut: `/weather <lat,lon> [hourly]`
    - uses `api.weather.gov/points → forecast/forecastHourly`
    - proper `User-Agent` header with contact email
    - friendly error messages for grid/timeout conditions

- pentest (detection-only, allowlisted)
  - global mode: `/pentest on|off|status`
  - allowlist targets/domains: `/pentest allow <target>` / `/pentest revoke <target>` / `/pentest list`
  - diagnostics: `/pentest doctor` (checks availability of `nmap`, `nikto`, `nuclei`, `nmcli`)
  - built-in wrappers (enumeration/detection only; no exploitation):
    - `/run nmap_scan {"target":"tgt","options":"-sV -T4"}`
    - `/run nikto_scan {"url":"https://tgt"}`
    - `/run nuclei_scan {"target":"https://tgt"}`
    - `/run wifi_scan {}`
  - OWASP ZAP API stub: `/run zap_scan {"url":"https://tgt"}` (starts scan; returns alerts summary when available)
  - Metasploit integration intentionally restricted to a stubbed `msf_check` (no exploitation)
  - runs only when pentest mode is on AND target is allowlisted

- autonomy
  - darkmode (tool autonomy + terse style): `/darkmode on|off|status`
    - supports up to 5 `<<call:NAME args='{{...}}'>>` tool steps per turn
  - darkvision (background self-improvement loop):
    - `/darkvision start|status|stop`
    - periodic dataset snapshots to `train/alice_sft.jsonl`
    - proposes tiny, safe skills based on conversation; tests in sandbox before saving

- training & datasets
  - `/train status|now` to snapshot current dialog pairs for SFT
  - stores to `train/alice_sft.jsonl` (overwrites each snapshot by design)

- install helper
  - `/install <package>` — pip-installs a Python package and returns result text

- response hygiene
  - strips “Thought:”/“Chain:” style meta-lines from model outputs
  - enforces concise, direct answers with your dash-bullet style

- security defaults
  - web access: OFF by default
  - pentest: OFF by default; detection-only; explicit allowlist required
  - skills: sandboxed by default
  - msf: exploitation disabled (stub only)

- files added/updated
  - `alice-p0.1.5.fixed6.py` — consolidated, compile-clean main
  - `alice-cnc.md` — operator “command & control” sheet (usage, workflows, safety)

- quick start
  - `python3 alice-p0.1.5.fixed6.py --model llama3.2:3b --enable-web`
  - try: `/weather 38.8977,-77.0365 hourly`
  - enable pentest safely: `/pentest on` → `/pentest allow example.com` → `/pentest doctor`
