# Alice — A Continuously‑Learning Local Agent
![alice-in-wonderland-gif-2](https://github.com/user-attachments/assets/d29695db-77d6-4242-a19c-f302b900b5d8)


## Why Alice
- local‑first assistant you fully control
- model‑agnostic via Ollama
- durable memory and searchable recall
- safe tool use with an opt‑in Python skill sandbox
- optional autonomous mode (“Darkmode”) for multi‑step tool use
- self‑training hooks for ongoing improvement
- pentest mode with allowlists, evidence logging, and gated exploitation
- NOAA/NWS weather via api.weather.gov (opt‑in web access)

---

## Quick Start

### Prerequisites
- Python 3.9+ and SQLite (bundled with Python)
- Ollama installed locally with at least one model pulled
- Linux/macOS terminal familiarity
- optional (weather)
  - internet access enabled in Alice (`/web on`) and the `requests` library installed
- optional (pentesting)
  - Nmap, Nikto, Nuclei, NetworkManager (for wifi surveys)
  - OWASP ZAP running locally (GUI or daemon) if you want automated web scanning
  - Metasploit Framework with RPC enabled, and the Python client `pymetasploit3`

### Install core dependencies
- pull one or more Ollama models
  - `ollama pull llama3.2:3b`
  - `ollama pull gpt-oss:20b`
- verify Ollama is running (`ollama serve` if not already started)

### Run Alice
- basic
  - `python3 alice.py --model llama3.2:3b`
- if your file name includes a patch tag, use that
  - `python3 alice-pX.Y.Z.py --model llama3.2:3b`
- first steps
  - `/help`
  - `/darkmode on` to allow autonomous tool calls
  - `/model list` to see installed models

---

## Core Concepts

### Modes
- normal
  - tools run only when you call them
  - full‑sentence explanations
- darkmode
  - autonomous tool calls enabled
  - terse, tactical output
  - toggle with `/darkmode on|off|status` or `--autonomous-tools` launch flag

### Memory
- durable facts live in a local SQLite DB
- short‑term chat log and optional embeddings for semantic recall
- commands
  - `/facts` — show top remembered facts
  - `/recall <query>` — search past content
  - `set_fact` / `get_facts` available to tools

### Models (Ollama)
- list and switch at runtime
  - `/model list`
  - `/model info`
  - `/model set <tag>`
- decoding and context settings can be tuned via CLI flags

### Skills API (Python)
- teach Alice a safe, stdlib‑only Python function
- pattern
  - `/teach my_tool \\`\\`\\`python\ndef skill_main(**kwargs):\n    # ...\n    return {\"ok\": True, \"data\": 123}\n\\`\\`\\``
  - `/run my_tool {"k": "v"}`
- sandbox
  - stdlib only, no raw OS/network/file I/O, no `subprocess`, no `eval/exec`
  - promote to “dangerous” only when you fully understand the risks

### Web Access & Browsing
- disabled by default for privacy
- enable
  - `/web on`
- disable
  - `/web off`
- tool
  - `/run browse {"url":"https://example.com"}` — fetches and summarizes page text (requires `requests`)

### NOAA Weather (NWS api.weather.gov)
- purpose
  - quick official forecast by latitude/longitude, with hourly or period summaries
- requirements
  - `/web on` (internet enabled) and the `requests` library installed
- usage
  - shortcut
    - `/weather <lat,lon>` — e.g., `/weather 38.8977,-77.0365`
    - `/weather <lat,lon> hourly` — hourly gridpoint forecast
  - tool form
    - `/run noaa_weather {"lat":38.8977,"lon":-77.0365}`
    - `/run noaa_weather {"lat":38.8977,"lon":-77.0365,"hourly":true}`
- output
  - `location` (city, state when available) and a trimmed list of forecast periods with name, start/end, temperature (+ unit), wind, and short/detailed descriptions

### Self‑Training
- snapshot recent dialog into a local SFT dataset
  - `/train now` or `/train status`
- autonomous improvement loop
  - `/darkvision start|status|stop`

---
![darkvision](https://github.com/user-attachments/assets/3318379d-2e4f-4743-bacf-12efb2f10970)

## Commands (Cheatsheet)

- `/help` — show this help
- `/facts` — list top remembered facts
- `/recall <query>` — search past chat/content
- `/good` | `/bad` — reward/punish last answer
- `/teach <name> \\`\\`\\`python ... \\`\\`\\`` — add a Python skill with `skill_main(**kwargs)`
- `/run <name> {json}` — run a built‑in tool or an approved skill
- `/ingest <path>` — ingest `.txt`/`.md` files recursively
- `/train status|now` — snapshot dataset for fine‑tuning workflows
- `/darkvision start|status|stop` — autonomous self‑improvement loop
- `/darkmode on|off|status` — toggle autonomous tool use + terse style
- `/model list|info|set <tag>` — list/info/set Ollama models
- `/embed on|off|model <name>` — enable/disable embeddings or set embed model
- `/web on|off` — toggle browsing/internet access
- `/weather <lat,lon> [hourly]` — NOAA forecast shortcut (uses `noaa_weather` tool)
- `/pentest on|off|status` — enable pentest mode (see below)
- `/pentest allow <target>` — add an IP or domain to the allowlist
- `/pentest revoke <target>` — remove from allowlist
- `/pentest list` — show allowlist
- `/exploit arm <target> <eng_id>` — create a short‑lived arming token
- `/exploit confirm <token> I HAVE AUTHORIZATION FOR <target> [<eng_id>]` — confirm authorization
- `/exploit on|off|status` — toggle exploit mode (requires arm + confirm)
- `/doctor` — dependency diagnostics

---

## Configuration & Flags

- `--db alice.db` — SQLite path
- `--model <tag>` — default Ollama model
- `--ollama-host 127.0.0.1` / `--ollama-port 11434`
- `--num-ctx 4096` — prompt window
- `--temperature 0.4 --top-p 0.95 --top-k 40` — decoding
- `--no-embed` — disable semantic embeddings
- `--embed-model nomic-embed-text` — set embedding model
- `--train-mode` — allow background LoRA attempts when supported
- `--train-interval <sec>` — cadence for training checks
- `--darkvision-interval <sec>` — cadence for auto‑improvement cycles
- `--autonomous-tools` — start with Darkmode enabled
- `--allow-dangerous_skills` — lift the skill sandbox (not recommended)
- `--pentest-cli` — optional mini‑CLI for pentest wrappers

---

## Data Layout

- `alice.db`
  - `messages` — full dialog log
  - `messages_fts` — FTS5 index for fast search
  - `facts` — durable key/value memory with weights
  - `embeddings` — optional vectors per message
  - `skills` — user‑taught functions with approval flag and usage stats
- `train/`
  - `alice_sft.jsonl` — rolling supervised dataset from recent dialog
  - `darkvision.log` — auto‑improvement logs
  - `lora_adapter/` — optional adapter output if LoRA is run
- `evidence/`
  - timestamped artifacts from pentest tools and checks

---

## Pentest Mode (Opt‑In, Safety‑First)

Alice can assist with network, web application, and wireless security assessments. Pentest features are **off by default**, require an explicit allowlist of targets, and gate exploitation behind a two‑step confirmation.

### Safety Gates
- pentest mode must be ON to use scanners
- target must be present in the allowlist
- exploit mode is OFF by default
  - requires `/exploit arm` then `/exploit confirm` then `/exploit on`
- evidence is logged to `./evidence/…` for auditability

### Legal Notice
- only test systems you own or are explicitly authorized to assess
- unauthorized scanning or exploitation may be illegal and unethical
- by enabling exploit mode you confirm you have proper authorization

### Dependencies (Fedora examples)
- `sudo dnf install nmap nikto nuclei NetworkManager`
- optional
  - OWASP ZAP — run the ZAP daemon or GUI locally
  - Metasploit Framework with RPC enabled, plus `pip install pymetasploit3`

### Enabling Pentest Mode
- `/pentest on`
- `/pentest allow 192.168.1.50`
- `/pentest allow example.local`
- `/pentest list`

### Built‑In Pentest Tools
- `nmap_scan` — network ports/services
  - `/run nmap_scan {"target":"192.168.1.50","options":"-sV -T4"}`
- `nikto_scan` — basic web server checks
  - `/run nikto_scan {"url":"http://TARGET_HOST"}`
- `nuclei_scan` — template‑based CVE checks
  - `/run nuclei_scan {"target":"http://TARGET_HOST"}`
- `zap_scan` — OWASP ZAP spider + active scan via API
  - optional env: `ZAP_ADDR` (default `http://127.0.0.1:8080`), `ZAP_API_KEY`
  - `/run zap_scan {"url":"http://TARGET_HOST"}`
- `wifi_scan` — non‑invasive nearby network inventory (uses `nmcli`)
  - `/run wifi_scan {}`
- `msf_check` — Metasploit “check” action for modules
  - requires Metasploit RPC credentials in env: `MSF_RPC_HOST`, `MSF_RPC_PORT`, `MSF_RPC_USER`, `MSF_RPC_PASS`
  - `/run msf_check {"module":"auxiliary/scanner/http/title","options":{"RHOST":"192.168.1.50","RPORT":80}}`
- `msf_exploit` — Metasploit exploitation (guarded)
  - exploit mode must be enabled and arming token confirmed
  - `/run msf_exploit {"module":"exploit/multi/http/struts2_content_type","options":{"RHOST":"192.168.1.50","RPORT":8080}}`

### Exploit Mode Workflow
- arm
  - `/exploit arm 192.168.1.50 ENG-001`
- confirm
  - copy the exact line printed by the arm step:
  - `/exploit confirm <token> I HAVE AUTHORIZATION FOR 192.168.1.50 [ENG-001]`
- enable
  - `/exploit on`
- run an exploit module (example above)
- disable any time
  - `/exploit off`

---

## Typical Workflows

### Network Recon
- `/pentest on`
- `/pentest allow 192.168.1.50`
- `/run nmap_scan {"target":"192.168.1.50","options":"-sV -T4"}`
- interpret open ports and versions; chain to `msf_check` where appropriate

### Web App Assessment
- `/pentest on`
- `/pentest allow TARGET_HOST`
- `/run nikto_scan {"url":"http://TARGET_HOST"}`
- `/run nuclei_scan {"target":"http://TARGET_HOST"}`
- `/run zap_scan {"url":"http://TARGET_HOST"}`
- report summarizes issues with evidence paths under `evidence/`

### Wireless Survey
- `/pentest on`
- `/run wifi_scan {}`
- review SSIDs, security types, and signal strength

### Weather Check
- `/web on`
- `/weather 38.8977,-77.0365`
- optional hourly
  - `/weather 38.8977,-77.0365 hourly`

### Guided Exploitation (Authorized)
- follow the exploit workflow (arm → confirm → on)
- run `msf_exploit` with explicit module + options
- review evidence JSON to track outcomes and sessions

---

## Troubleshooting

- no models found
  - pull at least one Ollama model; ensure `ollama serve` is active
- timeouts or truncated replies
  - try a smaller model or ask for concise output
  - ensure the selected model fits your hardware
- embeddings missing
  - `/embed on` and ensure the embed model is installed
- pentest tool not found
  - run `/doctor` for a dependency summary
  - install missing packages using your OS package manager
- ZAP errors
  - start ZAP locally and set `ZAP_ADDR` or `ZAP_API_KEY` if needed
- Metasploit errors
  - ensure RPC is enabled and credentials exported to environment
- NOAA weather errors
  - `/web on`, ensure `requests` is installed, and pass numeric `lat` and `lon`

---

## Design Principles
- local, private, auditable
- minimal external dependencies; degrade gracefully
- make the safe path the default
- composable skills and tools
- evidence‑first workflows and clear confirmations for risky actions

---
<img width="1024" height="1024" alt="alice-ghidra" src="https://github.com/user-attachments/assets/42b80835-4810-4934-bcb5-07455f7c3c5b" />

## License
- GPL‑3.0

## Acknowledgments
- born to be a sturdy, private, always‑learning lab companion
![alice_darkmode](https://github.com/user-attachments/assets/74df3431-f5b2-4cc6-8dd9-d87a77ea3f33)
