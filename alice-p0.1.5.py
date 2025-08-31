#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alice p0.1.5 — Pentest Assistant (Standalone)
- Fedora-friendly, free/open-source tool wrappers
- Pentest Mode + Target Allowlist
- Exploit Mode with two-step confirmation ("arming" + explicit phrase)
- Evidence logging
- Detection-first defaults; exploitation requires explicit enable

This is a focused CLI harness to accompany Alice 0.1.x while remaining usable alone.
It does NOT require an LLM; you interact via slash-commands.
Integrate later with alice.py by importing register_tools() and using run_tool().
"""

import os, sys, re, json, shlex, subprocess, datetime, ipaddress, uuid, base64
from pathlib import Path
from typing import Dict, Any, Optional

APP_NAME = "alice-p0.1.5"
STATE_PATH = Path.home() / f".{APP_NAME}-state.json"
EVIDENCE_DIR = Path.cwd() / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

BANNER = f"""\
{APP_NAME} — Pentest Assistant
- type /help for commands
- safety: pentest mode + allowlist; exploit mode requires two-step confirm
"""

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

DEFAULT_STATE = {
    "pentest_mode": False,
    "exploit_mode": False,
    "allowlist": [],           # list of targets (exact strings: ip or domain)
    "armed_token": None,       # dict: {"token": str, "target": str, "engagement": str, "expires": ts}
    "notes": {},
    "log": []
}

state = load_json(STATE_PATH, DEFAULT_STATE)

def log_event(kind: str, detail: Dict[str, Any]):
    entry = {"ts": now_iso(), "kind": kind, **detail}
    state["log"].append(entry)
    # keep last 1000
    if len(state["log"]) > 1000:
        state["log"] = state["log"][-1000:]
    save_json(STATE_PATH, state)

def die(msg: str):
    print(f"- error: {msg}")
    sys.exit(1)

# ---------------- Safety / Validation ----------------

DOMAIN_RE = re.compile(r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z0-9-]{1,63})+$")
def is_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        return False

def is_domain(s: str) -> bool:
    return bool(DOMAIN_RE.match(s))

def normalize_target(t: str) -> str:
    t = t.strip()
    # strip scheme if URL
    if "://" in t:
        t = t.split("://", 1)[1]
    # strip path
    t = t.split("/", 1)[0]
    return t

def require_pentest_enabled():
    if not state.get("pentest_mode"):
        raise RuntimeError("Pentest mode is OFF. Use /pentest on")

def require_target_allowed(target: str):
    target = normalize_target(target)
    if target not in state.get("allowlist", []):
        raise RuntimeError(f"Target '{target}' not in allowlist. /pentest allow {target}")

def require_exploit_enabled():
    if not state.get("exploit_mode"):
        raise RuntimeError("Exploit mode is OFF. Use /exploit arm + /exploit confirm + /exploit on")

# ---------------- Evidence ----------------

def write_evidence(tool: str, target: str, content: str, ext: str = "txt") -> Path:
    safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", target or "none")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    p = EVIDENCE_DIR / f"{ts}_{tool}_{safe_target}.{ext}"
    p.write_text(content, encoding="utf-8", errors="ignore")
    return p

# ---------------- Dependencies ----------------

def which(cmd: str) -> Optional[str]:
    from shutil import which as _w
    return _w(cmd)

def check_deps():
    deps = {
        "nmap": bool(which("nmap")),
        "nikto": bool(which("nikto")),
        "nuclei": bool(which("nuclei")),
        "nmcli": bool(which("nmcli")),  # for wifi scan
        "zap": bool(which("zap.sh")) or bool(which("zaproxy")),
        "pymetasploit3": _has_pymetasploit3()
    }
    print("- dependencies:")
    for k, v in deps.items():
        print(f"  - {k}: {'OK' if v else 'missing'}")
    return deps

def _has_pymetasploit3() -> bool:
    try:
        import pymetasploit3  # type: ignore
        return True
    except Exception:
        return False

# ---------------- Tool wrappers ----------------

def run_cmd(args, timeout=900) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)

def tool_nmap_scan(target: str, options: str = "-sV -T4") -> Dict[str, Any]:
    require_pentest_enabled(); require_target_allowed(target)
    args = ["nmap"] + shlex.split(options) + [target]
    cp = run_cmd(args)
    out = cp.stdout
    path = write_evidence("nmap", target, out, "txt")
    # crude parse
    open_ports = []
    for line in out.splitlines():
        # e.g., "80/tcp open  http    Apache httpd 2.4.41"
        if re.match(r"^\d+/(tcp|udp)\s+open", line):
            parts = line.split()
            port_proto = parts[0]
            service = parts[2] if len(parts) > 2 else ""
            info = " ".join(parts[3:]) if len(parts) > 3 else ""
            open_ports.append({"port_proto": port_proto, "service": service, "info": info})
    result = {"tool":"nmap","target":target,"options":options,"open_ports":open_ports,"evidence":str(path)}
    log_event("nmap_scan", result)
    return result

def tool_nikto_scan(url: str) -> Dict[str, Any]:
    require_pentest_enabled(); require_target_allowed(url)
    args = ["nikto", "-ask", "no", "-h", url]
    cp = run_cmd(args, timeout=3600)
    out = cp.stdout
    path = write_evidence("nikto", url, out, "txt")
    # minimal summary
    findings = [ln.strip() for ln in out.splitlines() if "OSVDB" in ln or "CVE-" in ln or "X-" in ln or "/admin" in ln]
    result = {"tool":"nikto","url":url,"findings":findings[:50],"evidence":str(path)}
    log_event("nikto_scan", result)
    return result

def tool_nuclei_scan(target: str, templates: Optional[str]=None) -> Dict[str, Any]:
    require_pentest_enabled(); require_target_allowed(target)
    out_file = EVIDENCE_DIR / f"nuclei_{uuid.uuid4().hex}.jsonl"
    args = ["nuclei", "-u", target, "-jsonl", "-o", str(out_file)]
    if templates:
        args += ["-t", templates]
    cp = run_cmd(args, timeout=7200)
    # parse jsonl
    vulns = []
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                j = json.loads(line)
                vulns.append({"template": j.get("template-id"), "info": j.get("info", {}), "matcher": j.get("matcher-name")})
            except Exception:
                continue
    path = write_evidence("nuclei", target, "\n".join([json.dumps(v) for v in vulns]), "jsonl")
    result = {"tool":"nuclei","target":target,"count":len(vulns),"evidence":str(path)}
    log_event("nuclei_scan", result)
    return result

def tool_wifi_scan() -> Dict[str, Any]:
    require_pentest_enabled()
    if not which("nmcli"):
        raise RuntimeError("nmcli missing. Install NetworkManager.")
    cp = run_cmd(["nmcli", "-f", "SSID,SECURITY,SIGNAL,CHAN,FREQ", "dev", "wifi", "list", "--rescan", "yes"], timeout=120)
    out = cp.stdout
    path = write_evidence("wifi", "local", out, "txt")
    # parse simple table
    nets = []
    for ln in out.splitlines()[1:]:
        parts = [p.strip() for p in re.split(r"\s{2,}", ln.strip())]
        if len(parts) >= 3:
            nets.append({"ssid": parts[0], "security": parts[1], "signal": parts[2]})
    result = {"tool":"wifi_scan","nets":nets[:100],"evidence":str(path)}
    log_event("wifi_scan", result)
    return result

# ---- ZAP ----

def tool_zap_scan(url: str, zap_addr: str="http://127.0.0.1:8080", api_key: Optional[str]=None) -> Dict[str, Any]:
    require_pentest_enabled(); require_target_allowed(url)
    # Simple passive+active scan via REST
    import urllib.parse, urllib.request, time as _t, json as _json
    def zap_call(path, params=None):
        if params is None: params = {}
        if api_key: params["apikey"] = api_key
        qs = urllib.parse.urlencode(params)
        full = f"{zap_addr}{path}?{qs}"
        with urllib.request.urlopen(full) as r:
            return _json.loads(r.read().decode("utf-8", "ignore"))
    # start spider
    spider = zap_call("/JSON/spider/action/scan", {"url": url})
    sid = spider.get("scan")
    # wait spider
    while True:
        stat = zap_call("/JSON/spider/view/status", {"scanId": sid})
        if stat.get("status") == "100":
            break
        print(f"- zap spider {stat.get('status')}%")
        _t.sleep(1)
    # passive alerts so far
    alerts = zap_call("/JSON/core/view/alerts", {"baseurl": url, "start": 0, "count": 9999}).get("alerts", [])
    # active scan
    ascan = zap_call("/JSON/ascan/action/scan", {"url": url})
    aid = ascan.get("scan")
    while True:
        stat = zap_call("/JSON/ascan/view/status", {"scanId": aid})
        if stat.get("status") == "100":
            break
        print(f"- zap ascan {stat.get('status')}%")
        _t.sleep(2)
    alerts2 = zap_call("/JSON/core/view/alerts", {"baseurl": url, "start": 0, "count": 9999}).get("alerts", [])
    all_alerts = alerts + alerts2
    sev_counts = {}
    for a in all_alerts:
        sev = a.get("risk", "info").lower()
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    ev = write_evidence("zap", url, json.dumps(all_alerts, indent=2), "json")
    res = {"tool":"zap_scan","url":url,"alerts":sev_counts,"evidence":str(ev)}
    log_event("zap_scan", res)
    return res

# ---- Metasploit (check + exploit) ----

def _msf_connect(host: str=None, port: int=None, user: str=None, password: str=None):
    if not _has_pymetasploit3():
        raise RuntimeError("pymetasploit3 not installed. pip install pymetasploit3")
    from pymetasploit3.msfrpc import MsfRpcClient  # type: ignore
    host = host or os.environ.get("MSF_RPC_HOST", "127.0.0.1")
    port = port or int(os.environ.get("MSF_RPC_PORT", "55552"))
    user = user or os.environ.get("MSF_RPC_USER", "msf")
    password = password or os.environ.get("MSF_RPC_PASS", "")
    if not password:
        raise RuntimeError("MSF_RPC_PASS not set.")
    client = MsfRpcClient(password, server=host, port=port, username=user, ssl=False)
    return client

def tool_msf_check(module: str, options: Dict[str, Any]) -> Dict[str, Any]:
    require_pentest_enabled()
    target = options.get("RHOST") or options.get("RHOSTS") or ""
    if target:
        require_target_allowed(target)
    client = _msf_connect()
    mod = client.modules.use(module.split('/')[0], '/'.join(module.split('/')[1:]))
    for k,v in options.items():
        mod[k] = v
    res = mod.execute("check")
    ev = write_evidence("msf_check", target or "none", json.dumps(res, indent=2), "json")
    out = {"tool":"msf_check","module":module,"options":options,"result":res,"evidence":str(ev)}
    log_event("msf_check", out)
    return out

def tool_msf_exploit(module: str, options: Dict[str, Any]) -> Dict[str, Any]:
    require_pentest_enabled(); require_exploit_enabled()
    target = options.get("RHOST") or options.get("RHOSTS") or ""
    if target:
        require_target_allowed(target)
    # log arming context
    if not state.get("armed_token"):
        raise RuntimeError("Exploit arming token missing. Use /exploit arm + /exploit confirm.")
    client = _msf_connect()
    mod = client.modules.use(module.split('/')[0], '/'.join(module.split('/')[1:]))
    for k,v in options.items():
        mod[k] = v
    res = mod.execute("exploit")
    ev = write_evidence("msf_exploit", target or "none", json.dumps(res, indent=2), "json")
    out = {"tool":"msf_exploit","module":module,"options":options,"result":res,"evidence":str(ev)}
    log_event("msf_exploit", out)
    return out

TOOLS = {
    "nmap_scan": tool_nmap_scan,
    "nikto_scan": tool_nikto_scan,
    "nuclei_scan": tool_nuclei_scan,
    "wifi_scan": lambda **kw: tool_wifi_scan(),
    "zap_scan": tool_zap_scan,
    "msf_check": tool_msf_check,
    "msf_exploit": tool_msf_exploit
}

def register_tools() -> Dict[str, Any]:
    """Import hook for alice.py integration; returns tool mapping."""
    return TOOLS

# ---------------- CLI ----------------

HELP = """\
/help                          — show this
/doctor                        — check dependencies
/pentest on|off|status         — toggle pentest mode
/pentest allow <target>        — add target to allowlist (ip or domain or url)
/pentest revoke <target>       — remove target from allowlist
/pentest list                  — show allowlist
/exploit arm <target> <eng_id> — produce arming token for target+engagement
/exploit confirm <token> I HAVE AUTHORIZATION FOR <target> [<eng_id>]
/exploit on|off|status         — toggle exploit mode (requires arm+confirm)
/run <tool> <json>             — run a tool (nmap_scan, nikto_scan, nuclei_scan, zap_scan, msf_check, msf_exploit)
/log                           — show recent actions
/quit                          — exit
"""

def cmd_pentest(args: list):
    if not args:
        print("- usage: /pentest on|off|status|allow|revoke|list"); return
    op = args[0]
    if op == "on":
        state["pentest_mode"] = True; save_json(STATE_PATH, state)
        print("- pentest mode: ON")
    elif op == "off":
        state["pentest_mode"] = False; save_json(STATE_PATH, state)
        print("- pentest mode: OFF")
    elif op == "status":
        print(f"- pentest mode: {'ON' if state.get('pentest_mode') else 'OFF'}")
    elif op == "allow" and len(args)>=2:
        tgt = normalize_target(" ".join(args[1:]))
        if tgt not in state["allowlist"]:
            state["allowlist"].append(tgt); save_json(STATE_PATH, state)
        print(f"- allowlist add: {tgt}")
    elif op == "revoke" and len(args)>=2:
        tgt = normalize_target(" ".join(args[1:]))
        if tgt in state["allowlist"]:
            state["allowlist"].remove(tgt); save_json(STATE_PATH, state)
        print(f"- allowlist remove: {tgt}")
    elif op == "list":
        lst = state.get("allowlist", [])
        print("- allowlist:"); 
        for x in lst: print(f"  - {x}")
    else:
        print("- unknown /pentest subcommand")

def cmd_exploit(args: list):
    if not args:
        print("- usage: /exploit arm|confirm|on|off|status"); return
    op = args[0]
    if op == "status":
        print(f"- exploit mode: {'ON' if state.get('exploit_mode') else 'OFF'}")
        print(f"- armed token: {'present' if state.get('armed_token') else 'none'}")
        return
    if op == "on":
        if not state.get("armed_token"):
            print("- error: must /exploit arm + /exploit confirm first")
            return
        # ensure not expired
        if state["armed_token"]["expires"] < time.time():
            print("- error: arming token expired; arm again")
            return
        state["exploit_mode"] = True; save_json(STATE_PATH, state)
        print("- exploit mode: ON")
    elif op == "off":
        state["exploit_mode"] = False; save_json(STATE_PATH, state)
        print("- exploit mode: OFF")
    elif op == "arm":
        if len(args) < 3:
            print("- usage: /exploit arm <target> <engagement_id>")
            return
        tgt = normalize_target(args[1]); eng = args[2]
        require_pentest_enabled()
        require_target_allowed(tgt)
        token = base64.urlsafe_b64encode(os.urandom(12)).decode().rstrip("=")
        state["armed_token"] = {
            "token": token,
            "target": tgt,
            "engagement": eng,
            "expires": time.time() + 15*60  # 15 minutes
        }
        save_json(STATE_PATH, state)
        print("- ARMED. To confirm, type EXACTLY:")
        print(f"/exploit confirm {token} I HAVE AUTHORIZATION FOR {tgt} [{eng}]")
    elif op == "confirm":
        if len(args) < 6:
            print("- usage: /exploit confirm <token> I HAVE AUTHORIZATION FOR <target> [<eng_id>]")
            return
        token = args[1]
        phrase = " ".join(args[2:6])
        if phrase != "I HAVE AUTHORIZATION":
            print("- error: phrase must begin with: I HAVE AUTHORIZATION FOR ..."); return
        # Reconstruct expected trailing parts
        try:
            idx_for = args.index("FOR")
        except ValueError:
            print("- error: missing FOR"); return
        tgt = normalize_target(args[idx_for+1])
        eng = None
        if len(args) > idx_for+2:
            maybe = " ".join(args[idx_for+2:]).strip()
            m = re.match(r"^\[(.+)\]$", maybe)
            if m: eng = m.group(1)
        tok = state.get("armed_token")
        if not tok:
            print("- error: no token armed"); return
        if tok["expires"] < time.time():
            print("- error: token expired"); return
        if tok["token"] != token:
            print("- error: token mismatch"); return
        if tok["target"] != tgt:
            print("- error: target mismatch"); return
        if eng and tok["engagement"] != eng:
            print("- error: engagement mismatch"); return
        print("- confirmation OK. You may now /exploit on")
    else:
        print("- unknown /exploit subcommand")

def cmd_run(args: list):
    if len(args) < 2:
        print("- usage: /run <tool> <json>"); return
    name = args[0]
    js = " ".join(args[1:])
    try:
        payload = json.loads(js) if js else {}
    except Exception as e:
        print(f"- error: bad json: {e}"); return
    fn = TOOLS.get(name)
    if not fn:
        print(f"- error: unknown tool: {name}"); return
    try:
        out = fn(**payload)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"- error: {e}")

def cmd_log():
    for e in state.get("log", [])[-50:]:
        print(f"- {e['ts']} {e['kind']} { {k:v for k,v in e.items() if k not in ('ts','kind')} }")

def repl():
    print(BANNER)
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line: 
            continue
        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0].lower()
            args = parts[1:]
            if cmd == "/help":
                print(HELP)
            elif cmd == "/doctor":
                check_deps()
            elif cmd == "/pentest":
                try:
                    cmd_pentest(args)
                except Exception as e:
                    print(f"- error: {e}")
            elif cmd == "/exploit":
                try:
                    cmd_exploit(args)
                except Exception as e:
                    print(f"- error: {e}")
            elif cmd == "/run":
                cmd_run(args)
            elif cmd == "/log":
                cmd_log()
            elif cmd == "/quit":
                break
            else:
                print("- unknown command; /help")
        else:
            print("- tip: use slash-commands; /help")
    print("- bye")

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP); sys.exit(0)
    repl()
