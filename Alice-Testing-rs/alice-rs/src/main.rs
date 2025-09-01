use anyhow::{anyhow, Result};
use clap::Parser;
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

const DB_SCHEMA: &str = r#"
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
"#;

const BASE_PROMPT: &str = r#"You are Alice — a continuously-learning AI assistant.
Style:
- concise; prefer dashed bullets for lists
- cite sources only if explicitly asked
Memory:
- you remember important facts and files the user ingests
"#;

const SAFETY_PROMPT: &str = r#"Safety:
- never execute OS commands; use only provided tools
- do not claim to have run code unless host confirms
"#;

#[derive(Parser, Debug, Clone)]
#[command(name="alice-gp-rs", about="Alice gp (Rust) — local-first assistant")]
struct Args {
    #[arg(long, default_value = "alice.db")]
    db: String,
    #[arg(long, default_value = "gpt-oss:20b")]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    ollama_host: String,
    #[arg(long, default_value_t = 11434)]
    ollama_port: u16,
    #[arg(long, default_value_t = 0.4)]
    temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 40)]
    top_k: i32,
    #[arg(long, default_value_t = 4096)]
    num_ctx: i32,
    #[arg(long, default_value_t = 600)]
    reflect: u64,
    #[arg(long, default_value_t = false)]
    allow_dangerous_skills: bool,
    #[arg(long, default_value_t = false)]
    no_embed: bool,
    #[arg(long, default_value = "nomic-embed-text")]
    embed_model: String,
    #[arg(long, default_value_t = false)]
    train_mode: bool,
    #[arg(long, default_value_t = 3600)]
    train_interval: u64,
    #[arg(long, default_value_t = 900)]
    darkvision_interval: u64,
    #[arg(long, default_value_t = false, help="Start with Darkmode on (autonomous tool use)")]
    autonomous_tools: bool,
    #[arg(long, default_value_t = false, help="Start with web browsing enabled")]
    enable_web: bool,
    #[arg(long, default_value = "", help="Contact email for NOAA User-Agent header")]
    contact_email: String,
    #[arg(long, default_value = "", help="Path to Ghidra analyzeHeadless")]
    ghidra_headless: String,
}

#[derive(Debug, Clone)]
struct Config {
    args: Args,
}

#[derive(Debug)]
struct Memory {
    conn: Connection,
}

impl Memory {
    fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(DB_SCHEMA)?;
        Ok(Self { conn })
    }

    fn now_iso() -> String {
        Utc::now().to_rfc3339()
    }

    fn add_message(&self, role: &str, content: &str) -> Result<i64> {
        let ts = Self::now_iso();
        self.conn.execute(
            "INSERT INTO messages(role, content, ts) VALUES (?,?,?)",
            params![role, content, ts],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    fn list_facts(&self, n: usize) -> Result<Vec<(String,String,f64)>> {
        let mut stmt = self.conn.prepare("SELECT key, value, weight FROM facts ORDER BY weight DESC LIMIT ?1")?;
        let rows = stmt.query_map([n as i64], |row| {
            Ok((row.get::<_,String>(0)?, row.get::<_,String>(1)?, row.get::<_,f64>(2)?))
        })?;
        let mut out = Vec::new();
        for r in rows { out.push(r?) }
        Ok(out)
    }

    fn upsert_fact(&self, key: &str, value: &str, weight_delta: f64) -> Result<()> {
        let ts = Self::now_iso();
        let existing: Option<(String,f64)> = self.conn.query_row(
            "SELECT key, weight FROM facts WHERE key=?1",
            [key],
            |row| Ok((row.get(0)?, row.get(1)?))
        ).optional()?;
        if let Some((_k, w)) = existing {
            let nw = w + weight_delta;
            self.conn.execute("UPDATE facts SET value=?, weight=?, ts=? WHERE key=?",
                params![value, nw, ts, key])?;
        } else {
            let w = 1.0 + weight_delta;
            self.conn.execute("INSERT OR REPLACE INTO facts(key,value,weight,ts) VALUES (?,?,?,?)",
                params![key, value, w, ts])?;
        }
        Ok(())
    }

    fn recall(&self, query: &str, k: usize) -> Result<Vec<(i64,String)>> {
        let mut out: Vec<(i64,String)> = Vec::new();
        let re = Regex::new(r"[A-Za-z0-9_]{2,}")?;
        let tokens: Vec<&str> = re.find_iter(query).map(|m| m.as_str()).collect();
        if !tokens.is_empty() {
            let fts_q = tokens.join(" OR ");
            let mut stmt = self.conn.prepare(
                "SELECT message_id, content FROM messages_fts WHERE messages_fts MATCH ?1 ORDER BY bm25(messages_fts) LIMIT ?2"
            )?;
            let rows = stmt.query_map(params![fts_q, k as i64], |row| {
                Ok((row.get::<_,i64>(0)?, row.get::<_,String>(1)?))
            })?;
            for r in rows { out.push(r?) }
        }
        if out.is_empty() {
            let mut stmt = self.conn.prepare("SELECT id, content FROM messages ORDER BY id DESC LIMIT 1000")?;
            let rows = stmt.query_map([], |row| Ok((row.get::<_,i64>(0)?, row.get::<_,String>(1)?)))?;
            let ql = query.to_lowercase();
            let mut scored: Vec<(i64,String,i32)> = Vec::new();
            for r in rows {
                let (mid, content) = r?;
                let lc = content.to_lowercase();
                let score = ql.split_whitespace().filter(|t| lc.contains(*t)).count() as i32;
                if score > 0 {
                    scored.push((mid, content, score));
                }
            }
            scored.sort_by(|a,b| b.2.cmp(&a.2).then(b.0.cmp(&a.0)));
            for (mid, content, _) in scored.into_iter().take(k) {
                out.push((mid, content));
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct LlmClient {
    base: String,
    model: String,
    http: reqwest::blocking::Client,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    num_ctx: i32,
    _embed_model: String,
}

impl LlmClient {
    fn new(cfg: &Config) -> Result<Self> {
        let base = format!("http://{}:{}", cfg.args.ollama_host, cfg.args.ollama_port);
        Ok(Self {
            base,
            model: cfg.args.model.clone(),
            http: reqwest::blocking::Client::builder().timeout(Duration::from_secs(240)).build()?,
            temperature: cfg.args.temperature,
            top_p: cfg.args.top_p,
            top_k: cfg.args.top_k,
            num_ctx: cfg.args.num_ctx,
            _embed_model: cfg.args.embed_model.clone(),
        })
    }

    fn complete(&self, prompt: &str, system: Option<&str>, max_tokens: i32) -> Result<String> {
        let mut payload = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens
            }
        });
        if let Some(sys) = system {
            payload["system"] = json!(sys);
        }
        let url = format!("{}/api/generate", self.base);
        let resp: serde_json::Value = self.http.post(url).json(&payload).send()?.json()?;
        Ok(resp.get("response").and_then(|v| v.as_str()).unwrap_or("").to_string())
    }

    fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base);
        let resp: serde_json::Value = self.http.get(url).send()?.json()?;
        let mut names = Vec::new();
        if let Some(models) = resp.get("models").and_then(|v| v.as_array()) {
            for m in models {
                if let Some(tag) = m.get("model").or_else(|| m.get("name")).and_then(|v| v.as_str()) {
                    names.push(tag.to_string());
                }
            }
        }
        names.sort();
        names.dedup();
        Ok(names)
    }

    fn show_model(&self, name: Option<&str>) -> Result<serde_json::Value> {
        let url = format!("{}/api/show", self.base);
        let resp: serde_json::Value = self.http.post(url).json(&json!({
            "name": name.unwrap_or(&self.model)
        })).send()?.json()?;
        Ok(resp)
    }
}

// ---------- top-level structs for NOAA output ----------
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct ForecastPeriod {
    name: Option<String>,
    #[serde(rename = "startTime")]
    start_time: Option<String>,
    #[serde(rename = "endTime")]
    end_time: Option<String>,
    temperature: Option<i64>,
    unit: Option<String>,
    wind: Option<String>,
    #[serde(rename = "shortForecast")]
    short_forecast: Option<String>,
    #[serde(rename = "detailedForecast")]
    detailed_forecast: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct WeatherOut {
    ok: bool,
    location: Option<String>,
    hourly: bool,
    periods: Option<Vec<ForecastPeriod>>,
    error: Option<String>,
}

#[derive(Debug)]
struct Alice {
    cfg: Config,
    mem: Memory,
    llm: LlmClient,
    darkmode: bool,
    web_access: bool,
    last_assistant_id: Option<i64>,
}

impl Alice {
    fn new(cfg: Config) -> Result<Self> {
        let mem = Memory::new(&cfg.args.db)?;
        mem.add_message("system", &(String::from(BASE_PROMPT) + SAFETY_PROMPT))?;
        mem.upsert_fact("agent.name", "Alice", 0.2)?;
        mem.upsert_fact("agent.style", "concise-with-dashed-bullets", 0.2)?;
        let llm = LlmClient::new(&cfg)?;
        Ok(Self {
            darkmode: cfg.args.autonomous_tools,
            web_access: cfg.args.enable_web,
            cfg,
            mem,
            llm,
            last_assistant_id: None,
        })
    }

    fn system_prompt(&self) -> String {
        let mut s = String::from(BASE_PROMPT);
        if self.darkmode {
            s.push_str("Tools are available via /run commands.\n");
        }
        s.push_str(SAFETY_PROMPT);
        s
    }

    fn compose_prompt(&self, user_text: &str) -> Result<String> {
        let facts = self.mem.list_facts(12)?;
        let fact_lines: Vec<String> = facts.into_iter()
            .map(|(k,v,w)| format!("- {}: {} (w={:.1})", k, v, w))
            .collect();
        let mut recent = String::new();
        let mut stmt = self.mem.conn.prepare("SELECT role, content FROM messages ORDER BY id DESC LIMIT 14")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_,String>(0)?, row.get::<_,String>(1)?)))?;
        let mut tmp = Vec::new();
        for r in rows { tmp.push(r?) }
        tmp.reverse();
        for (role, content) in tmp {
            recent.push_str(&format!("{}: {}\n", role.to_uppercase(), content));
        }
        let recall_hits = self.mem.recall(user_text, 8)?;
        let recall_text: String = if recall_hits.is_empty() {
            String::from("(none)")
        } else {
            recall_hits.into_iter().map(|(_id, c)| format!("• {}", truncate(&c, 400))).collect::<Vec<_>>().join("\n")
        };
        let guidance = "Answer succinctly. Use FACTS and RECALL if relevant. Prefer dashed bullets for lists.\n";
        Ok(format!("{}\nFACTS (top-weighted):\n{}\nRECALL (similar topics):\n{}\nRECENT CHAT:\n{}\nUSER: {}\nASSISTANT:", guidance, if fact_lines.is_empty() { "(none)".to_string() } else { fact_lines.join("\n") }, recall_text, recent, user_text))
    }

    fn handle_line(&mut self, line: &str) -> Result<String> {
        let t = line.trim();
        if t.starts_with("/help") {
            return Ok(self.help_text());
        }
        if t.starts_with("/facts") {
            let facts = self.mem.list_facts(30)?;
            if facts.is_empty() { return Ok(String::from("- (none)")); }
            let mut out = String::from("Known facts (top weights):\n");
            for (k,v,w) in facts {
                out.push_str(&format!("- {}: {} (w={:.2})\n", k, v, w));
            }
            return Ok(out.trim_end().to_string());
        }
        if t.starts_with("/recall") {
            let q = t.strip_prefix("/recall").unwrap().trim();
            let q = if q.is_empty() { "*" } else { q };
            let hits = self.mem.recall(q, 12)?;
            if hits.is_empty() { return Ok(String::from("- (no matches)")); }
            let mut out = String::new();
            for (mid, snip) in hits {
                out.push_str(&format!("- {}: {}\n", mid, truncate(&snip, 160)));
            }
            return Ok(out.trim_end().to_string());
        }
        if t.starts_with("/good") {
            if let Some(id) = self.last_assistant_id {
                self.mem.conn.execute("INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?,?,?,?)",
                    params![id, 1.0_f64, "user_mark_good", Memory::now_iso()])?;
            }
            return Ok(String::from("- thanks — feedback recorded"));
        }
        if t.starts_with("/bad") {
            if let Some(id) = self.last_assistant_id {
                self.mem.conn.execute("INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?,?,?,?)",
                    params![id, -1.0_f64, "user_mark_bad", Memory::now_iso()])?;
            }
            return Ok(String::from("- got it — I’ll adjust"));
        }
        if t.starts_with("/web") {
            let parts: Vec<&str> = t.split_whitespace().collect();
            if parts.len() == 1 { return Ok("Usage: /web on | off".into()); }
            match parts[1].to_lowercase().as_str() {
                "on" | "enable" | "start" => { self.web_access = true; return Ok("- Web access: ENABLED (Alice can browse & reach NOAA)".into()); }
                "off" | "disable" | "stop" => { self.web_access = false; return Ok("- Web access: DISABLED".into()); }
                _ => return Ok("Usage: /web on | off".into()),
            }
        }
        if t.starts_with("/model") {
            return self.cmd_model(t);
        }
        if t.starts_with("/weather") {
            return self.cmd_weather(t);
        }
        if t.starts_with("/run ") {
            return self.cmd_run(t);
        }
        if t.starts_with("/ingest ") {
            return self.cmd_ingest(t);
        }

        // simple echo
        if t.to_lowercase().starts_with("say ") || t.to_lowercase().starts_with("/say ") {
            let parts: Vec<&str> = t.splitn(2, ' ').collect();
            if parts.len() > 1 {
                let mut echo = parts[1].trim().to_string();
                // robust quotes trim without triple-quote issues
                let is_quoted = (echo.starts_with('"') && echo.ends_with('"')) ||
                                (echo.starts_with('\'') && echo.ends_with('\''));
                if is_quoted && echo.len() >= 2 {
                    echo = echo[1..echo.len()-1].to_string();
                }
                return Ok(echo);
            }
        }

        // regular chat
        self.mem.add_message("user", t)?;
        let prompt = self.compose_prompt(t)?;
        let system = self.system_prompt();
        let reply = self.llm.complete(&prompt, Some(&system), 700)?;
        let final_text = strip_thoughts(&reply).trim().to_string();
        let id = self.mem.add_message("assistant", &final_text)?;
        self.last_assistant_id = Some(id);
        Ok(final_text)
    }

    fn help_text(&self) -> String {
        r#"Commands:
- /help                  — show this help
- /facts                 — list top remembered facts
- /recall <query>        — search past chat/content
- /good | /bad           — reward/punish last answer
- /run <name> {json}     — run a built-in tool (see below)
- /ingest <path>         — ingest .txt/.md file(s) into memory
- /darkmode on|off|status — toggle autonomous style (no auto-tooling in Rust MVP)
- /model list|info|set <tag>
- /web on|off            — enable or disable internet access
- /weather <lat,lon> [hourly] — NOAA forecast shortcut
Built-in tools: browse, noaa_weather, nmap_scan, nikto_scan, nuclei_scan, wifi_scan, ghidra_functions (MVP)
"#.to_string()
    }

    fn cmd_ingest(&self, raw: &str) -> Result<String> {
        let path = raw.trim().strip_prefix("/ingest").unwrap().trim();
        if path.is_empty() {
            return Ok("Usage: /ingest <file-or-directory>".into());
        }
        let path = shellexpand::tilde(path).to_string();
        let p = Path::new(&path);
        if !p.exists() {
            return Ok(format!("- Path not found: {}", path));
        }
        let mut files: Vec<PathBuf> = Vec::new();
        if p.is_dir() {
            for entry in walkdir::WalkDir::new(p).min_depth(1) {
                let e = entry?;
                if e.path().is_file() {
                    if let Some(ext) = e.path().extension().and_then(|s| s.to_str()) {
                        let ext = ext.to_lowercase();
                        if ext == "txt" || ext == "md" {
                            files.push(e.path().to_path_buf());
                        }
                    }
                }
            }
        } else {
            files.push(p.to_path_buf());
        }
        let mut count = 0;
        for fp in files {
            if let Ok(text) = fs::read_to_string(&fp) {
                let name = fp.file_name().and_then(|s| s.to_str()).unwrap_or("file");
                let payload = format!("[ingested:{}]\n{}", name, text);
                let _ = self.mem.add_message("system", &payload);
                count += 1;
            }
        }
        Ok(format!("- Ingested {} file(s). Use /recall to search.", count))
    }

    fn cmd_model(&mut self, raw: &str) -> Result<String> {
        let parts: Vec<&str> = raw.split_whitespace().collect();
        if parts.len() == 1 { return Ok("Usage: /model list | info | set <tag>".into()); }
        match parts[1].to_lowercase().as_str() {
            "list" => {
                let models = self.llm.list_models()?;
                if models.is_empty() { return Ok("- no models found (is ollama running and models pulled?)".into()); }
                let mut out = String::from("- Installed models:\n");
                for m in models { out.push_str(&format!("- {}\n", m)); }
                Ok(out.trim_end().to_string())
            }
            "info" => {
                let info = self.llm.show_model(None)?;
                let details = info.get("details").cloned().unwrap_or(json!({}));
                let ctx_len = details.get("context_length").and_then(|v| v.as_i64()).unwrap_or(0);
                let family = details.get("family").or_else(|| details.get("families")).cloned().unwrap_or(json!("?"));
                Ok(format!("- model={}\n- context_length={}\n- family={}", self.llm.model, ctx_len, family))
            }
            "set" if parts.len() >= 3 => {
                let tag = parts[2..].join(" ");
                self.llm.model = tag.clone();
                let _ = self.llm.show_model(Some(&tag));
                Ok(format!("- model switched to {}", tag))
            }
            _ => Ok("Usage: /model list | info | set <tag>".into())
        }
    }

    fn cmd_weather(&self, raw: &str) -> Result<String> {
        let re = Regex::new(r"/weather\\s+(-?\\d+(\\.\\d+)?)\\s*,\\s*(-?\\d+(\\.\\d+)?)(\\s+hourly)?")?;
        if let Some(c) = re.captures(raw) {
            let lat: f64 = c.get(1).unwrap().as_str().parse()?;
            let lon: f64 = c.get(3).unwrap().as_str().parse()?;
            let hourly = c.get(5).is_some();
            let r = self.tool_noaa_weather(lat, lon, hourly)?;
            if !r.ok { return Ok(format!("- NOAA error: {}", r.error.unwrap_or_else(|| "unknown".into()))); }
            let loc = r.location.clone().unwrap_or(format!("{:.4},{:.4}", lat, lon));
            let mut lines = vec![format!("- location: {}", loc), format!("- mode: {}", if hourly { "hourly" } else { "periods" })];
            for p in r.periods.clone().unwrap_or_default() {
                lines.push(format!("- {}: {} — {}{} | wind {}", p.name.unwrap_or_default(), p.short_forecast.unwrap_or_default(), p.temperature.unwrap_or(0), p.unit.unwrap_or_default(), p.wind.unwrap_or_default()));
            }
            Ok(lines.join("\n"))
        } else {
            Ok("Usage: /weather <lat,lon> [hourly] (e.g., /weather 38.8977,-77.0365 hourly)".into())
        }
    }

    fn cmd_run(&self, raw: &str) -> Result<String> {
        let re = Regex::new(r#"^/run\\s+([A-Za-z0-9_\\-]+)\\s+(\\{.*\\})\\s*$"#)?;
        if let Some(c) = re.captures(raw) {
            let name = c.get(1).unwrap().as_str();
            let args: serde_json::Value = serde_json::from_str(c.get(2).unwrap().as_str())?;
            match name {
                "browse" => {
                    let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
                    let out = self.tool_browse(url)?;
                    return Ok(out);
                }
                "noaa_weather" => {
                    let lat = args.get("lat").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let lon = args.get("lon").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let hourly = args.get("hourly").and_then(|v| v.as_bool()).unwrap_or(false);
                    let r = self.tool_noaa_weather(lat, lon, hourly)?;
                    return Ok(serde_json::to_string_pretty(&r)?);
                }
                "nmap_scan" => {
                    let tgt = args.get("target").and_then(|v| v.as_str()).unwrap_or("");
                    let opts = args.get("options").and_then(|v| v.as_str()).unwrap_or("-sV -T4");
                    let res = self.tool_nmap_scan(tgt, opts)?;
                    return Ok(res);
                }
                "nikto_scan" => {
                    let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
                    let res = self.tool_nikto_scan(url)?;
                    return Ok(res);
                }
                "nuclei_scan" => {
                    let tgt = args.get("target").and_then(|v| v.as_str()).unwrap_or("");
                    let res = self.tool_nuclei_scan(tgt)?;
                    return Ok(res);
                }
                "wifi_scan" => {
                    let res = self.tool_wifi_scan()?;
                    return Ok(res);
                }
                "ghidra_functions" => {
                    let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");
                    let res = self.tool_ghidra_functions(file)?;
                    return Ok(res);
                }
                _ => return Ok(format!("- No such tool: {}", name))
            }
        }
        Ok("Usage: /run <name> {json-args}".into())
    }

    // ---------------- tools ----------------

    fn require_web(&self) -> Result<()> {
        if !self.web_access {
            return Err(anyhow!("web access is disabled"));
        }
        Ok(())
    }

    fn tool_browse(&self, url: &str) -> Result<String> {
        self.require_web()?;
        if url.is_empty() { return Ok(String::from(r#"{"ok":false,"error":"missing url"}"#)); }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Ok(String::from(r#"{"ok":false,"error":"invalid URL scheme"}"#));
        }
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(15)).build()?;
        let resp = client.get(url).header("User-Agent", "AliceBot/0.1").send()?;
        let text = resp.text()?;
        let rendered = html2text::from_read(text.as_bytes(), 80);
        let snippet = if rendered.len() > 1000 { format!("{}...", &rendered[..1000]) } else { rendered };
        Ok(serde_json::to_string(&json!({"ok": true, "content": snippet}))?)
    }

    fn noaa_headers(&self) -> Vec<(&'static str, String)> {
        let contact = if !self.cfg.args.contact_email.trim().is_empty() {
            self.cfg.args.contact_email.clone()
        } else {
            format!("{}@example.invalid", whoami::username())
        };
        vec![("User-Agent", format!("AliceWeather/0.1 (+{})", contact)),
             ("Accept", "application/geo+json, application/json;q=0.9".into())]
    }

    fn tool_noaa_weather(&self, lat: f64, lon: f64, hourly: bool) -> Result<WeatherOut> {
        self.require_web()?;
        if !( (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) ) {
            return Ok(WeatherOut{ ok:false, hourly, error:Some("lat/lon out of range".into()), ..Default::default() });
        }
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(15)).build()?;
        let p_url = format!("https://api.weather.gov/points/{:.4},{:.4}", lat, lon);
        let mut req = client.get(&p_url);
        for (k,v) in self.noaa_headers() { req = req.header(k, v); }
        let p_resp: serde_json::Value = req.send()?.json()?;
        let props = p_resp.get("properties").cloned().unwrap_or(json!({}));
        let rel = props.get("relativeLocation").and_then(|v| v.get("properties")).cloned().unwrap_or(json!({}));
        let location = match (rel.get("city").and_then(|v| v.as_str()), rel.get("state").and_then(|v| v.as_str())) {
            (Some(c), Some(s)) => format!("{}, {}", c, s),
            _ => format!("{:.4},{:.4}", lat, lon),
        };
        let forecast_url = if hourly { props.get("forecastHourly") } else { props.get("forecast") }
            .and_then(|v| v.as_str()).unwrap_or("");
        if forecast_url.is_empty() {
            return Ok(WeatherOut{ ok:false, hourly, error:Some("forecast URL not available".into()), ..Default::default() });
        }
        let mut reqf = client.get(forecast_url);
        for (k,v) in self.noaa_headers() { reqf = reqf.header(k, v); }
        let f_resp: serde_json::Value = reqf.send()?.json()?;
        let periods = f_resp.get("properties").and_then(|v| v.get("periods")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let take = if hourly { 6 } else { 4 };
        let mut out_periods = Vec::new();
        for p in periods.into_iter().take(take) {
            let temp = p.get("temperature").and_then(|v| v.as_i64());
            let unit = p.get("temperatureUnit").and_then(|v| v.as_str()).map(|s| s.to_string());
            let wind = format!("{} {}", p.get("windSpeed").and_then(|v| v.as_str()).unwrap_or(""), p.get("windDirection").and_then(|v| v.as_str()).unwrap_or("")).trim().to_string();
            out_periods.push(ForecastPeriod {
                name: p.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()),
                start_time: p.get("startTime").and_then(|v| v.as_str()).map(|s| s.to_string()),
                end_time: p.get("endTime").and_then(|v| v.as_str()).map(|s| s.to_string()),
                temperature: temp,
                unit,
                wind: if wind.is_empty() { None } else { Some(wind) },
                short_forecast: p.get("shortForecast").and_then(|v| v.as_str()).map(|s| s.to_string()),
                detailed_forecast: p.get("detailedForecast").and_then(|v| v.as_str()).map(|s| truncate(s, 240)),
            });
        }
        Ok(WeatherOut{ ok:true, location:Some(location), hourly, periods:Some(out_periods), error:None })
    }

    fn pentest_allowed(&self, target: &str) -> bool {
        if target.trim().is_empty() { return false; }
        let mut stmt = match self.mem.conn.prepare("SELECT key FROM facts WHERE key=?1") { Ok(s) => s, Err(_) => return false };
        let row: Option<String> = match stmt.query_row([format!("pentest.allow:{}", target.to_lowercase())], |row| row.get(0)).optional() { Ok(r) => r, Err(_) => None };
        row.is_some()
    }

    fn tool_nmap_scan(&self, target: &str, options: &str) -> Result<String> {
        if !self.pentest_allowed(target) { return Ok(r#"{"ok":false,"error":"target not allowed; /pentest allow <target>"}"#.into()); }
        let bin = which::which("nmap").map_err(|_| anyhow!("nmap not found"))?;
        let output = Command::new(bin).args(options.split_whitespace()).arg(target).output()?;
        let txt = format!("{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        Ok(txt.trim().to_string())
    }

    fn tool_nikto_scan(&self, url: &str) -> Result<String> {
        let host = url.split("://").last().unwrap_or("").split('/').next().unwrap_or("").to_lowercase();
        if !self.pentest_allowed(&host) { return Ok(r#"{"ok":false,"error":"target not allowed; /pentest allow <host>"}"#.into()); }
        let bin = which::which("nikto").map_err(|_| anyhow!("nikto not found"))?;
        let output = Command::new(bin).args(&["-host", url, "-ask", "no", "-maxtime", "600"]).output()?;
        let txt = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(txt.trim().to_string())
    }

    fn tool_nuclei_scan(&self, target: &str) -> Result<String> {
        if !self.pentest_allowed(target) { return Ok(r#"{"ok":false,"error":"target not allowed; /pentest allow <target>"}"#.into()); }
        let bin = which::which("nuclei").map_err(|_| anyhow!("nuclei not found"))?;
        let output = Command::new(bin).args(&["-u", target, "-silent", "-timeout", "10"]).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn tool_wifi_scan(&self) -> Result<String> {
        let bin = which::which("nmcli").map_err(|_| anyhow!("nmcli not found"))?;
        let output = Command::new(bin).args(&["-t","-f","SSID,SIGNAL,SECURITY","dev","wifi"]).output()?;
        let mut nets = Vec::new();
        for ln in String::from_utf8_lossy(&output.stdout).lines() {
            if ln.trim().is_empty() { continue; }
            let parts: Vec<&str> = ln.split(':').collect();
            let ssid = parts.get(0).copied().unwrap_or("");
            let signal = parts.get(1).copied().unwrap_or("");
            let sec = if parts.len() > 2 { parts[2..].join(":") } else { "".into() };
            nets.push(json!({"ssid": ssid, "signal": signal, "security": sec}));
        }
        Ok(serde_json::to_string_pretty(&json!({"ok": true, "networks": nets}))?)
    }

    fn tool_ghidra_functions(&self, file: &str) -> Result<String> {
        let _headless = if !self.cfg.args.ghidra_headless.trim().is_empty() {
            PathBuf::from(&self.cfg.args.ghidra_headless)
        } else {
            which::which("analyzeHeadless").or_else(|_| which::which("analyzeHeadless.bat")).map_err(|_| anyhow!("Ghidra analyzeHeadless not found; set --ghidra-headless"))?
        };
        let file = dunce::canonicalize(Path::new(file))?;
        if !file.exists() { return Err(anyhow!("file not found: {}", file.display())); }
        Ok(r#"{"ok":false,"error":"ghidra headless function listing is stubbed in Rust MVP"}"#.into())
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}...", &s[..n]) }
}

fn strip_thoughts(s: &str) -> String {
    let re = Regex::new(r"(?m)^\\s*(Thought|Chain|Internal):.*$").unwrap();
    re.replace_all(s, "").to_string()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut alice = Alice::new(Config{ args: args.clone() })?;

    ctrlc::set_handler(move || {
        eprintln!("\n[shutting down]");
        std::process::exit(0);
    }).ok();

    println!("Alice gp (Rust) MVP is awake. Type /help for commands. (Ctrl+C to exit.)");

    let spinner_style = ProgressStyle::with_template("{spinner:.green} {msg}").unwrap();
    loop {
        print!("You> ");
        io::stdout().flush().ok();
        let mut buf = String::new();
        if io::stdin().read_line(&mut buf).is_err() { break; }
        let line = buf.trim().to_string();
        if line.is_empty() { continue; }

        let pb = ProgressBar::new_spinner();
        pb.set_style(spinner_style.clone());
        pb.set_message("thinking");
        pb.enable_steady_tick(Duration::from_millis(80));

        let out = alice.handle_line(&line);

        pb.finish_and_clear();

        match out {
            Ok(s) => println!("Alice> {}", s),
            Err(e) => println!("Alice> - error: {}", e),
        }
    }
    Ok(())
}