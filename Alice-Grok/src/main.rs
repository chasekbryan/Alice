use anyhow::{anyhow, Result};
use clap::Parser;
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use walkdir::WalkDir;
use which::which;
use shellexpand;
use url::Url;
use html2text;
use reqwest;
use evalexpr;

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

const TOOLS_PROMPT: &str = r#"Tools:
- Built-ins: calc(expr), recall(query), set_fact(key,value), get_facts()
- browse(url): fetch content from a webpage (only if web access is enabled)
- noaa_weather(lat, lon, hourly=False): get National Weather Service forecast for coordinates
- Pentest (detection-only by default; target must be authorized):
- nmap_scan(target, options) — enumerate ports/services (requires nmap)
- nikto_scan(url) — web server checks (requires nikto)
- zap_scan(url) — OWASP ZAP active scan via API (ZAP must be running)
- nuclei_scan(target) — run nuclei templates against target (requires nuclei)
- msf_check(module, rhost, rport, opts) — run Metasploit 'check' only (RPC)
- wifi_scan(interface=None) — list nearby Wi‑Fi networks (nmcli or airodump-ng)
- User-taught skills may exist; only call tools you are told exist.
To use a tool, emit exactly one line:
<<call:NAME args='{"param": "value"}'>>
Host will reply with TOOL_RESULT; then continue.
"#;

const SAFETY_PROMPT: &str = r#"Safety:
- never execute OS commands; use only provided tools
- do not claim to have run code unless host confirms
"#;

#[derive(Parser, Debug, Clone)]
#[command(name = "alice-gp-rs", version = "0.2.0", about = "Alice gp v0.2.0 (Rust) — local-first AI assistant with tools, memory, and autonomy")]
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
    reflect_every_sec: u64,
    #[arg(long, default_value_t = 12)]
    recall_k: usize,
    #[arg(long, default_value_t = 14)]
    max_history: usize,
    #[arg(long, default_value_t = false)]
    allow_dangerous_skills: bool,
    #[arg(long, default_value = "127.0.0.1")]
    msfrpc_host: String,
    #[arg(long, default_value_t = 55552)]
    msfrpc_port: u16,
    #[arg(long, default_value = "msf")]
    msfrpc_user: String,
    #[arg(long, default_value = "")]
    msfrpc_pass: String,
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    zap_api: String,
    #[arg(long, default_value = "")]
    zap_api_key: String,
    #[arg(long, default_value = "nmap")]
    nmap_binary: String,
    #[arg(long, default_value = "nikto")]
    nikto_binary: String,
    #[arg(long, default_value = "nuclei")]
    nuclei_binary: String,
    #[arg(long, default_value = "sqlmap")]
    sqlmap_binary: String,
    #[arg(long, default_value = "nmcli")]
    nmcli_binary: String,
    #[arg(long, default_value = "airodump-ng")]
    airodump_binary: String,
    #[arg(long, default_value = "")]
    ghidra_headless: String,
    #[arg(long, default_value_t = true)]
    embed_enable: bool,
    #[arg(long, default_value = "nomic-embed-text")]
    embed_model: String,
    #[arg(long, default_value_t = false)]
    train_mode: bool,
    #[arg(long, default_value_t = 3600)]
    train_interval_sec: u64,
    #[arg(long, default_value_t = 900)]
    darkvision_interval_sec: u64,
    #[arg(long, default_value_t = false, help = "Start with Darkmode on (autonomous tool use)")]
    autonomous_tools: bool,
    #[arg(long, default_value_t = false, help = "Start with web browsing enabled (internet access)")]
    web_access: bool,
    #[arg(long, default_value = "", help = "Contact email for NOAA User-Agent header")]
    contact_email: String,
}

#[derive(Debug, Clone)]
struct Config {
    args: Args,
}

#[derive(Debug, Clone)]
struct Memory {
    conn: Arc<Mutex<Connection>>,
    embed_q: Arc<Mutex<Vec<(i64, String)>>>, // Queue for background embedding
}

impl Memory {
    fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(DB_SCHEMA)?;
        conn.pragma_update(None, "busy_timeout", &5000)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embed_q: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn now_iso() -> String {
        Utc::now().to_rfc3339()
    }

    fn add_message(&self, role: &str, content: &str) -> Result<i64> {
        let ts = Self::now_iso();
        let mut conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO messages(role, content, ts) VALUES (?,?,?)",
            params![role, content, ts],
        )?;
        let id = conn.last_insert_rowid();
        if !content.trim().is_empty() {
            let mut q = self.embed_q.lock().unwrap();
            q.push((id, content.to_string()));
        }
        Ok(id)
    }

    fn list_facts(&self, n: usize) -> Result<Vec<(String, String, f64)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT key, value, weight FROM facts ORDER BY weight DESC LIMIT ?1")?;
        let rows = stmt.query_map([n as i64], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect::<Result<Vec<_>, rusqlite::Error>>().map_err(anyhow::Error::from)
    }

    fn upsert_fact(&self, key: &str, value: &str, weight_delta: f64) -> Result<()> {
        let ts = Self::now_iso();
        let conn = self.conn.lock().unwrap();
        let existing: Option<f64> = conn.query_row(
            "SELECT weight FROM facts WHERE key=?1",
            [key],
            |row| row.get(0)
        ).optional()?;
        if let Some(w) = existing {
            let nw = w + weight_delta;
            conn.execute("UPDATE facts SET value=?, weight=?, ts=? WHERE key=?",
                params![value, nw, ts, key])?;
        } else {
            let w = 1.0 + weight_delta;
            conn.execute("INSERT OR REPLACE INTO facts(key,value,weight,ts) VALUES (?,?,?,?)",
                params![key, value, w, ts])?;
        }
        Ok(())
    }

    fn get_facts(&self) -> Result<String> {
        let facts = self.list_facts(20)?;
        let mut out = String::new();
        for (k, v, w) in facts {
            out.push_str(&format!("- {}: {} (w={:.1})\n", k, v, w));
        }
        Ok(out)
    }

    fn recall(&self, query: &str, k: usize) -> Result<String> {
        let conn = self.conn.lock().unwrap();
        let mut out = String::new();
        // FTS search
        let mut stmt = conn.prepare(
            "SELECT message_id, content FROM messages_fts WHERE messages_fts MATCH ?1 ORDER BY bm25(messages_fts) LIMIT ?2"
        )?;
        let fts_q = query.split_whitespace().map(|t| format!("\"{}\"", t)).collect::<Vec<_>>().join(" OR ");
        let rows = stmt.query_map(params![fts_q, k as i64], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut results: Vec<(i64, String)> = rows.collect::<Result<_, rusqlite::Error>>().map_err(anyhow::Error::from)?;
        
        // Fallback if no FTS results
        if results.is_empty() {
            let mut stmt_fb = conn.prepare("SELECT id, content FROM messages ORDER BY id DESC LIMIT 1000")?;
            let rows_fb = stmt_fb.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            let hist: Vec<(i64, String)> = rows_fb.collect::<Result<_, rusqlite::Error>>().map_err(anyhow::Error::from)?;
            let ql = query.to_lowercase();
            let mut scored: Vec<(i64, String, usize)> = hist.into_iter()
                .map(|(id, content)| {
                    let score = ql.split_whitespace().filter(|t| content.to_lowercase().contains(t)).count();
                    (id, content, score)
                })
                .filter(|(_, _, s)| *s > 0)
                .collect();
            scored.sort_by(|a, b| b.2.cmp(&a.2).then(b.0.cmp(&a.0)));
            results = scored.into_iter().take(k).map(|(id, c, _)| (id, c)).collect();
        }

        for (_, content) in results {
            out.push_str(&format!("- {}\n", truncate(&content, 200)));
        }
        Ok(out)
    }

    fn add_embedding(&self, mid: i64, model: &str, vec: Vec<f32>) -> Result<()> {
        let ts = Self::now_iso();
        let dim = vec.len() as i32;
        let vec_json = serde_json::to_string(&vec)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(message_id, model, dim, vec_json, ts) VALUES (?,?,?,?,?)",
            params![mid, model, dim, vec_json, ts],
        )?;
        Ok(())
    }

    fn shutdown(&self) {
        // Drain queue if needed
    }
}

#[derive(Debug, Clone)]
struct LlmClient {
    base: String,
    model: String,
    embed_model: String,
    http: reqwest::blocking::Client,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    num_ctx: i32,
}

impl LlmClient {
    fn new(cfg: &Config) -> Result<Self> {
        let base = format!("http://{}:{}", cfg.args.ollama_host, cfg.args.ollama_port);
        Ok(Self {
            base,
            model: cfg.args.model.clone(),
            embed_model: cfg.args.embed_model.clone(),
            http: reqwest::blocking::Client::builder().timeout(Duration::from_secs(240)).build()?,
            temperature: cfg.args.temperature,
            top_p: cfg.args.top_p,
            top_k: cfg.args.top_k,
            num_ctx: cfg.args.num_ctx,
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
                "num_predict": max_tokens,
            }
        });
        if let Some(sys) = system {
            payload["system"] = json!(sys);
        }
        let url = format!("{}/api/generate", self.base);
        let resp: JsonValue = self.http.post(&url).json(&payload).send()?.json()?;
        Ok(resp["response"].as_str().unwrap_or("").to_string())
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(anyhow!("empty text"));
        }
        let payload = json!({
            "model": self.embed_model,
            "prompt": text,
        });
        let url = format!("{}/api/embeddings", self.base);
        let resp: JsonValue = self.http.post(&url).json(&payload).send()?.json()?;
        let emb = resp["embedding"].as_array().ok_or(anyhow!("no embedding"))?;
        Ok(emb.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
    }

    fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base);
        let resp: JsonValue = self.http.get(&url).send()?.json()?;
        let empty_vec = vec![];
        let models = resp["models"].as_array().unwrap_or(&empty_vec);
        let mut names: Vec<String> = models.iter().filter_map(|m| m["model"].as_str().or(m["name"].as_str())).map(|s| s.to_string()).collect();
        names.sort();
        names.dedup();
        Ok(names)
    }

    fn show_model(&self, name: Option<&str>) -> Result<JsonValue> {
        let payload = json!({ "name": name.unwrap_or(&self.model) });
        let url = format!("{}/api/show", self.base);
        let resp: JsonValue = self.http.post(&url).json(&payload).send()?.json()?;
        Ok(resp)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ForecastPeriod {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    short_forecast: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detailed_forecast: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct WeatherOut {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    location: Option<String>,
    hourly: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    periods: Option<Vec<ForecastPeriod>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug)]
struct Alice {
    cfg: Config,
    mem: Memory,
    llm: LlmClient,
    web_access: bool,
    pentest_enabled: bool,
    darkmode: bool,
    embed_enable: bool,
    train_mode: bool,
    reflect_thread: Option<JoinHandle<()>>,
    embed_thread: Option<JoinHandle<()>>,
    darkvision_thread: Option<JoinHandle<()>>,
    trainer_thread: Option<JoinHandle<()>>,
    stop_event: Arc<Mutex<bool>>,
}

impl Alice {
    fn new(cfg: Config) -> Result<Self> {
        let mem = Memory::new(&cfg.args.db)?;
        let llm = LlmClient::new(&cfg)?;
        let stop_event = Arc::new(Mutex::new(false));

        let mut alice = Self {
            cfg: cfg.clone(),
            mem,
            llm,
            web_access: cfg.args.web_access,
            pentest_enabled: false,
            darkmode: cfg.args.autonomous_tools,
            embed_enable: cfg.args.embed_enable,
            train_mode: cfg.args.train_mode,
            reflect_thread: None,
            embed_thread: None,
            darkvision_thread: None,
            trainer_thread: None,
            stop_event: stop_event.clone(),
        };

        // Start background threads
        alice.start_embed_indexer()?;
        alice.start_reflector()?;
        if alice.train_mode {
            alice.start_trainer()?;
        }
        if alice.darkmode {
            alice.start_darkvision()?;
        }

        Ok(alice)
    }

    fn start_embed_indexer(&mut self) -> Result<()> {
        let mem = self.mem.clone();
        let llm = self.llm.clone();
        let enable = self.embed_enable;
        let stop = self.stop_event.clone();
        let t = thread::spawn(move || {
            while !*stop.lock().unwrap() {
                let mut q = mem.embed_q.lock().unwrap();
                if let Some((mid, text)) = q.pop() {
                    if enable {
                        if let Ok(vec) = llm.embed(&text) {
                            let _ = mem.add_embedding(mid, &llm.embed_model, vec);
                        }
                    }
                } else {
                    drop(q);
                    thread::sleep(Duration::from_secs(1));
                }
            }
        });
        self.embed_thread = Some(t);
        Ok(())
    }

    fn start_reflector(&mut self) -> Result<()> {
        let mem = self.mem.clone();
        let llm = self.llm.clone();
        let interval = Duration::from_secs(self.cfg.args.reflect_every_sec);
        let stop = self.stop_event.clone();
        let t = thread::spawn(move || {
            let mut last = Instant::now();
            while !*stop.lock().unwrap() {
                if last.elapsed() >= interval {
                    // Simple reflection: find low-score, reflect via LLM, upsert fact
                    let prompt = "Reflect on recent interactions and extract key facts.";
                    if let Ok(refl) = llm.complete(prompt, None, 200) {
                        let ts = Memory::now_iso();
                        let conn = mem.conn.lock().unwrap();
                        let _ = conn.execute("INSERT INTO reflections(text, score, ts) VALUES (?,?,?)", params![refl, 0.0, ts]);
                    }
                    last = Instant::now();
                }
                thread::sleep(Duration::from_secs(10));
            }
        });
        self.reflect_thread = Some(t);
        Ok(())
    }

    fn start_trainer(&mut self) -> Result<()> {
        let mem = self.mem.clone();
        let interval = Duration::from_secs(self.cfg.args.train_interval_sec);
        let stop = self.stop_event.clone();
        let train_dir = "train".to_string();
        let t = thread::spawn(move || {
            fs::create_dir_all(&train_dir).ok();
            let mut last = Instant::now();
            while !*stop.lock().unwrap() {
                if last.elapsed() >= interval {
                    if let Ok(_) = Self::train_snapshot(&mem, &train_dir) {
                        last = Instant::now();
                    }
                }
                thread::sleep(Duration::from_secs(60));
            }
        });
        self.trainer_thread = Some(t);
        Ok(())
    }

    fn start_darkvision(&mut self) -> Result<()> {
        let mem = self.mem.clone();
        let llm = self.llm.clone();
        let interval = Duration::from_secs(self.cfg.args.darkvision_interval_sec);
        let stop = self.stop_event.clone();
        let t = thread::spawn(move || {
            let mut last = Instant::now();
            while !*stop.lock().unwrap() {
                if last.elapsed() >= interval {
                    // Autonomous improvement: snapshot, propose skills, test
                    let recent = "Summarize recent dialog and propose a safe skill.";
                    if let Ok(prop) = llm.complete(recent, None, 300) {
                        // Parse and test proposed skill (stub)
                        let _ = mem.upsert_fact("darkvision_proposal", &prop, 1.0);
                    }
                    last = Instant::now();
                }
                thread::sleep(Duration::from_secs(30));
            }
        });
        self.darkvision_thread = Some(t);
        Ok(())
    }

    fn shutdown(&mut self) {
        *self.stop_event.lock().unwrap() = true;
        if let Some(t) = self.reflect_thread.take() { let _ = t.join(); }
        if let Some(t) = self.embed_thread.take() { let _ = t.join(); }
        if let Some(t) = self.darkvision_thread.take() { let _ = t.join(); }
        if let Some(t) = self.trainer_thread.take() { let _ = t.join(); }
        self.mem.shutdown();
    }

    fn handle_line(&mut self, line: &str) -> Result<String> {
        let trimmed = line.trim();
        if trimmed.starts_with('/') {
            return self.handle_command(trimmed);
        }
        self.mem.add_message("user", trimmed)?;

        let mut system = format!("{}\n{}", BASE_PROMPT, SAFETY_PROMPT);
        if self.darkmode {
            system.push_str(TOOLS_PROMPT);
        }
        let mut prompt = self.build_history_context()?;
        prompt.push_str(&format!("\nUser: {}\nAlice: ", trimmed));

        let mut out = self.llm.complete(&prompt, Some(&system), 700)?;
        out = strip_thoughts(&out);

        let mut tool_loop = 0;
        let max_loops = if self.darkmode { 5 } else { 0 };
        while let Some((name, args)) = self.parse_tool_call(&out) {
            if tool_loop >= max_loops { break; }
            let tool_res = self.exec_tool(&name, &args)?;
            prompt.push_str(&format!("\nTOOL_RESULT: {}\n", tool_res));
            out = self.llm.complete(&prompt, Some(&system), 700)?;
            out = strip_thoughts(&out);
            tool_loop += 1;
        }

        self.mem.add_message("assistant", &out)?;
        Ok(out)
    }

    fn build_history_context(&self) -> Result<String> {
        let conn = self.mem.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT role, content FROM messages ORDER BY id DESC LIMIT ?1")?;
        let rows = stmt.query_map([self.cfg.args.max_history as i64], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut hist: Vec<(String, String)> = rows.collect::<Result<_, rusqlite::Error>>().map_err(anyhow::Error::from)?;
        hist.reverse();
        let mut prompt = String::new();
        for (role, content) in hist {
            let role_cap = if !role.is_empty() {
                format!("{}{}", role[0..1].to_uppercase(), &role[1..])
            } else {
                "Unknown".to_string()
            };
            prompt.push_str(&format!("{}: {}\n", role_cap, content));
        }
        Ok(prompt)
    }

    fn parse_tool_call(&self, out: &str) -> Option<(String, HashMap<String, JsonValue>)> {
        let re = Regex::new(r"<<call:(?P<name>[A-Za-z0-9_-]+)\s+args='(?P<args>.*?)'>>").ok()?;
        re.captures(out).map(|c| {
            let name = c.name("name").unwrap().as_str().to_string();
            let args_str = c.name("args").unwrap().as_str();
            let args: HashMap<String, JsonValue> = serde_json::from_str(args_str).unwrap_or_default();
            (name, args)
        })
    }

    fn exec_tool(&mut self, name: &str, args: &HashMap<String, JsonValue>) -> Result<String> {
        match name {
            "calc" => {
                let expr = args.get("expr").and_then(|v| v.as_str()).unwrap_or("");
                let res = evalexpr::eval(expr).map(|v| v.to_string()).map_err(|e| anyhow!(e.to_string()))?;
                Ok(json!({"ok": true, "result": res}).to_string())
            }
            "recall" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                self.mem.recall(query, self.cfg.args.recall_k)
            }
            "set_fact" => {
                let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
                let value = args.get("value").and_then(|v| v.as_str()).unwrap_or("");
                self.mem.upsert_fact(key, value, 0.5)?;
                Ok(json!({"ok": true}).to_string())
            }
            "get_facts" => self.mem.get_facts(),
            "browse" => {
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
                self.tool_browse(url)
            }
            "noaa_weather" => {
                let lat = args.get("lat").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let lon = args.get("lon").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let hourly = args.get("hourly").and_then(|v| v.as_bool()).unwrap_or(false);
                let out = self.tool_noaa_weather(lat, lon, hourly)?;
                Ok(serde_json::to_string(&out)?)
            }
            "nmap_scan" => {
                let target = args.get("target").and_then(|v| v.as_str()).unwrap_or("");
                let options = args.get("options").and_then(|v| v.as_str()).unwrap_or("");
                self.tool_nmap_scan(target, options)
            }
            "nikto_scan" => {
                let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
                self.tool_nikto_scan(url)
            }
            "nuclei_scan" => {
                let target = args.get("target").and_then(|v| v.as_str()).unwrap_or("");
                self.tool_nuclei_scan(target)
            }
            "wifi_scan" => self.tool_wifi_scan(),
            "ghidra_functions" => {
                let file = args.get("file").and_then(|v| v.as_str()).unwrap_or("");
                self.tool_ghidra_functions(file)
            }
            // Add more: zap_scan, msf_check (stubbed for safety)
            _ => {
                // Check skills
                let conn = self.mem.conn.lock().unwrap();
                let code: Option<String> = conn.query_row(
                    "SELECT code FROM skills WHERE name=?1 AND approved=1",
                    [name],
                    |row| row.get(0)
                ).optional()?;
                if let Some(code) = code {
                    // Sandbox execution; stub for v0.2.0 (use rustpython-vm in future)
                    Ok(json!({"ok": false, "error": "skill execution stubbed"}).to_string())
                } else {
                    Err(anyhow!("unknown tool or unapproved skill: {}", name))
                }
            }
        }
    }

    fn handle_command(&mut self, cmd: &str) -> Result<String> {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return Ok("".to_string());
        }
        match parts[0] {
            "/help" => Ok(self.help_text()),
            "/facts" => self.mem.get_facts(),
            "/recall" => {
                let query = parts.get(1..).unwrap_or(&[]).join(" ");
                self.mem.recall(&query, self.cfg.args.recall_k)
            }
            "/good" | "/bad" => {
                let delta = if parts[0] == "/good" { 1.0 } else { -1.0 };
                let reason = parts.get(1..).unwrap_or(&[]).join(" ");
                self.add_reward(delta, &reason)
            }
            "/teach" => self.cmd_teach(&parts),
            "/run" => self.cmd_run(&parts),
            "/ingest" => self.cmd_ingest(&parts),
            "/train" => self.cmd_train(&parts),
            "/darkvision" => self.cmd_darkvision(&parts),
            "/darkmode" => self.cmd_darkmode(&parts),
            "/model" => self.cmd_model(&parts),
            "/embed" => self.cmd_embed(&parts),
            "/web" => self.cmd_web(&parts),
            "/weather" => self.cmd_weather(&parts),
            "/pentest" => self.cmd_pentest(&parts),
            "/doctor" => self.cmd_doctor(),
            _ => Err(anyhow!("unknown command: {}", parts[0])),
        }
    }

    fn help_text(&self) -> String {
        r#"- /help — show this help
- /facts — list top remembered facts
- /recall <query> — search past chat/content
- /good | /bad — reward/punish last answer
- /teach <name> ```python ... ``` — add a Python skill (stubbed)
- /run <name> {json} — run a tool or skill
- /ingest <path> — ingest text files
- /train status|now — snapshot for training
- /darkvision start|status|stop — auto-improvement
- /darkmode on|off|status — autonomous tools
- /model list|info|set <tag> — manage models
- /embed on|off|model <name> — embeddings
- /web on|off — web access
- /weather <lat,lon> [hourly] — NOAA forecast
- /pentest on|off|status|allow <target>|revoke <target>|list|doctor — pentest mode
- /doctor — diagnostics"#.to_string()
    }

    fn add_reward(&self, delta: f64, reason: &str) -> Result<String> {
        let ts = Memory::now_iso();
        let conn = self.mem.conn.lock().unwrap();
        let last_mid: Option<i64> = conn.query_row(
            "SELECT id FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get(0)
        ).optional()?;
        if let Some(mid) = last_mid {
            conn.execute(
                "INSERT INTO rewards(message_id, delta, reason, ts) VALUES (?,?,?,?)",
                params![mid, delta, reason, ts],
            )?;
            Ok(format!("- reward applied: {:.1} ({})", delta, reason))
        } else {
            Err(anyhow!("no recent assistant message"))
        }
    }

    fn cmd_teach(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /teach <name> ```python code```"));
        }
        let name = parts[1];
        let code = parts[2..].join(" ");
        // Validate code (stub: check for def skill_main)
        if !code.contains("def skill_main(") {
            return Err(anyhow!("skill must define skill_main(**kwargs)"));
        }
        let ts = Memory::now_iso();
        let approved = if self.cfg.args.allow_dangerous_skills { 1 } else { 0 };
        let conn = self.mem.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO skills(name, code, approved, usage_count, created_ts) VALUES (?,?,?,?,?)",
            params![name, code, approved, 0, ts],
        )?;
        Ok(format!("- skill taught: {} (approved={})", name, approved))
    }

    fn cmd_run(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /run <name> {{json}}"));
        }
        let name = parts[1];
        let args_str = parts[2..].join(" ");
        let args: HashMap<String, JsonValue> = serde_json::from_str(&args_str).unwrap_or_default();
        self.exec_tool(name, &args)
    }

    fn cmd_ingest(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /ingest <path>"));
        }
        let path = shellexpand::tilde(parts[1]).to_string();
        let mut count = 0;
        for entry in WalkDir::new(&path).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() && entry.path().extension().map(|e| e == "txt" || e == "md").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    let key = format!("ingest:{}", entry.path().display());
                    self.mem.upsert_fact(&key, &content, 1.0)?;
                    count += 1;
                }
            }
        }
        Ok(format!("- ingested {} files", count))
    }

    fn cmd_train(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /train status|now"));
        }
        match parts[1] {
            "status" => Ok("- training status: stub".to_string()),
            "now" => {
                Self::train_snapshot(&self.mem, "train")?;
                Ok("- snapshot taken".to_string())
            }
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn train_snapshot(mem: &Memory, dir: &str) -> Result<()> {
        fs::create_dir_all(dir)?;
        let path = Path::new(dir).join("alice_sft.jsonl");
        let mut f = File::create(path)?;
        let conn = mem.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT role, content FROM messages ORDER BY id")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        let mut pairs: Vec<JsonValue> = Vec::new();
        let mut sys = json!({"role": "system", "content": BASE_PROMPT});
        let mut user = None;
        for r in rows {
            let (role, content): (String, String) = r.map_err(anyhow::Error::from)?;
            if role == "system" {
                sys["content"] = json!(format!("{} {}", sys["content"].as_str().unwrap_or(""), content));
            } else if role == "user" {
                user = Some(json!({"role": "user", "content": content}));
            } else if role == "assistant" && user.is_some() {
                let mut msgs = vec![sys.clone()];
                msgs.push(user.take().unwrap());
                msgs.push(json!({"role": "assistant", "content": content}));
                pairs.push(json!({"messages": msgs}));
            }
        }
        for p in pairs {
            writeln!(f, "{}", p.to_string())?;
        }
        Ok(())
    }

    fn cmd_darkvision(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /darkvision start|status|stop"));
        }
        match parts[1] {
            "start" => {
                self.start_darkvision()?;
                Ok("- darkvision started".to_string())
            }
            "status" => Ok(format!("- darkvision: {}", if self.darkvision_thread.is_some() { "running" } else { "stopped" })),
            "stop" => {
                if let Some(t) = self.darkvision_thread.take() {
                    *self.stop_event.lock().unwrap() = true;
                    let _ = t.join();
                    *self.stop_event.lock().unwrap() = false;
                }
                Ok("- darkvision stopped".to_string())
            }
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn cmd_darkmode(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /darkmode on|off|status"));
        }
        match parts[1] {
            "on" => {
                self.darkmode = true;
                Ok("- darkmode: ON".to_string())
            }
            "off" => {
                self.darkmode = false;
                Ok("- darkmode: OFF".to_string())
            }
            "status" => Ok(format!("- darkmode: {}", if self.darkmode { "ON" } else { "OFF" })),
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn cmd_model(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /model list|info|set <tag>"));
        }
        match parts[1] {
            "list" => {
                let models = self.llm.list_models()?;
                Ok(models.join("\n"))
            }
            "info" => {
                let info = self.llm.show_model(None)?;
                Ok(info.to_string())
            }
            "set" => {
                if parts.len() < 3 {
                    return Err(anyhow!("missing tag"));
                }
                self.llm.model = parts[2].to_string();
                Ok(format!("- model set: {}", self.llm.model))
            }
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn cmd_embed(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /embed on|off|model <name>"));
        }
        match parts[1] {
            "on" => {
                self.embed_enable = true;
                Ok("- embeddings: ON".to_string())
            }
            "off" => {
                self.embed_enable = false;
                Ok("- embeddings: OFF".to_string())
            }
            "model" => {
                if parts.len() < 3 {
                    return Err(anyhow!("missing name"));
                }
                self.llm.embed_model = parts[2].to_string();
                Ok(format!("- embed model: {}", self.llm.embed_model))
            }
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn cmd_web(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /web on|off"));
        }
        match parts[1] {
            "on" => {
                self.web_access = true;
                Ok("- web: ON".to_string())
            }
            "off" => {
                self.web_access = false;
                Ok("- web: OFF".to_string())
            }
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn cmd_weather(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /weather <lat,lon> [hourly]"));
        }
        let coords = parts[1].split(',').collect::<Vec<_>>();
        if coords.len() != 2 {
            return Err(anyhow!("invalid lat,lon"));
        }
        let lat: f64 = coords[0].parse().unwrap_or(0.0);
        let lon: f64 = coords[1].parse().unwrap_or(0.0);
        let hourly = parts.get(2).map_or(false, |&s| s == "hourly");
        let out = self.tool_noaa_weather(lat, lon, hourly)?;
        Ok(serde_json::to_string_pretty(&out)?)
    }

    fn cmd_pentest(&mut self, parts: &[&str]) -> Result<String> {
        if parts.len() < 2 {
            return Err(anyhow!("usage: /pentest on|off|status|allow <target>|revoke <target>|list|doctor"));
        }
        match parts[1] {
            "on" => {
                self.pentest_enabled = true;
                Ok("- pentest: ON".to_string())
            }
            "off" => {
                self.pentest_enabled = false;
                Ok("- pentest: OFF".to_string())
            }
            "status" => {
                let allow = self.mem.list_facts(50)?.into_iter().filter(|(k, _, _)| k.starts_with("pentest.allow:")).map(|(k, _, _)| k.replace("pentest.allow:", "")).collect::<Vec<_>>();
                Ok(format!("- pentest: {}\n- allowlist: {}", if self.pentest_enabled { "ON" } else { "OFF" }, allow.join(", ")))
            }
            "allow" => {
                if parts.len() < 3 {
                    return Err(anyhow!("missing target"));
                }
                let tgt = parts[2].to_lowercase();
                self.mem.upsert_fact(&format!("pentest.allow:{}", tgt), "1", 0.5)?;
                Ok(format!("- allowlisted: {}", tgt))
            }
            "revoke" => {
                if parts.len() < 3 {
                    return Err(anyhow!("missing target"));
                }
                let tgt = parts[2].to_lowercase();
                let conn = self.mem.conn.lock().unwrap();
                conn.execute("DELETE FROM facts WHERE key=?", [format!("pentest.allow:{}", tgt)])?;
                Ok(format!("- revoked: {}", tgt))
            }
            "list" => {
                let allow = self.mem.list_facts(50)?.into_iter().filter(|(k, _, _)| k.starts_with("pentest.allow:")).map(|(k, _, _)| k.replace("pentest.allow:", "")).collect::<Vec<_>>();
                Ok(allow.join("\n"))
            }
            "doctor" => self.cmd_doctor(),
            _ => Err(anyhow!("invalid subcommand")),
        }
    }

    fn cmd_doctor(&self) -> Result<String> {
        let bins = vec![
            ("nmap", &self.cfg.args.nmap_binary),
            ("nikto", &self.cfg.args.nikto_binary),
            ("nuclei", &self.cfg.args.nuclei_binary),
            ("nmcli", &self.cfg.args.nmcli_binary),
        ];
        let mut out = String::new();
        for (name, bin) in bins {
            let path = which(bin).ok();
            out.push_str(&format!("- {}: {} ({})\n", name, if path.is_some() { "ok" } else { "missing" }, path.map(|p| p.display().to_string()).unwrap_or_default()));
        }
        Ok(out)
    }

    // Tools impl
    fn tool_browse(&self, url: &str) -> Result<String> {
        if !self.web_access {
            return Err(anyhow!("web access disabled"));
        }
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(15)).build()?;
        let resp = client.get(url).header("User-Agent", "AliceBot/0.2").send()?;
        let text = resp.text()?;
        let rendered = html2text::from_read(text.as_bytes(), 80);
        let snippet = truncate(&rendered, 1000);
        Ok(json!({"ok": true, "content": snippet}).to_string())
    }

    fn tool_noaa_weather(&self, lat: f64, lon: f64, hourly: bool) -> Result<WeatherOut> {
        if !self.web_access {
            return Ok(WeatherOut { ok: false, error: Some("web access disabled".into()), ..Default::default() });
        }
        // Similar to existing; reuse code
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(15)).build()?;
        let contact = if !self.cfg.args.contact_email.is_empty() { self.cfg.args.contact_email.clone() } else { "no-contact@example.com".into() };
        let ua = format!("AliceWeather/0.2 ({})", contact);
        let p_url = format!("https://api.weather.gov/points/{:.4},{:.4}", lat, lon);
        let p_resp: JsonValue = client.get(p_url).header("User-Agent", ua.clone()).send()?.json()?;
        let props = p_resp.get("properties").cloned().unwrap_or(json!({}));
        let location = props.get("relativeLocation").and_then(|rl| rl.get("properties")).and_then(|p| {
            let city = p.get("city").and_then(|v| v.as_str());
            let state = p.get("state").and_then(|v| v.as_str());
            Some(format!("{}, {}", city?, state?))
        }).unwrap_or(format!("{:.4},{:.4}", lat, lon));
        let forecast_url = if hourly { props.get("forecastHourly") } else { props.get("forecast") }.and_then(|v| v.as_str()).unwrap_or("");
        if forecast_url.is_empty() {
            return Ok(WeatherOut { ok: false, error: Some("no forecast URL".into()), ..Default::default() });
        }
        let f_resp: JsonValue = client.get(forecast_url).header("User-Agent", ua).send()?.json()?;
        let periods_arr = f_resp.get("properties").and_then(|p| p.get("periods")).and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let take = if hourly { 6 } else { 4 };
        let mut periods = Vec::new();
        for p in periods_arr.into_iter().take(take) {
            periods.push(ForecastPeriod {
                name: p.get("name").and_then(|v| v.as_str()).map(Into::into),
                start_time: p.get("startTime").and_then(|v| v.as_str()).map(Into::into),
                end_time: p.get("endTime").and_then(|v| v.as_str()).map(Into::into),
                temperature: p.get("temperature").and_then(|v| v.as_i64()),
                unit: p.get("temperatureUnit").and_then(|v| v.as_str()).map(Into::into),
                wind: p.get("windSpeed").and_then(|v| v.as_str()).map(|ws| format!("{} {}", ws, p.get("windDirection").and_then(|vd| vd.as_str()).unwrap_or(""))).map(|s| s.trim().to_string()),
                short_forecast: p.get("shortForecast").and_then(|v| v.as_str()).map(Into::into),
                detailed_forecast: p.get("detailedForecast").and_then(|v| v.as_str()).map(|s| truncate(s, 240)),
            });
        }
        Ok(WeatherOut { ok: true, location: Some(location), periods: Some(periods), hourly, error: None })
    }

    fn pentest_allowed(&self, target: &str) -> bool {
        if !self.pentest_enabled { return false; }
        self.mem.list_facts(50).unwrap_or_default().iter().any(|(k, _, _)| k == &format!("pentest.allow:{}", target.to_lowercase()))
    }

    fn tool_nmap_scan(&self, target: &str, options: &str) -> Result<String> {
        if !self.pentest_allowed(target) {
            return Err(anyhow!("target not allowed"));
        }
        let bin = which(&self.cfg.args.nmap_binary).map_err(|_| anyhow!("nmap not found"))?;
        let output = Command::new(bin).args(options.split_whitespace()).arg(target).output()?;
        Ok(format!("stdout: {}\nstderr: {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr)))
    }

    fn tool_nikto_scan(&self, url: &str) -> Result<String> {
        let parsed_url = Url::parse(url).map_err(|_| anyhow!("invalid url"))?;
        let host = parsed_url.host_str().unwrap_or("").to_lowercase();
        if !self.pentest_allowed(&host) {
            return Err(anyhow!("target not allowed"));
        }
        let bin = which(&self.cfg.args.nikto_binary).map_err(|_| anyhow!("nikto not found"))?;
        let output = Command::new(bin).args(["-h", url, "-ask", "no"]).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn tool_nuclei_scan(&self, target: &str) -> Result<String> {
        if !self.pentest_allowed(target) {
            return Err(anyhow!("target not allowed"));
        }
        let bin = which(&self.cfg.args.nuclei_binary).map_err(|_| anyhow!("nuclei not found"))?;
        let output = Command::new(bin).args(["-u", target, "-silent"]).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn tool_wifi_scan(&self) -> Result<String> {
        let bin = which(&self.cfg.args.nmcli_binary).map_err(|_| anyhow!("nmcli not found"))?;
        let output = Command::new(bin).args(["-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi"]).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut nets = Vec::new();
        for line in stdout.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() >= 2 {
                nets.push(json!({
                    "ssid": parts[0],
                    "signal": parts[1],
                    "security": parts.get(2..).unwrap_or(&[]).join(":")
                }));
            }
        }
        Ok(json!({"ok": true, "networks": nets}).to_string())
    }

    fn tool_ghidra_functions(&self, file: &str) -> Result<String> {
        if self.cfg.args.ghidra_headless.is_empty() {
            return Err(anyhow!("ghidra headless not configured"));
        }
        let headless = which(&self.cfg.args.ghidra_headless).map_err(|_| anyhow!("ghidra not found"))?;
        // Stub: run command, capture output
        let output = Command::new(headless).args(["-import", file, "-scriptPath", "/path/to/scripts", "-postScript", "ListFunctions.java"]).output()?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_string() } else { format!("{}...", &s[..n]) }
}

fn strip_thoughts(s: &str) -> String {
    let re = Regex::new(r"(?m)^\s*(Thought|Chain|Internal):.*$").unwrap();
    re.replace_all(s, "").trim().to_string()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let alice = Arc::new(Mutex::new(Alice::new(Config { args })?));
    let alice_clone = alice.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n[shutting down]");
        alice_clone.lock().unwrap().shutdown();
        std::process::exit(0);
    })?;

    println!("Alice gp v0.2.0 (Rust) is awake. Type /help for commands. (Ctrl+C to exit.)");

    let spinner_style = ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap();
    loop {
        print!("You> ");
        io::stdout().flush()?;
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        let line = buf.trim().to_string();
        if line.is_empty() { continue; }

        let pb = ProgressBar::new_spinner();
        pb.set_style(spinner_style.clone());
        pb.set_message("thinking");
        pb.enable_steady_tick(Duration::from_millis(80));

        let out = alice.lock().unwrap().handle_line(&line);

        pb.finish_and_clear();

        match out {
            Ok(s) if !s.is_empty() => println!("Alice> {}", s),
            Ok(_) => {},
            Err(e) => println!("Alice> error: {}", e),
        }
    }
}