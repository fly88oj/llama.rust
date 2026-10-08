//! server_mcp.rs — port of `tools/server/server-mcp.cpp` (pinned
//! bd4f514db1, 820 lines): the MCP (Model Context Protocol) stdio bridge —
//! `--mcp-servers-config`/`--mcp-servers-json` server definitions, the
//! JSON-RPC handshake, warmup tool discovery, and the per-call transports
//! the `/tools` endpoint proxies to.
//!
//! Mapping (server-mcp.cpp → Rust):
//!   `server_mcp_server_config::parse_cursor_format` (:139-172) -> [`ServerConfig::parse_cursor_format`]
//!   `MCP_PROTOCOL_VERSION` (:179)                          -> const below
//!   `rpc_error_message` (:181-192)                         -> [`rpc_error_message`]
//!   `mcp_result_to_response` (:196-212)                    -> [`mcp_result_to_response`]
//!   `server_mcp_transport::send_rpc` (:214-252)            -> [`StdioTransport::send_rpc`]
//!   `server_mcp_transport::ensure_init` (:254-281)         -> [`StdioTransport::ensure_init`]
//!   `server_mcp_transport::list_tools` (:283-313)          -> [`StdioTransport::list_tools`]
//!   `server_mcp_transport::call_tool` (:315-337)           -> [`StdioTransport::call_tool`]
//!   `server_mcp_stdio::start` (:473-502)                   -> [`StdioTransport::start`]
//!   `mcp_pump_ndjson` (:26-129)                            -> the reader/errlog threads'
//!                                                           bounded-line read loop
//!   `mcp_write_all` (:538-602)                             -> the writer thread's
//!                                                           blocking write (the child
//!                                                           dies into SIGPIPE-free
//!                                                           teardown the same way)
//!   `server_mcp::start/warmup` (:693-749)                  -> [`ServerMcp::start`]
//!   `server_mcp::get_or_create` (:771-819)                 -> [`ServerMcp::get_or_create`]
//!
//! Documented deviation: the C pumps its pipes with poll() so teardown never
//! blocks on a grandchild holding the write end; the port's reader threads
//! park in blocking reads that end at the terminate()-driven EOF (the same
//! unblock the C's poll loop is defending against, reached one step later),
//! and the 8 MiB per-line cap is kept.

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use llama::json_schema::Json;

use crate::subproc::Subproc;

/// `MCP_PROTOCOL_VERSION` (server-mcp.cpp:179)
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// drop any single NDJSON line larger than this (:32)
const MAX_LINE: usize = 8 * 1024 * 1024;

/// `MCP_COOLDOWN_SECONDS` (:657)
const MCP_COOLDOWN_SECONDS: u64 = 5;
/// `MCP_WARMUP_TIMEOUT_SECONDS` (:658)
const MCP_WARMUP_TIMEOUT_SECONDS: u64 = 10;

// ---------------------------------------------------------------------------
// config — server_mcp_server_config (server-mcp.h/.cpp)
// ---------------------------------------------------------------------------

/// `server_mcp_server_config`
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: String,
    pub timeout_ms: u64,
}

/// the C's `timeout_ms` member initializer (server-mcp.h); 30 s is the
/// reference's default
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

impl ServerConfig {
    /// `parse_from_json` (:135-137)
    pub fn parse_from_json(json_str: &str) -> Result<Vec<ServerConfig>, String> {
        let j = Json::parse(json_str)?;
        Ok(Self::parse_cursor_format(&j))
    }

    /// `parse_cursor_format` (:139-172) — the `{"mcpServers": {...}}` shape
    pub fn parse_cursor_format(j: &Json) -> Vec<ServerConfig> {
        let mut result = Vec::new();

        let Some(servers) = j.at("mcpServers") else {
            return result;
        };
        if !servers.is_object() {
            return result;
        }
        let Json::Object(entries) = servers else { return result };

        for (name, cfg) in entries {
            let mut sc = ServerConfig {
                name: name.clone(),
                timeout_ms: DEFAULT_TIMEOUT_MS,
                command: String::new(),
                args: Vec::new(),
                env: Vec::new(),
                cwd: String::new(),
            };
            sc.command = cfg
                .at("command")
                .and_then(|v| v.get_str().ok())
                .unwrap_or_default()
                .to_string();
            sc.cwd = cfg
                .at("cwd")
                .and_then(|v| v.get_str().ok())
                .unwrap_or_default()
                .to_string();
            sc.timeout_ms = cfg
                .at("timeout_ms")
                .and_then(|v| v.get_i64().ok())
                .map(|v| v.max(0) as u64)
                .unwrap_or(DEFAULT_TIMEOUT_MS);

            if let Some(args) = cfg.at("args") {
                if let Json::Array(items) = args {
                    for a in items {
                        if let Ok(s) = a.get_str() {
                            sc.args.push(s.to_string());
                        }
                    }
                }
            }
            if let Some(env) = cfg.at("env") {
                if let Json::Object(kvs) = env {
                    for (k, v) in kvs {
                        if let Ok(s) = v.get_str() {
                            sc.env.push((k.clone(), s.to_string()));
                        }
                    }
                }
            }

            if sc.command.is_empty() {
                eprintln!("MCP server '{name}' has no command, skipping");
                continue;
            }
            result.push(sc);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// `rpc_error_message` (:181-192)
fn rpc_error_message(resp: &Json) -> String {
    if let Some(e) = resp.at("error") {
        if e.is_object() {
            return e
                .at("message")
                .and_then(|m| m.get_str().ok())
                .unwrap_or("unknown error")
                .to_string();
        }
        if let Ok(s) = e.get_str() {
            return s.to_string();
        }
    }
    "unknown error".to_string()
}

/// `mcp_result_to_response` (:196-212) — concat text parts of
/// `result.content[]`, surface an `isError` result
fn mcp_result_to_response(result: &Json) -> Json {
    let mut text = String::new();
    if let Some(Json::Array(parts)) = result.at("content") {
        for part in parts {
            if part.is_object()
                && part
                    .at("type")
                    .and_then(|t| t.get_str().ok())
                    .is_some_and(|t| t == "text")
            {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(part.at("text").and_then(|t| t.get_str().ok()).unwrap_or(""));
            }
        }
    }
    let is_error = matches!(result.at("isError"), Some(Json::Bool(true)));
    if is_error {
        return Json::Object(vec![(
            "error".into(),
            Json::String(if text.is_empty() {
                "MCP tool returned an error".into()
            } else {
                text
            }),
        )]);
    }
    Json::Object(vec![("plain_text_response".into(), Json::String(text))])
}

/// one NDJSON frame queue between the reader thread and send_rpc — the C's
/// bounded `thread_pool` channel (`from_server.max_size = 65536`, :466)
struct FrameQueue {
    frames: std::collections::VecDeque<String>,
    closed: bool,
}

impl FrameQueue {
    fn push(&mut self, s: String) -> bool {
        if self.closed || self.frames.len() >= 65536 {
            return false; // consumer gone / saturated
        }
        self.frames.push_back(s);
        true
    }
}

// ---------------------------------------------------------------------------
// stdio transport — server_mcp_stdio
// ---------------------------------------------------------------------------

/// one discovered tool of one server (`server_mcp_tool_def`)
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub server_name: String,
    pub name: String,
    pub description: String,
    pub input_schema: Json,
}

/// `server_mcp_stdio` (:343-650): the child process + three pump threads.
pub struct StdioTransport {
    name: String,
    timeout_ms: u64,
    config: ServerConfig,

    proc: Mutex<Option<Subproc>>,
    running: Arc<AtomicBool>,
    /// the reader thread pushes reply/notification frames here
    from_server: Arc<Mutex<FrameQueue>>,
    /// send_rpc waits on this condvar for a matching frame
    frame_cv: std::sync::Condvar,
    /// the writer thread's outbound queue
    to_server: Arc<Mutex<std::collections::VecDeque<String>>>,
    to_cv: std::sync::Condvar,
    initialized: AtomicBool,
    next_id: AtomicU64,
    last_error: Mutex<String>,
    err_tail: Mutex<String>,
    rpc_mutex: Mutex<()>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl StdioTransport {
    /// `server_mcp_stdio::server_mcp_stdio` (:462-467)
    pub fn new(config: ServerConfig) -> Self {
        StdioTransport {
            name: config.name.clone(),
            timeout_ms: config.timeout_ms,
            config,
            proc: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            from_server: Arc::new(Mutex::new(FrameQueue {
                frames: std::collections::VecDeque::new(),
                closed: false,
            })),
            frame_cv: std::sync::Condvar::new(),
            to_server: Arc::new(Mutex::new(std::collections::VecDeque::new())),
            to_cv: std::sync::Condvar::new(),
            initialized: AtomicBool::new(false),
            next_id: AtomicU64::new(0),
            last_error: Mutex::new(String::new()),
            err_tail: Mutex::new(String::new()),
            rpc_mutex: Mutex::new(()),
            threads: Mutex::new(Vec::new()),
        }
    }

    /// `server_mcp_stdio::start` (:473-502)
    pub fn start(&self) -> bool {
        let mut argv_s = vec![self.config.command.clone()];
        argv_s.extend(self.config.args.iter().cloned());

        // env: parent env with the config overrides (:426-459 mcp_build_env)
        let env: Vec<(String, String)> = if self.config.env.is_empty() {
            Vec::new()
        } else {
            let mut env: Vec<(String, String)> = std::env::vars().collect();
            for (k, v) in &self.config.env {
                if let Some(slot) = env.iter_mut().find(|(ek, _)| ek == k) {
                    slot.1 = v.clone();
                } else {
                    env.push((k.clone(), v.clone()));
                }
            }
            env
        };

        let mut proc = Subproc::new();
        let cwd = if self.config.cwd.is_empty() {
            None
        } else {
            Some(self.config.cwd.as_str())
        };
        if !proc.create(&argv_s, &env, cwd) {
            eprintln!("MCP '{}': failed to spawn '{}'", self.config.name, self.config.command);
            return false;
        }
        let mut stdin = proc.take_stdin();
        let stdout = proc.take_stdout();
        let stderr = proc.take_stderr();

        self.running.store(true, Ordering::SeqCst);
        *self.proc.lock().unwrap() = Some(proc);

        // reader thread — mcp_pump_ndjson + reader_loop (:528-535)
        if let Some(out) = stdout {
            let running = self.running.clone();
            let from_server = self.from_server.clone();
            let frame_cv = &self.frame_cv as *const std::sync::Condvar as usize;
            let handle = std::thread::spawn(move || {
                // SAFETY: the Condvar outlives the thread (close() joins the
                // pumps before the transport drops); the raw pointer is only
                // used to notify waiters after each frame lands
                let cv: &std::sync::Condvar = unsafe { &*(frame_cv as *const _) };
                let mut reader = std::io::BufReader::new(out);
                loop {
                    let mut line = Vec::new();
                    // byte-wise bounded read: drop an over-long line's bytes
                    // but keep scanning to its newline (:94-127)
                    let mut eof = false;
                    let mut discarded = false;
                    loop {
                        let mut byte = [0u8; 1];
                        match reader.read(&mut byte) {
                            Ok(0) => {
                                eof = true;
                                break;
                            }
                            Err(_) => {
                                eof = true;
                                break;
                            }
                            Ok(_) => {
                                if byte[0] == b'\n' {
                                    break;
                                }
                                if line.len() < MAX_LINE {
                                    line.push(byte[0]);
                                } else if !discarded {
                                    discarded = true;
                                    eprintln!(
                                        "MCP: dropping oversized line (> {MAX_LINE} bytes) from child pipe"
                                    );
                                }
                            }
                        }
                    }
                    if eof && line.is_empty() {
                        break;
                    }
                    if line.is_empty() {
                        continue;
                    }
                    let mut s = String::from_utf8_lossy(&line).into_owned();
                    if s.ends_with('\r') {
                        s.pop();
                    }
                    if s.is_empty() {
                        continue;
                    }
                    let ok = {
                        let mut q = from_server.lock().unwrap();
                        q.push(s)
                    };
                    if !ok {
                        break; // consumer gone, stop (:530)
                    }
                    cv.notify_all();
                }
                running.store(false, Ordering::SeqCst);
                // EOF to any waiting caller (:533-534)
                if let Ok(mut q) = from_server.lock() {
                    q.closed = true;
                }
                cv.notify_all();
            });
            self.threads.lock().unwrap().push(handle);
        }

        // writer thread — writer_loop + mcp_write_all (:604-616)
        if let Some(mut stdin) = stdin {
            let running = self.running.clone();
            let to_server = self.to_server.clone();
            let to_cv = &self.to_cv as *const std::sync::Condvar as usize;
            let from_server = self.from_server.clone();
            let frame_cv = &self.frame_cv as *const std::sync::Condvar as usize;
            let handle = std::thread::spawn(move || {
                // SAFETY: same lifetime argument as the reader thread above
                let cv: &std::sync::Condvar = unsafe { &*(frame_cv as *const _) };
                let to_cv: &std::sync::Condvar = unsafe { &*(to_cv as *const _) };
                loop {
                    let msg = {
                        let mut q = to_server.lock().unwrap();
                        loop {
                            if let Some(m) = q.pop_front() {
                                break m;
                            }
                            if !running.load(Ordering::SeqCst) {
                                break String::new();
                            }
                            let (guard, _) =
                                to_cv.wait_timeout(q, Duration::from_millis(50)).unwrap();
                            q = guard;
                        }
                    };
                    if msg.is_empty() {
                        if !running.load(Ordering::SeqCst) {
                            break;
                        }
                        continue;
                    }
                    let mut frame = msg;
                    frame.push('\n');
                    if stdin.write_all(frame.as_bytes()).is_err() {
                        break; // child gone or shutting down
                    }
                    let _ = stdin.flush();
                }
                running.store(false, Ordering::SeqCst);
                // fail fast on any further send_rpc write + wake waiters
                // (:613-615)
                if let Ok(mut q) = from_server.lock() {
                    q.closed = true;
                }
                cv.notify_all();
            });
            self.threads.lock().unwrap().push(handle);
        }

        // errlog thread — errlog_loop (:618-632): drain stderr, keep a tail
        if let Some(err) = stderr {
            let running = self.running.clone();
            let name = self.name.clone();
            let err_tail = &self.err_tail as *const Mutex<String> as usize;
            let handle = std::thread::spawn(move || {
                // SAFETY: same lifetime argument as the pump threads above
                // (close() joins them before the transport drops)
                let err_tail: &Mutex<String> = unsafe { &*(err_tail as *const _) };
                let mut reader = std::io::BufReader::new(err);
                const ERR_TAIL_MAX: usize = 4096;
                loop {
                    let mut line = Vec::new();
                    match read_line_bounded(&mut reader, &mut line) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                    if !line.is_empty() {
                        let s = String::from_utf8_lossy(&line).into_owned();
                        eprintln!("MCP '{name}' stderr: {s}");
                        let mut tail = err_tail.lock().unwrap();
                        tail.push_str(&s);
                        tail.push('\n');
                        if tail.len() > ERR_TAIL_MAX {
                            let cut = tail.len() - ERR_TAIL_MAX;
                            tail.drain(..cut);
                        }
                    }
                    if !running.load(Ordering::SeqCst) {
                        break;
                    }
                }
            });
            self.threads.lock().unwrap().push(handle);
        }

        true
    }

    /// `is_alive` (:508-510)
    pub fn is_alive(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// `diagnostics` (:512-526)
    pub fn diagnostics(&self) -> String {
        let mut out = self.last_error.lock().unwrap().clone();
        let tail = self.err_tail.lock().unwrap().clone();
        if !tail.is_empty() {
            if !out.is_empty() {
                out.push_str("; ");
            }
            out.push_str("last stderr: ");
            out.push_str(&tail);
        }
        out
    }

    /// `close` / `join_pumps` (:504-506, :634-650)
    pub fn close(&self) {
        self.running.store(false, Ordering::SeqCst);
        // wake the writer and any waiting caller
        self.to_cv.notify_all();
        if let Ok(mut q) = self.from_server.try_lock() {
            q.closed = true;
        }
        if let Ok(mut guard) = self.proc.try_lock() {
            if let Some(p) = guard.as_mut() {
                p.terminate(); // child death unblocks the readers
            }
            // reap the child
            if let Some(mut p) = guard.take() {
                p.join();
            }
        }
        // join the pumps
        let mut threads = self.threads.lock().unwrap();
        for t in threads.drain(..) {
            let _ = t.join();
        }
    }

    /// `send_rpc` (:214-252)
    fn send_rpc(&self, request: &Json, should_stop: &dyn Fn() -> bool) -> Json {
        {
            let mut q = self.to_server.lock().unwrap();
            q.push_back(request.dump());
        }
        self.to_cv.notify_all();

        let has_id = request.at("id").is_some();
        let req_id = request.at("id").cloned();
        let deadline = Instant::now() + Duration::from_millis(self.timeout_ms);

        loop {
            if (should_stop)() || Instant::now() >= deadline {
                break;
            }
            // wait a bounded time for a frame
            let frame = {
                let mut q = self.from_server.lock().unwrap();
                if q.frames.is_empty() && !q.closed {
                    let (guard, _) = self
                        .frame_cv
                        .wait_timeout(q, Duration::from_millis(50))
                        .unwrap();
                    guard
                } else {
                    q
                }
            };
            let mut q = frame;
            while let Some(frame) = q.frames.pop_front() {
                let reply = match Json::parse(&frame) {
                    Ok(r) => r,
                    Err(_) => {
                        if Instant::now() >= deadline {
                            break;
                        }
                        continue; // skip malformed frame
                    }
                };
                // no id: a notification. mismatched id: a stale reply
                // (:236-242)
                let matches = match (&has_id, &req_id) {
                    (false, _) => true,
                    (true, Some(id)) => reply.at("id") == Some(id),
                    (true, None) => false,
                };
                if matches {
                    return reply;
                }
                if Instant::now() >= deadline {
                    break;
                }
            }
        }

        if (should_stop)() {
            return Json::Object(vec![(
                "error".into(),
                Json::Object(vec![
                    ("code".into(), Json::Int(-32603)),
                    ("message".into(), Json::String("cancelled".into())),
                ]),
            )]);
        }
        if Instant::now() >= deadline {
            return Json::Object(vec![(
                "error".into(),
                Json::Object(vec![
                    ("code".into(), Json::Int(-32603)),
                    ("message".into(), Json::String("request timed out".into())),
                ]),
            )]);
        }
        Json::Object(vec![(
            "error".into(),
            Json::Object(vec![
                ("code".into(), Json::Int(-32603)),
                ("message".into(), Json::String("transport closed".into())),
            ]),
        )])
    }

    /// `ensure_init` (:254-281) — the initialize handshake
    fn ensure_init(&self, should_stop: &dyn Fn() -> bool) -> bool {
        if self.initialized.load(Ordering::SeqCst) {
            return true;
        }

        let init_req = Json::Object(vec![
            ("jsonrpc".into(), Json::String("2.0".into())),
            ("id".into(), Json::Int(self.next_id.fetch_add(1, Ordering::SeqCst) as i64)),
            ("method".into(), Json::String("initialize".into())),
            (
                "params".into(),
                Json::Object(vec![
                    (
                        "protocolVersion".into(),
                        Json::String(MCP_PROTOCOL_VERSION.into()),
                    ),
                    ("capabilities".into(), Json::Object(vec![])),
                    (
                        "clientInfo".into(),
                        Json::Object(vec![
                            ("name".into(), Json::String("llama.cpp".into())),
                            ("version".into(), Json::String("1.0".into())),
                        ]),
                    ),
                ]),
            ),
        ]);
        let resp = self.send_rpc(&init_req, should_stop);
        if resp.at("result").is_none() {
            *self.last_error.lock().unwrap() =
                format!("initialize failed: {}", rpc_error_message(&resp));
            return false;
        }

        // notifications/initialized: no id, no reply expected
        let notif = Json::Object(vec![
            ("jsonrpc".into(), Json::String("2.0".into())),
            ("method".into(), Json::String("notifications/initialized".into())),
        ]);
        {
            let mut q = self.to_server.lock().unwrap();
            q.push_back(notif.dump());
        }
        self.to_cv.notify_all();

        self.initialized.store(true, Ordering::SeqCst);
        true
    }

    /// `list_tools` (:283-313)
    pub fn list_tools(&self, should_stop: &dyn Fn() -> bool) -> Vec<ToolDef> {
        let _lock = self.rpc_mutex.lock().unwrap();
        if !self.ensure_init(should_stop) {
            return Vec::new();
        }

        let req = Json::Object(vec![
            ("jsonrpc".into(), Json::String("2.0".into())),
            ("id".into(), Json::Int(self.next_id.fetch_add(1, Ordering::SeqCst) as i64)),
            ("method".into(), Json::String("tools/list".into())),
        ]);
        let resp = self.send_rpc(&req, should_stop);
        let Some(result) = resp.at("result") else {
            *self.last_error.lock().unwrap() =
                format!("tools/list failed: {}", rpc_error_message(&resp));
            return Vec::new();
        };

        let mut tools = Vec::new();
        if let Some(Json::Array(items)) = result.at("tools") {
            for t in items {
                tools.push(ToolDef {
                    server_name: self.name.clone(),
                    name: t.at("name")
                        .and_then(|v| v.get_str().ok())
                        .unwrap_or("")
                        .to_string(),
                    description: t.at("description")
                        .and_then(|v| v.get_str().ok())
                        .unwrap_or("")
                        .to_string(),
                    input_schema: t.at("inputSchema").cloned().unwrap_or(Json::Null),
                });
            }
        }
        tools
    }

    /// `call_tool` (:315-337)
    pub fn call_tool(&self, tool_name: &str, arguments: &Json, should_stop: &dyn Fn() -> bool) -> Json {
        let _lock = self.rpc_mutex.lock().unwrap();
        if !self.ensure_init(should_stop) {
            let err = self.last_error.lock().unwrap().clone();
            return Json::Object(vec![("error".into(), Json::String(err))]);
        }

        let req = Json::Object(vec![
            ("jsonrpc".into(), Json::String("2.0".into())),
            ("id".into(), Json::Int(self.next_id.fetch_add(1, Ordering::SeqCst) as i64)),
            ("method".into(), Json::String("tools/call".into())),
            (
                "params".into(),
                Json::Object(vec![
                    ("name".into(), Json::String(tool_name.into())),
                    ("arguments".into(), arguments.clone()),
                ]),
            ),
        ]);
        let resp = self.send_rpc(&req, should_stop);
        if resp.at("error").is_some() {
            return Json::Object(vec![(
                "error".into(),
                Json::String(rpc_error_message(&resp)),
            )]);
        }
        if let Some(result) = resp.at("result") {
            return mcp_result_to_response(result);
        }
        Json::Object(vec![(
            "error".into(),
            Json::String("invalid response from MCP server".into()),
        )])
    }
}

/// bounded getline for the errlog drain
fn read_line_bounded<R: Read>(reader: &mut std::io::BufReader<R>, out: &mut Vec<u8>) -> std::io::Result<usize> {
    let mut byte = [0u8; 1];
    let mut n = 0;
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Ok(n),
            Ok(_) => {
                n += 1;
                if byte[0] == b'\n' {
                    return Ok(n);
                }
                if out.len() < MAX_LINE {
                    out.push(byte[0]);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// server_mcp — the manager
// ---------------------------------------------------------------------------

/// `server_mcp` (server-mcp.h + :656-820)
pub struct ServerMcp {
    configs: Vec<ServerConfig>,
    transports: Mutex<HashMap<String, Arc<StdioTransport>>>,
    dead_servers: Mutex<HashMap<String, Instant>>,
    registry: Mutex<Vec<ToolDef>>,
    stopping: AtomicBool,
}

impl Default for ServerMcp {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerMcp {
    pub fn new() -> Self {
        ServerMcp {
            configs: Vec::new(),
            transports: Mutex::new(HashMap::new()),
            dead_servers: Mutex::new(HashMap::new()),
            registry: Mutex::new(Vec::new()),
            stopping: AtomicBool::new(false),
        }
    }

    /// `find_config` (:684-691)
    fn find_config(&self, name: &str) -> Option<&ServerConfig> {
        self.configs.iter().find(|c| c.name == name)
    }

    /// `shutdown` (:680-682)
    pub fn shutdown(&self) {
        self.stopping.store(true, Ordering::SeqCst);
    }

    /// `create_transport` (:676-678) — the stdio transport is the only one
    fn create_transport(&self, cfg: &ServerConfig) -> Arc<StdioTransport> {
        Arc::new(StdioTransport::new(cfg.clone()))
    }

    /// `start` (:693-749): parse both config sources, warm every server up
    /// (discovering tools), and keep only the discovered registry — calls
    /// re-spawn transports on demand.
    pub fn start(
        &mut self,
        mcp_servers_config: &str,
        mcp_servers_json: &str,
    ) -> Result<(), String> {
        let mut append = |json_str: &str, configs: &mut Vec<ServerConfig>| -> Result<(), String> {
            let parsed = ServerConfig::parse_from_json(json_str)
                .map_err(|e| format!("failed to parse MCP config JSON: {e}"))?;
            if parsed.is_empty() {
                eprintln!("MCP config: no servers found in JSON");
            }
            for p in parsed {
                // names must be unique across both config sources
                if configs.iter().any(|c| c.name == p.name) {
                    eprintln!("MCP config: duplicate server name '{}', skipping", p.name);
                    continue;
                }
                configs.push(p);
            }
            Ok(())
        };

        if !mcp_servers_config.is_empty() {
            let text = std::fs::read_to_string(mcp_servers_config).map_err(|_| {
                format!("failed to open MCP config file: {mcp_servers_config}")
            })?;
            append(&text, &mut self.configs)?;
        }
        if !mcp_servers_json.is_empty() {
            append(mcp_servers_json, &mut self.configs)?;
        }

        if self.configs.is_empty() {
            return Ok(());
        }

        let mut discovered = Vec::new();
        for cfg in &self.configs {
            let t = self.create_transport(cfg);
            if !t.start() {
                eprintln!("MCP warmup: failed to spawn '{}': {}", cfg.name, t.diagnostics());
                continue;
            }
            // bound warmup per server so an unresponsive one can't stall
            // startup for the full per-call timeout (:736-740)
            let deadline = Instant::now() + Duration::from_secs(MCP_WARMUP_TIMEOUT_SECONDS);
            let stopping = &self.stopping;
            let should_stop = move || stopping.load(Ordering::SeqCst) || Instant::now() >= deadline;
            let tools = t.list_tools(&should_stop);
            eprintln!("MCP warmup: '{}' discovered {} tools", cfg.name, tools.len());
            discovered.extend(tools);
            t.close();
        }

        *self.registry.lock().unwrap() = discovered;
        Ok(())
    }

    /// `empty` — any tools discovered
    pub fn is_empty(&self) -> bool {
        self.registry.lock().unwrap().is_empty()
    }

    /// `list_tools` (:751-754)
    pub fn list_tools(&self) -> Vec<ToolDef> {
        self.registry.lock().unwrap().clone()
    }

    /// `call_tool` (:756-769)
    pub fn call_tool(&self, server_name: &str, tool_name: &str, arguments: &Json) -> Json {
        let Some(transport) = self.get_or_create(server_name) else {
            return Json::Object(vec![(
                "error".into(),
                Json::String(format!("MCP server unavailable: {server_name}")),
            )]);
        };
        let stopping = &self.stopping;
        transport.call_tool(tool_name, arguments, &|| stopping.load(Ordering::SeqCst))
    }

    /// `get_or_create` (:771-819) — reuse the live transport, respawn a dead
    /// one, cooldown a failing one
    fn get_or_create(&self, name: &str) -> Option<Arc<StdioTransport>> {
        let mut to_close: Vec<Arc<StdioTransport>> = Vec::new();
        let result;

        {
            let mut transports = self.transports.lock().unwrap();
            if self.stopping.load(Ordering::SeqCst) {
                return None;
            }

            let now = Instant::now();
            {
                let mut dead = self.dead_servers.lock().unwrap();
                if let Some(until) = dead.get(name).copied() {
                    if now < until {
                        return None;
                    }
                    dead.remove(name);
                }
            }

            if let Some(t) = transports.get(name) {
                if t.is_alive() {
                    return Some(t.clone());
                }
                eprintln!("MCP '{name}' is no longer alive: {}", t.diagnostics());
                to_close.push(t.clone());
                transports.remove(name);
            }

            if let Some(cfg) = self.find_config(name) {
                let fresh = self.create_transport(cfg);
                if fresh.start() && fresh.is_alive() {
                    transports.insert(name.to_string(), fresh.clone());
                    result = Some(fresh);
                } else {
                    eprintln!("MCP '{name}': failed to start: {}", fresh.diagnostics());
                    to_close.push(fresh);
                    self.dead_servers
                        .lock()
                        .unwrap()
                        .insert(name.to_string(), now + Duration::from_secs(MCP_COOLDOWN_SECONDS));
                    result = None;
                }
            } else {
                result = None;
            }
        }

        for t in to_close {
            t.close(); // blocking call, no leaks
        }

        result
    }
}

impl Drop for ServerMcp {
    /// `~server_mcp` (:660-674)
    fn drop(&mut self) {
        self.shutdown();
        let to_close: Vec<Arc<StdioTransport>> = {
            let mut transports = self.transports.lock().unwrap();
            transports.drain().map(|(_, v)| v).collect()
        };
        for t in to_close {
            t.close();
        }
    }
}

// ---------------------------------------------------------------------------
// tests — the config parser and the result normalization against the C's
// tests. The transport's wire behavior is exercised end-to-end by
// parity/server_tools_mcp_parity.sh against the reference server.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_format_parse() {
        // server-mcp.cpp:139-172
        let j = Json::parse(
            r#"{"mcpServers": {
                "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "timeout_ms": 1500},
                "bad": {"args": ["x"]},
                "envy": {"command": "c", "env": {"A": "1", "B": "2"}, "cwd": "/tmp"}}
        }"#,
        )
        .unwrap();
        let cfgs = ServerConfig::parse_cursor_format(&j);
        assert_eq!(cfgs.len(), 2); // "bad" has no command -> skipped
        let fetch = &cfgs[0];
        assert_eq!(fetch.name, "fetch");
        assert_eq!(fetch.command, "uvx");
        assert_eq!(fetch.args, vec!["mcp-server-fetch"]);
        assert_eq!(fetch.timeout_ms, 1500);
        let envy = &cfgs[1];
        assert_eq!(envy.env, vec![("A".to_string(), "1".into()), ("B".into(), "2".into())]);
        assert_eq!(envy.cwd, "/tmp");
        assert_eq!(envy.timeout_ms, DEFAULT_TIMEOUT_MS);

        // no mcpServers key -> empty
        assert!(ServerConfig::parse_cursor_format(&Json::parse("{}").unwrap()).is_empty());
    }

    #[test]
    fn rpc_error_message_shapes() {
        // server-mcp.cpp:181-192
        let j = Json::parse(r#"{"error": {"code": -1, "message": "boom"}}"#).unwrap();
        assert_eq!(rpc_error_message(&j), "boom");
        let j = Json::parse(r#"{"error": "plain"}"#).unwrap();
        assert_eq!(rpc_error_message(&j), "plain");
        let j = Json::parse(r#"{"result": {}}"#).unwrap();
        assert_eq!(rpc_error_message(&j), "unknown error");
    }

    #[test]
    fn mcp_result_to_response_contract() {
        // server-mcp.cpp:196-212 — the /tools contract normalization
        let j = Json::parse(
            r#"{"content": [{"type": "text", "text": "a"}, {"type": "image", "text": "x"}, {"type": "text", "text": "b"}]}"#,
        )
        .unwrap();
        let r = mcp_result_to_response(&j);
        assert_eq!(
            r.at("plain_text_response").and_then(|v| v.get_str().ok()),
            Some("a\nb")
        );

        let j = Json::parse(r#"{"content": [{"type": "text", "text": "oops"}], "isError": true}"#).unwrap();
        let r = mcp_result_to_response(&j);
        assert_eq!(r.at("error").and_then(|v| v.get_str().ok()), Some("oops"));

        let j = Json::parse(r#"{"isError": true}"#).unwrap();
        let r = mcp_result_to_response(&j);
        assert_eq!(
            r.at("error").and_then(|v| v.get_str().ok()),
            Some("MCP tool returned an error")
        );
    }

    /// a stdio round-trip against a shell "MCP server": initialize +
    /// tools/list through the real transport threads
    #[test]
    fn stdio_transport_handshake_and_call() {
        // an NDJSON JSON-RPC echo server implemented in sh: replies with the
        // id it was asked, echoing a canned result
        let script = r#"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  method=$(printf '%s' "$line" | sed -n 's/.*"method":"\([a-z/]*\)".*/\1/p')
  if [ "$method" = "initialize" ]; then
    printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05"}}\n' "$id"
  elif [ "$method" = "tools/list" ]; then
    printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echo tool","inputSchema":{"type":"object"}}]}}\n' "$id"
  elif [ "$method" = "tools/call" ]; then
    printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"called"}]}}\n' "$id"
  fi
done
"#;
        let cfg = ServerConfig {
            name: "shellmcp".into(),
            command: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            env: vec![],
            cwd: String::new(),
            timeout_ms: 10_000,
        };
        let t = StdioTransport::new(cfg);
        assert!(t.start());
        let tools = t.list_tools(&|| false);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].server_name, "shellmcp");
        assert_eq!(tools[0].description, "Echo tool");

        let args = Json::parse(r#"{"x": 1}"#).unwrap();
        let r = t.call_tool("echo", &args, &|| false);
        assert_eq!(
            r.at("plain_text_response").and_then(|v| v.get_str().ok()),
            Some("called")
        );
        t.close();
    }
}
