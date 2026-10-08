//! Minimal HTTP/1.1 server on `std::net::TcpListener` — the transport of
//! tools/server/server-http.cpp.
//!
//! The reference uses cpp-httplib; the port keeps the same shape with the
//! standard library only: a thread per connection (the reference's own decode
//! loop is single-threaded, so requests serialise on the engine lock), one
//! handler per method+path, the JSON error envelope of `format_error_response`
//! (server-common.cpp:36-60) and chunked `text/event-stream` responses
//! (server-context.cpp:4413-4490).
//!
//! Security: the listener binds exactly the address from `--host`
//! (default 127.0.0.1, server-http.cpp:120-150). The server never opens an
//! outbound connection, never follows a redirect and takes no credentials.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use crate::api::json_error;
use llama::json_schema::Json;

/// `uploaded_file` (server-http.h:42-47): the raw bytes of one multipart
/// file part plus its disposition metadata. `std::string`/`raw_buffer` carry
/// arbitrary bytes — no UTF-8 constraint rides this path.
#[derive(Clone, Default, Debug)]
pub struct UploadedFile {
    pub data: Vec<u8>,
    pub filename: String,
    pub content_type: String,
}

/// `server_http_req` (server-http.h:45-66). `params`/`headers` mirror the C
/// (`req.params`, `req.headers`); `body` is the C `std::string` — arbitrary
/// bytes, decoded to text only where a handler asks (`body_str`).
#[allow(dead_code)]
pub struct Request {
    pub method: String,
    pub path: String,
    /// path + query params (`req.params`)
    pub params: HashMap<String, String>,
    pub headers: HashMap<String, String>,
    /// `req.body` — the request body as raw bytes (a POST whose
    /// Content-Type is multipart/form-data arrives already translated: text
    /// fields become the JSON body, file parts land in `files` —
    /// server-http.cpp:683-717)
    pub body: Vec<u8>,
    /// `req.files` (server-http.h:55) — the multipart file parts, keyed and
    /// iterated like the C's `std::map` (sorted by field name)
    pub files: BTreeMap<String, UploadedFile>,
}

#[allow(dead_code)]
impl Request {
    /// `req.get_param(key, def)` (server-http.h:59-65)
    pub fn get_param(&self, key: &str, def: &str) -> String {
        self.params.get(key).cloned().unwrap_or_else(|| def.to_string())
    }

    /// the text view of `req.body` — lossless for the JSON surface (the C
    /// hands nlohmann the same bytes; invalid UTF-8 fails its parse the way
    /// the lossy copy fails the port's)
    pub fn body_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.body)
    }
}

/// `server_http_res` (server-http.h:22-38): a full body, or a generator of
/// stream chunks (`next()`, i.e. SSE).
pub enum Body {
    /// `res->data` — the complete response body (text; the JSON handlers)
    Full(String),
    /// `res->data`'s `std::string` bytes semantics — non-UTF-8 bodies (the
    /// /cors-proxy relay, the mounted/embedded binary assets) ride this
    /// variant so the wire stays byte-exact
    Bytes(Vec<u8>),
    /// `res->next` — one `data: {...}\n\n` frame per item, in order
    Events(Receiver<StreamEvent>),
}

/// One streamed frame. `Done` terminates the stream (the reference's
/// `next() -> false`), `Ping` is the `:\n\n` keep-alive comment
/// (`sse_ping_interval`, server-context.cpp:4470-4474 — the port emits no pings
/// yet, see PARITY.md).
#[allow(dead_code)]
pub enum StreamEvent {
    Frame(String),
    Ping,
    Done,
}

pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Body,
    pub headers: Vec<(String, String)>,
    /// OAI streams end with `data: [DONE]` after the last frame
    /// (`handle_completions_impl`'s `output = "data: [DONE]\n\n"`,
    /// server-context.cpp:4425-4433); the native `/completion` stream ends
    /// silently.
    pub terminal_done: bool,
}

impl Response {
    /// `server_res_generator::ok` — 200 + JSON
    pub fn ok(json: impl Into<String>) -> Self {
        Response {
            status: 200,
            content_type: "application/json; charset=utf-8".into(),
            body: Body::Full(json.into()),
            headers: Vec::new(),
            terminal_done: false,
        }
    }

    /// `server_res_generator::error` (server-context.cpp:821-834 `set_error`)
    pub fn error(json: impl Into<String>) -> Self {
        let body = json.into();
        // the status is the `code` inside the error object (server.cpp:71-77)
        let status = crate::api::json_error_code(&body).unwrap_or(500);
        Response {
            status,
            content_type: "application/json; charset=utf-8".into(),
            body: Body::Full(body),
            headers: Vec::new(),
            terminal_done: false,
        }
    }

    /// `res->content_type` replacement + `set_next` (server-http.cpp:649)
    pub fn stream(rx: Receiver<StreamEvent>) -> Self {
        Response {
            status: 200,
            content_type: "text/event-stream".into(),
            body: Body::Events(rx),
            headers: vec![("Cache-Control".into(), "no-cache".into())],
            terminal_done: false,
        }
    }

    /// the reference's 404 body — server-http.cpp:201-213's error handler,
    /// whose `safe_json_to_str` keeps the INSERTION order (message, type,
    /// code) rather than nlohmann's alphabetical dump; byte-identical 404s
    /// are what the ui-parity compares assert
    pub fn not_found() -> Self {
        Response {
            status: 404,
            content_type: "application/json; charset=utf-8".into(),
            body: Body::Full(format!(
                "{{\"error\":{{\"message\":\"File Not Found\",\"type\":\"not_found_error\",\"code\":404}}}}"
            )),
            headers: Vec::new(),
            terminal_done: false,
        }
    }

    /// 503 while the model is loading (`middleware_server_state`,
    /// server-http.cpp:307-325)
    pub fn loading() -> Self {
        Response {
            status: 503,
            content_type: "application/json; charset=utf-8".into(),
            body: Body::Full(json_error("Loading model", "unavailable_error", 503)),
            headers: Vec::new(),
            terminal_done: false,
        }
    }
}

/// Handler type: `server_http_context::handler_t` (server-http.h:72)
pub type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

/// The routing table of `server_http_context` (server-http.h:70-101), keyed
/// by "METHOD /path", plus the static-file mount (`set_mount_point`,
/// server-http.cpp:379-385 — the `--path` web UI serving).
///
/// The C keeps TWO tables: httplib's routes under
/// `path_prefix + path` (server-http.cpp:673/724/745 — `--api-prefix`
/// re-mounts EVERY route, API surface included) and its own `handlers` map
/// under the bare `path` (server-http.cpp:675/726/747 — what the GCP predict
/// dispatch looks up, server-http.cpp:881-895). The port mirrors both.
#[derive(Clone, Default)]
pub struct Routes {
    /// the HTTP routes, under `path_prefix + path`
    pub handlers: HashMap<String, Handler>,
    /// the internal-dispatch table, under the bare `path`
    pub dispatch: HashMap<String, Handler>,
    /// `server_http_context::path_prefix` (server-http.h:77)
    pub path_prefix: String,
    /// (mount point, directory) — cpp-httplib `base_dirs_`
    pub mount: Option<(String, std::path::PathBuf)>,
}

impl Routes {
    pub fn get(&self, key: &str) -> Option<&Handler> {
        self.handlers.get(key)
    }

    /// the internal `handlers` map lookup (server-http.cpp:881-895) — the
    /// bare path, prefix-free
    pub fn dispatch_get(&self, key: &str) -> Option<&Handler> {
        self.dispatch.get(key)
    }
}

/// `server_http_context` (server-http.h:70-101): routing table + listener.
pub struct HttpServer {
    handlers: HashMap<String, Handler>,
    dispatch: HashMap<String, Handler>,
    /// `path_prefix` (server-http.h:77) — `params.api_prefix`
    /// (server-http.cpp:114)
    path_prefix: String,
    mount: Option<(String, std::path::PathBuf)>,
    pub ready: Arc<std::sync::atomic::AtomicBool>,
}

impl HttpServer {
    pub fn new() -> Self {
        HttpServer {
            handlers: HashMap::new(),
            dispatch: HashMap::new(),
            path_prefix: String::new(),
            mount: None,
            ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// `path_prefix = params.api_prefix` (server-http.cpp:114) — every
    /// subsequent `add` registers under the prefix
    pub fn set_path_prefix(&mut self, prefix: &str) {
        self.path_prefix = prefix.to_string();
    }

    /// `ctx_http.get/post(path, handler)` (server.cpp:251-284): the httplib
    /// route under `path_prefix + path` (server-http.cpp:673/724/745), the
    /// internal `handlers.emplace(path, ...)` under the bare path
    /// (server-http.cpp:675/726/747)
    pub fn add(&mut self, method: &str, path: &str, handler: Handler) {
        let full_path = format!("{}{}", self.path_prefix, path);
        self.handlers.insert(format!("{method} {full_path}"), handler.clone());
        self.dispatch.insert(format!("{method} {path}"), handler);
    }

    /// `srv->set_mount_point(mount_point, dir)` (server-http.cpp:379-385) —
    /// the `--path` static file serving the web UI rides
    pub fn set_mount(&mut self, mount_point: String, dir: std::path::PathBuf) {
        self.mount = Some((mount_point, dir));
    }

    /// The immutable routing table handed to the listener.
    pub fn into_routes(self) -> Routes {
        Routes {
            handlers: self.handlers,
            dispatch: self.dispatch,
            path_prefix: self.path_prefix,
            mount: self.mount,
        }
    }
}

impl Default for HttpServer {
    fn default() -> Self {
        Self::new()
    }
}

/// `ctx_http.init` + `start` (server-http.cpp:120-160): bind `host:port` and
/// accept for the lifetime of the process. `ready` gates every route with the
/// 503 "Loading model" answer (server-http.cpp:307-325).
pub fn serve_routes(
    host: &str,
    port: u16,
    routes: Routes,
    ready: Arc<std::sync::atomic::AtomicBool>,
) -> std::io::Result<()> {
    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    let local = listener.local_addr()?;
    eprintln!("llama-server: listening on http://{local}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let routes = routes.clone();
                let ready = ready.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_conn(stream, &routes, &ready) {
                        if e.kind() != std::io::ErrorKind::BrokenPipe
                            && e.kind() != std::io::ErrorKind::ConnectionReset
                        {
                            eprintln!("llama-server: connection error: {e}");
                        }
                    }
                });
            }
            Err(e) => eprintln!("llama-server: accept error: {e}"),
        }
    }
    Ok(())
}

fn handle_conn(
    stream: TcpStream,
    handlers: &Routes,
    ready: &Arc<std::sync::atomic::AtomicBool>,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let req = match read_request(&mut reader)? {
        Some(r) => r,
        None => return Ok(()), // empty connection
    };

    let res = if !ready.load(std::sync::atomic::Ordering::Relaxed) {
        Response::loading()
    } else {
        match route(&handlers.handlers, &req) {
            // httplib's path params land in the same `req.params` map the
            // query string fills (server-http.cpp's `get_param`)
            Some((h, req)) => h(&req),
            None => {
                // the static-file mount (`--path`): GET/HEAD requests below
                // the mount point fall through to the directory
                // (cpp-httplib's `base_dirs_` routing, server-http.cpp:379).
                // The prefix match is on a path-segment boundary: a
                // '/'-terminated mount (the `api_prefix + "/"` of
                // server-http.cpp:379) carries its own boundary, so "/pfx"
                // does NOT match the "/pfx/" mount (httplib.cpp:8917-8923)
                if req.method == "GET" || req.method == "HEAD" {
                    if let Some((mnt, dir)) = &handlers.mount {
                        let m = mnt.as_str();
                        let boundary = m.ends_with('/')
                            || req.path.len() == m.len()
                            || req.path.as_bytes().get(m.len()) == Some(&b'/');
                        if req.path.starts_with(m) && boundary {
                            // sub_path = "/" + the path past the mount point
                            let sub_path =
                                format!("/{}", &req.path[m.len().min(req.path.len())..]);
                            if let Some(r) = crate::ui::serve_mounted(dir, &sub_path) {
                                r
                            } else {
                                Response::not_found()
                            }
                        } else {
                            Response::not_found()
                        }
                    } else {
                        Response::not_found()
                    }
                } else {
                    Response::not_found()
                }
            }
        }
    };

    write_response(stream, res)
}

/// `ctx_http.post("/slots/:id_slot", ...)` — the exact "METHOD /path" key
/// first (every route the port had before path params), then the patterns
/// with `:name` segments (cpp-httplib's regex-free path matching). Returns
/// the handler plus the request with the captured params merged in.
fn route<'a>(
    handlers: &'a HashMap<String, Handler>,
    req: &Request,
) -> Option<(&'a Handler, Request)> {
    if let Some(h) = handlers.get(&format!("{} {}", req.method, req.path)) {
        return Some((h, clone_req(req)));
    }
    let req_segs: Vec<&str> = req.path.trim_matches('/').split('/').collect();
    for (key, h) in handlers {
        let Some((method, pattern)) = key.split_once(' ') else { continue };
        if method != req.method || !pattern.contains(':') {
            continue;
        }
        let pat_segs: Vec<&str> = pattern.trim_matches('/').split('/').collect();
        if pat_segs.len() != req_segs.len() {
            continue;
        }
        let mut params = HashMap::new();
        let mut ok = true;
        for (p, r) in pat_segs.iter().zip(req_segs.iter()) {
            if let Some(name) = p.strip_prefix(':') {
                params.insert(name.to_string(), r.to_string());
            } else if p != r {
                ok = false;
                break;
            }
        }
        if ok {
            let mut captured = clone_req(req);
            captured.params.extend(params);
            return Some((h, captured));
        }
    }
    None
}

/// a `Request` copy for the param merge (the original stays borrowed)
fn clone_req(req: &Request) -> Request {
    Request {
        method: req.method.clone(),
        path: req.path.clone(),
        params: req.params.clone(),
        headers: req.headers.clone(),
        body: req.body.clone(),
        files: req.files.clone(),
    }
}

/// Request line + headers + `Content-Length` body (the only body framing the
/// API uses; the reference accepts anything httplib does).
fn read_request(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.trim_end().split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    if method.is_empty() {
        return Ok(None);
    }

    let mut headers: HashMap<String, String> = HashMap::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end().to_string();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.clone(), String::new()),
    };
    let mut params: HashMap<String, String> = HashMap::new();
    for kv in query.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        params.insert(k.to_string(), v.to_string());
    }

    let mut body: Vec<u8> = Vec::new();
    if let Some(len) = headers.get("content-length") {
        let n: usize = len.trim().parse().unwrap_or(0);
        body.resize(n, 0);
        reader.read_exact(&mut body)?;
    }

    // the multipart POST surface (server-http.cpp:683-717): httplib's
    // read_content parses the form (httplib.cpp:8746-8810 — a part with a
    // filename becomes a file, without one a text field), and ctx_http's
    // post() callback translates the fields to a JSON body before the
    // handler runs. get/del never translate (server-http.cpp:666-678/739-751).
    let mut files: BTreeMap<String, UploadedFile> = BTreeMap::new();
    let content_type = headers.get("content-type").cloned().unwrap_or_default();
    if method == "POST" && is_multipart_form_data(&content_type) {
        let boundary = parse_multipart_boundary(&content_type).unwrap_or_default();
        if let Some((fields, files_parsed)) = parse_multipart_form(&body, &boundary) {
            // json form_json = json::object(); form_json[key] = field.content
            // — nlohmann's default map sorts the keys; duplicates become
            // arrays in first-seen order (server-http.cpp:686-700)
            body = multipart_fields_json(&fields).into_bytes();
            files = files_parsed;
        }
    }

    Ok(Some(Request { method, path, params, headers, body, files }))
}

/// `Request::is_multipart_form_data` (httplib.cpp:6821-6823): the
/// Content-Type starts with `multipart/form-data` (case-insensitive)
fn is_multipart_form_data(content_type: &str) -> bool {
    content_type
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
}

/// `detail::parse_multipart_boundary` (httplib.cpp:3682-3698): the
/// `boundary=` parameter of the Content-Type, quoted or bare
fn parse_multipart_boundary(content_type: &str) -> Option<String> {
    let lowered = content_type.to_ascii_lowercase();
    let pos = lowered.find("boundary=").map(|p| p + "boundary=".len())?;
    let rest = &content_type[pos..];
    let value = if let Some(stripped) = rest.strip_prefix('"') {
        stripped.split('"').next().unwrap_or("")
    } else {
        rest.split(';').next().unwrap_or("").trim()
    };
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// one `Content-Disposition: form-data; name="..."; filename="..."`
/// (httplib.cpp:4741-4790's param scan, quoted values unescaped)
fn parse_content_disposition(value: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // scan to the next ';' then read `key=value`
        while i < bytes.len() && bytes[i] != b';' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        i += 1; // the ';'
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' && bytes[i] != b';' {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            continue;
        }
        let key = value[key_start..i].trim().to_string();
        i += 1;
        let val = if i < bytes.len() && bytes[i] == b'"' {
            i += 1;
            let start = i;
            let mut unescaped = String::new();
            while i < bytes.len() {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    unescaped.push(bytes[i + 1] as char);
                    i += 2;
                    continue;
                }
                if bytes[i] == b'"' {
                    break;
                }
                unescaped.push(bytes[i] as char);
                i += 1;
            }
            i += 1; // the closing quote
            let _ = start;
            unescaped
        } else {
            let start = i;
            while i < bytes.len() && bytes[i] != b';' {
                i += 1;
            }
            value[start..i].trim().to_string()
        };
        out.push((key, val));
    }
    out
}

/// parse the multipart body into (text fields in part order, files keyed for
/// the C's `std::map` iteration order) — the receiving half of
/// httplib.cpp:8746-8810
fn parse_multipart_form(
    body: &[u8],
    boundary: &str,
) -> Option<(Vec<(String, String)>, BTreeMap<String, UploadedFile>)> {
    if boundary.is_empty() {
        return None;
    }
    let delim = format!("--{boundary}");
    let delim_b = delim.as_bytes();
    // the preamble before the first delimiter is skipped
    let mut pos = find_sub(body, delim_b, 0)?;
    pos += delim_b.len();
    let mut fields: Vec<(String, String)> = Vec::new();
    let mut files: BTreeMap<String, UploadedFile> = BTreeMap::new();
    loop {
        // "--" right after the delimiter ends the form
        if body[pos..].starts_with(b"--") {
            break;
        }
        // the CRLF after the delimiter (tolerate a bare LF)
        if body[pos..].starts_with(b"\r\n") {
            pos += 2;
        } else if body[pos..].starts_with(b"\n") {
            pos += 1;
        }
        // part headers up to the blank line
        let hdr_end = find_sub(body, b"\r\n\r\n", pos)
            .map(|p| (p, 4))
            .or_else(|| find_sub(body, b"\n\n", pos).map(|p| (p, 2)))?;
        let headers_block = String::from_utf8_lossy(&body[pos..hdr_end.0]).into_owned();
        let mut name = String::new();
        let mut filename = String::new();
        let mut part_content_type = String::new();
        for h in headers_block.split("\r\n").flat_map(|l| l.split('\n')) {
            if let Some((k, v)) = h.split_once(':') {
                let k = k.trim().to_ascii_lowercase();
                let v = v.trim();
                if k == "content-disposition" && v.to_ascii_lowercase().starts_with("form-data") {
                    for (pk, pv) in parse_content_disposition(v) {
                        if pk == "name" {
                            name = pv;
                        } else if pk == "filename" {
                            filename = pv;
                        }
                    }
                } else if k == "content-type" {
                    part_content_type = v.to_string();
                }
            }
        }
        // the part body runs to the next CRLF+delimiter
        let content_start = hdr_end.0 + hdr_end.1;
        let next = find_sub(body, delim_b, content_start)?;
        let mut content_end = next;
        if content_end >= 2 && &body[content_end - 2..content_end] == b"\r\n" {
            content_end -= 2;
        } else if content_end >= 1 && body[content_end - 1] == b'\n' {
            content_end -= 1;
        }
        let content = body[content_start..content_end].to_vec();
        if filename.is_empty() {
            // a text field (httplib.cpp:8785-8790)
            fields.push((name, String::from_utf8_lossy(&content).into_owned()));
        } else {
            // a file part (httplib.cpp:8790-8793)
            files.insert(
                name,
                UploadedFile {
                    data: content,
                    filename,
                    content_type: part_content_type,
                },
            );
        }
        pos = next + delim_b.len();
    }
    Some((fields, files))
}

/// the fields-as-JSON translation of server-http.cpp:686-700 — `json` is
/// `common_json`, nlohmann **ordered**_json (json.cpp:14): keys keep
/// first-seen part order, duplicate keys fold to arrays in part order
fn multipart_fields_json(fields: &[(String, String)]) -> String {
    let mut keys: Vec<String> = Vec::new();
    for (k, _) in fields {
        if !keys.iter().any(|x| x == k) {
            keys.push(k.clone());
        }
    }
    let mut entries: Vec<(String, Json)> = Vec::new();
    for key in keys {
        let values: Vec<&(String, String)> = fields.iter().filter(|(k, _)| *k == key).collect();
        if values.len() == 1 {
            entries.push((key, Json::String(values[0].1.clone())));
        } else {
            entries.push((
                key,
                Json::Array(values.iter().map(|(_, v)| Json::String(v.clone())).collect()),
            ));
        }
    }
    Json::Object(entries).dump()
}

fn find_sub(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

fn write_response(mut stream: TcpStream, res: Response) -> std::io::Result<()> {
    let reason = match res.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "OK",
    };
    match res.body {
        Body::Full(body) => {
            write_full_body(
                &mut stream,
                res.status,
                reason,
                &res.content_type,
                &res.headers,
                body.as_bytes(),
            )
        }
        Body::Bytes(body) => {
            write_full_body(&mut stream, res.status, reason, &res.content_type, &res.headers, &body)
        }
        Body::Events(rx) => {
            // httplib's set_chunked_content_provider: chunked framing, one
            // chunk per `data:` frame (server-http.cpp:629-649)
            let mut head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n",
                res.status, reason, res.content_type
            );
            for (k, v) in &res.headers {
                head.push_str(&format!("{k}: {v}\r\n"));
            }
            head.push_str("\r\n");
            stream.write_all(head.as_bytes())?;
            stream.flush()?;
            while let Ok(ev) = rx.recv() {
                // the reference's stream generator frames each result as
                // `data: {json}\n\n` (server-context.cpp:4416-4430)
                let frame = match ev {
                    StreamEvent::Frame(f) => format!("data: {f}\n\n"),
                    StreamEvent::Ping => ":\n\n".to_string(),
                    StreamEvent::Done => break,
                };
                if frame.is_empty() {
                    continue;
                }
                let mut chunk = format!("{:x}\r\n", frame.as_bytes().len()).into_bytes();
                chunk.extend_from_slice(frame.as_bytes());
                chunk.extend_from_slice(b"\r\n");
                stream.write_all(&chunk)?;
                stream.flush()?;
            }
            // `data: [DONE]` — the OAI terminal frame
            // (server-context.cpp:4425-4433)
            if res.terminal_done {
                let frame = "data: [DONE]\n\n";
                let mut chunk = format!("{:x}\r\n", frame.len()).into_bytes();
                chunk.extend_from_slice(frame.as_bytes());
                chunk.extend_from_slice(b"\r\n");
                stream.write_all(&chunk)?;
            }
            stream.write_all(b"0\r\n\r\n")?;
            stream.flush()
        }
    }
}

/// the full-body response head + bytes (the `Full`/`Bytes` arms share it —
/// `Bytes` is the byte-exact wire path)
fn write_full_body(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    let mut out = head.into_bytes();
    for (k, v) in headers {
        out.extend_from_slice(format!("{k}: {v}\r\n").as_bytes());
    }
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(body);
    stream.write_all(&out)?;
    stream.flush()
}
// ---------------------------------------------------------------------------
// tests — the multipart surface, the bytes semantics and the api-prefix
// routing (the cross-server round-trips live in
// parity/server_ui_proxy_parity.sh)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_boundary_parsing() {
        // detail::parse_multipart_boundary (httplib.cpp:3682-3698)
        assert_eq!(
            parse_multipart_boundary("multipart/form-data; boundary=abc123"),
            Some("abc123".into())
        );
        assert_eq!(
            parse_multipart_boundary("multipart/form-data; boundary=\"quoted b\""),
            Some("quoted b".into())
        );
        assert_eq!(parse_multipart_boundary("application/json"), None);
        assert_eq!(parse_multipart_boundary("multipart/form-data"), None);
        assert!(is_multipart_form_data("Multipart/Form-Data; boundary=x"));
        assert!(!is_multipart_form_data("application/json"));
    }

    #[test]
    fn multipart_form_parse_fields_and_binary_files() {
        // one text field, a duplicate key, and a binary file part — the
        // receiving half of httplib.cpp:8746-8810
        let bin = [0x00u8, 0xff, 0xfe, 0x80, 0x7f, 0x0d, 0x0a, 0x2d];
        let body = format!(
            "--B\r\nContent-Disposition: form-data; name=\"q\"\r\n\r\nhello\r\n\
             --B\r\nContent-Disposition: form-data; name=\"q\"\r\n\r\nworld\r\n\
             --B\r\nContent-Disposition: form-data; name=\"z\"\r\n\r\nfirst\r\n\
             --B\r\nContent-Disposition: form-data; name=\"f\"; filename=\"a\\\"b.bin\"\r\n\
             Content-Type: application/x-binary\r\n\r\n"
        )
        .into_bytes();
        let body = [body, bin.to_vec(), b"\r\n--B--\r\n".to_vec()].concat();
        let (fields, files) = parse_multipart_form(&body, "B").expect("parses");
        assert_eq!(
            fields,
            vec![
                ("q".to_string(), "hello".to_string()),
                ("q".to_string(), "world".to_string()),
                ("z".to_string(), "first".to_string()),
            ]
        );
        let f = files.get("f").expect("file part");
        assert_eq!(f.filename, "a\"b.bin");
        assert_eq!(f.content_type, "application/x-binary");
        assert_eq!(f.data, bin.to_vec()); // bytes survive verbatim

        // the fields→JSON translation (server-http.cpp:686-700): ordered_json
        // keeps first-seen key order, duplicates fold to arrays
        assert_eq!(
            multipart_fields_json(&fields),
            r#"{"q":["hello","world"],"z":"first"}"#
        );
    }

    #[test]
    fn multipart_form_parse_escapes_and_crlf_in_content() {
        // a field whose content contains a CRLF and a lone "--" (must not
        // terminate the part early — only \r\n--boundary does)
        let body = b"--X\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nline1\r\n--not-bound\r\nline2\r\n--X--\r\n";
        let (fields, files) = parse_multipart_form(body, "X").expect("parses");
        assert_eq!(files.len(), 0);
        assert_eq!(fields[0].1, "line1\r\n--not-bound\r\nline2");

        // a broken form (no terminator) parses to None — the request keeps
        // its raw body (the port's conservative fallthrough)
        assert!(parse_multipart_form(b"--X\r\n", "X").is_none());
    }

    #[test]
    fn api_prefix_routes_and_dispatch_table() {
        // server-http.cpp:673/724/745 — the httplib route under
        // path_prefix + path; :675/726/747 — the internal handlers map under
        // the bare path
        let mut srv = HttpServer::new();
        srv.set_path_prefix("/pfx");
        srv.add("GET", "/health", Arc::new(|_| Response::ok("{}")));
        let routes = srv.into_routes();
        assert!(routes.get("GET /pfx/health").is_some());
        assert!(routes.get("GET /health").is_none()); // unprefixed is gone
        assert!(routes.dispatch_get("GET /health").is_some()); // internal map
        assert_eq!(routes.path_prefix, "/pfx");
    }
}
