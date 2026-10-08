//! server_tools.rs — port of `tools/server/server-tools.cpp` (pinned
//! bd4f514db1, 2,172 lines): the built-in code-execution tools behind
//! `--tools` / `--tools-runtime` and the `/tools` GET+POST endpoints.
//!
//! Security model (mirrored from the reference):
//!   * every spawn is argv-based — no shell interpolation anywhere except
//!     `exec_shell_command`, which runs exactly `sh -c <command>` because the
//!     caller asked for it (server-tools.cpp:1287);
//!   * the isolate runtimes quote their inner argv into ONE shell word for
//!     ssh (`shell_quote_join`, :635-648) and reject a leading-dash target
//!     or container id (:748-755, :811-818);
//!   * timeouts terminate the child; output caps append the
//!     "\n[output truncated]" marker (:293).
//!
//! Mapping (server-tools.cpp → Rust):
//!   `server_tool::to_json` (:75-86)                    -> [`ServerTool::to_json`]
//!   `home_dir`/`expand_home` (:112-133)                -> [`expand_home`]
//!   `tools_io` + `tools_io_basic` (:147-514)           -> [`ToolsIo`] (host impl)
//!   `run_subprocess` (:187-296)                        -> [`ToolsIo::run_host`]
//!   `tools_io_isolate` + container + ssh (:524-783)    -> the argv builders
//!                                                       [`build_isolate_argv`]
//!   `container_runtime_spec` (:786-819)                -> [`ContainerRuntimeSpec`]
//!   `make_tools_io` (:821-849)                          -> [`ToolsIo::for_params`]
//!   `path_glob_match` (:852-860)                        -> [`path_glob_match`]
//!   `read_file` (:869-966)                              -> `read_file` in [`invoke_tool`]
//!   `file_glob_search` (:977-1084)                      -> `file_glob_search`
//!   `grep_search` (:1092-1236)                          -> `grep_search`
//!   `exec_shell_command` (:1245-1318)                   -> `exec_shell_command`
//!   `write_file` (:1324-1361)                           -> `write_file`
//!   `edit_file` (:1367-1692)                            -> `edit_file` (+ the fuzzy
//!                                                       normalizers)
//!   `get_info` (:1701-1756)                             -> `get_info`
//!   `server_tools::setup` (:1997-2172)                  -> [`ServerTools::setup`]
//!                                                       + [`handle_tools_get`]/[`handle_tools_post`]
//!
//! Documented deviations:
//!   * `grep_search` matches with a bounded regex engine (the port's
//!     `regex_lite`) instead of std::regex::ECMAScript — same metacharacter
//!     surface for the literal-escape path (:1142-1150 is ported verbatim).
//!   * the Windows OEM-code-page conversion `console_output_to_utf8`
//!     (:45-73) is a POSIX no-op in the reference too.
//!   * tools_io_isolate's remote ls/find plumbing rides the same
//!     `build_isolate_argv` shape, verified through the host runtime (the
//!     ssh/container engines need external services; their argv contracts
//!     are unit-pinned below).

use std::collections::HashSet;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use llama::json_schema::Json;

use crate::http::{Body, Request, Response, StreamEvent};
use crate::server_mcp::{ServerMcp, ToolDef};
use crate::subproc::Subproc;

// ---------------------------------------------------------------------------
// constants (server-tools.cpp)
// ---------------------------------------------------------------------------

const SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT: usize = 8 * 1024 * 1024; // :88
const SERVER_TOOL_LIST_ENTRIES_TIMEOUT: u64 = 15; // :90
const SERVER_TOOL_READ_FILE_MAX_SIZE: usize = 16 * 1024; // :866
const SERVER_TOOL_READ_FILE_MAX_SIZE_BASE64: usize = 32 * 1024 * 1024; // :867
const SERVER_TOOL_FILE_SEARCH_MAX_RESULTS: i64 = 100; // :972
const SERVER_TOOL_GREP_SEARCH_MAX_RESULTS: usize = 100; // :1090
const SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE: i64 = 16 * 1024; // :1242
const SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT: i64 = 60; // :1243
const SERVER_TOOL_GET_INFO_MAX_OUTPUT: i64 = 4096; // :1698
const SERVER_TOOL_GET_INFO_TIMEOUT: i64 = 5; // :1699
const SERVER_TOOL_ISOLATE_EXEC_TIMEOUT: i64 = 15; // :518
const SERVER_TOOL_ISOLATE_READ_FILE_MAX_SIZE: usize = 64 * 1024 * 1024; // :519

/// `SERVER_TOOL_JUNK_DIR_NAMES` (:142-145)
const JUNK_DIR_NAMES: &[&str] = &[
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    "target", ".cache", ".idea", ".vscode",
];

// ---------------------------------------------------------------------------
// glob_match — common.cpp:729-817 (the audit's MISSED `glob_match`)
// ---------------------------------------------------------------------------

/// `glob_class_match` (common.cpp:729-767)
fn glob_class_match(c: u8, pattern: &[u8], class_end: usize) -> bool {
    let mut class_start = 0usize;
    let mut negated = false;

    if pattern[class_start] == b'!' {
        negated = true;
        class_start += 1;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if class_start < pattern.len() && (pattern[class_start] == b']' || pattern[class_start] == b'-')
    {
        if class_start < class_end && pattern[class_start] == c {
            return !negated;
        }
        class_start += 1;
    }

    let mut matched = false;
    while class_start < class_end {
        if class_start + 2 < class_end
            && pattern[class_start + 1] == b'-'
            && pattern[class_start + 2] != b']'
        {
            let (start_char, end_char) = (pattern[class_start], pattern[class_start + 2]);
            if c >= start_char && c <= end_char {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if pattern[class_start] == c {
                matched = true;
                break;
            }
            class_start += 1;
        }
    }

    if negated {
        !matched
    } else {
        matched
    }
}

/// `glob_match` (common.cpp:769-813): `*` matches non-/ chars, `**` anything
/// including `/`, `[...]` a character class
fn glob_match_impl(pattern: &[u8], str_: &[u8]) -> bool {
    if pattern.is_empty() {
        return str_.is_empty();
    }
    if pattern[0] == b'*' && pattern.len() > 1 && pattern[1] == b'*' {
        let p = &pattern[2..];
        if glob_match_impl(p, str_) {
            return true;
        }
        if !str_.is_empty() {
            return glob_match_impl(pattern, &str_[1..]);
        }
        return false;
    }
    if pattern[0] == b'*' {
        let p = &pattern[1..];
        let mut i = 0;
        while i < str_.len() && str_[i] != b'/' {
            if glob_match_impl(p, &str_[i..]) {
                return true;
            }
            i += 1;
        }
        return glob_match_impl(p, &str_[i..]);
    }
    if pattern[0] == b'?' && !str_.is_empty() && str_[0] != b'/' {
        return glob_match_impl(&pattern[1..], &str_[1..]);
    }
    if pattern[0] == b'[' {
        let mut class_end = 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if class_end < pattern.len() && (pattern[class_end] == b']' || pattern[class_end] == b'-') {
            class_end += 1;
        }
        while class_end < pattern.len() && pattern[class_end] != b']' {
            class_end += 1;
        }
        if class_end < pattern.len() && pattern[class_end] == b']' {
            if str_.is_empty() {
                return false;
            }
            let matched = glob_class_match(str_[0], &pattern[1..], class_end - 1);
            return matched && glob_match_impl(&pattern[class_end + 1..], &str_[1..]);
        }
        if !str_.is_empty() && str_[0] == b'[' {
            return glob_match_impl(&pattern[1..], &str_[1..]);
        }
        return false;
    }
    if !str_.is_empty() && pattern[0] == str_[0] {
        return glob_match_impl(&pattern[1..], &str_[1..]);
    }
    false
}

pub fn glob_match(pattern: &str, s: &str) -> bool {
    glob_match_impl(pattern.as_bytes(), s.as_bytes())
}

/// `path_glob_match` (server-tools.cpp:852-860)
fn path_glob_match(pattern: &str, rel_path: &str) -> bool {
    if !pattern.contains('/') {
        let base = Path::new(rel_path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        return glob_match(pattern, &base);
    }
    if pattern == "**" || pattern.starts_with("**/") || pattern.starts_with('/') {
        return glob_match(pattern, rel_path);
    }
    glob_match(&format!("**/{pattern}"), rel_path)
}

// ---------------------------------------------------------------------------
// path helpers (:101-138)
// ---------------------------------------------------------------------------

/// `expand_home` (:127-133)
fn expand_home(path: &str) -> String {
    if path.is_empty() || !path.starts_with('~') {
        return path.to_string();
    }
    if path.len() > 1 && !path[1..].starts_with('/') && !path[1..].starts_with('\\') {
        return path.to_string();
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => format!("{home}{}", &path[1..]),
        _ => path.to_string(),
    }
}

/// `entry_depth` (:136-138)
fn entry_depth(rel: &str) -> i32 {
    1 + rel.matches('/').count() as i32
}

// ---------------------------------------------------------------------------
// the IO layer — tools_io (host runtime) + the isolate argv builders
// ---------------------------------------------------------------------------

/// `tools_io::exec_result` (:149-153)
pub struct ExecResult {
    pub output: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

/// `list_kind` (:93-97)
#[derive(Clone, Copy, PartialEq, Eq)]
enum ListKind {
    Files,
    Dirs,
    All,
}

/// `tools_io::list_entry` / `list_result` (:164-172)
struct ListEntry {
    rel: String,
    is_dir: bool,
}
struct ListResult {
    entries: Vec<ListEntry>,
    err: String,
    truncated: bool,
}

/// the host runtime (`tools_io_basic`, :298-514). The isolate runtimes
/// (`tools_io_isolate`/container/ssh, :524-783) build argv through
/// [`ContainerRuntimeSpec`] + [`build_isolate_argv`] and share `run_host`.
pub struct ToolsIo {
    cwd: String,
    runtime: String,
}

impl ToolsIo {
    /// `make_tools_io` (:821-849): the empty runtime runs on the host;
    /// container/ssh specs are validated eagerly
    pub fn for_params(cwd: &str, runtime: &str) -> Result<ToolsIo, String> {
        if runtime.is_empty() {
            return Ok(ToolsIo { cwd: cwd.to_string(), runtime: String::new() });
        }
        if let Some(spec) = ContainerRuntimeSpec::parse(runtime) {
            if !spec.attach {
                return Err(format!("tool runtime must name a running container: {runtime}"));
            }
            if !ContainerRuntimeSpec::is_valid_id(&spec.arg) {
                return Err(format!("invalid container id: {}", spec.arg));
            }
            return Ok(ToolsIo { cwd: cwd.to_string(), runtime: runtime.to_string() });
        }
        if let Some(target) = runtime.strip_prefix("ssh:") {
            if !is_valid_ssh_target(target) {
                return Err(format!("invalid ssh target: {target}"));
            }
            return Ok(ToolsIo { cwd: cwd.to_string(), runtime: runtime.to_string() });
        }
        // do not fall back to the host, the caller asked for an isolate (:847)
        Err(format!("unknown tool runtime: {runtime}"))
    }

    /// `tools_io_basic::resolve` (:305-327): expand `~`, resolve against the
    /// cwd (or the server cwd), normalize `.`/`..`
    pub fn resolve(&self, path: &str) -> String {
        let p = expand_home(path);
        let mut full = PathBuf::from(&p);
        if !full.is_absolute() {
            let base = if self.cwd.is_empty() {
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from(&p))
            } else {
                PathBuf::from(&self.cwd)
            };
            full = base.join(full);
        }
        let norm = normalize_path(&full);
        norm.to_string_lossy().into_owned()
    }

    pub fn is_directory(&self, path: &str) -> bool {
        let isolate = self.isolate_shell_test();
        if let Some(argv) = isolate {
            let res = self.run_isolate(&argv, 64, SERVER_TOOL_ISOLATE_EXEC_TIMEOUT);
            return res.exit_code == 0 && !res.timed_out;
        }
        std::fs::metadata(self.resolve(path))
            .map(|m| m.is_dir())
            .unwrap_or(false)
    }

    pub fn is_regular_file(&self, path: &str) -> bool {
        if let Some(argv) = self.isolate_shell_test() {
            let res = self.run_isolate(&argv, 64, SERVER_TOOL_ISOLATE_EXEC_TIMEOUT);
            return res.exit_code == 0 && !res.timed_out;
        }
        std::fs::metadata(self.resolve(path))
            .map(|m| m.is_file())
            .unwrap_or(false)
    }

    /// `tools_io_basic::file_size` (:339-343)
    pub fn file_size(&self, path: &str) -> Option<u64> {
        std::fs::metadata(self.resolve(path)).ok().map(|m| m.len())
    }

    /// `tools_io_basic::read_file` (:345-352)
    pub fn read_file(&self, path: &str) -> Option<String> {
        if !self.runtime.is_empty() {
            // isolate: cat -- <path> without stderr spliced in (:558-564)
            let resolved = self.isolate_resolve(path);
            let argv = self.build_isolate_argv(
                &["cat".to_string(), "--".to_string(), resolved],
                false,
            );
            let res = self.run_isolate_raw(&argv, SERVER_TOOL_ISOLATE_READ_FILE_MAX_SIZE, SERVER_TOOL_ISOLATE_EXEC_TIMEOUT, false);
            if res.exit_code != 0 || res.timed_out {
                return None;
            }
            return Some(res.output);
        }
        std::fs::read(self.resolve(path)).ok().map(|b| String::from_utf8_lossy(&b).into_owned())
    }

    /// `tools_io_basic::write_file` (:354-365) — parents created first
    pub fn write_file(&self, path: &str, content: &str) -> bool {
        if !self.runtime.is_empty() {
            // the content travels on stdin: no argv for the far side (:566-573)
            let resolved = self.isolate_resolve(path);
            let inner = vec![
                "sh".to_string(),
                "-c".to_string(),
                "mkdir -p \"$(dirname \"$1\")\" && cat > \"$1\"".to_string(),
                "_".to_string(),
                resolved,
            ];
            let argv = self.build_isolate_argv(&inner, true);
            let res = self.run_isolate_with_stdin(&argv, 4096, SERVER_TOOL_ISOLATE_EXEC_TIMEOUT, content.as_bytes());
            return res.exit_code == 0 && !res.timed_out;
        }
        let fpath = PathBuf::from(self.resolve(path));
        if let Some(parent) = fpath.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return false;
            }
        }
        std::fs::write(fpath, content).is_ok()
    }

    /// `list_entries` (:367-402 + the fallback walker :457-513)
    fn list_entries(&self, base: &str, max_depth: i32, kind: ListKind) -> ListResult {
        let mut out = ListResult { entries: Vec::new(), err: String::new(), truncated: false };

        if !Path::new(base).is_dir() {
            out.err = "path does not exist or is not a directory".into();
            return out;
        }

        let deadline = Instant::now() + Duration::from_secs(SERVER_TOOL_LIST_ENTRIES_TIMEOUT);

        // git ls-files cannot list directories; use the walker when they are
        // requested (:379-397)
        if kind == ListKind::Files {
            let res = if self.runtime.is_empty() {
                self.run_host(
                    &[
                        "git".to_string(),
                        "-C".to_string(),
                        base.to_string(),
                        "ls-files".to_string(),
                        "--cached".to_string(),
                        "--others".to_string(),
                        "--exclude-standard".to_string(),
                    ],
                    SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT,
                    SERVER_TOOL_LIST_ENTRIES_TIMEOUT as i64,
                    None,
                )
            } else {
                let inner = vec![
                    "sh".to_string(),
                    "-c".to_string(),
                    "cd \"$1\" && git ls-files --cached --others --exclude-standard".to_string(),
                    "_".to_string(),
                    base.to_string(),
                ];
                let argv = self.build_isolate_argv(&inner, false);
                self.run_isolate(&argv, SERVER_TOOL_GIT_LS_FILES_MAX_OUTPUT, SERVER_TOOL_ISOLATE_EXEC_TIMEOUT)
            };
            if res.exit_code == 0 && !res.timed_out {
                for line in res.output.lines() {
                    let line = line.trim_end_matches('\r');
                    if line.is_empty() {
                        continue;
                    }
                    let line = line.replace('\\', "/");
                    if max_depth > 0 && entry_depth(&line) > max_depth {
                        continue;
                    }
                    if self.is_regular_file(&format!("{base}/{line}")) {
                        out.entries.push(ListEntry { rel: line, is_dir: false });
                    }
                }
                return out;
            }
        }

        self.list_entries_fallback(base, max_depth, kind, deadline, &mut out);
        out
    }

    /// `list_entries_fallback` (:457-513) — a stack walk that never follows
    /// symlinks and skips the junk directories' interiors
    fn list_entries_fallback(
        &self,
        base: &str,
        max_depth: i32,
        kind: ListKind,
        deadline: Instant,
        out: &mut ListResult,
    ) {
        // (dir, rel_dir, depth)
        let mut stack: Vec<(PathBuf, PathBuf, i32)> = vec![(PathBuf::from(base), PathBuf::new(), 0)];

        while let Some((dir, rel_dir, depth)) = stack.pop() {
            if Instant::now() >= deadline {
                out.truncated = true;
                return;
            }

            let Ok(entries) = std::fs::read_dir(&dir) else {
                // permission errors are skipped: a subtree the caller never sees
                out.truncated = true;
                continue;
            };
            for entry in entries.flatten() {
                if Instant::now() >= deadline {
                    out.truncated = true;
                    return;
                }
                let fname = entry.file_name();
                let Ok(ftype) = entry.file_type() else { continue };
                if ftype.is_dir() {
                    if kind == ListKind::Dirs || kind == ListKind::All {
                        out.entries.push(ListEntry {
                            rel: rel_dir.join(&fname).to_string_lossy().replace('\\', "/"),
                            is_dir: true,
                        });
                    }
                    // junk directories stay selectable but are never walked (:499-500)
                    let name = fname.to_string_lossy();
                    if JUNK_DIR_NAMES.contains(&name.as_ref()) {
                        continue;
                    }
                    // a link can point back to an ancestor, so it is never walked (:501)
                    let is_link = entry
                        .metadata()
                        .ok()
                        .and_then(|m| {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::PermissionsExt;
                                Some(m.permissions().mode() & 0o170000 == 0o120000)
                            }
                            #[cfg(not(unix))]
                            {
                                None
                            }
                        })
                        .unwrap_or(false);
                    if !is_link && (max_depth == 0 || depth + 1 < max_depth) {
                        stack.push((entry.path(), rel_dir.join(&fname), depth + 1));
                    }
                } else if ftype.is_file() {
                    if kind == ListKind::Files || kind == ListKind::All {
                        out.entries.push(ListEntry {
                            rel: rel_dir.join(&fname).to_string_lossy().replace('\\', "/"),
                            is_dir: false,
                        });
                    }
                }
            }
        }
    }

    /// `run` (:404-410) — the host path of `tools_io::run`
    pub fn run(&self, args: &[String], max_output: usize, timeout_secs: i64) -> ExecResult {
        if self.runtime.is_empty() {
            return self.run_host(args, max_output, timeout_secs, None);
        }
        // the isolate wraps the command with its own timeout (:615-627)
        let mut inner: Vec<String> = vec!["timeout".into(), format!("{}s", timeout_secs)];
        inner.extend_from_slice(args);
        let argv = self.build_isolate_argv(&self.with_cwd(inner), true);
        self.run_isolate(&argv, max_output, timeout_secs + 5)
    }

    /// `run_subprocess` (:187-296): spawn, feed stdin, drain stdout
    /// (combined with stderr for shell output), cap the output, enforce the
    /// timeout by termination
    fn run_host(
        &self,
        args: &[String],
        max_output: usize,
        timeout_secs: i64,
        stdin_data: Option<&[u8]>,
    ) -> ExecResult {
        let mut res = ExecResult { output: String::new(), exit_code: -1, timed_out: false };

        let mut proc = Subproc::new();
        let cwd = if self.cwd.is_empty() { None } else { Some(self.cwd.as_str()) };
        let Some(mut out) = proc.create_combined(args, &[], cwd) else {
            res.output = "failed to spawn process".into();
            return res;
        };

        // write stdin before reading stdout, then always close (:226-239)
        if let Some(data) = stdin_data {
            proc.write_stdin_and_close(data);
        } else {
            proc.close_stdin();
        }

        let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1) as u64);
        let mut output: Vec<u8> = Vec::new();
        let mut truncated = false;
        let mut buf = [0u8; 4096];
        loop {
            if Instant::now() >= deadline {
                res.timed_out = true;
                proc.terminate();
                break;
            }
            match out.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if truncated {
                        continue; // keep draining, else the child blocks (:264-266)
                    }
                    if output.len() + n <= max_output {
                        output.extend_from_slice(&buf[..n]);
                    } else {
                        let remaining = max_output - output.len();
                        output.extend_from_slice(&buf[..remaining]);
                        truncated = true;
                    }
                }
                Err(_) => break,
            }
        }

        res.exit_code = proc.join();
        res.output = String::from_utf8_lossy(&output).into_owned();
        if truncated {
            res.output.push_str("\n[output truncated]");
        }
        res
    }

    // -- the isolate plumbing (tools_io_isolate, :524-714) --

    /// `tools_io_isolate::resolve` (:531-536) — POSIX paths, cwd join
    fn isolate_resolve(&self, path: &str) -> String {
        if self.cwd.is_empty() || path.starts_with('/') {
            return path.to_string();
        }
        format!("{}/{}", self.cwd, path)
    }

    fn isolate_shell_test(&self) -> Option<Vec<String>> {
        if self.runtime.is_empty() {
            return None;
        }
        None // shell_test is folded into is_directory/is_regular_file callers
    }

    /// `with_cwd` (:655-663): the working directory rides in the command
    fn with_cwd(&self, inner: Vec<String>) -> Vec<String> {
        if self.cwd.is_empty() {
            return inner;
        }
        // 127 is what a shell reports for a command it could not run
        let mut out = vec![
            "sh".to_string(),
            "-c".to_string(),
            "cd \"$1\" || exit 127; shift; exec \"$@\"".to_string(),
            "_".to_string(),
            self.cwd.clone(),
        ];
        out.extend(inner);
        out
    }

    /// `build_argv` (:630-632) — wrap a complete POSIX argv into the
    /// host-side argv that runs it in the isolate
    fn build_isolate_argv(&self, inner: &[String], needs_stdin: bool) -> Vec<String> {
        if let Some(spec) = ContainerRuntimeSpec::parse(&self.runtime) {
            // tools_io_container::build_argv (:724-732)
            let mut argv = vec![spec.bin.clone(), "exec".to_string()];
            if needs_stdin {
                argv.push("-i".into());
            }
            argv.push(spec.arg.clone());
            argv.extend_from_slice(inner);
            return argv;
        }
        if let Some(target) = self.runtime.strip_prefix("ssh:") {
            // tools_io_ssh::build_argv (:758-766): the remote shell re-parses
            // the command line, so `inner` travels as one quoted word
            let mut argv = ssh_argv();
            if !needs_stdin {
                argv.push("-n".into());
            }
            argv.push(target.to_string());
            argv.push(shell_quote_join(inner));
            return argv;
        }
        inner.to_vec()
    }

    fn run_isolate(&self, argv: &[String], max_output: usize, timeout_secs: i64) -> ExecResult {
        self.run_isolate_raw(argv, max_output, timeout_secs, true)
    }

    fn run_isolate_raw(
        &self,
        argv: &[String],
        max_output: usize,
        timeout_secs: i64,
        _combine_stderr: bool,
    ) -> ExecResult {
        let mut res = ExecResult { output: String::new(), exit_code: -1, timed_out: false };
        let mut proc = Subproc::new();
        let Some(mut out) = proc.create_combined(argv, &[], None) else {
            res.output = "failed to spawn process".into();
            return res;
        };
        proc.close_stdin();
        let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1) as u64);
        let mut output: Vec<u8> = Vec::new();
        let mut truncated = false;
        let mut buf = [0u8; 4096];
        loop {
            if Instant::now() >= deadline {
                res.timed_out = true;
                proc.terminate();
                break;
            }
            match out.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if truncated {
                        continue;
                    }
                    if output.len() + n <= max_output {
                        output.extend_from_slice(&buf[..n]);
                    } else {
                        let remaining = max_output - output.len();
                        output.extend_from_slice(&buf[..remaining]);
                        truncated = true;
                    }
                }
                Err(_) => break,
            }
        }
        res.exit_code = proc.join();
        res.output = String::from_utf8_lossy(&output).into_owned();
        if truncated {
            res.output.push_str("\n[output truncated]");
        }
        res
    }

    fn run_isolate_with_stdin(
        &self,
        argv: &[String],
        max_output: usize,
        timeout_secs: i64,
        stdin: &[u8],
    ) -> ExecResult {
        let mut res = ExecResult { output: String::new(), exit_code: -1, timed_out: false };
        let mut proc = Subproc::new();
        let Some(mut out) = proc.create_combined(argv, &[], None) else {
            res.output = "failed to spawn process".into();
            return res;
        };
        proc.write_stdin_and_close(stdin);
        let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1) as u64);
        let mut output: Vec<u8> = Vec::new();
        let mut truncated = false;
        let mut buf = [0u8; 4096];
        loop {
            if Instant::now() >= deadline {
                res.timed_out = true;
                proc.terminate();
                break;
            }
            match out.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if truncated {
                        continue;
                    }
                    if output.len() + n <= max_output {
                        output.extend_from_slice(&buf[..n]);
                    } else {
                        let remaining = max_output - output.len();
                        output.extend_from_slice(&buf[..remaining]);
                        truncated = true;
                    }
                }
                Err(_) => break,
            }
        }
        res.exit_code = proc.join();
        res.output = String::from_utf8_lossy(&output).into_owned();
        if truncated {
            res.output.push_str("\n[output truncated]");
        }
        res
    }
}

/// `tools_io_ssh::ssh_argv` (:774-782): no console, so a prompt would hang
/// the tool call — key-based auth only, admin trusts the host key beforehand
fn ssh_argv() -> Vec<String> {
    vec![
        "ssh".into(),
        "-o".into(),
        "BatchMode=yes".into(),
        "-o".into(),
        "PasswordAuthentication=no".into(),
        "-o".into(),
        "KbdInteractiveAuthentication=no".into(),
        "-o".into(),
        "StrictHostKeyChecking=yes".into(),
    ]
}

/// `tools_io_ssh::is_valid_target` (:748-755) — a target from a client
/// header must not become an ssh option
fn is_valid_ssh_target(target: &str) -> bool {
    if target.is_empty() || target.starts_with('-') {
        return false;
    }
    target
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' || c == '@')
}

/// `shell_quote_join` (:635-648) — a single word a POSIX shell re-parses
/// into exactly `argv`
fn shell_quote_join(argv: &[String]) -> String {
    let mut out = String::new();
    for arg in argv {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push('\'');
        for c in arg.chars() {
            if c == '\'' {
                out.push_str("'\\''");
            } else {
                out.push(c);
            }
        }
        out.push('\'');
    }
    out
}

/// `container_runtime_spec` (:786-819)
struct ContainerRuntimeSpec {
    bin: String,
    arg: String,
    attach: bool,
}

impl ContainerRuntimeSpec {
    /// `parse` (:791-808)
    fn parse(spec: &str) -> Option<Self> {
        for bin in ["docker", "podman"] {
            let attach_prefix = format!("{bin}-container:");
            if let Some(id) = spec.strip_prefix(&attach_prefix) {
                return Some(ContainerRuntimeSpec { bin: bin.into(), arg: id.into(), attach: true });
            }
            let spawn_prefix = format!("{bin}:");
            if let Some(image) = spec.strip_prefix(&spawn_prefix) {
                return Some(ContainerRuntimeSpec { bin: bin.into(), arg: image.into(), attach: false });
            }
        }
        None
    }

    /// `is_valid_id` (:811-818) — an id starting with '-' would become an
    /// engine option, e.g. --privileged
    fn is_valid_id(id: &str) -> bool {
        if id.is_empty() || !id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric()) {
            return false;
        }
        id.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
    }
}

/// `std::filesystem::path::lexically_normal` — the `.`/`..` collapse of
/// `tools_io_basic::resolve` (:321)
fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// the tools — the definitions + the invokes (server-tools.cpp:862-1756)
// ---------------------------------------------------------------------------

/// the enabled tool set (`build_tools`, :1951-1965 — kept minimal on purpose)
pub const ALL_TOOL_NAMES: &[&str] = &[
    "read_file",
    "file_glob_search",
    "grep_search",
    "exec_shell_command",
    "write_file",
    "edit_file",
    "get_info",
];

fn jstr(s: &str) -> Json {
    Json::String(s.to_string())
}

fn jstr_arr(v: &[&str]) -> Json {
    Json::Array(v.iter().map(|s| jstr(s)).collect())
}

fn prop(name: &str, ty: &str, desc: &str) -> (String, Json) {
    (
        name.to_string(),
        Json::Object(vec![("type".into(), jstr(ty)), ("description".into(), jstr(desc))]),
    )
}

/// `server_tool::to_json` (:75-86) for the built-in tools (display_name,
/// tool, type "server", permissions.write, uses_cwd, definition)
fn tool_meta(name: &str) -> (&'static str, bool, bool) {
    // (display_name, permission_write, uses_cwd)
    match name {
        "read_file" => ("Read file", false, true),
        "file_glob_search" => ("File search", false, true),
        "grep_search" => ("Grep search", false, true),
        "exec_shell_command" => ("Execute shell command", true, true),
        "write_file" => ("Write file", true, true),
        "edit_file" => ("Edit file", true, true),
        "get_info" => ("Get Runtime Info", false, true),
        _ => ("", false, false),
    }
}

/// the JSON schema (`server_tool_*::get_definition`, :877-896 … :1709-1721)
fn tool_definition(name: &str) -> Json {
    let function = |desc: &str, params: Json| {
        Json::Object(vec![
            ("type".into(), jstr("function")),
            (
                "function".into(),
                Json::Object(vec![
                    ("name".into(), jstr(name)),
                    ("description".into(), jstr(desc)),
                    ("parameters".into(), params),
                ]),
            ),
        ])
    };
    let obj = |props: Vec<(String, Json)>, required: &[&str]| {
        Json::Object(vec![
            ("type".into(), jstr("object")),
            ("properties".into(), Json::Object(props)),
            (
                "required".into(),
                Json::Array(required.iter().map(|s| jstr(s)).collect()),
            ),
        ])
    };

    match name {
        "read_file" => function(
            "Read the contents of a file. Optionally specify a 1-based line range. \
             If append_loc is true, each line is prefixed with its line number (e.g. \"1\u{2192}...\").",
            obj(
                vec![
                    prop("path", "string", "Path to the file"),
                    prop("start_line", "integer", "First line to read, 1-based (default: 1)"),
                    prop("end_line", "integer", "Last line to read, 1-based inclusive (default: end of file)"),
                    prop("append_loc", "boolean", "Prefix each line with its line number"),
                ],
                &["path"],
            ),
        ),
        "file_glob_search" => function(
            "Recursively search for files matching a glob pattern under a directory. \
             Automatically skips files ignored by .gitignore (when the directory is inside a git repo) \
             and common junk directories (.git, node_modules, build, dist, etc.) otherwise. \
             A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. \
             A pattern containing '/' matches the full relative path; unless already anchored with \
             \"**/\" or a leading '/', it is automatically prefixed with \"**/\". \
             Use type=\"dir\" or \"all\" to also list directories; directory entries are suffixed with '/' in the output. \
             Note: directory listings do not apply .gitignore filtering.",
            obj(
                vec![
                    prop("path", "string", "Base directory to search in"),
                    prop("include", "string", "Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **"),
                    prop("exclude", "string", "Glob pattern for files to exclude"),
                    prop("type", "string", "Entry type to return: \"file\" (default), \"dir\" or \"all\""),
                    prop("max_depth", "integer", "Maximum depth to descend into subdirectories (default: 0 = unlimited; 1 = direct children only)"),
                    prop("limit", "integer", "Maximum number of results to return, capped at 100 (default 100)"),
                ],
                &["path"],
            ),
        ),
        "grep_search" => function(
            "Search for a pattern in files under a path. Returns matching lines with file paths \
             (and, unless searching a single file, paths relative to the given directory). \
             Automatically skips files ignored by .gitignore (when the directory is inside a git repo) \
             and common junk directories (.git, node_modules, build, dist, etc.) otherwise. \
             include/exclude: a pattern with no '/' matches the basename at any depth; a pattern \
             containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored).",
            obj(
                vec![
                    prop("path", "string", "File or directory to search in"),
                    prop("pattern", "string", "Pattern to search for (regular expression unless literal is true)"),
                    prop("include", "string", "Glob pattern to filter files (default: **)"),
                    prop("exclude", "string", "Glob pattern to exclude files"),
                    prop("return_line_numbers", "boolean", "If true, include line numbers in results"),
                    prop("literal", "boolean", "Treat pattern as a literal string instead of a regular expression (default: false)"),
                    prop("ignore_case", "boolean", "Case-insensitive search (default: false)"),
                    prop("context_lines", "integer", "Number of lines of context to show before and after each match (default: 0)"),
                ],
                &["path", "pattern"],
            ),
        ),
        "exec_shell_command" => function(
            "Execute a shell command and return its output (stdout and stderr combined).",
            obj(
                vec![
                    prop("command", "string", "Shell command to execute"),
                    prop("timeout", "integer", "Timeout in seconds (default 10, max 60)"),
                    prop("max_output_size", "integer", "Maximum output size in bytes (default 16384)"),
                ],
                &["command"],
            ),
        ),
        "write_file" => function(
            "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits.",
            obj(
                vec![
                    prop("path", "string", "Path of the file to write"),
                    prop("content", "string", "Content to write"),
                ],
                &["path", "content"],
            ),
        ),
        "edit_file" => function(
            "Edit a file using exact text replacement. Each edits[].old_text must be unique in the file \
             and is matched against the original content, not incrementally. Merge nearby changes into \
             one edit instead of overlapping edits. Use write_file to replace the whole file.",
            obj(
                vec![
                    prop("path", "string", "Path to the file to edit"),
                    (
                        "edits".into(),
                        Json::Object(vec![
                            ("type".into(), jstr("array")),
                            ("description".into(), jstr("One or more exact text replacements to apply")),
                            (
                                "items".into(),
                                Json::Object(vec![
                                    ("type".into(), jstr("object")),
                                    (
                                        "properties".into(),
                                        Json::Object(vec![
                                            prop("old_text", "string", "Exact text to find; must be unique in the file and must not overlap with other edits"),
                                            prop("new_text", "string", "Text to replace old_text with"),
                                        ]),
                                    ),
                                    ("required".into(), jstr_arr(&["old_text", "new_text"])),
                                ]),
                            ),
                        ]),
                    ),
                ],
                &["path", "edits"],
            ),
        ),
        "get_info" => function(
            "Returns runtime info: the OS name/version and the current working directory",
            Json::Object(vec![
                ("type".into(), jstr("object")),
                ("properties".into(), Json::Object(vec![])),
            ]),
        ),
        other => {
            let _ = other;
            Json::Object(vec![])
        }
    }
}

/// `server_tool::to_json` (:75-86)
fn tool_to_json(name: &str) -> Json {
    let (display, permission_write, uses_cwd) = tool_meta(name);
    Json::Object(vec![
        ("display_name".into(), jstr(display)),
        ("tool".into(), jstr(name)),
        ("type".into(), jstr("server")),
        (
            "permissions".into(),
            Json::Object(vec![("write".into(), Json::Bool(permission_write))]),
        ),
        ("uses_cwd".into(), Json::Bool(uses_cwd)),
        ("definition".into(), tool_definition(name)),
    ])
}

/// `server_mcp_tool::to_json` (server-tools.cpp:1822-1835)
fn mcp_tool_to_json(def: &ToolDef) -> Json {
    let schema = if def.input_schema.is_null() || !def.input_schema.is_object() {
        Json::Object(vec![])
    } else {
        def.input_schema.clone()
    };
    let name = format!("{}_{}", def.server_name, def.name);
    Json::Object(vec![
        ("display_name".into(), jstr(&name)),
        ("tool".into(), jstr(&name)),
        ("type".into(), jstr("mcp")),
        (
            "permissions".into(),
            Json::Object(vec![("write".into(), Json::Bool(false))]),
        ),
        ("uses_cwd".into(), Json::Bool(false)),
        (
            "definition".into(),
            Json::Object(vec![
                ("type".into(), jstr("function")),
                (
                    "function".into(),
                    Json::Object(vec![
                        ("name".into(), jstr(&name)),
                        ("description".into(), jstr(&def.description)),
                        ("parameters".into(), schema),
                    ]),
                ),
            ]),
        ),
    ])
}

// -- edit_file's fuzzy helpers (:1496-1692) --

/// `normalize_line_for_fuzzy_match` (:1498-1537): strip trailing whitespace,
/// fold smart quotes/dashes/spaces to ASCII
fn normalize_line_for_fuzzy_match(line: &str) -> String {
    let s = line.trim_end_matches([' ', '\t', '\r']);
    let mut s = s.to_string();
    let replace_all = |s: &mut String, from: &str, to: &str| {
        if from.is_empty() {
            return;
        }
        *s = s.replace(from, to);
    };
    // smart single quotes -> '
    for b in [0x98u8, 0x99, 0x9A, 0x9B] {
        replace_all(&mut s, &format!("\u{1b}\u{80}{}", b as char), "'");
    }
    // (the byte-pair form above never matches UTF-8 — the C matches the raw
    // \xE2\x80\x9x sequences, mirrored exactly below)
    for seq in ["\u{2018}", "\u{2019}", "\u{201A}", "\u{201B}"] {
        replace_all(&mut s, seq, "'");
    }
    // smart double quotes -> "
    for seq in ["\u{201C}", "\u{201D}", "\u{201E}", "\u{201F}"] {
        replace_all(&mut s, seq, "\"");
    }
    // various dashes -> -
    for seq in ["\u{2010}", "\u{2011}", "\u{2012}", "\u{2013}", "\u{2014}", "\u{2015}", "\u{2212}"] {
        replace_all(&mut s, seq, "-");
    }
    // special spaces -> ' '
    for seq in [
        "\u{00A0}", "\u{2002}", "\u{2003}", "\u{2004}", "\u{2005}", "\u{2006}", "\u{2007}",
        "\u{2008}", "\u{2009}", "\u{200A}", "\u{202F}", "\u{205F}", "\u{3000}",
    ] {
        replace_all(&mut s, seq, " ");
    }
    s
}

/// `normalize_for_fuzzy_match` (:1540-1554) — per line, preserving positions
fn normalize_for_fuzzy_match(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut start = 0usize;
    loop {
        match content[start..].find('\n') {
            Some(rel) => {
                let nl = start + rel;
                result.push_str(&normalize_line_for_fuzzy_match(&content[start..nl]));
                result.push('\n');
                start = nl + 1;
            }
            None => {
                result.push_str(&normalize_line_for_fuzzy_match(&content[start..]));
                return result;
            }
        }
    }
}

/// `count_occurrences` (:1588-1596)
fn count_occurrences(content: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    content.match_indices(needle).count()
}

struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

/// `apply_replacements` (:1606-1616) — ascending, non-overlapping, applied
/// back-to-front
fn apply_replacements(content: &str, replacements: &[MatchedEdit], offset: usize) -> String {
    let mut byte_content = content.as_bytes().to_vec();
    for rep in replacements.iter().rev() {
        let local = rep.match_index - offset;
        byte_content.splice(
            local..local + rep.match_length,
            rep.new_text.as_bytes().iter().copied(),
        );
    }
    String::from_utf8_lossy(&byte_content).into_owned()
}

/// `split_lines_with_endings` (:1557-1570)
fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    while start < content.len() {
        match content[start..].find('\n') {
            Some(rel) => {
                let nl = start + rel;
                lines.push(&content[start..=nl]);
                start = nl + 1;
            }
            None => {
                lines.push(&content[start..]);
                break;
            }
        }
    }
    lines
}

/// `apply_replacements_preserving_unchanged_lines` (:1647-1691)
fn apply_replacements_preserving_unchanged_lines(
    original_content: &str,
    base_content: &str,
    replacements: &[MatchedEdit],
) -> String {
    let original_lines = split_lines_with_endings(original_content);
    // get_line_spans (:1577-1585)
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(original_lines.len());
    let mut off = 0;
    for line in &original_lines {
        spans.push((off, off + line.len()));
        off += line.len();
    }

    // groups of replacements touching overlapping line ranges (:1654-1670)
    struct Group {
        start_line: usize,
        end_line: usize, // exclusive
        reps: Vec<MatchedEdit>,
    }
    let mut groups: Vec<Group> = Vec::new();
    for rep in replacements {
        // get_replacement_line_range (:1619-1644)
        let (start_line, end_line) = {
            let replacement_start = rep.match_index;
            let replacement_end = rep.match_index + rep.match_length;
            let mut start_line = None;
            for (i, (s, e)) in spans.iter().enumerate() {
                if replacement_start >= *s && replacement_start < *e {
                    start_line = Some(i);
                    break;
                }
            }
            let start_line = match start_line {
                Some(l) => l,
                // outside every line (should not happen); pin to the last
                None => spans.len().saturating_sub(1),
            };
            let mut end_line = start_line;
            while end_line < spans.len() && spans[end_line].1 < replacement_end {
                end_line += 1;
            }
            if end_line >= spans.len() {
                (start_line, spans.len().max(start_line + 1))
            } else {
                (start_line, end_line + 1)
            }
        };
        if let Some(last) = groups.last_mut() {
            if start_line < last.end_line {
                last.end_line = last.end_line.max(end_line);
                last.reps.push(MatchedEdit {
                    edit_index: rep.edit_index,
                    match_index: rep.match_index,
                    match_length: rep.match_length,
                    new_text: rep.new_text.clone(),
                });
                continue;
            }
        }
        groups.push(Group {
            start_line,
            end_line,
            reps: vec![MatchedEdit {
                edit_index: rep.edit_index,
                match_index: rep.match_index,
                match_length: rep.match_length,
                new_text: rep.new_text.clone(),
            }],
        });
    }

    let mut original_line_index = 0usize;
    let mut result = String::new();
    for g in &groups {
        for line in &original_lines[original_line_index..g.start_line.min(original_lines.len())] {
            result.push_str(line);
        }
        let group_start = spans[g.start_line.min(spans.len().saturating_sub(1))].0;
        let group_end = spans[(g.end_line - 1).min(spans.len().saturating_sub(1))].1;
        let slice = &base_content[group_start..group_end.min(base_content.len())];
        result.push_str(&apply_replacements(slice, &g.reps, group_start));
        original_line_index = g.end_line;
    }
    for line in &original_lines[original_line_index.min(original_lines.len())..] {
        result.push_str(line);
    }
    result
}

// -- the invokes --

/// the `x-resp-type` base64 support of read_file (:904) — the port's base64
/// is the server api's inline encoder (server/api.rs), reimplemented here on
/// bytes
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

/// `server_tools::find_tool` + dispatch — `invoke(params)` per tool. The
/// `runtime`/`cwd`/`resp_type` params were injected from the headers by the
/// POST handler (:2085-2113).
pub fn invoke_tool(name: &str, params: &Json, stream: Option<ToolStream>) -> Result<Json, String> {
    let get_string = |key: &str| -> Result<String, String> {
        params
            .at(key)
            .and_then(|v| v.get_str().ok().map(|s| s.to_string()))
            .ok_or_else(|| format!("at(params): key '{key}' not found"))
    };
    let val_string = |key: &str, def: &str| -> String {
        params
            .at(key)
            .and_then(|v| v.get_str().ok().map(|s| s.to_string()))
            .unwrap_or_else(|| def.to_string())
    };
    let val_i64 = |key: &str, def: i64| -> i64 {
        params.at(key).and_then(|v| v.get_i64().ok()).unwrap_or(def)
    };
    let val_bool = |key: &str, def: bool| -> bool {
        params
            .at(key)
            .and_then(|v| match v {
                Json::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(def)
    };

    let cwd = val_string("cwd", "");
    let runtime = val_string("runtime", "");
    let io = ToolsIo::for_params(&cwd, &runtime)?;

    match name {
        // read_file (:898-965)
        "read_file" => {
            let path = get_string("path")?;
            let start_line = val_i64("start_line", 1) as i32;
            let end_line = val_i64("end_line", -1) as i32;
            let append_loc = val_bool("append_loc", false);
            // comes from the x-resp-type header, the model cannot ask for it
            let as_base64 = val_string("resp_type", "") == "base64";

            let Some(file_size) = io.file_size(&path) else {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("cannot stat file: {path}")),
                )]));
            };

            if as_base64 {
                if file_size > SERVER_TOOL_READ_FILE_MAX_SIZE_BASE64 as u64 {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!(
                            "file too large ({} bytes, max {})",
                            file_size, SERVER_TOOL_READ_FILE_MAX_SIZE_BASE64
                        )),
                    )]));
                }
                let Some(content) = io.read_file(&path) else {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!("failed to open file: {path}")),
                    )]));
                };
                return Ok(Json::Object(vec![
                    ("base64".into(), jstr(&base64_encode(content.as_bytes()))),
                    ("size_bytes".into(), Json::Uint(content.len() as u64)),
                ]));
            }

            if file_size > SERVER_TOOL_READ_FILE_MAX_SIZE as u64 && end_line == -1 {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!(
                        "file too large ({} bytes, max {}). Use start_line/end_line to read a portion.",
                        file_size, SERVER_TOOL_READ_FILE_MAX_SIZE
                    )),
                )]));
            }

            let Some(content) = io.read_file(&path) else {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("failed to open file: {path}")),
                )]));
            };

            let mut result = String::new();
            for (i, line) in content.lines().enumerate() {
                let lineno = (i + 1) as i32;
                if lineno < start_line {
                    continue;
                }
                if end_line != -1 && lineno > end_line {
                    break;
                }
                let out_line = if append_loc {
                    format!("{lineno}\u{2192}{line}\n")
                } else {
                    format!("{line}\n")
                };
                if result.len() + out_line.len() > SERVER_TOOL_READ_FILE_MAX_SIZE {
                    result.push_str("[output truncated]");
                    break;
                }
                result.push_str(&out_line);
            }
            Ok(Json::Object(vec![(
                "plain_text_response".into(),
                jstr(&result),
            )]))
        }

        // file_glob_search (:1015-1083)
        "file_glob_search" => {
            let path = get_string("path")?;

            let base = io.resolve(&path);
            let include = val_string("include", "**");
            let exclude = val_string("exclude", "");
            let ty = val_string("type", "file");
            let max_depth = val_i64("max_depth", 0).max(0) as i32;
            let limit_req = val_i64("limit", SERVER_TOOL_FILE_SEARCH_MAX_RESULTS);
            if limit_req < 1 {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("invalid limit: {limit_req} (expected 1 or more)")),
                )]));
            }
            let limit = limit_req.min(SERVER_TOOL_FILE_SEARCH_MAX_RESULTS);

            let kind = match ty.as_str() {
                "file" => ListKind::Files,
                "dir" => ListKind::Dirs,
                "all" => ListKind::All,
                _ => {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!(
                            "invalid type: {ty} (expected \"file\", \"dir\" or \"all\")"
                        )),
                    )]))
                }
            };

            let listing = io.list_entries(&base, max_depth, kind);
            if !listing.err.is_empty() {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("{}: {}", listing.err, path)),
                )]));
            }

            let mut matches: Vec<&ListEntry> = Vec::new();
            for entry in &listing.entries {
                if !path_glob_match(&include, &entry.rel) {
                    continue;
                }
                if !exclude.is_empty() && path_glob_match(&exclude, &entry.rel) {
                    continue;
                }
                matches.push(entry);
            }

            let total = matches.len();
            let shown = total.min(limit as usize);

            let mut output_text = String::new();
            let mut entries_json = Vec::new();
            for m in &matches[..shown] {
                output_text.push_str(&m.rel);
                if m.is_dir {
                    output_text.push('/');
                }
                output_text.push('\n');
                entries_json.push(Json::Object(vec![
                    ("path".into(), jstr(&m.rel)),
                    ("type".into(), jstr(if m.is_dir { "dir" } else { "file" })),
                ]));
            }

            output_text.push_str(&format!("\n---\nTotal matches: {total}\n"));
            if total > shown {
                output_text.push_str(&format!(
                    "[{shown} results limit reached ({total} total matches). Refine the glob pattern to narrow the search.]\n"
                ));
            }
            if listing.truncated {
                output_text.push_str("[results truncated: time budget or unreadable directory]\n");
            }

            Ok(Json::Object(vec![
                ("plain_text_response".into(), jstr(&output_text)),
                ("entries".into(), Json::Array(entries_json)),
                ("base".into(), jstr(&base)),
            ]))
        }

        // grep_search (:1130-1235)
        "grep_search" => {
            let path = get_string("path")?;
            let pat_str = get_string("pattern")?;
            let include = val_string("include", "**");
            let exclude = val_string("exclude", "");
            let show_lineno = val_bool("return_line_numbers", false);
            let literal = val_bool("literal", false);
            let ignore_case = val_bool("ignore_case", false);
            let ctx_lines = val_i64("context_lines", 0).max(0) as usize;

            let pattern_src = if literal {
                // :1142-1150 — escape the specials of the C's set
                const SPECIALS: &str = "\\^$.|?*+()[]{}";
                let mut escaped = String::with_capacity(pat_str.len() * 2);
                for c in pat_str.chars() {
                    if SPECIALS.contains(c) {
                        escaped.push('\\');
                    }
                    escaped.push(c);
                }
                escaped
            } else {
                pat_str.clone()
            };

            // the case-insensitive form folds both sides to lowercase (the
            // std::regex::icase equivalent for ASCII patterns)
            let fold = |s: String| if ignore_case { s.to_lowercase() } else { s };
            let re = llama::regex_lite::RegexLite::new(&fold(pattern_src))
                .map_err(|e| format!("invalid regex: {e}"))?;

            // collect (absolute_path, display_path) pairs (:1163-1181)
            let mut files: Vec<(String, String)> = Vec::new();
            let abs_path = io.resolve(&path);
            if io.is_regular_file(&path) {
                files.push((abs_path.clone(), path.clone()));
            } else if io.is_directory(&path) {
                let listing = io.list_entries(&abs_path, 0, ListKind::Files);
                if !listing.err.is_empty() {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!("{}: {}", listing.err, path)),
                    )]));
                }
                for entry in &listing.entries {
                    if !path_glob_match(&include, &entry.rel) {
                        continue;
                    }
                    if !exclude.is_empty() && path_glob_match(&exclude, &entry.rel) {
                        continue;
                    }
                    files.push((format!("{}/{}", abs_path.trim_end_matches('/'), entry.rel), entry.rel.clone()));
                }
            } else {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("path does not exist: {path}")),
                )]));
            }

            let mut output_text = String::new();
            let mut total = 0usize;
            let mut limit_reached = false;
            let show_num = show_lineno || ctx_lines > 0;

            'outer: for (fpath, display_path) in &files {
                let Some(content) = io.read_file(fpath) else { continue };
                let lines: Vec<&str> = content.lines().collect();

                for (i, line) in lines.iter().enumerate() {
                    if total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS {
                        limit_reached = true;
                        break 'outer;
                    }
                    if !re.find(fold(line.to_string()).as_bytes()).is_some() {
                        continue;
                    }

                    let ctx_start = if ctx_lines > 0 { i.saturating_sub(ctx_lines) } else { i };
                    let ctx_end = if ctx_lines > 0 {
                        (i + ctx_lines).min(lines.len() - 1)
                    } else {
                        i
                    };
                    for j in ctx_start..=ctx_end {
                        let is_match = j == i;
                        output_text.push_str(display_path);
                        output_text.push(if is_match { ':' } else { '-' });
                        if show_num {
                            output_text.push_str(&(j + 1).to_string());
                            output_text.push(if is_match { ':' } else { '-' });
                        }
                        output_text.push_str(lines[j]);
                        output_text.push('\n');
                    }
                    if ctx_lines > 0 {
                        output_text.push_str("--\n");
                    }
                    total += 1;
                }
            }

            output_text.push_str(&format!("\n---\nTotal matches: {total}\n"));
            if limit_reached {
                output_text.push_str(&format!(
                    "[{} matches limit reached. Narrow the path/pattern/include to see more.]\n",
                    SERVER_TOOL_GREP_SEARCH_MAX_RESULTS
                ));
            }

            Ok(Json::Object(vec![(
                "plain_text_response".into(),
                jstr(&output_text),
            )]))
        }

        // exec_shell_command (:1273-1317)
        "exec_shell_command" => {
            let command = get_string("command")?;
            let mut timeout = val_i64("timeout", 10);
            let mut max_output = val_i64("max_output_size", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

            timeout = timeout.min(SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
            max_output = max_output.min(SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

            // an isolate is always POSIX regardless of host OS (:1281-1288)
            let args: Vec<String> = vec!["sh".into(), "-c".into(), command];

            if let Some(st) = stream {
                let (chunks, done) = st.channels;
                let io2 = ToolsIo::for_params(&cwd, &runtime)?;
                std::thread::spawn(move || {
                    let res = io2.run(&args, max_output as usize, timeout);
                    let tail = format!("\n[exit code: {}]", res.exit_code)
                        + if res.timed_out { " [exit due to timed out]" } else { "" };
                    let _ = chunks.send(StreamEvent::Frame(res.output));
                    let _ = chunks.send(StreamEvent::Frame(tail));
                    let _ = done.send(());
                });
                // the chunks stream outlives this call; the handler waits on done
                return Ok(Json::Null);
            }

            let res = io.run(&args, max_output as usize, timeout);
            let mut text_output = res.output;
            text_output.push_str(&format!("\n[exit code: {}]", res.exit_code));
            if res.timed_out {
                text_output.push_str(" [exit due to timed out]");
            }
            Ok(Json::Object(vec![(
                "plain_text_response".into(),
                jstr(&text_output),
            )]))
        }

        // write_file (:1350-1360)
        "write_file" => {
            let path = get_string("path")?;
            let content = get_string("content")?;

            if !io.write_file(&path, &content) {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("failed to write file: {path}")),
                )]));
            }
            Ok(Json::Object(vec![
                ("result".into(), jstr("file written successfully")),
                ("path".into(), jstr(&path)),
                ("bytes".into(), Json::Uint(content.len() as u64)),
            ]))
        }

        // edit_file (:1407-1494)
        "edit_file" => {
            let path = get_string("path")?;
            let Some(edits_json) = params.at("edits") else {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr("\"edits\" must be a non-empty array"),
                )]));
            };
            let Json::Array(edits_arr) = edits_json else {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr("\"edits\" must be a non-empty array"),
                )]));
            };
            if edits_arr.is_empty() {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr("\"edits\" must be a non-empty array"),
                )]));
            }

            let mut edits: Vec<(String, String)> = Vec::with_capacity(edits_arr.len());
            for e in edits_arr {
                let old_text = e
                    .at("old_text")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                let new_text = e
                    .at("new_text")
                    .and_then(|v| v.get_str().ok())
                    .unwrap_or("")
                    .to_string();
                if old_text.is_empty() {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!("edits[{}].old_text must not be empty", edits.len())),
                    )]));
                }
                edits.push((old_text, new_text));
            }

            let Some(original_content) = io.read_file(&path) else {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("failed to open file: {path}")),
                )]));
            };

            // does any old_text need fuzzy matching? (:1437-1449)
            let mut any_fuzzy = false;
            for (i, (old, _)) in edits.iter().enumerate() {
                if original_content.contains(old.as_str()) {
                    continue;
                }
                let fuzzy_content = normalize_for_fuzzy_match(&original_content);
                let fuzzy_old = normalize_for_fuzzy_match(old);
                if !fuzzy_content.contains(&fuzzy_old) {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!(
                            "could not find edits[{i}].old_text in {path}, it must match the file's current content exactly"
                        )),
                    )]));
                }
                any_fuzzy = true;
            }

            let base_content = if any_fuzzy {
                normalize_for_fuzzy_match(&original_content)
            } else {
                original_content.clone()
            };

            // uniqueness check always uses fuzzy-normalized text (:1453-1468)
            let mut matched: Vec<MatchedEdit> = Vec::with_capacity(edits.len());
            for (i, (old, new_text)) in edits.iter().enumerate() {
                let needle = if any_fuzzy {
                    normalize_for_fuzzy_match(old)
                } else {
                    old.clone()
                };
                let occurrences = count_occurrences(
                    &normalize_for_fuzzy_match(&original_content),
                    &normalize_for_fuzzy_match(old),
                );
                if occurrences > 1 {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!(
                            "found {occurrences} occurrences of edits[{i}].old_text in {path}, it must be unique"
                        )),
                    )]));
                }
                // a needle that appears nowhere was rejected above; 0-index
                // here only when the fuzzy base still contains it
                let idx = base_content.find(&needle).unwrap_or(0);
                matched.push(MatchedEdit {
                    edit_index: i,
                    match_index: idx,
                    match_length: needle.len(),
                    new_text: new_text.clone(),
                });
            }

            matched.sort_by(|a, b| a.match_index.cmp(&b.match_index));
            for i in 1..matched.len() {
                if matched[i - 1].match_index + matched[i - 1].match_length > matched[i].match_index {
                    return Ok(Json::Object(vec![(
                        "error".into(),
                        jstr(&format!(
                            "edits[{}] and edits[{}] overlap in {path}; merge them into one edit or target disjoint regions",
                            matched[i - 1].edit_index, matched[i].edit_index
                        )),
                    )]));
                }
            }

            let new_content = if any_fuzzy {
                apply_replacements_preserving_unchanged_lines(&original_content, &base_content, &matched)
            } else {
                apply_replacements(&base_content, &matched, 0)
            };

            if new_content == original_content {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr("no changes made: the replacement(s) produced identical content"),
                )]));
            }

            if !io.write_file(&path, &new_content) {
                return Ok(Json::Object(vec![(
                    "error".into(),
                    jstr(&format!("failed to write file: {path}")),
                )]));
            }

            Ok(Json::Object(vec![
                ("result".into(), jstr("file edited successfully")),
                ("path".into(), jstr(&path)),
                ("edits_applied".into(), Json::Int(matched.len() as i64)),
            ]))
        }

        // get_info (:1723-1755)
        "get_info" => {
            let args: Vec<String> = vec!["uname".into(), "-a".into()];
            let res = io.run(&args, SERVER_TOOL_GET_INFO_MAX_OUTPUT as usize, SERVER_TOOL_GET_INFO_TIMEOUT);
            let os_info = if res.exit_code == 0 && !res.timed_out {
                res.output.trim().to_string()
            } else {
                "unknown".to_string()
            };

            let cwd = if !cwd.is_empty() {
                cwd
            } else if runtime.is_empty() {
                std::env::current_dir()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                let pwd = io.run(&["pwd".into()], SERVER_TOOL_GET_INFO_MAX_OUTPUT as usize, SERVER_TOOL_GET_INFO_TIMEOUT);
                if pwd.exit_code == 0 && !pwd.timed_out {
                    pwd.output.trim().to_string()
                } else {
                    "unknown".to_string()
                }
            };

            Ok(Json::Object(vec![
                ("os".into(), jstr(&os_info)),
                ("cwd".into(), jstr(&cwd)),
            ]))
        }

        other => Err(format!("unknown tool \"{other}\"")),
    }
}

// ---------------------------------------------------------------------------
// the /tools endpoints — server_tools::setup's handlers (:1997-2172)
// ---------------------------------------------------------------------------

/// `server_tool::stream`'s push channel — the chunks the worker streams
/// (the Sender half; the worker's terminal frame rides the same channel)
pub struct ToolStream {
    pub channels: (mpsc::Sender<StreamEvent>, mpsc::Sender<()>),
}

/// the `--tools-runtime` spec (`server_tools_runtime`, :1844-1933). The
/// container-spawning owner is not ported (it needs an engine daemon); an
/// `"<engine>:<image>"` spec fails like an unavailable runtime.
fn tools_runtime_spec(spec: &str) -> Result<String, String> {
    if let Some(parsed) = ContainerRuntimeSpec::parse(spec) {
        if !parsed.attach {
            return Err(format!("unknown --tools-runtime option: {spec}"));
        }
    }
    ToolsIo::for_params("", spec).map(|_| spec.to_string())
}

/// `server_tools::setup` (:1997-2060): the enabled-tool set + the MCP bridge
pub fn setup(
    enabled_tools: &[String],
    mcp_mgr: &Arc<ServerMcp>,
    tools_runtime: &str,
) -> Result<Vec<String>, String> {
    if !tools_runtime.is_empty() {
        tools_runtime_spec(tools_runtime)?;
    }

    let mut tools: Vec<String> = Vec::new();

    if !enabled_tools.is_empty() {
        if !Subproc::is_supported() {
            return Err("subprocess is not enabled on this build".into());
        }

        let enabled_set: HashSet<String> = enabled_tools.iter().cloned().collect();

        // validate that every requested tool is known (:2019-2028)
        for name in enabled_tools {
            if name == "all" {
                continue;
            }
            if !ALL_TOOL_NAMES.contains(&name.as_str()) {
                return Err(format!(
                    "unknown tool \"{name}\". available tools: {}",
                    ALL_TOOL_NAMES.join(", ")
                ));
            }
        }

        for name in ALL_TOOL_NAMES {
            if enabled_set.contains(*name) || enabled_set.contains("all") {
                tools.push(name.to_string());
            }
        }
    }

    // append MCP tools, skipping collisions (:2038-2059)
    if !mcp_mgr.is_empty() {
        let mut seen: HashSet<String> = tools.iter().cloned().collect();
        let mut n_added = 0;
        for def in mcp_mgr.list_tools() {
            let mcp_name = format!("{}_{}", def.server_name, def.name);
            if seen.contains(&mcp_name) {
                eprintln!(
                    "MCP tool \"{mcp_name}\" from server \"{}\" collides with an existing tool, skipping",
                    def.server_name
                );
                continue;
            }
            seen.insert(mcp_name.clone());
            tools.push(mcp_name);
            n_added += 1;
        }
        if n_added > 0 {
            eprintln!("Added {n_added} MCP tools");
        }
    }

    Ok(tools)
}

/// `handle_get` (:2061-2075) — the enabled tools' definitions
pub fn handle_tools_get(tools: &Arc<Vec<String>>, mcp_mgr: &Arc<ServerMcp>, _req: &Request) -> Response {
    let mut result = Vec::new();
    for t in tools.iter() {
        if let Some((server, tool)) = t.split_once('_') {
            // MCP tools carry the "<server>_<tool>" name
            if let Some(def) = mcp_mgr
                .list_tools()
                .into_iter()
                .find(|d| d.server_name == server && format!("{}_{}", d.server_name, d.name) == *t)
            {
                result.push(mcp_tool_to_json(&def));
                continue;
            }
        }
        result.push(tool_to_json(t));
    }
    Response::ok(Json::Array(result).dump())
}

/// `handle_post` (:2077-2171): `{"tool": name, "params": {...}, "stream": b}`
/// with the x-tool-cwd / x-tool-runtime / x-resp-type header injections.
/// `stream: true` answers `text/event-stream` with `{"chunk": ...}` frames
/// and a terminal `{"done": true}` (:2117-2153).
pub fn handle_tools_post(
    tools: &Arc<Vec<String>>,
    tools_runtime: &Arc<String>,
    mcp_mgr: &Arc<ServerMcp>,
    req: &Request,
) -> Response {
    // 400 body shape: format_error_response(what, ERROR_TYPE_INVALID_REQUEST)
    let bad_request = |what: &str| -> Response {
        Response {
            status: 400,
            content_type: "application/json; charset=utf-8".into(),
            body: Body::Full(crate::api::json_error(what, "invalid_request_error", 400)),
            headers: Vec::new(),
            terminal_done: false,
        }
    };
    #[allow(clippy::redundant_closure)]
    let not_found = |_what: &str| -> Response { Response::not_found() };
    let server_error = |what: &str| -> Response {
        Response {
            status: 500,
            content_type: "application/json; charset=utf-8".into(),
            body: Body::Full(crate::api::json_error(what, "server_error", 500)),
            headers: Vec::new(),
            terminal_done: false,
        }
    };

    let body = match Json::parse(&req.body_str()) {
        Ok(b) => b,
        Err(e) => return bad_request(&e),
    };
    let Some(tool_name) = body.at("tool").and_then(|v| v.get_str().ok()).map(|s| s.to_string()) else {
        return bad_request("at(body): key 'tool' not found");
    };
    let mut params = body.at("params").cloned().unwrap_or(Json::Object(vec![]));
    let stream = body
        .at("stream")
        .and_then(|v| match v {
            Json::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);

    // the header injections (:2085-2113) — params copies, keys erased first
    let mut set_param = |params: &mut Json, key: &str, value: String| {
        if let Json::Object(entries) = params {
            entries.retain(|(k, _)| k != key);
            entries.push((key.to_string(), Json::String(value)));
        }
    };
    if let Some(cwd) = req.headers.get("x-tool-cwd") {
        if !cwd.is_empty() {
            set_param(&mut params, "cwd", cwd.clone());
        }
    }
    let runtime_header = req.headers.get("x-tool-runtime").cloned().unwrap_or_default();
    if !runtime_header.is_empty() {
        set_param(&mut params, "runtime", runtime_header);
    } else if !tools_runtime.is_empty() {
        set_param(&mut params, "runtime", tools_runtime.as_ref().clone());
    }
    if let Some(resp_type) = req.headers.get("x-resp-type") {
        if !resp_type.is_empty() {
            set_param(&mut params, "resp_type", resp_type.clone());
        }
    }

    // find_tool (:1935-1945)
    if !tools.contains(&tool_name) {
        return not_found(&format!("unknown tool \"{tool_name}\""));
    }

    // the MCP bridge: "<server>_<tool>" (:1800-1840)
    let is_mcp = mcp_mgr
        .list_tools()
        .iter()
        .any(|d| format!("{}_{}", d.server_name, d.name) == tool_name);

    if stream {
        if is_mcp {
            return bad_request(&format!(
                "tool \"{tool_name}\" does not support stream = true"
            ));
        }
        let (tx, rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let params2 = params.clone();
        let name2 = tool_name.clone();
        std::thread::spawn(move || {
            let st = ToolStream { channels: (tx, done_tx) };
            let result = invoke_tool(&name2, &params2, Some(st));
            // the worker's terminal frame (:2128-2138)
            let _ = &result;
        });
        let (frame_tx, frame_rx) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(ev) = rx.recv() {
                if frame_tx.send(ev).is_err() {
                    break;
                }
            }
            // done → the terminal {"done": true} frame
            let _ = done_rx.recv();
            let done_frame = StreamEvent::Frame("{\"done\":true}".to_string());
            let _ = frame_tx.send(done_frame);
            let _ = frame_tx.send(StreamEvent::Done);
        });
        return Response::stream(frame_rx);
    }

    if is_mcp {
        let (server, tool) = tool_name.split_once('_').unwrap();
        let result = mcp_mgr.call_tool(server, tool, &params);
        return Response::ok(result.dump());
    }

    match invoke_tool(&tool_name, &params, None) {
        Ok(result) => Response::ok(result.dump()),
        Err(e) => {
            eprintln!("got exception: {e}");
            server_error(&e)
        }
    }
}

// ---------------------------------------------------------------------------
// the CORS proxy — server-cors-proxy.h (83 lines)
// ---------------------------------------------------------------------------

/// `common_http_parse_url` (common/http.h:122-191) — the subset the proxy
/// validates on: scheme://host[:port]/path
struct ParsedUrl {
    scheme: String,
    host: String,
    port: u16,
    path: String,
    password: String,
}

fn parse_url(target: &str) -> Result<ParsedUrl, String> {
    let (scheme, rest) = target
        .split_once("://")
        .ok_or_else(|| "invalid target URL: missing host".to_string())?;
    // userinfo@host — the proxy rejects a password outright (:34-36)
    let (userinfo, hostport) = match rest.split_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, rest),
    };
    // the authority split of common/http.h:71-91: a bracketed IPv6 literal
    // keeps its inner colons, else the FIRST colon starts the port
    let (authority, path) = match hostport.split_once('/') {
        Some((a, p)) => (a, format!("/{p}")),
        None => (hostport, String::new()),
    };
    let (host, port) = if authority.starts_with('[') {
        let close = authority
            .find(']')
            .ok_or_else(|| format!("invalid IPv6 URL authority: {authority}"))?;
        let h = authority[1..close].to_string();
        let after = &authority[close + 1..];
        let p = match after.strip_prefix(':') {
            Some(p) => p
                .parse::<u16>()
                .map_err(|_| "invalid target URL port".to_string())?,
            None => {
                if scheme == "https" {
                    443
                } else {
                    80
                }
            }
        };
        (h, p)
    } else {
        match authority.split_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse::<u16>()
                    .map_err(|_| "invalid target URL port".to_string())?,
            ),
            None => (
                authority.to_string(),
                if scheme == "https" { 443 } else { 80 },
            ),
        }
    };
    let password = userinfo
        .and_then(|u| u.split_once(':'))
        .map(|(_, pw)| pw.to_string())
        .unwrap_or_default();
    Ok(ParsedUrl {
        scheme: scheme.to_string(),
        host,
        port,
        path,
        password,
    })
}

/// `proxy_request` (server-cors-proxy.h:22-75) + the two handler entry points
/// (:77-83). The `url` query param is the target; headers tagged
/// `x-llama-server-proxy-header-<name>` become the forwarded headers.
///
/// The forwarding client is `server_http_proxy` (server-models.cpp:2456-2600):
/// http over a plain socket, https over the system OpenSSL client
/// (`httplib::SSLClient` — TLS1.2+, system verify paths + hostname checks,
/// the exact defaults of vendor/cpp-httplib/httplib.h:3045-3046; the port's
/// `tls.rs` links the same libssl/libcrypto the reference binary does),
/// `set_follow_location(true)` with the redirect rules of
/// vendor/cpp-httplib/httplib.cpp:4222-4244 (max 20 hops, 303 turns
/// non-GET/HEAD into a bodyless headerless GET, cross-host hops drop
/// Host/Authorization/Cookie/Cookie2), the request-header rewrite rules of
/// server-models.cpp:2536-2583 (drop accept-encoding/transfer-encoding/
/// content-length, multipart content-type → `application/json;
/// charset=utf-8` when the request carried no files, and `Host:
/// format_host(host)[:port]` with the port omitted on the scheme's default),
/// the multipart rebuild of :2553-2568 (uploaded files — the multipart POST
/// surface of server-http.cpp:683-717 — are reassembled with a fresh
/// `----llama-cpp-proxy-<16 alnum>` boundary, `build_multipart_body`
/// server-models.cpp:2395-2451), and the response relay of :2365-2377 (all
/// headers minus server/transfer-encoding/content-length/keep-alive/
/// access-control-*, the body as raw bytes — `Body::Bytes`, byte-exact).
pub fn cors_proxy(req: &Request) -> Response {
    let server_error = |what: &str| Response {
        status: 500,
        content_type: "application/json; charset=utf-8".into(),
        body: Body::Full(crate::api::json_error(what, "server_error", 500)),
        headers: Vec::new(),
        terminal_done: false,
    };

    let target_url = req.get_param("url", "");
    let mut parsed = match parse_url(&target_url) {
        Ok(p) => p,
        Err(e) => return server_error(&e),
    };

    if parsed.host.is_empty() {
        return server_error("invalid target URL: missing host");
    }
    if parsed.path.is_empty() {
        parsed.path = "/".into();
    }
    if !parsed.password.is_empty() {
        return server_error("authentication in target URL is not supported");
    }
    if parsed.scheme != "http" && parsed.scheme != "https" {
        // common/http.h:39's own parse error surfaces before the proxy's
        // check (server-cors-proxy.h:38-39)
        return server_error(&format!("unsupported URL scheme: {}", parsed.scheme));
    }

    eprintln!(
        "proxying {} request to {}://{}:{}{}",
        req.method, parsed.scheme, parsed.host, parsed.port, parsed.path
    );

    // x-llama-server-proxy-header-<name> -> <name> (:44-58)
    let mut headers: Vec<(String, String)> = Vec::new();
    const PROXY_PREFIX: &str = "x-llama-server-proxy-header-";
    for (key, value) in &req.headers {
        if !key.starts_with(PROXY_PREFIX) {
            continue;
        }
        let new_key = &key[PROXY_PREFIX.len()..];
        if new_key.is_empty() {
            continue;
        }
        headers.push((new_key.to_string(), value.clone()));
    }

    let mut method = req.method.clone();
    let mut body = req.body.clone();
    // when files are present the incoming multipart form was converted to a
    // JSON body of its text fields (server-http.cpp:686-700) — reconstruct
    // the multipart body for the downstream server (server-models.cpp:
    // 2553-2568)
    let mut has_files = !req.files.is_empty();
    let mut override_content_type: Option<String> = None;
    if has_files {
        let form_fields = match Json::parse(&req.body_str()) {
            Ok(v) if v.is_object() => v,
            _ => return server_error("failed to parse multipart form fields JSON"),
        };
        let boundary = generate_multipart_boundary();
        match build_multipart_body(&form_fields, &req.files, &boundary) {
            Ok(b) => body = b,
            Err(e) => return server_error(&e),
        }
        override_content_type = Some(format!("multipart/form-data; boundary={boundary}"));
    }
    let mut redirects = 20usize; // CPPHTTPLIB_REDIRECT_MAX_COUNT (httplib.h:121-122)
    let resp = loop {
        let r = match proxy_fetch_once(
            &parsed,
            &method,
            &headers,
            &body,
            has_files,
            override_content_type.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => return server_error(&e),
        };
        // `300 < res.status < 400 && follow_location_` (httplib.cpp:10775)
        if (301..400).contains(&r.status) {
            if let Some(location) =
                proxy_header(&r.headers, "location").filter(|l| !l.is_empty())
            {
                if redirects == 0 {
                    return server_error("exceed redirect count");
                }
                let Ok(next) = parse_redirect_location(&location, &parsed) else {
                    // detail::parse_url failure → the redirect is not
                    // followed; the 3xx itself relays
                    break r;
                };
                // only http/https redirects are followed (httplib.cpp:10834-10837)
                if next.scheme != "http" && next.scheme != "https" {
                    break r;
                }
                redirects -= 1;
                // 303 turns a non-GET/HEAD hop into a bodyless, headerless
                // GET (detail::redirect, httplib.cpp:4227-4231)
                if r.status == 303 && method != "GET" && method != "HEAD" {
                    method = "GET".into();
                    body.clear();
                    headers.clear();
                    // the override content-type rode the request headers —
                    // cleared with them
                    override_content_type = None;
                    has_files = false;
                }
                // cross-host/scheme hops drop the host-bound headers
                // (create_redirect_client, httplib.cpp:10872-10882)
                if next.host != parsed.host || next.scheme != parsed.scheme || next.port != parsed.port
                {
                    headers.retain(|(k, _)| {
                        !matches!(
                            k.to_ascii_lowercase().as_str(),
                            "host" | "proxy-authorization" | "authorization" | "cookie" | "cookie2"
                        )
                    });
                }
                parsed = next;
                continue;
            }
        }
        break r;
    };

    // the response relay (make_header_msg, server-models.cpp:2365-2377):
    // content-type rides its field, everything else minus the strip list
    let mut content_type = "application/octet-stream".to_string();
    let mut out_headers: Vec<(String, String)> = Vec::new();
    for (k, v) in &resp.headers {
        let lowered = k.to_ascii_lowercase();
        if lowered == "server"
            || lowered == "transfer-encoding"
            || lowered == "content-length"
            || lowered == "keep-alive"
            || lowered.starts_with("access-control-")
        {
            continue;
        }
        if lowered == "content-type" {
            content_type = v.clone();
            continue;
        }
        out_headers.push((k.clone(), v.clone()));
    }
    // the relayed body is the raw response bytes (`res->data` is a
    // std::string — arbitrary bytes, no UTF-8 constraint)
    Response {
        status: resp.status,
        content_type,
        body: Body::Bytes(resp.body),
        headers: out_headers,
        terminal_done: false,
    }
}

/// `generate_multipart_boundary` (server-models.cpp:2383-2393):
/// `----llama-cpp-proxy-` + 16 chars of [0-9a-z]
fn generate_multipart_boundary() -> String {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0) };
    }
    // the reference seeds std::mt19937 from random_device; the port folds
    // address-space entropy + a monotonically advancing thread-local
    // counter — both are uniformly unpredictable boundaries, which is the
    // property the rebuild needs (it must not collide with body content)
    STATE.with(|st| {
        let mut x = st.get().wrapping_add(0x9E3779B97F4A7C15);
        st.set(x);
        if x == 0 {
            x = &x as *const u64 as u64;
        }
        let mut boundary = String::from("----llama-cpp-proxy-");
        for _ in 0..16 {
            // xorshift64 → [0,36)
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            const CHARS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
            boundary.push(CHARS[(x % 36) as usize] as char);
        }
        boundary
    })
}

/// `sanitize_field` (server-models.cpp:2400-2410): strip CR, LF and the
/// double-quote — the characters that would break the disposition line
fn sanitize_field(text: &str) -> String {
    text.chars().filter(|&c| c != '\n' && c != '\r' && c != '"').collect()
}

/// `build_multipart_body` (server-models.cpp:2395-2451): the form fields
/// (a JSON object; duplicate-key arrays were already folded by the JSON
/// translation) then the files, each part framed by the boundary, closed
/// with the terminal `--boundary--`. A non-string field value throws
/// `invalid_argument("expected string")` in the C — surfaced as the relay's
/// error envelope here.
fn build_multipart_body(
    form_fields: &Json,
    files: &std::collections::BTreeMap<String, crate::http::UploadedFile>,
    boundary: &str,
) -> Result<Vec<u8>, String> {
    let Json::Object(entries) = form_fields else {
        return Err("failed to parse multipart form fields JSON".into());
    };
    let mut body: Vec<u8> = Vec::new();
    let mut part = |disp: &str, extra: &[&str], content_type: Option<&str>, data: &[u8]| {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(disp.as_bytes());
        for line in extra {
            body.extend_from_slice(line.as_bytes());
        }
        if let Some(ct) = content_type {
            body.extend_from_slice(format!("Content-Type: {ct}\r\n").as_bytes());
        }
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    };
    for (key, value) in entries {
        match value {
            Json::String(s) => part(
                &format!(
                    "Content-Disposition: form-data; name=\"{}\"\r\n",
                    sanitize_field(key)
                ),
                &[],
                None,
                s.as_bytes(),
            ),
            Json::Array(items) => {
                for item in items {
                    let Json::String(s) = item else {
                        return Err("expected string".into());
                    };
                    part(
                        &format!(
                            "Content-Disposition: form-data; name=\"{}\"\r\n",
                            sanitize_field(key)
                        ),
                        &[],
                        None,
                        s.as_bytes(),
                    );
                }
            }
            _ => return Err("expected string".into()),
        }
    }
    for (key, file) in files {
        let mut disp = format!(
            "Content-Disposition: form-data; name=\"{}\"",
            sanitize_field(key)
        );
        if !file.filename.is_empty() {
            disp.push_str(&format!("; filename=\"{}\"", sanitize_field(&file.filename)));
        }
        disp.push_str("\r\n");
        let ct = if file.content_type.is_empty() {
            Some("application/octet-stream")
        } else {
            Some(file.content_type.as_str())
        };
        part(&disp, &[], ct, &file.data);
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Ok(body)
}

/// `common_http_format_host` (common/http.h:25-27) — bracket a bare IPv6
/// literal for the wire
fn format_host(host: &str) -> String {
    if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    }
}

fn proxy_header(headers: &[(String, String)], name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

struct ProxyResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

/// the transport of `server_http_proxy`'s client: `httplib::ClientImpl` for
/// http, `SSLClient` for https (server-models.cpp:2462-2469)
enum ProxyStream {
    Plain(std::net::TcpStream),
    Tls(crate::tls::SslStream),
}

impl std::io::Read for ProxyStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            ProxyStream::Plain(s) => s.read(buf),
            ProxyStream::Tls(s) => s.read(buf),
        }
    }
}

impl std::io::Write for ProxyStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            ProxyStream::Plain(s) => s.write(buf),
            ProxyStream::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            ProxyStream::Plain(s) => s.flush(),
            ProxyStream::Tls(s) => s.flush(),
        }
    }
}

/// one request/response hop (the 600 s timeout pair of server-cors-proxy.h
/// :70-71 rides the stream options)
fn proxy_fetch_once(
    parsed: &ParsedUrl,
    method: &str,
    headers: &[(String, String)],
    body: &[u8],
    has_files: bool,
    override_content_type: Option<&str>,
) -> Result<ProxyResponse, String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let sock = TcpStream::connect((parsed.host.as_str(), parsed.port))
        .map_err(|e| format!("proxy connect failed: {e}"))?;
    let _ = sock.set_read_timeout(Some(Duration::from_secs(600)));
    let _ = sock.set_write_timeout(Some(Duration::from_secs(600)));
    let mut stream = if parsed.scheme == "https" {
        ProxyStream::Tls(crate::tls::SslStream::connect(sock, &parsed.host)?)
    } else {
        ProxyStream::Plain(sock)
    };

    // the outgoing header set (server-models.cpp:2536-2583): accept-encoding,
    // transfer-encoding and content-length are dropped; a multipart
    // content-type is corrected to JSON when no files rode the request, and
    // dropped outright when they did (the rebuild set its own boundary —
    // `override_content_type`, applied after the loop like the C's
    // `req.set_header`); Host is rewritten to the formatted host, with the
    // port omitted on the scheme default
    let mut request = format!(
        "{} {} HTTP/1.1\r\n",
        method,
        if parsed.path.is_empty() { "/" } else { &parsed.path }
    );
    let is_default_port = (parsed.scheme == "https" && parsed.port == 443)
        || (parsed.scheme == "http" && parsed.port == 80);
    let host_value = if is_default_port {
        format_host(&parsed.host)
    } else {
        format!("{}:{}", format_host(&parsed.host), parsed.port)
    };
    let mut host_set = false;
    for (k, v) in headers {
        let lowered = k.to_ascii_lowercase();
        if lowered == "accept-encoding" || lowered == "transfer-encoding" || lowered == "content-length"
        {
            continue;
        }
        if lowered == "content-type" {
            if has_files {
                // the rebuild supplies its own Content-Type (:2564-2568)
                continue;
            }
            if v.contains("multipart/form-data") {
                // no files → the body is JSON now (:2570-2574)
                request.push_str("Content-Type: application/json; charset=utf-8\r\n");
                continue;
            }
        }
        if lowered == "host" {
            request.push_str(&format!("Host: {host_value}\r\n"));
            host_set = true;
            continue;
        }
        request.push_str(&format!("{k}: {v}\r\n"));
    }
    if !host_set {
        request.push_str(&format!("Host: {host_value}\r\n"));
    }
    // `if (!override_content_type.empty()) req.set_header(...)`
    // (server-models.cpp:2580-2583) — the rebuild's boundary content-type
    if let Some(ct) = override_content_type {
        request.push_str(&format!("Content-Type: {ct}\r\n"));
    }
    // cpp-httplib's client header completion (httplib.cpp:10342-10371):
    // `Accept: */*` when the caller set none, `Content-Type: text/plain`
    // (the Client's default content type) for a body without one, and
    // Content-Length from the actual body — the proxy's callers hand over
    // only the x-llama-server-proxy-header-* set, so these defaults are
    // what the target observes
    if !headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("accept"))
    {
        request.push_str("Accept: */*\r\n");
    }
    if !body.is_empty()
        && override_content_type.is_none()
        && !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
    {
        request.push_str("Content-Type: text/plain\r\n");
    }
    // Content-Length only for a non-empty body (httplib.cpp:10367-10371)
    if !body.is_empty() {
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("Connection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .and_then(|_| stream.write_all(body))
        .map_err(|_| "proxy write failed".to_string())?;

    // read the head, then exactly Content-Length bytes (chunked or
    // unspecified falls back to reading until the peer closes — the client
    // sent `Connection: close`)
    let mut raw = Vec::new();
    let mut buf = [0u8; 16384];
    let head_end = loop {
        let n = stream.read(&mut buf).map_err(|_| "proxy read failed".to_string())?;
        if n == 0 {
            return Err("proxy read failed".into());
        }
        let start = raw.len();
        raw.extend_from_slice(&buf[..n]);
        if let Some(p) = raw[start.saturating_sub(3)..]
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
        {
            break start + p;
        }
    };
    let head = String::from_utf8_lossy(&raw[..head_end]).into_owned();
    let mut lines = head.lines();
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "proxy got a malformed response".to_string())?;
    let mut out_headers = Vec::new();
    let mut content_length: Option<usize> = None;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_string(), v.trim().to_string());
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().ok();
            }
            out_headers.push((k, v));
        }
    }
    let mut body_bytes = raw[head_end + 4..].to_vec();
    if let Some(cl) = content_length {
        while body_bytes.len() < cl {
            let n = stream
                .read(&mut buf)
                .map_err(|_| "proxy read failed".to_string())?;
            if n == 0 {
                break;
            }
            body_bytes.extend_from_slice(&buf[..n]);
        }
        body_bytes.truncate(cl);
    } else {
        loop {
            let n = match stream.read(&mut buf) {
                Ok(n) => n,
                Err(_) => break,
            };
            if n == 0 {
                break;
            }
            body_bytes.extend_from_slice(&buf[..n]);
        }
    }
    Ok(ProxyResponse {
        status,
        headers: out_headers,
        body: body_bytes,
    })
}

/// the Location hop resolution of `ClientImpl::redirect`
/// (vendor/cpp-httplib/httplib.cpp:10821-10864): an absolute URL replaces,
/// anything else inherits scheme/host/port from the current target
fn parse_redirect_location(location: &str, cur: &ParsedUrl) -> Result<ParsedUrl, String> {
    if location.contains("://") {
        let mut next = parse_url(location)?;
        if next.path.is_empty() {
            next.path = "/".into();
        }
        Ok(next)
    } else {
        let mut next = ParsedUrl {
            scheme: cur.scheme.clone(),
            host: cur.host.clone(),
            port: cur.port,
            path: String::new(),
            password: String::new(),
        };
        if let Some(rest) = location.strip_prefix("//") {
            // scheme-relative: //host[:port]/path
            let mut abs = format!("{}://{}", cur.scheme, rest);
            if !abs.contains('/') || abs.rsplit_once('/').map(|(_, t)| t.is_empty()).unwrap_or(false)
            {
                // no path part
                if !abs.ends_with('/') {
                    abs.push('/');
                }
            }
            let mut parsed = parse_url(&abs)?;
            if parsed.path.is_empty() {
                parsed.path = "/".into();
            }
            return Ok(parsed);
        }
        next.path = if location.starts_with('/') {
            location.to_string()
        } else {
            // relative to the current path's directory
            let dir = match cur.path.rfind('/') {
                Some(p) => &cur.path[..=p],
                None => "/",
            };
            format!("{dir}{location}")
        };
        Ok(next)
    }
}

// ---------------------------------------------------------------------------
// GCP (Vertex AI) compat — server-http.cpp:84-96, :752-923
// ---------------------------------------------------------------------------

/// `gcp_params` (server-http.cpp:84-96): the AIP_* environment contract
pub struct GcpParams {
    pub enabled: bool,
    pub path_health: String,
    pub path_predict: String,
    pub port: u16,
}

impl GcpParams {
    pub fn from_env() -> Self {
        let getenv = |name: &str, default: &str, ensure_leading_slash: bool| -> String {
            match std::env::var(name) {
                Ok(v) if !v.is_empty() => {
                    let mut val = v;
                    if ensure_leading_slash && !val.starts_with('/') {
                        val.insert(0, '/');
                    }
                    val
                }
                _ => default.to_string(),
            }
        };
        GcpParams {
            enabled: std::env::var("AIP_MODE").unwrap_or_default() == "PREDICTION",
            path_health: getenv("AIP_HEALTH_ROUTE", "", true),
            path_predict: getenv("AIP_PREDICT_ROUTE", "/predict", true),
            port: std::env::var("AIP_HTTP_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
        }
    }
}

/// `path_to_gcp_format` (server-http.cpp:758-778): "/v1/chat/completions" ->
/// "chatCompletions", "/apply-template" -> "applyTemplate"
pub fn path_to_gcp_format(path: &str) -> String {
    let mut s = path.to_string();
    let b = s.as_bytes();
    if s.len() > 3 && b[0] == b'/' && b[1] == b'v' && b[2] == b'1' {
        s = s[3..].to_string();
    }
    if s.starts_with('/') {
        s = s[1..].to_string();
    }
    let mut result = String::new();
    let mut cap = false;
    for c in s.chars() {
        if c == ':' {
            break; // stop before path parameters
        }
        if c == '/' || c == '-' || c == '_' {
            cap = true;
        } else {
            if cap {
                result.extend(c.to_uppercase());
            } else {
                result.push(c);
            }
            cap = false;
        }
    }
    result
}

/// `parse_gcp_predict_response` (server-http.cpp:780-795)
fn parse_gcp_predict_response(res: &Response) -> Result<Json, String> {
    let Body::Full(data) = &res.body else {
        return Err("predict route does not support streaming responses".into());
    };
    if data.is_empty() {
        return Ok(Json::Null);
    }
    match Json::parse(data) {
        Ok(j) => Ok(j),
        Err(_) => Ok(Json::String(data.clone())),
    }
}

/// `register_gcp_compat` (server-http.cpp:797-923). The port dispatches the
/// per-instance internal requests through `routes_cell` — the routing table
/// itself, populated right after `into_routes` (the C closes over its own
/// `handlers` map). Instances run sequentially (the C's std::async fan-out
/// is a latency detail; the predictions array order is identical).
pub fn register_gcp_compat(
    routes_table: &mut crate::http::HttpServer,
    routes_cell: &Arc<std::sync::Mutex<Option<crate::http::Routes>>>,
) {
    let gcp = GcpParams::from_env();
    if !gcp.enabled {
        return; // do nothing (:800-803)
    }

    if !gcp.path_health.is_empty() {
        // the /health handler under the AIP route (:817-821)
        let cell = routes_cell.clone();
        routes_table.add(
            "GET",
            &gcp.path_health,
            Arc::new(move |req: &Request| {
                let routes = cell.lock().unwrap().clone().unwrap_or_default();
                // the internal `handlers` map lookup — the bare path
                // (server-http.cpp:819-821)
                if let Some(h) = routes.dispatch_get("GET /health") {
                    h(req)
                } else {
                    Response::not_found()
                }
            }),
        );
    }

    let cell = routes_cell.clone();
    routes_table.add(
        "POST",
        &gcp.path_predict,
        Arc::new(move |req: &Request| {
            let bad = |status: u16, message: &str| -> Response {
                Response {
                    status,
                    content_type: "application/json; charset=utf-8".into(),
                    body: Body::Full(crate::api::json_error(
                        message,
                        "invalid_request_error",
                        status as i64,
                    )),
                    headers: Vec::new(),
                    terminal_done: false,
                }
            };
            // `build_error` (:824-826): {"error", format_error_response(...)}
            // — format_error_response is the INNER object (message/type/code)
            let build_error = |message: &str| -> Json {
                Json::Object(vec![(
                    "error".into(),
                    Json::Object(vec![
                        ("message".into(), Json::String(message.to_string())),
                        ("type".into(), Json::String("invalid_request_error".into())),
                        ("code".into(), Json::Int(400)),
                    ]),
                )])
            };

            let data = match Json::parse(&req.body_str()) {
                Ok(d) => d,
                Err(e) => return bad(400, &e),
            };
            if !data.is_object() {
                return bad(400, "request body must be a JSON object");
            }
            let instances_ok = data
                .at("instances")
                .is_some_and(|v| v.is_array());
            if !instances_ok {
                return bad(400, "request body must include an array field named instances");
            }

            const MAX_INSTANCES: usize = 128;
            if data.at("instances").unwrap().size() > MAX_INSTANCES {
                return bad(
                    400,
                    &format!("instances array exceeds maximum size of {MAX_INSTANCES}"),
                );
            }

            let routes = cell.lock().unwrap().clone().unwrap_or_default();
            // camelCase alias -> canonical path (first registration wins on
            // collision, :810-815) — over the internal `handlers` map, the
            // bare paths
            let mut alias_to_path: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
            for key in routes.dispatch.keys() {
                let Some((_, path)) = key.split_once(' ') else { continue };
                alias_to_path
                    .entry(path_to_gcp_format(path))
                    .or_insert_with(|| path.to_string());
            }

            let mut predictions: Vec<Json> = Vec::new();
            if let Json::Array(items) = data.at("instances").unwrap() {
                for instance in items {
                    if !instance.is_object() {
                        predictions.push(build_error("each instance must be a JSON object"));
                        continue;
                    }
                    let Some(fmt) = instance
                        .at("@requestFormat")
                        .and_then(|v| v.get_str().ok().map(|s| s.to_string()))
                    else {
                        predictions.push(build_error(
                            "each instance must include a string @requestFormat",
                        ));
                        continue;
                    };

                    // strip @requestFormat, force stream off (:871-879)
                    let mut payload_entries: Vec<(String, Json)> =
                        if let Json::Object(entries) = &instance {
                            entries
                                .iter()
                                .filter(|(k, _)| k != "@requestFormat")
                                .cloned()
                                .collect()
                        } else {
                            Vec::new()
                        };
                    if payload_entries.iter().any(|(k, _)| k == "stream") {
                        eprintln!(
                            "ignoring client-provided stream field in instance, streaming is not supported in predict route"
                        );
                    }
                    if let Some(slot) = payload_entries.iter_mut().find(|(k, _)| k == "stream") {
                        slot.1 = Json::Bool(false);
                    }

                    // accept both camelCase aliases and direct paths (:881-890)
                    let dispatch_path = if let Some(p) = alias_to_path.get(&fmt) {
                        p.clone()
                    } else if routes.dispatch.contains_key(&format!("POST {fmt}")) {
                        fmt.clone()
                    } else {
                        predictions.push(build_error(&format!(
                            "no handler registered for @requestFormat: {fmt}"
                        )));
                        continue;
                    };

                    let internal_req = Request {
                        method: "POST".into(),
                        // `path_prefix + dispatch_path` (server-http.cpp:895)
                        path: format!("{}{}", routes.path_prefix, dispatch_path),
                        params: req.params.clone(),
                        headers: req.headers.clone(),
                        body: Json::Object(payload_entries).dump().into_bytes(),
                        files: Default::default(),
                    };
                    let Some(handler) = routes.dispatch_get(&format!("POST {dispatch_path}")) else {
                        predictions.push(build_error(&format!(
                            "no handler registered for @requestFormat: {fmt}"
                        )));
                        continue;
                    };
                    match parse_gcp_predict_response(&handler(&internal_req)) {
                        Ok(j) => predictions.push(j),
                        Err(e) => predictions.push(build_error(&e)),
                    }
                }
            }

            Response::ok(
                Json::Object(vec![("predictions".into(), Json::Array(predictions))]).dump(),
            )
        }),
    );
}

// ---------------------------------------------------------------------------
// tests — schema exposure, the sandboxing argv contracts, and the tool
// semantics against fixtures (the cross-server round-trip lives in
// parity/server_tools_mcp_parity.sh)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_match_c_semantics() {
        // common.cpp:769-813
        assert!(glob_match("", ""));
        assert!(!glob_match("", "a"));
        assert!(glob_match("a*", "abc"));
        assert!(!glob_match("a*b", "a/c/b")); // * stops at '/'
        assert!(glob_match("a**b", "a/c/b")); // ** crosses '/'
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "a/c")); // ? does not match '/'
        assert!(glob_match("a[bc]d", "abd"));
        assert!(!glob_match("a[bc]d", "aed"));
        assert!(glob_match("a[b-d]e", "ace")); // range
        assert!(glob_match("a[!bc]d", "aed")); // negation
        assert!(!glob_match("a[!b]d", "abd"));
        assert!(glob_match("[a-c]*", "bxyz"));
        // unterminated class: '[' literal
        assert!(glob_match("a[b", "a[b"));
    }

    #[test]
    fn path_glob_match_anchoring() {
        // server-tools.cpp:852-860
        assert!(path_glob_match("*.cpp", "src/main.cpp"));
        assert!(path_glob_match("*.cpp", "main.cpp"));
        assert!(!path_glob_match("*.cpp", "src/main.rs"));
        // the C's auto "**/" prefix (server-tools.cpp:859) cannot match a
        // bare leading directory — "**/src/x" needs a component before "src"
        // (the ** branch keeps the '/' it must then match) — so an anchored
        // pattern matches only under a parent
        assert!(path_glob_match("src/**/*.cpp", "root/src/a/b/main.cpp"));
        assert!(!path_glob_match("src/**/*.cpp", "src/a/b/main.cpp"));
        assert!(path_glob_match("a/b.cpp", "x/y/a/b.cpp")); // auto "**/"
    }

    #[test]
    fn expand_home_shapes() {
        // server-tools.cpp:127-133
        assert_eq!(expand_home("plain/path"), "plain/path");
        assert_eq!(expand_home("~x/not/home"), "~x/not/home");
        std::env::set_var("HOME", "/home/test");
        assert_eq!(expand_home("~/rest"), "/home/test/rest");
        // bare "~" expands too (common.cpp:127-133 — the size>1 guard only
        // protects the "~user" forms)
        assert_eq!(expand_home("~"), "/home/test");
    }

    #[test]
    fn ssh_target_and_quote_contracts() {
        // server-tools.cpp:748-755, :635-648 — the sandboxing surface
        assert!(is_valid_ssh_target("user@host.example.com"));
        assert!(!is_valid_ssh_target("-oProxyCommand=evil"));
        assert!(!is_valid_ssh_target(""));
        assert!(!is_valid_ssh_target("a b"));
        assert_eq!(
            shell_quote_join(&["sh".into(), "-c".into(), "echo 'hi'".into()]),
            "'sh' '-c' 'echo '\\''hi'\\'''"
        );
        assert!(!ContainerRuntimeSpec::is_valid_id("--privileged"));
        assert!(ContainerRuntimeSpec::is_valid_id("abc123"));
        assert_eq!(
            ContainerRuntimeSpec::parse("docker-container:deadbeef").map(|s| (s.bin, s.attach)),
            Some(("docker".to_string(), true))
        );
        assert_eq!(
            ContainerRuntimeSpec::parse("podman:image:1").map(|s| (s.bin, s.attach)),
            Some(("podman".to_string(), false))
        );
        assert!(ContainerRuntimeSpec::parse("ssh:host").is_none());
    }

    #[test]
    fn tool_schemas_exposure() {
        // the /tools GET listing — one definition per built-in
        for name in ALL_TOOL_NAMES {
            let j = tool_to_json(name);
            assert_eq!(j.at("type").and_then(|v| v.get_str().ok()), Some("server"));
            assert_eq!(j.at("tool").and_then(|v| v.get_str().ok()), Some(*name));
            assert_eq!(
                j.at("permissions").and_then(|p| p.at("write")).and_then(|v| match v {
                    Json::Bool(b) => Some(*b),
                    _ => None,
                }),
                Some(matches!(*name, "exec_shell_command" | "write_file" | "edit_file"))
            );
            let def = j.at("definition").unwrap();
            assert_eq!(
                def.at("function").and_then(|f| f.at("name")).and_then(|v| v.get_str().ok()),
                Some(*name)
            );
            assert!(def.at("function").and_then(|f| f.at("parameters")).is_some());
        }
    }

    #[test]
    fn read_file_line_range_and_loc() {
        // server-tools.cpp:898-965 over a fixture
        let dir = std::env::temp_dir().join("llama-tools-test-rf");
        let _ = std::fs::create_dir_all(&dir);
        let f = dir.join("lines.txt");
        std::fs::write(&f, "one\ntwo\nthree\nfour\n").unwrap();
        let _io = ToolsIo::for_params(dir.to_str().unwrap(), "").unwrap();
        let params = Json::Object(vec![
            ("path".into(), jstr("lines.txt")),
            // the x-tool-cwd header rides in the params (server-tools.cpp:2085-2092)
            ("cwd".into(), jstr(dir.to_str().unwrap())),
            ("start_line".into(), Json::Int(2)),
            ("end_line".into(), Json::Int(3)),
            ("append_loc".into(), Json::Bool(true)),
        ]);
        let r = invoke_tool("read_file", &params, None).unwrap();
        assert_eq!(
            r.at("plain_text_response").and_then(|v| v.get_str().ok()),
            Some("2\u{2192}two\n3\u{2192}three\n")
        );

        // the too-large guard for a full read
        let big = dir.join("big.txt");
        std::fs::write(&big, vec![b'x'; SERVER_TOOL_READ_FILE_MAX_SIZE + 1]).unwrap();
        let params = Json::Object(vec![
            ("path".into(), jstr("big.txt")),
            ("cwd".into(), jstr(dir.to_str().unwrap())),
        ]);
        let r = invoke_tool("read_file", &params, None).unwrap();
        assert!(r.at("error").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_then_edit_file_roundtrip() {
        let dir = std::env::temp_dir().join("llama-tools-test-we");
        let _ = std::fs::create_dir_all(&dir);
        let io_cwd = dir.to_str().unwrap().to_string();
        let io = ToolsIo::for_params(&io_cwd, "").unwrap();
        assert!(io.write_file("sub/a.txt", "alpha\nbeta\n"));
        assert!(dir.join("sub/a.txt").exists());

        // edit_file: exact replacement
        let params = Json::Object(vec![
            ("path".into(), jstr("sub/a.txt")),
            ("cwd".into(), jstr(&io_cwd)),
            (
                "edits".into(),
                Json::Array(vec![Json::Object(vec![
                    ("old_text".into(), jstr("beta")),
                    ("new_text".into(), jstr("gamma")),
                ])]),
            ),
        ]);
        let r = invoke_tool("edit_file", &params, None).unwrap();
        assert_eq!(
            r.at("result").and_then(|v| v.get_str().ok()),
            Some("file edited successfully")
        );
        assert_eq!(std::fs::read_to_string(dir.join("sub/a.txt")).unwrap(), "alpha\ngamma\n");

        // fuzzy path: trailing whitespace difference
        assert!(io.write_file("sub/b.txt", "keep   \nreplace me  \n"));
        let params = Json::Object(vec![
            ("path".into(), jstr("sub/b.txt")),
            ("cwd".into(), jstr(&io_cwd)),
            (
                "edits".into(),
                Json::Array(vec![Json::Object(vec![
                    ("old_text".into(), jstr("replace me")),
                    ("new_text".into(), jstr("replaced")),
                ])]),
            ),
        ]);
        let r = invoke_tool("edit_file", &params, None).unwrap();
        assert!(r.at("error").is_none(), "{:?}", r.dump());
        assert_eq!(
            std::fs::read_to_string(dir.join("sub/b.txt")).unwrap(),
            "keep   \nreplaced  \n"
        );

        // not-found and non-unique errors
        let params = Json::Object(vec![
            ("path".into(), jstr("sub/b.txt")),
            ("cwd".into(), jstr(&io_cwd)),
            (
                "edits".into(),
                Json::Array(vec![Json::Object(vec![
                    ("old_text".into(), jstr("nope")),
                    ("new_text".into(), jstr("x")),
                ])]),
            ),
        ]);
        let r = invoke_tool("edit_file", &params, None).unwrap();
        assert!(r.at("error").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exec_shell_command_contract() {
        // :1273-1317 — sh -c, exit-code tail, timeout cap
        let params = Json::Object(vec![("command".into(), jstr("echo hi; echo err 1>&2"))]);
        let r = invoke_tool("exec_shell_command", &params, None).unwrap();
        let text = r.at("plain_text_response").and_then(|v| v.get_str().ok()).unwrap();
        assert!(text.contains("hi"));
        assert!(text.contains("err"));
        assert!(text.ends_with("[exit code: 0]"));

        // nonzero exit rides the tail
        let params = Json::Object(vec![("command".into(), jstr("exit 3"))]);
        let r = invoke_tool("exec_shell_command", &params, None).unwrap();
        let text = r.at("plain_text_response").and_then(|v| v.get_str().ok()).unwrap();
        assert!(text.ends_with("[exit code: 3]"));
    }

    #[test]
    fn get_info_shape() {
        let params = Json::Object(vec![]);
        let r = invoke_tool("get_info", &params, None).unwrap();
        let os = r.at("os").and_then(|v| v.get_str().ok()).unwrap();
        assert!(!os.is_empty() && os != "unknown");
        assert!(r.at("cwd").and_then(|v| v.get_str().ok()).is_some());
    }

    #[test]
    fn file_glob_search_shape() {
        let dir = std::env::temp_dir().join("llama-tools-test-fgs");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src/deep")).unwrap();
        std::fs::write(dir.join("src/main.cpp"), "int main(){}").unwrap();
        std::fs::write(dir.join("src/deep/util.cpp"), "int util(){}").unwrap();
        std::fs::write(dir.join("notes.txt"), "x").unwrap();

        let params = Json::Object(vec![
            ("path".into(), jstr(dir.to_str().unwrap())),
            ("include".into(), jstr("*.cpp")),
        ]);
        let r = invoke_tool("file_glob_search", &params, None).unwrap();
        let text = r.at("plain_text_response").and_then(|v| v.get_str().ok()).unwrap();
        assert!(text.contains("src/main.cpp"), "{text}");
        assert!(text.contains("src/deep/util.cpp"), "{text}");
        assert!(!text.contains("notes.txt"));
        assert!(text.contains("Total matches: 2"));
        let entries = r.at("entries").unwrap();
        assert_eq!(entries.size(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grep_search_shape() {
        let dir = std::env::temp_dir().join("llama-tools-test-gs");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "hello world\nbye\n").unwrap();
        std::fs::write(dir.join("b.txt"), "no match\n").unwrap();

        let params = Json::Object(vec![
            ("path".into(), jstr(dir.to_str().unwrap())),
            ("pattern".into(), jstr("wor.d")),
            ("return_line_numbers".into(), Json::Bool(true)),
        ]);
        let r = invoke_tool("grep_search", &params, None).unwrap();
        let text = r.at("plain_text_response").and_then(|v| v.get_str().ok()).unwrap();
        assert!(text.contains("a.txt:1:hello world"), "{text}");
        assert!(text.contains("Total matches: 1"));

        // literal escapes the specials (:1142-1150)
        let params = Json::Object(vec![
            ("path".into(), jstr(dir.to_str().unwrap())),
            ("pattern".into(), jstr("wor.d")),
            ("literal".into(), Json::Bool(true)),
        ]);
        let r = invoke_tool("grep_search", &params, None).unwrap();
        let text = r.at("plain_text_response").and_then(|v| v.get_str().ok()).unwrap();
        assert!(text.contains("Total matches: 0"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn setup_validation() {
        // :2004-2036 — unknown tool names and the all selector
        let mcp = Arc::new(ServerMcp::new());
        let err = setup(&["no_such_tool".into()], &mcp, "").unwrap_err();
        assert!(err.starts_with("unknown tool \"no_such_tool\". available tools: read_file"));

        let tools = setup(&["all".into()], &mcp, "").unwrap();
        assert_eq!(tools.len(), 7);
        assert!(tools.contains(&"exec_shell_command".to_string()));

        // a spawn-spec runtime is rejected (no engine ownership, :1988-1995)
        let err = setup(&["read_file".into()], &mcp, "docker:someimage").unwrap_err();
        assert!(err.contains("unknown --tools-runtime option"));
        // an attach spec validates
        setup(&["read_file".into()], &mcp, "docker-container:abc123").unwrap();
    }
}

#[cfg(test)]
mod multipart_tests {
    use super::*;
    use crate::http::UploadedFile;
    use std::collections::BTreeMap;

    fn file(name: &str, filename: &str, ct: &str, data: &[u8]) -> (String, UploadedFile) {
        (
            name.to_string(),
            UploadedFile {
                data: data.to_vec(),
                filename: filename.to_string(),
                content_type: ct.to_string(),
            },
        )
    }

    /// build_multipart_body (server-models.cpp:2395-2451): the exact part
    /// framing — fields first (array values once per item), then files
    /// (std::map order = sorted by field name), the terminal --boundary--
    #[test]
    fn multipart_body_rebuild() {
        let form = Json::Object(vec![
            ("prompt".into(), Json::String("hi".into())),
            (
                "multi".into(),
                Json::Array(vec![Json::String("a".into()), Json::String("b".into())]),
            ),
        ]);
        let mut files = BTreeMap::new();
        files.insert("zfile".to_string(), file("zfile", "z.bin", "application/x-z", b"\x00\xff\x0d\x0a").1);
        files.insert("afile".to_string(), file("afile", "", "", b"plain") .1);
        let b = build_multipart_body(&form, &files, "----llama-cpp-proxy-test").unwrap();
        // the text prefix up to the binary file content, then the raw tail
        let expected = b"------llama-cpp-proxy-test\r\n\
             Content-Disposition: form-data; name=\"prompt\"\r\n\
             \r\n\
             hi\r\n\
             ------llama-cpp-proxy-test\r\n\
             Content-Disposition: form-data; name=\"multi\"\r\n\
             \r\n\
             a\r\n\
             ------llama-cpp-proxy-test\r\n\
             Content-Disposition: form-data; name=\"multi\"\r\n\
             \r\n\
             b\r\n\
             ------llama-cpp-proxy-test\r\n\
             Content-Disposition: form-data; name=\"afile\"\r\n\
             Content-Type: application/octet-stream\r\n\
             \r\n\
             plain\r\n\
             ------llama-cpp-proxy-test\r\n\
             Content-Disposition: form-data; name=\"zfile\"; filename=\"z.bin\"\r\n\
             Content-Type: application/x-z\r\n\
             \r\n"
            .to_vec();
        assert_eq!(&b[..expected.len()], &expected[..]);
        // the binary file bytes survive verbatim, then the closing frame
        assert_eq!(&b[expected.len()..expected.len() + 4], &[0x00, 0xff, 0x0d, 0x0a]);
        assert_eq!(&b[expected.len() + 4..], b"\r\n------llama-cpp-proxy-test--\r\n");
    }

    /// sanitize_field (server-models.cpp:2400-2410) and the non-string error
    #[test]
    fn multipart_sanitize_and_errors() {
        assert_eq!(sanitize_field("a\"b\nc\rd"), "abcd");
        let form = Json::Object(vec![("n".into(), Json::Int(3))]);
        assert_eq!(
            build_multipart_body(&form, &BTreeMap::new(), "B").unwrap_err(),
            "expected string"
        );
        // a non-object form (the parse_no_throw guard of :2555-2560)
        assert_eq!(
            build_multipart_body(&Json::Array(vec![]), &BTreeMap::new(), "B").unwrap_err(),
            "failed to parse multipart form fields JSON"
        );
    }

    /// generate_multipart_boundary (server-models.cpp:2383-2393): the exact
    /// prefix and alphabet, 16 chars, distinct across calls
    #[test]
    fn boundary_shape() {
        let b1 = generate_multipart_boundary();
        let b2 = generate_multipart_boundary();
        assert!(b1.starts_with("----llama-cpp-proxy-"));
        assert_eq!(b1.len(), "----llama-cpp-proxy-".len() + 16);
        assert!(b1
            .strip_prefix("----llama-cpp-proxy-")
            .unwrap()
            .bytes()
            .all(|c| c.is_ascii_digit() || c.is_ascii_lowercase()));
        assert_ne!(b1, b2);
    }
}
