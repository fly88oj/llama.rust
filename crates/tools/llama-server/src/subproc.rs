//! subproc.rs — port of `common/subproc.cpp` (pinned bd4f514db1, 143 lines):
//! the cross-platform subprocess wrapper the server's tool/MCP machinery
//! builds on (server-tools.cpp, server-mcp.cpp, server-models.cpp).
//!
//! The C wraps sheredom/subprocess.h behind `LLAMA_SUBPROCESS`; the port
//! wraps `std::process` with the same semantics:
//!
//!   common_subproc::is_supported (:3-9)      -> [`Subproc::is_supported`]
//!   create(args, options, env, cwd) (:30-47) -> [`Subproc::create`] — argv
//!       passed as-is (NO shell interpolation anywhere in this module), env
//!       replaces the parent's when non-empty, cwd optional
//!   has_handle (:49-57)                      -> [`Subproc::has_handle`]
//!   alive (:60-62)                           -> [`Subproc::alive`]
//!   stdin_file/stdout_file/stderr_file       -> [`Subproc::take_stdin`] etc.
//!       (:64-74)                                (Rust hands the pipe once)
//!   close_stdin (:76-81)                     -> [`Subproc::close_stdin`]
//!   terminate (:83-87)                       -> [`Subproc::terminate`] (SIGTERM)
//!   join (:89-97)                            -> [`Subproc::join`]
//!
//! Security notes (mirroring the reference): every spawn is argv-based —
//! `exec_shell_command` is the only place a shell runs, and only because the
//! caller asked for it with `sh -c <command>` (server-tools.cpp:1287); the
//! isolate runtimes pass a *quoted single word* to ssh (server-tools.cpp:758-766).

use std::io::Write as _;
use std::process::{Child, Command, Stdio};

// pipe(2)/dup(2)/close(2) for the combined stdout+stderr stream — the C's
// `subprocess_option_combined_stdout_stderr` duplicates the write end of
// stdout's pipe onto the child's stderr fd before exec. No libc crate
// (mlock.rs precedent).
extern "C" {
    fn pipe(fds: *mut i32) -> i32;
    fn dup(fd: i32) -> i32;
    fn close(fd: i32) -> i32;
}

/// the read end of a pipe whose two write ends became the child's stdout and
/// stderr (see [`Subproc::create_combined`])
pub struct CombinedOutput(std::fs::File);

impl std::io::Read for CombinedOutput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

pub struct Subproc {
    child: Option<Child>,
    /// the C keeps FILE* handles in `proc`; the port hands each pipe out once
    stdin: Option<std::process::ChildStdin>,
}

impl Default for Subproc {
    fn default() -> Self {
        Self::new()
    }
}

impl Subproc {
    pub fn new() -> Self {
        Subproc { child: None, stdin: None }
    }

    /// `common_subproc::is_supported` — the reference's LLAMA_SUBPROCESS
    /// build flag; the port's std implementation is always on
    pub fn is_supported() -> bool {
        true
    }

    /// `create` (:30-47). `env` empty → inherit the parent environment (the
    /// C's `subprocess_option_inherit_environment`).
    pub fn create(
        &mut self,
        args: &[String],
        env: &[(String, String)],
        cwd: Option<&str>,
    ) -> bool {
        self.join(); // create() overwrites the handle — release the previous one (:1904)

        let mut cmd = Command::new(&args[0]);
        cmd.args(&args[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        if !env.is_empty() {
            cmd.env_clear();
            cmd.envs(env.iter().cloned());
        }

        match cmd.spawn() {
            Ok(mut child) => {
                self.stdin = child.stdin.take();
                self.child = Some(child);
                true
            }
            Err(_) => false,
        }
    }

    /// spawn with stderr merged onto stdout (`combined_stdout_stderr`,
    /// server-tools.cpp:203) — one pipe carries both streams, like the C
    pub fn create_combined(
        &mut self,
        args: &[String],
        env: &[(String, String)],
        cwd: Option<&str>,
    ) -> Option<CombinedOutput> {
        use std::os::unix::io::FromRawFd;

        self.join();

        let mut fds = [0i32; 2];
        if unsafe { pipe(fds.as_mut_ptr()) } != 0 {
            return None;
        }
        let w2 = unsafe { dup(fds[1]) };
        if w2 < 0 {
            unsafe {
                close(fds[0]);
                close(fds[1]);
            }
            return None;
        }

        let mut cmd = Command::new(&args[0]);
        cmd.args(&args[1..])
            .stdin(Stdio::piped())
            .stdout(unsafe { Stdio::from_raw_fd(fds[1]) })
            .stderr(unsafe { Stdio::from_raw_fd(w2) });
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        if !env.is_empty() {
            cmd.env_clear();
            cmd.envs(env.iter().cloned());
        }

        match cmd.spawn() {
            Ok(mut child) => {
                // the write-end Stdio objects were consumed by spawn(); their
                // Drop closes the parent copies — no manual close (an
                // owned-fd double close aborts the runtime)
                self.stdin = child.stdin.take();
                self.child = Some(child);
                Some(CombinedOutput(unsafe { std::fs::File::from_raw_fd(fds[0]) }))
            }
            Err(_) => {
                // cmd still owns the Stdio wrappers; dropping it closes them.
                // fds[0] is wrapped so its Drop closes the read end too
                drop(cmd);
                drop(unsafe { std::fs::File::from_raw_fd(fds[0]) });
                None
            }
        }
    }

    /// `has_handle` (:49-57)
    pub fn has_handle(&self) -> bool {
        self.child.is_some()
    }

    /// `alive` (:60-62) — the transport layer tracks liveness on its own
    /// flag (the C's `subprocess_alive` poll), kept for API parity
    #[allow(dead_code)]
    pub fn alive(&mut self) -> bool {
        match self.child.as_mut() {
            Some(c) => matches!(c.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// `stdin_file` (:64-66) — the pipe is handed out once (Rust ownership)
    pub fn take_stdin(&mut self) -> Option<std::process::ChildStdin> {
        self.stdin.take()
    }

    /// `stdout_file` (:68-70)
    pub fn take_stdout(&mut self) -> Option<std::process::ChildStdout> {
        self.child.as_mut().and_then(|c| c.stdout.take())
    }

    /// `stderr_file` (:72-74)
    pub fn take_stderr(&mut self) -> Option<std::process::ChildStderr> {
        self.child.as_mut().and_then(|c| c.stderr.take())
    }

    /// `close_stdin` (:76-81) — an open stdin pipe blocks transport children
    pub fn close_stdin(&mut self) {
        if let Some(mut s) = self.stdin.take() {
            let _ = s.flush();
        }
    }

    /// write then close (the tools' stdin_data flow, server-tools.cpp:228-239)
    pub fn write_stdin_and_close(&mut self, data: &[u8]) {
        if let Some(mut s) = self.stdin.take() {
            if !data.is_empty() {
                let _ = s.write_all(data);
            }
            let _ = s.flush();
        }
    }

    /// `terminate` (:83-87) — SIGTERM
    pub fn terminate(&mut self) {
        if let Some(c) = self.child.as_mut() {
            let _ = c.kill();
        }
    }

    /// `join` (:89-97) — wait and reap; returns the exit code (-1 = none)
    pub fn join(&mut self) -> i32 {
        match self.child.take() {
            Some(mut c) => match c.wait() {
                Ok(status) => status.code().unwrap_or(-1),
                Err(_) => -1,
            },
            None => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_argv_capture_and_join() {
        let mut p = Subproc::new();
        assert!(p.create(&["/bin/sh".into(), "-c".into(), "echo hello".into()], &[], None));
        assert!(p.has_handle());
        p.close_stdin();
        let mut out = p.take_stdout().expect("stdout");
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut out, &mut buf).unwrap();
        assert_eq!(p.join(), 0);
        assert_eq!(buf, "hello\n");
        assert!(!p.has_handle());
    }

    #[test]
    fn combined_stderr_merges_onto_stdout() {
        // server-tools.cpp:203 combined_stdout_stderr — both streams in one pipe
        let mut p = Subproc::new();
        let mut out = p
            .create_combined(
                &[
                    "/bin/sh".into(),
                    "-c".into(),
                    "echo out; echo err 1>&2".into(),
                ],
                &[],
                None,
            )
            .expect("spawn");
        p.close_stdin();
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut out, &mut buf).unwrap();
        assert_eq!(p.join(), 0);
        let mut lines: Vec<&str> = buf.lines().collect();
        lines.sort();
        assert_eq!(lines, vec!["err", "out"]);
    }

    #[test]
    fn terminate_kills_a_hung_child() {
        let mut p = Subproc::new();
        assert!(p.create(&["/bin/sh".into(), "-c".into(), "sleep 30".into()], &[], None));
        p.terminate();
        let code = p.join();
        assert_ne!(code, 0);
    }

    #[test]
    fn env_override_reaches_the_child() {
        let mut p = Subproc::new();
        assert!(p.create(
            &["/bin/sh".into(), "-c".into(), "printf %s \"$FOO\"".into()],
            &[("FOO".into(), "bar".into())],
            None
        ));
        p.close_stdin();
        let mut out = p.take_stdout().expect("stdout");
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut out, &mut buf).unwrap();
        p.join();
        assert_eq!(buf, "bar");
    }

    #[test]
    fn missing_binary_fails_to_spawn() {
        let mut p = Subproc::new();
        assert!(!p.create(&["/nonexistent/llama-test-bin".into()], &[], None));
        assert!(!p.has_handle());
        assert_eq!(p.join(), -1);
    }
}
