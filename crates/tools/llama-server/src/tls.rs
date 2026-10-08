//! tls.rs — the OpenSSL client session behind the `/cors-proxy` https arm.
//!
//! The reference's `server_http_proxy` (server-models.cpp:2456-2526) builds
//! an `httplib::SSLClient` when the target scheme is https
//! (`#ifdef CPPHTTPLIB_OPENSSL_SUPPORT`, vendor/cpp-httplib/httplib.cpp:
//! 13746-13792 `initialize_ssl`): a TLS 1.2+ client context, system default
//! verify paths, server-certificate **and** hostname verification both on by
//! default (httplib.h:3045-3046), then a blocking BIO over the connected
//! socket. This module is the same client over the same system library —
//! raw FFI to `libssl`/`libcrypto`, linked the same way the reference links
//! them (a system library, not a crate dependency). Everything the proxy
//! needs is four calls: handshake, write, read, shutdown.

use std::io;
use std::net::TcpStream;
use std::os::fd::AsRawFd;

// ---- the OpenSSL C surface (openssl/ssl.h + openssl/err.h) ---------------
// Linking a native system library via #[link] is not a crate dependency —
// the reference binary links these exact .so's (ldd llama-server:
// libssl.so.3 / libcrypto.so.3).
#[link(name = "ssl")]
#[link(name = "crypto")]
extern "C" {
    fn TLS_client_method() -> *const std::ffi::c_void;
    fn SSL_CTX_new(method: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn SSL_CTX_free(ctx: *mut std::ffi::c_void);
    fn SSL_CTX_ctrl(
        ctx: *mut std::ffi::c_void,
        cmd: std::ffi::c_int,
        larg: std::ffi::c_long,
        parg: *mut std::ffi::c_void,
    ) -> std::ffi::c_long;
    fn SSL_CTX_set_default_verify_paths(ctx: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn SSL_new(ctx: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    fn SSL_free(ssl: *mut std::ffi::c_void);
    fn SSL_set_fd(ssl: *mut std::ffi::c_void, fd: std::ffi::c_int) -> std::ffi::c_int;
    fn SSL_set1_host(ssl: *mut std::ffi::c_void, hostname: *const u8) -> std::ffi::c_int;
    fn SSL_ctrl(
        ssl: *mut std::ffi::c_void,
        cmd: std::ffi::c_int,
        larg: std::ffi::c_long,
        parg: *mut std::ffi::c_void,
    ) -> std::ffi::c_long;
    fn SSL_connect(ssl: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn SSL_read(ssl: *mut std::ffi::c_void, buf: *mut u8, num: std::ffi::c_int)
        -> std::ffi::c_int;
    fn SSL_write(ssl: *mut std::ffi::c_void, buf: *const u8, num: std::ffi::c_int)
        -> std::ffi::c_int;
    fn SSL_shutdown(ssl: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn SSL_get_verify_result(ssl: *const std::ffi::c_void) -> std::ffi::c_long;
    fn ERR_get_error() -> std::ffi::c_ulong;
    fn ERR_error_string_n(e: std::ffi::c_ulong, buf: *mut u8, len: usize);
    fn ERR_clear_error();
}

// SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION)
const SSL_CTRL_SET_MIN_PROTO_VERSION: std::ffi::c_int = 123;
const TLS1_2_VERSION: std::ffi::c_long = 0x0303;
// SSL_set_tlsext_host_name(ssl, host) — the SNI extension
const SSL_CTRL_SET_TLSEXT_HOSTNAME: std::ffi::c_int = 55;
const TLSEXT_NAMETYPE_host_name: std::ffi::c_long = 0;
const X509_V_OK: std::ffi::c_long = 0;

fn ssl_error_string() -> String {
    unsafe {
        let mut buf = [0u8; 256];
        let e = ERR_get_error();
        ERR_error_string_n(e, buf.as_mut_ptr(), buf.len());
        ERR_clear_error();
        String::from_utf8_lossy(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())])
            .into_owned()
    }
}

/// A client TLS session over a connected socket. Verification mirrors
/// httplib's defaults: `SSL_CTX_set_default_verify_paths` (httplib.cpp:14332
/// `load_certs`' system store) + `SSL_set1_host` (hostname verification) —
/// an untrusted or mismatched peer fails the handshake like the reference's
/// `Error::SSLServerVerification`.
pub struct SslStream {
    ssl: *mut std::ffi::c_void,
    ctx: *mut std::ffi::c_void,
    _sock: TcpStream,
}

// the session is used from one handler thread at a time; OpenSSL is not
// thread-affine
unsafe impl Send for SslStream {}

impl SslStream {
    pub fn connect(sock: TcpStream, host: &str) -> Result<Self, String> {
        // SAFETY: the FFI calls below follow ssl-client usage from
        // SSL_CTX_new to SSL_connect; every failure path frees what it made
        unsafe {
            ERR_clear_error();
            let ctx = SSL_CTX_new(TLS_client_method());
            if ctx.is_null() {
                return Err(format!("TLS context: {}", ssl_error_string()));
            }
            // min proto TLS1.2 (httplib.cpp:14466 create_client_context)
            SSL_CTX_ctrl(
                ctx,
                SSL_CTRL_SET_MIN_PROTO_VERSION,
                TLS1_2_VERSION,
                std::ptr::null_mut(),
            );
            if SSL_CTX_set_default_verify_paths(ctx) != 1 {
                let e = ssl_error_string();
                SSL_CTX_free(ctx);
                return Err(format!("TLS verify paths: {e}"));
            }
            let ssl = SSL_new(ctx);
            if ssl.is_null() {
                let e = ssl_error_string();
                SSL_CTX_free(ctx);
                return Err(format!("TLS session: {e}"));
            }
            let mut ok = SSL_set_fd(ssl, sock.as_raw_fd()) == 1;
            // both SSL_set1_host and the SNI extension take a NUL-terminated
            // hostname; non-UTF8-free host names cannot appear here (the URL
            // parser hands out ASCII)
            let host_c = std::ffi::CString::new(host).unwrap_or_default();
            if ok {
                // hostname verification (X509_check_host through set1_host)
                ok = SSL_set1_host(ssl, host_c.as_ptr() as *const u8) == 1;
            }
            if ok {
                // SNI — best effort like SSLClient::initialize_ssl
                SSL_ctrl(
                    ssl,
                    SSL_CTRL_SET_TLSEXT_HOSTNAME,
                    TLSEXT_NAMETYPE_host_name,
                    host_c.as_ptr() as *const u8 as *mut std::ffi::c_void,
                );
            }
            if !ok || SSL_connect(ssl) != 1 {
                // a failed verify leaves a non-zero verify result
                let vr = SSL_get_verify_result(ssl);
                let e = ssl_error_string();
                SSL_free(ssl);
                SSL_CTX_free(ctx);
                if vr != X509_V_OK {
                    return Err(format!("TLS certificate verification failed ({vr})"));
                }
                return Err(format!("TLS handshake: {e}"));
            }
            let vr = SSL_get_verify_result(ssl);
            if vr != X509_V_OK {
                let e = format!("TLS certificate verification failed ({vr})");
                SSL_free(ssl);
                SSL_CTX_free(ctx);
                return Err(e);
            }
            Ok(SslStream {
                ssl,
                ctx,
                _sock: sock,
            })
        }
    }
}

impl Drop for SslStream {
    fn drop(&mut self) {
        // SAFETY: frees this stream's own session/context exactly once
        unsafe {
            SSL_shutdown(self.ssl);
            SSL_free(self.ssl);
            SSL_CTX_free(self.ctx);
        }
    }
}

impl io::Read for SslStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // SAFETY: SSL_read into the caller's buffer, len < INT_MAX
        let n = unsafe {
            SSL_read(self.ssl, buf.as_mut_ptr(), buf.len().min(i32::MAX as usize) as i32)
        };
        if n < 0 {
            return Err(io::Error::new(io::ErrorKind::Other, ssl_error_string()));
        }
        Ok(n as usize)
    }
}

impl io::Write for SslStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // SAFETY: SSL_write from the caller's buffer, len < INT_MAX
        let n = unsafe {
            SSL_write(self.ssl, buf.as_ptr(), buf.len().min(i32::MAX as usize) as i32)
        };
        if n < 0 {
            return Err(io::Error::new(io::ErrorKind::Other, ssl_error_string()));
        }
        Ok(n as usize)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
