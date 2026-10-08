//! ui.rs — the embedded web client (llama-server's "cli-client shell"):
//! the `llama-ui-assets` serving surface of tools/server.
//!
//! Reference shape (pinned bd4f514db1):
//!   * `tools/ui/CMakeLists.txt`'s `llama-ui-assets` target provisions a
//!     dist tree (a pre-built `tools/ui/dist` > an npm build > an HF
//!     download) and `scripts/ui-assets.cmake:emit_files` (lines 143-269)
//!     generates `ui.cpp`/`ui.h`: one gzipped-with-zeroed-timestamp byte
//!     array per file (`file(ARCHIVE_CREATE ... COMPRESSION GZip)` with
//!     `SOURCE_DATE_EPOCH=0`, :176-181/:204-219), a quoted-SHA-256 ETag
//!     (`file(SHA256 "${embed_dir}/${f}" etag)` + the `"\\"${etag}\\""`
//!     table row, :234-245), and the extension→MIME table of
//!     `mime_from_ext` (:17-60) — an EMPTY table when the tree has no
//!     index.html (:150-159). The port's generated artifact is
//!     `ui_assets.rs` (parity/gen_ui_assets.py) — empty by default, the
//!     exact state of this machine's reference build
//!     (build-rust-ref/tools/ui/ui.cpp embeds 0 assets).
//!   * `server_http_context::init`'s Web UI block (server-http.cpp:360-478)
//!     serves it: `--ui/--no-ui` (arg.cpp:3464-3470, params.ui default
//!     true) gates the whole block; `--path/--api-prefix` (arg.cpp:3339-3345
//!     / :3382-3388) redirect to a directory mount / prefix the routes; the
//!     embedded routes are "/" + "/index.html" (ETag-revalidated, COOP/COEP
//!     isolated) and every table entry (hashed assets cached immutable,
//!     `sw.js` / `manifest.webmanifest` / `_app/version.json` /
//!     `build.json` revalidated) with the gzip gate (415 when the client
//!     lacks `Accept-Encoding: gzip`) and If-None-Match → 304.

use crate::http::{HttpServer, Request, Response};
use std::sync::Arc;

/// `struct llama_ui_asset` (tools/ui/ui.h.in) — one embedded file. The
/// generated table carries `&'static` pieces.
pub struct UiAsset {
    pub name: &'static str,
    pub data: &'static [u8],
    /// quoted sha256 of the embedded (possibly gzipped) bytes — the wire
    /// ETag verbatim
    pub etag: &'static str,
    /// mime_from_ext's product (scripts/ui-assets.cmake:17-60)
    pub ty: &'static str,
}

/// `llama_ui_get_assets` — the (possibly empty) embedded table.
pub fn assets() -> &'static [UiAsset] {
    crate::ui_assets::ASSETS
}

/// `llama_ui_use_gzip` — true when the build embedded the gzip'd variants.
pub fn use_gzip() -> bool {
    crate::ui_assets::USE_GZIP
}

/// `llama_ui_find_asset` (ui.cpp.in) — exact name match.
pub fn find_asset(name: &str) -> Option<&'static UiAsset> {
    assets().iter().find(|a| a.name == name)
}

// `mime_from_ext` (scripts/ui-assets.cmake:17-60) — the asset-table's MIME
// rules (differs from cpp-httplib's serving map on purpose: charset on html,
// `application/javascript`, `application/manifest+json`)
pub fn asset_mime(name: &str) -> &'static str {
    let ext = name.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    match ext {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css",
        "js" => "application/javascript",
        "json" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn asset_not_found() -> Response {
    Response {
        status: 404,
        content_type: "application/json; charset=utf-8".into(),
        body: crate::http::Body::Full(crate::api::json_error(
            "File Not Found",
            "not_found_error",
            404,
        )),
        headers: Vec::new(),
        terminal_done: false,
    }
}

/// the binary-body answer with the extra headers the route collected — the
/// raw asset bytes (`Bytes`): png/woff2 survive the wire byte-exactly, the
/// same path the /cors-proxy relay rides
fn asset_response(asset: &UiAsset, headers: Vec<(String, String)>) -> Response {
    Response {
        status: 200,
        content_type: asset.ty.to_string(),
        body: crate::http::Body::Bytes(asset.data.to_vec()),
        headers,
        terminal_done: false,
    }
}

/// `serve_asset_cached` (server-http.cpp:413-433): the gzip gate, then
/// ETag/If-None-Match → 304, then the isolation + caching headers and the
/// bytes.
pub fn serve_asset_cached(req: &Request, name: &str, isolation: bool, cache_control: &str) -> Response {
    serve_from(assets(), use_gzip(), req, name, Some((isolation, cache_control)))
}

/// `serve_asset_nocache` (server-http.cpp:435-446).
pub fn serve_asset_nocache(req: &Request, name: &str) -> Response {
    serve_from(assets(), use_gzip(), req, name, None)
}

/// the shared body of the two serve lambdas — table-parameterized so the
/// semantics are unit-testable against a synthetic table (the generated
/// `ui_assets.rs` default is empty, exactly the reference build's state).
fn serve_from(
    table: &[UiAsset],
    gzip: bool,
    req: &Request,
    name: &str,
    cached: Option<(bool, &str)>,
) -> Response {
    // handle_gzip_header (server-http.cpp:381-394)
    let mut headers: Vec<(String, String)> = Vec::new();
    if gzip {
        let accepts = req
            .headers
            .get("accept-encoding")
            .map(|v| v.contains("gzip"))
            .unwrap_or(false);
        if !accepts {
            return Response {
                status: 415, // unsupported media type
                content_type: "text/plain".into(),
                body: crate::http::Body::Full(
                    "Error: gzip is not supported by this browser".into(),
                ),
                headers: Vec::new(),
                terminal_done: false,
            };
        }
        headers.push(("Content-Encoding".into(), "gzip".into()));
    }
    let Some(a) = table.iter().find(|a| a.name == name) else {
        return asset_not_found();
    };
    let Some((isolation, cache_control)) = cached else {
        headers.push(("Cache-Control".into(), "no-cache".into()));
        return asset_response(a, headers);
    };
    headers.push(("ETag".into(), a.etag.to_string()));
    let inm = req.headers.get("if-none-match").cloned().unwrap_or_default();
    if !inm.is_empty() && (inm == a.etag || inm == format!("W/{}", a.etag)) {
        return Response {
            status: 304,
            content_type: "text/html; charset=utf-8".into(),
            body: crate::http::Body::Full(String::new()),
            headers,
            terminal_done: false,
        };
    }
    if isolation {
        headers.push((
            "Cross-Origin-Embedder-Policy".into(),
            "require-corp".into(),
        ));
        headers.push(("Cross-Origin-Opener-Policy".into(), "same-origin".into()));
    }
    headers.push(("Cache-Control".into(), cache_control.to_string()));
    asset_response(a, headers)
}

/// hashed assets never change under a name; `index.html` revalidates
const CACHE_IMMUTABLE: &str = "public, max-age=31536000, immutable";
const CACHE_REVALIDATE: &str = "no-cache";

/// the `no_cache_names` set (server-http.cpp:452-457) — the PWA
/// revalidation files
fn is_nocache_name(name: &str) -> bool {
    matches!(
        name,
        "sw.js" | "manifest.webmanifest" | "_app/version.json" | "build.json"
    )
}

/// the self-removing service worker served at `{api_prefix}/sw.js` when the
/// built-in UI is not served — a browser that used the built-in UI keeps its
/// service worker, so it shows the old UI even after the UI is replaced or
/// disabled; answering the worker's update check with this unregisters it
/// (server-http.cpp:473-489)
pub const SW_REMOVE_JS: &str = "
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil((async () => {
    await self.registration.unregister();
    for (const key of await caches.keys()) await caches.delete(key);
    for (const c of await self.clients.matchAll({ type: 'window' })) c.navigate(c.url);
})()));
";

/// the `sw.js` route of `init_listener` (server-http.cpp:473-489). Registered
/// whenever the embedded UI is not served (`!ui` or a `--path` mount); a real
/// `sw.js` in the mounted directory wins, exactly like cpp-httplib's
/// file-before-handler routing.
fn sw_remove_response() -> Response {
    Response {
        status: 200,
        content_type: "application/javascript".into(),
        body: crate::http::Body::Full(SW_REMOVE_JS.to_string()),
        headers: vec![("Cache-Control".into(), "no-cache".into())],
        terminal_done: false,
    }
}

/// `register_sw_removal` — the `!params.ui || !params.public_path.empty()`
/// registration of `init_listener`.
fn register_sw_removal(routes: &mut HttpServer, mounted_dir: Option<&std::path::Path>) {
    let mounted_dir = mounted_dir.map(|p| p.to_path_buf());
    let handler: crate::http::Handler = Arc::new(move |_req: &Request| {
        // cpp-httplib answers from the mount point first; only a mount
        // without sw.js falls through to the removal worker
        if let Some(dir) = &mounted_dir {
            if let Some(r) = serve_mounted(dir, "/sw.js") {
                return r;
            }
        }
        sw_remove_response()
    });
    routes.add("GET", "/sw.js", handler);
}

/// the Web UI block of `server_http_context::init`
/// (server-http.cpp:366-478): the flag logs, the `--path` mount, or the
/// embedded-asset routes under `--api-prefix`.
pub fn register_ui(routes: &mut HttpServer, api_prefix: &str, public_path: &str, ui_enabled: bool) {
    if !ui_enabled {
        // SRV_INF pair (server-http.cpp:373-374)
        eprintln!("info: The UI is disabled");
        eprintln!("info: Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable");
        // the UI is not served — remove any service worker a browser kept
        // from a previous session (init_listener, server-http.cpp:473-489)
        register_sw_removal(routes, None);
        let _ = api_prefix;
        return;
    }
    if !public_path.is_empty() {
        // srv->set_mount_point(api_prefix + "/", params.public_path)
        // (server-http.cpp:379-385) — a missing directory fails init
        let dir = std::path::Path::new(public_path);
        if !dir.is_dir() {
            eprintln!("error: static assets path not found: {public_path}");
            std::process::exit(1);
        }
        // the embedded UI is not served behind a mount — remove any kept
        // service worker (a real sw.js in the mount is served first)
        register_sw_removal(routes, Some(dir));
        let mnt = if api_prefix.is_empty() {
            "/".to_string()
        } else {
            format!("{api_prefix}/")
        };
        routes.set_mount(mnt, dir.to_path_buf());
        return;
    }
    if assets().is_empty() {
        // #else of LLAMA_UI_HAS_ASSETS: no routes registered; "/" keeps the
        // 404 the reference's empty-table build answers (the state of
        // build-rust-ref)
        return;
    }
    // main index file — revalidated, isolation headers on
    // (server-http.cpp:449-450). The paths are bare: `add` mounts them under
    // `path_prefix + path`, the reference's
    // `srv->Get(params.api_prefix + "/", ...)` (server-http.cpp:447-448)
    for path in ["/", "/index.html"] {
        let handler: crate::http::Handler = Arc::new(move |req: &Request| {
            serve_asset_cached(req, "index.html", true, CACHE_REVALIDATE)
        });
        routes.add("GET", path, handler);
    }
    // every remaining table entry (server-http.cpp:459-473) — bare names;
    // `add` prefixes (`params.api_prefix + "/" + a.name`, server-http.cpp:464)
    for a in assets() {
        if a.name == "index.html" {
            continue;
        }
        let route = format!("/{}", a.name);
        let name: &'static str = a.name;
        if is_nocache_name(name) {
            let handler: crate::http::Handler =
                Arc::new(move |req: &Request| serve_asset_nocache(req, name));
            routes.add("GET", &route, handler);
        } else {
            let handler: crate::http::Handler = Arc::new(move |req: &Request| {
                serve_asset_cached(req, name, false, CACHE_IMMUTABLE)
            });
            routes.add("GET", &route, handler);
        }
    }
}

// ---------------------------------------------------------------------------
// the `--path` mount — cpp-httplib's static file serving subset the
// reference rides (Server::set_mount_point, httplib.cpp:8286-8312, plus
// detail::content_type's extension map, httplib.cpp:2759-2814)
// ---------------------------------------------------------------------------

/// detail::content_type's extension map with its `text/plain` default
pub fn mount_mime(path: &str) -> &'static str {
    let ext = path.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    match ext {
        "css" => "text/css",
        "csv" => "text/csv",
        "htm" | "html" => "text/html",
        "js" | "mjs" => "text/javascript",
        "txt" => "text/plain",
        "vtt" => "text/vtt",
        "apng" => "image/apng",
        "avif" => "image/avif",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "tif" | "tiff" => "image/tiff",
        "jpg" | "jpeg" => "image/jpeg",
        "mp4" => "video/mp4",
        "mpeg" => "video/mpeg",
        "webm" => "video/webm",
        "mp3" => "audio/mp3",
        "mpga" => "audio/mpeg",
        "weba" => "audio/webm",
        "wav" => "audio/wave",
        "otf" => "font/otf",
        "ttf" => "font/ttf",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "7z" => "application/x-7z-compressed",
        "atom" => "application/atom+xml",
        "pdf" => "application/pdf",
        "json" => "application/json",
        "rss" => "application/rss+xml",
        "tar" => "application/x-tar",
        "xht" | "xhtml" => "application/xhtml+xml",
        "xslt" => "application/xslt+xml",
        "xml" => "application/xml",
        "gz" => "application/gzip",
        "zip" => "application/zip",
        "wasm" => "application/wasm",
        _ => "text/plain",
    }
}

/// one mounted file answer: path traversal clamped inside the mount root, a
/// directory resolves to its index.html (httplib's
/// `prepare_static_file_request` behavior), missing files fall through to
/// the 404 the router answers
pub fn serve_mounted(mount_root: &std::path::Path, url_path: &str) -> Option<Response> {
    // strip the query half if any slipped through, then percent-decode is
    // skipped (the router hands over the raw path)
    let rel = url_path.trim_start_matches('/');
    if rel.split('/').any(|seg| seg == "..") {
        return None;
    }
    let mut full = mount_root.to_path_buf();
    for seg in rel.split('/').filter(|s| !s.is_empty()) {
        full.push(seg);
    }
    if full.is_dir() {
        full.push("index.html");
    }
    if !full.is_file() {
        return None;
    }
    let name = full.to_string_lossy().into_owned();
    let data = std::fs::read(&full).ok()?;
    Some(Response {
        status: 200,
        content_type: mount_mime(&name).to_string(),
        // the raw file bytes — binary mounts (png/woff2/…) are byte-exact
        body: crate::http::Body::Bytes(data),
        headers: Vec::new(),
        terminal_done: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{Body, Request};

    /// the self-removing service worker of `init_listener`
    /// (server-http.cpp:473-489, upstream ba0ba54d9): byte-exact body, the
    /// no-cache header and the javascript content type
    #[test]
    fn sw_removal_response_matches_reference() {
        let r = sw_remove_response();
        assert_eq!(r.status, 200);
        assert_eq!(r.content_type, "application/javascript");
        assert_eq!(
            r.headers,
            vec![("Cache-Control".to_string(), "no-cache".to_string())]
        );
        match r.body {
            Body::Full(s) => {
                // the C++ raw literal R"(...)" — starts with a newline and
                // ends with the newline before the closing paren
                assert!(s.starts_with('\n'));
                assert!(s.ends_with("})()));\n"));
                assert!(s.contains("self.skipWaiting()"));
                assert!(s.contains("self.registration.unregister()"));
                assert!(s.contains("caches.delete(key)"));
                assert!(s.contains("c.navigate(c.url)"));
                assert_eq!(s.len(), 336);
            }
            _ => panic!("expected a full body"),
        }
    }

    fn table() -> Vec<UiAsset> {
        vec![
            UiAsset {
                name: "index.html",
                data: b"<html>ui</html>\n",
                etag: "\"index-etag\"",
                ty: "text/html; charset=utf-8",
            },
            UiAsset {
                name: "bundle.abc123.js",
                data: b"console.log(1)\n",
                etag: "\"bundle-etag\"",
                ty: "application/javascript",
            },
            UiAsset {
                name: "sw.js",
                data: b"sw();\n",
                etag: "\"sw-etag\"",
                ty: "application/javascript",
            },
            UiAsset {
                name: "_app/version.json",
                data: b"{\"version\":\"1\"}",
                etag: "\"ver-etag\"",
                ty: "application/json",
            },
        ]
    }

    fn req(headers: &[(&str, &str)]) -> Request {
        let mut h = std::collections::HashMap::new();
        for (k, v) in headers {
            h.insert(k.to_string(), v.to_string());
        }
        Request {
            method: "GET".into(),
            path: "/".into(),
            params: Default::default(),
            headers: h,
            body: Vec::new(),
            files: Default::default(),
        }
    }

    fn body(r: &Response) -> String {
        match &r.body {
            Body::Full(s) => s.clone(),
            Body::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            _ => String::new(),
        }
    }

    fn hdr(r: &Response, name: &str) -> String {
        r.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    }

    /// the empty generated table: no assets, no gzip — the exact state of
    /// this machine's reference build (build-rust-ref/tools/ui/ui.cpp)
    #[test]
    fn empty_table_matches_reference_build() {
        assert!(assets().is_empty());
        assert!(!use_gzip());
        assert!(find_asset("index.html").is_none());
    }

    /// serve_asset_cached's ETag/304/COOP/COEP/Cache-Control contract
    /// (server-http.cpp:413-433)
    #[test]
    fn cached_asset_semantics() {
        let t = table();
        let r = serve_from(&t, false, &req(&[]), "index.html", Some((true, CACHE_REVALIDATE)));
        assert_eq!(r.status, 200);
        assert_eq!(r.content_type, "text/html; charset=utf-8");
        assert_eq!(body(&r), "<html>ui</html>\n");
        assert_eq!(hdr(&r, "ETag"), "\"index-etag\"");
        assert_eq!(hdr(&r, "Cross-Origin-Opener-Policy"), "same-origin");
        assert_eq!(hdr(&r, "Cross-Origin-Embedder-Policy"), "require-corp");
        assert_eq!(hdr(&r, "Cache-Control"), "no-cache");

        // If-None-Match → 304, both bare and W/-prefixed
        for inm in ["\"index-etag\"", "W/\"index-etag\""] {
            let r = serve_from(
                &t,
                false,
                &req(&[("if-none-match", inm)]),
                "index.html",
                Some((true, CACHE_REVALIDATE)),
            );
            assert_eq!(r.status, 304, "inm {inm}");
        }
        // a stale etag serves the bytes
        let r = serve_from(
            &t,
            false,
            &req(&[("if-none-match", "\"other\"")]),
            "index.html",
            Some((true, CACHE_REVALIDATE)),
        );
        assert_eq!(r.status, 200);

        // non-index assets: no isolation headers, the immutable policy
        let r = serve_from(
            &t,
            false,
            &req(&[]),
            "bundle.abc123.js",
            Some((false, CACHE_IMMUTABLE)),
        );
        assert_eq!(r.content_type, "application/javascript");
        assert!(hdr(&r, "Cross-Origin-Opener-Policy").is_empty());
        assert_eq!(hdr(&r, "Cache-Control"), "public, max-age=31536000, immutable");

        // missing name → 404
        let r = serve_from(&t, false, &req(&[]), "nope.js", Some((false, CACHE_IMMUTABLE)));
        assert_eq!(r.status, 404);
    }

    /// serve_asset_nocache + the no_cache_names set (server-http.cpp:435-457)
    #[test]
    fn nocache_names() {
        assert!(is_nocache_name("sw.js"));
        assert!(is_nocache_name("manifest.webmanifest"));
        assert!(is_nocache_name("_app/version.json"));
        assert!(is_nocache_name("build.json"));
        assert!(!is_nocache_name("bundle.abc123.js"));
        let t = table();
        let r = serve_from(&t, false, &req(&[]), "sw.js", None);
        assert_eq!(r.status, 200);
        assert_eq!(hdr(&r, "Cache-Control"), "no-cache");
        // no ETag on the nocache route (the reference's lambda sets none)
        assert!(hdr(&r, "ETag").is_empty());
    }

    /// handle_gzip_header (server-http.cpp:381-394): a gzip build answers
    /// 415 with the exact body when the client lacks Accept-Encoding: gzip,
    /// and marks the ok path Content-Encoding: gzip
    #[test]
    fn gzip_gate() {
        let t = table();
        let r = serve_from(&t, true, &req(&[]), "index.html", Some((true, CACHE_REVALIDATE)));
        assert_eq!(r.status, 415);
        assert_eq!(r.content_type, "text/plain");
        assert_eq!(body(&r), "Error: gzip is not supported by this browser");

        let r = serve_from(
            &t,
            true,
            &req(&[("accept-encoding", "deflate, gzip")]),
            "index.html",
            Some((true, CACHE_REVALIDATE)),
        );
        assert_eq!(r.status, 200);
        assert_eq!(hdr(&r, "Content-Encoding"), "gzip");
    }

    /// the `--path` mount: httplib's extension map (httplib.cpp:2759-2814
    /// with its text/plain default), the directory → index.html fallback,
    /// the 404 fallthrough and the traversal clamp
    #[test]
    fn mount_serving() {
        let dir = std::env::temp_dir().join("llama-ui-mount-test");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("index.html"), b"<h1>hi</h1>").unwrap();
        std::fs::write(dir.join("sub/a.js"), b"1").unwrap();
        std::fs::write(dir.join("sub/note.txt"), b"note").unwrap();

        let r = serve_mounted(&dir, "").unwrap(); // the directory → index.html
        assert_eq!(r.status, 200);
        assert_eq!(r.content_type, "text/html");
        assert_eq!(body(&r), "<h1>hi</h1>");

        let r = serve_mounted(&dir, "sub/a.js").unwrap();
        assert_eq!(r.content_type, "text/javascript");
        let r = serve_mounted(&dir, "sub/note.txt").unwrap();
        assert_eq!(r.content_type, "text/plain");
        std::fs::write(dir.join("sub/nothing"), b"x").unwrap();
        // unknown extension falls to the default mimetype
        let r = serve_mounted(&dir, "sub/nothing").unwrap();
        assert_eq!(r.content_type, "text/plain");

        assert!(serve_mounted(&dir, "missing.js").is_none());
        // traversal never escapes the root
        assert!(serve_mounted(&dir, "../../etc/passwd").is_none());
    }

    /// mime_from_ext (scripts/ui-assets.cmake:17-60) — the asset-table MIME
    /// rules differ from httplib's serving map on purpose
    #[test]
    fn asset_mime_table() {
        assert_eq!(asset_mime("index.html"), "text/html; charset=utf-8");
        assert_eq!(asset_mime("a.js"), "application/javascript");
        assert_eq!(asset_mime("m.webmanifest"), "application/manifest+json");
        assert_eq!(asset_mime("x.bin"), "application/octet-stream");
    }
}
