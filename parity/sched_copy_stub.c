/* parity/sched_copy_stub.c — stub scheduler for the foreign copy-callback
 * FFI round-trip test (crates/ggml/src/backend_emit.rs
 * `foreign_sched_copy_callback_stub_roundtrip`).
 *
 * Mirrors what ggml_backend_sched does with the copy callback at
 * ggml-backend.cpp:969-970 (fields callback_copy + callback_copy_user_data),
 * :2140-2143 (ggml_backend_sched_set_copy_callback) and :1835-1836 (the
 * dispatch — here replayed on demand through stub_sched_invoke_copy) of the
 * NEW pinned tree @ c35b66744 (6753a033f). No GPU is needed: the test only
 * proves the C-ABI trampoline round-trip.
 *
 * Build:
 *   cc -shared -fPIC -o libggml-schedcb-stub.so sched_copy_stub.c
 */
#include <stddef.h>

typedef int (*sched_copy_cb)(void * backend, const void * src, void * dst,
                             void * graph, void * user_data);

static sched_copy_cb g_cb;
static void * g_user_data;
static void * g_sched_seen;

/* ggml_backend_sched_set_copy_callback (ggml-backend.cpp:2140-2143) */
void ggml_backend_sched_set_copy_callback(void * sched, sched_copy_cb cb, void * user_data) {
    g_sched_seen = sched;
    g_cb = cb;
    g_user_data = user_data;
}

/* replay the scheduler's dispatch (ggml-backend.cpp:1835-1836) on demand */
int stub_sched_invoke_copy(void * backend, const void * src, void * dst, void * graph) {
    if (!g_cb) {
        return 0;
    }
    return g_cb(backend, src, dst, graph, g_user_data);
}

/* test hooks: what the stub recorded */
void * stub_seen_sched(void) { return g_sched_seen; }
void * stub_seen_user_data(void) { return g_user_data; }
