/* parity/backend_dl_stub.c — DL stub backend used by the ggml backend-layer
 * tests (crates/ggml/src/backend.rs `backend_dl_stub`).
 *
 * This is the kept artifact of the C source the test generates at runtime
 * (parameterized by {reg_name} / optional ggml_backend_score). It mirrors the
 * C-ABI layouts declared in crates/ggml/src/sysffi.rs — ggml-backend-impl.h
 * struct ggml_backend_reg / ggml_backend_device at pinned revision
 * bd4f514db1 — enough for the registry to load, score, version-check and
 * enumerate a dynamically loaded backend (ggml-backend-reg.cpp:220-264).
 * Driving foreign compute over this ABI is task ③ of the GPU plan.
 *
 * Build:
 *   cc -shared -fPIC [-DSCORE=n] -o libggml-stub.so backend_dl_stub.c
 * (the test builds its own copies into a temp dir with the same shape)
 */

#include <stddef.h>

/* mirrors sysffi.rs GgmlBackendRegI (ggml-backend-impl.h:230-240) */
typedef const char *(*get_name_fn)(const void *);
typedef size_t (*get_count_fn)(const void *);
typedef void *(*get_dev_fn)(const void *, size_t);
typedef struct {
    get_name_fn get_name;
    get_count_fn get_device_count;
    get_dev_fn get_device;
    void *(*get_proc_address)(const void *, const char *);
} reg_i_t;

/* mirrors sysffi.rs GgmlBackendReg (ggml-backend-impl.h:242-246) */
typedef struct {
    int api_version; /* GGML_BACKEND_API_VERSION == 2 */
    reg_i_t iface;
    void *context;
} reg_t;

/* mirrors sysffi.rs GgmlBackendDeviceI (first four slots of
 * ggml-backend-impl.h:176-218) */
typedef struct {
    get_name_fn get_name;
    get_name_fn get_description;
    void (*get_memory)(const void *, size_t *, size_t *);
    int (*get_type)(const void *);
} dev_i_t;

/* mirrors sysffi.rs GgmlBackendDevice (ggml-backend-impl.h:220-224) */
typedef struct {
    dev_i_t iface;
    const reg_t *reg;
    void *context;
} dev_t;

static const char *reg_name_fn(const void *r) { (void)r; return "STUB"; }
static size_t reg_count_fn(const void *r)     { (void)r; return 1; }

static const char *dev0_name_fn(const void *d) { (void)d; return "STUB0"; }
static void dev0_mem_fn(const void *d, size_t *f, size_t *t) { (void)d; *f = 1 << 30; *t = 1 << 30; }
static int dev0_type_fn(const void *d)        { (void)d; return 0; } /* GGML_BACKEND_DEVICE_TYPE_CPU */

static dev_t dev0 = { { dev0_name_fn, dev0_name_fn, dev0_mem_fn, dev0_type_fn }, NULL, NULL };

static void *reg_get_dev_fn(const void *r, size_t i) {
    (void)r;
    return i == 0 ? (void *)&dev0 : NULL;
}

static reg_t the_reg = { 2, { reg_name_fn, reg_count_fn, reg_get_dev_fn, NULL }, NULL };

#ifdef SCORE
/* impl.h:254 — 0 means "not supported on this system" */
int ggml_backend_score(void) { return SCORE; }
#endif

/* impl.h:251 — the registry's dlsym entry point */
void *ggml_backend_init(void) { return (void *)&the_reg; }
