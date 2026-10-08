/* gen_iq_tables.c — emits Rust lookup tables for quants_k.rs from the pinned
 * ggml-common.h table section (agent D tooling; output is pasted verbatim).
 *
 * Build like ref_quants_dump.c; run: ./gen_iq_tables > tables.rs
 */
#include <stdint.h>
#include <stdio.h>

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

static void emit_u8(const char *rust_name, const uint8_t *t, int n) {
    printf("pub static %s: [u8; %d] = [\n", rust_name, n);
    for (int i = 0; i < n; ++i) {
        printf(" %u,%s", t[i], (i % 16 == 15 || i == n - 1) ? "\n" : "");
    }
    printf("];\n\n");
}

static void emit_i8(const char *rust_name, const int8_t *t, int n) {
    printf("pub static %s: [i8; %d] = [\n", rust_name, n);
    for (int i = 0; i < n; ++i) {
        printf(" %d,%s", t[i], (i % 16 == 15 || i == n - 1) ? "\n" : "");
    }
    printf("];\n\n");
}

static void emit_u32(const char *rust_name, const uint32_t *t, int n) {
    printf("pub static %s: [u32; %d] = [\n", rust_name, n);
    for (int i = 0; i < n; ++i) {
        printf(" 0x%08x,%s", t[i], (i % 8 == 7 || i == n - 1) ? "\n" : "");
    }
    printf("];\n\n");
}

static void emit_u64(const char *rust_name, const uint64_t *t, int n) {
    printf("pub static %s: [u64; %d] = [\n", rust_name, n);
    for (int i = 0; i < n; ++i) {
        printf(" 0x%016llx,%s", (unsigned long long)t[i], (i % 4 == 3 || i == n - 1) ? "\n" : "");
    }
    printf("];\n\n");
}

int main(void) {
    emit_u8   ("KMASK_IQ2XS",    kmask_iq2xs,    8);
    emit_u8   ("KSIGNS_IQ2XS",   ksigns_iq2xs,   128);
    emit_u64  ("KSIGNS64",       ksigns64,       128);
    emit_u64  ("IQ2XXS_GRID",    iq2xxs_grid,    256);
    emit_u64  ("IQ2XS_GRID",     iq2xs_grid,     512);
    emit_u64  ("IQ2S_GRID",      iq2s_grid,      1024);
    emit_u32  ("IQ3XXS_GRID",    iq3xxs_grid,    256);
    emit_u32  ("IQ3S_GRID",      iq3s_grid,      512);
    emit_i8   ("KVALUES_IQ4NL",  kvalues_iq4nl,  16);
    emit_u64  ("IQ1S_GRID",      iq1s_grid,      NGRID_IQ1S);
    return 0;
}
