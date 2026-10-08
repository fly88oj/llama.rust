#!/usr/bin/env python3
"""gguf_pin_gemma4_assistant.py — re-pin a newer-convert gemma4-assistant
GGUF onto the pinned revision's naming (bd4f514db1).

The local head file (gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf, the AtomicChat
convert) uses the NEWER upstream naming — arch `gemma4_assistant`,
`{arch}.n_embd_backbone`, tensors `mtp.pre_projection` / `mtp.post_projection`,
no `nextn_predict_layers` — while the pinned reference expects arch
`gemma4-assistant` (llama-arch.cpp:60), reads `{arch}.embedding_length_out`
(llama-arch.cpp:194) and `{arch}.nextn_predict_layers` (llama-arch.cpp:223)
and creates the projections as `nextn.pre_projection` / `nextn.post_projection`
(llama-arch.cpp:580-581). This script rewrites ONLY the header (kv + tensor
names/offsets); the tensor data bytes are copied verbatim to the new data
section.

usage: python3 parity/gguf_pin_gemma4_assistant.py <in.gguf> <out.gguf>
"""
import struct
import sys

SZC = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B"}
ALIGN = 32  # GGML_DEFAULT_ALIGNMENT


def rstr(f):
    (l,) = struct.unpack("<Q", f.read(8))
    return f.read(l).decode(errors="replace")


def rd_kv(f):
    k = rstr(f)
    (t,) = struct.unpack("<I", f.read(4))
    if t in FMT:
        v = ("scalar", t, struct.unpack(FMT[t], f.read(SZC[t]))[0])
    elif t == 8:
        v = ("str", t, rstr(f))
    elif t == 10:
        v = ("scalar", t, struct.unpack("<Q", f.read(8))[0])
    elif t == 11:
        v = ("scalar", t, struct.unpack("<q", f.read(8))[0])
    elif t == 12:
        v = ("scalar", t, struct.unpack("<d", f.read(8))[0])
    elif t == 9:
        et, = struct.unpack("<I", f.read(4))
        n, = struct.unpack("<Q", f.read(8))
        if et == 8:
            v = ("arr", t, (et, [rstr(f) for _ in range(n)]))
        else:
            v = ("arr", t, (et, [struct.unpack(FMT[et], f.read(SZC[et]))[0] for _ in range(n)]))
    else:
        raise Exception(f"kv type {t}")
    return k, v


def wstr(out, s):
    b = s.encode()
    out += struct.pack("<Q", len(b))
    out += b
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    f = open(src, "rb")
    assert f.read(4) == b"GGUF"
    ver, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))
    kvs = []
    for _ in range(n_kv):
        kvs.append(rd_kv(f))
    tensors = []
    for _ in range(n_tensors):
        name = rstr(f)
        ne_n, = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{min(ne_n, 4)}q", f.read(8 * min(ne_n, 4)))
        if ne_n > 4:
            f.read(8 * (ne_n - 4))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append((name, ne_n, dims, ty, off))
    data_offset = f.tell()
    data_offset = (data_offset + ALIGN - 1) & ~(ALIGN - 1)
    f.seek(data_offset)
    blob = f.read()

    # ---- the pinned-renaming pass ----
    kvs2 = []
    for k, (kind, t, v) in kvs:
        if k == "general.architecture":
            assert v == "gemma4_assistant", v
            v = "gemma4-assistant"
        elif k.startswith("gemma4_assistant."):
            k = "gemma4-assistant." + k[len("gemma4_assistant."):]
        if k == "gemma4-assistant.n_embd_backbone":
            # superseded by the pinned embedding_length_out below
            continue
        kvs2.append((k, (kind, t, v)))
    # the pinned keys the newer converter does not write
    have = {k for k, _ in kvs2}
    backbone = next(
        v for k, (kind, t, v) in kvs if k == "gemma4_assistant.n_embd_backbone"
    )
    assert isinstance(backbone, int)
    kvs2.append(("gemma4-assistant.embedding_length_out", ("scalar", 4, backbone)))
    n_layers = next(
        v for k, (kind, t, v) in kvs if k == "gemma4_assistant.block_count"
    )
    kvs2.append(("gemma4-assistant.nextn_predict_layers", ("scalar", 4, n_layers)))

    tensors2 = []
    for (name, ne_n, dims, ty, off) in tensors:
        if name == "mtp.pre_projection.weight":
            name = "nextn.pre_projection.weight"
        elif name == "mtp.post_projection.weight":
            name = "nextn.post_projection.weight"
        tensors2.append((name, ne_n, dims, ty, off))

    # ---- write the patched file ----
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", ver)
    out += struct.pack("<Q", len(tensors2))
    out += struct.pack("<Q", len(kvs2))
    for k, (kind, t, v) in kvs2:
        out = wstr(out, k)
        out += struct.pack("<I", t)
        if kind == "scalar":
            out += struct.pack(FMT[t], v)
        elif kind == "str":
            out = wstr(out, v)
        else:
            # array: the element type travels with the parsed value
            et, vals = v
            out += struct.pack("<I", et)
            out += struct.pack("<Q", len(vals))
            for e in vals:
                if et == 8:
                    out = wstr(out, e)
                else:
                    out += struct.pack(FMT[et], e)
    # place the data section right after a 32B-aligned header end
    head_end = len(out) + sum(
        8 + len(n.encode()) + 4 + 8 * min(ne, 4) + 8 * max(0, ne - 4) + 4 + 8
        for (n, ne, _, _, _) in tensors2
    )
    new_data = (head_end + ALIGN - 1) & ~(ALIGN - 1)
    for (name, ne_n, dims, ty, off) in tensors2:
        out = wstr(out, name)
        out += struct.pack("<I", ne_n)
        out += struct.pack(f"<{min(ne_n, 4)}q", *dims)
        out += b"\x00" * (8 * max(0, ne_n - 4))
        out += struct.pack("<I", ty)
        out += struct.pack("<Q", off)  # relative to the (aligned) data start
    out += b"\x00" * (new_data - len(out))
    assert len(out) == new_data
    out += blob
    open(dst, "wb").write(bytes(out))
    print(
        f"pinned {src} -> {dst}: {len(kvs2)} kv, {len(tensors2)} tensors, "
        f"data at {new_data} (was {data_offset}), {len(blob)} bytes verbatim"
    )


if __name__ == "__main__":
    main()
