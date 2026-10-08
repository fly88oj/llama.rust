#!/usr/bin/env python3
"""Generate the multimodal parity fixture images.

The same bytes are fed to the reference `llama-mtmd-cli --image <file>` and to
the port, so the vision embeddings of both sides start from an identical
bitmap.  Only the standard library is used (struct + zlib), so the script runs
anywhere python3 does.

    python3 parity/gen_mtmd_fixture.py

Writes:
    parity/mtmd-fixture.png   320x224 RGB, 8-bit, non-interlaced, filter 0
    parity/mtmd-fixture.bmp   320x224 24-bit uncompressed BMP (same pixels)

The pattern is a handful of solid rectangles plus a per-pixel ramp, so a
resampling or channel-order mistake is visible in the decoded pixels.
"""

import struct
import zlib
import os

W, H = 320, 224


def make_pixels():
    """RGB8 rows: rectangles + a deterministic gradient."""
    px = bytearray(W * H * 3)
    # background ramp (exercises the resampler everywhere)
    for y in range(H):
        for x in range(W):
            i = (y * W + x) * 3
            px[i + 0] = (x * 255) // (W - 1)
            px[i + 1] = (y * 255) // (H - 1)
            px[i + 2] = ((x + y) * 255) // (W + H - 2)

    def rect(x0, y0, x1, y1, rgb):
        for y in range(y0, y1):
            for x in range(x0, x1):
                i = (y * W + x) * 3
                px[i : i + 3] = bytes(rgb)

    rect(10, 10, 90, 70, (255, 0, 0))       # red
    rect(110, 10, 190, 70, (0, 255, 0))     # green
    rect(210, 10, 310, 70, (0, 0, 255))     # blue
    rect(10, 100, 150, 210, (255, 255, 255))  # white
    rect(170, 100, 240, 150, (0, 0, 0))     # black
    rect(250, 160, 310, 210, (255, 255, 0))  # yellow
    return bytes(px)


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png(path, px):
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter type 0 (None)
        raw += px[y * W * 3 : (y + 1) * W * 3]
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)  # 8-bit RGB
    out = b"\x89PNG\r\n\x1a\n"
    out += png_chunk(b"IHDR", ihdr)
    out += png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    out += png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(out)


def write_bmp(path, px):
    row_stride = (W * 3 + 3) // 4 * 4
    data = bytearray()
    for y in range(H - 1, -1, -1):  # BMP stores bottom-up
        for x in range(W):
            i = (y * W + x) * 3
            data += bytes((px[i + 2], px[i + 1], px[i + 0]))  # BGR
        data += b"\x00" * (row_stride - W * 3)
    file_size = 54 + len(data)
    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    info = struct.pack(
        "<IiiHHIIiiII", 40, W, H, 1, 24, 0, len(data), 2835, 2835, 0, 0
    )
    with open(path, "wb") as f:
        f.write(header + info + bytes(data))


def write_ppm(path, px):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (W, H))
        f.write(px)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    px = make_pixels()
    write_png(os.path.join(here, "mtmd-fixture.png"), px)
    write_bmp(os.path.join(here, "mtmd-fixture.bmp"), px)
    write_ppm(os.path.join(here, "mtmd-fixture.ppm"), px)
    print("wrote mtmd-fixture.{png,bmp,ppm} (%dx%d)" % (W, H))


if __name__ == "__main__":
    main()