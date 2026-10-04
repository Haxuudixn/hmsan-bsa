"""Functional check: LRU eviction must not break image reads."""
import io, sys, tempfile, zipfile
from pathlib import Path
sys.path.insert(0, r"C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa\src")
from PIL import Image
import bid_slicing.data.dataset as ds

tmp = Path(tempfile.mkdtemp(prefix="ziplru_"))
n_zips = ds._ZIP_CACHE_MAX + 4           # strictly more than the cache can hold
zips = []
for i in range(n_zips):
    p = tmp / f"z{i}.zip"
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(i * 20 % 255, 0, 0)).save(buf, format="PNG")
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(f"blocks/block_{i:04d}.png", buf.getvalue())
    zips.append(str(p))

# Read every member once: this forces eviction of the earlier handles.
for i, zp in enumerate(zips):
    im = ds._read_image(zp, f"blocks/block_{i:04d}.png")
    assert im is not None and im.size == (16, 16), f"first pass failed at {i}"
assert len(ds._ZIP_CACHE) == ds._ZIP_CACHE_MAX, ds._ZIP_CACHE
print(f"pass 1: {n_zips} zips read, cache bounded at "
      f"{len(ds._ZIP_CACHE)}/{ds._ZIP_CACHE_MAX}")

# Read them again (all now evicted): ZipFile.open must transparently reopen.
for i, zp in enumerate(zips):
    im = ds._read_image(zp, f"blocks/block_{i:04d}.png")
    assert im is not None, f"re-read failed at {i}"
print(f"pass 2: all {n_zips} re-read successfully after eviction")

# Missing member and missing file must still degrade to None, not raise.
assert ds._read_image(zips[0], "blocks/nope.png") is None
assert ds._read_image(str(tmp / "missing.zip"), "blocks/block_0000.png") is None
assert ds._read_image("", "") is None
print("error paths: missing member / missing file / empty args all return None")

# Reuse of a cached handle returns the identical object.
a, b = ds._get_zip(zips[0]), ds._get_zip(zips[0])
assert a is b, "cache miss on a handle that is still resident"
print("handle reuse: cached ZipFile returned without reopening")
print("OK")