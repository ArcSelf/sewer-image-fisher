#!/usr/bin/env python3
"""SDXL LoRA-ready preprocessing for fished images.

Per-image pipeline:
  1. Perceptual-hash dedup (default ON; --no-dedup to disable)
  2. Strict min-size filter: BOTH dims must be ≥ min_size (default 512)
  3. EXIF orientation honored (ImageOps.exif_transpose)
  4. CMYK / RGBA / grayscale / P → RGB (white matte for transparency)
  5. Closest SDXL aspect-ratio bucket → resize + center-crop (LANCZOS)
  6. Write PNG + matching .txt caption sidecar (Kohya format)

Captions (fallback chain): DDG title → page <img alt> → og:title → page <h1>
  → schema_name → filename slug. Light cleanup: strip web suffixes,
  lowercase, comma-separate phrases.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps


SDXL_BUCKETS = [
    (1024, 1024),  # 1:1
    (1152, 896),   # 9:7
    (1216, 832),   # 19:13
    (1344, 768),   # 7:4
    (1536, 640),   # 12:5
    (896, 1152),   # 7:9
    (832, 1216),   # 13:19
    (768, 1344),   # 4:7
    (640, 1536),   # 5:12
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp", ".tiff"}

CAPTION_SUFFIX_PATTERNS = [
    r"\s*\|\s*[^|]+$",                       # " | Site Name"
    r"\s+[-–—]\s+[^-–—]+$",                  # " - Site Name" or " — Site"
    r"\s+at\s+[A-Z][\w\s]+\s+blog\b.*$",     # " at Judith Smith blog"
    r"\s*\(\d+\s*[x×]\s*\d+\)\s*$",          # "(1920x1080)"
    r"\s*[-–—]\s*wordpress.*$",              # " — WordPress"
    r"\s*[-–—]\s*pinterest.*$",
    r"\s*on\s+pinterest\b.*$",
]


def closest_bucket(w: int, h: int) -> tuple[int, int]:
    aspect = w / h
    return min(SDXL_BUCKETS, key=lambda b: abs(aspect - b[0] / b[1]))


def clean_caption(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    for pat in CAPTION_SUFFIX_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = s.lower()
    s = re.sub(r"\s+[-–—]\s+", ", ", s)
    s = re.sub(r"\s*[|:]\s*", ", ", s)
    s = re.sub(r"\.\s+", ", ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*(?:,\s*)+", ", ", s)
    s = re.sub(r"^[,\s]+|[,\s]+$", "", s)
    return s


def slug_from_filename(stem: str) -> str:
    s = re.sub(r"^\d+_", "", stem)
    s = re.sub(r"[_\-]+", " ", s)
    return clean_caption(s)


def pick_caption(entry: dict) -> str:
    page = entry.get("page_signals") or {}
    candidates = [
        entry.get("title"),
        page.get("img_alt"),
        page.get("og_title"),
        page.get("h1"),
        page.get("schema_name"),
    ]
    for c in candidates:
        cleaned = clean_caption(c or "")
        if cleaned:
            return cleaned
    p = entry.get("downloaded_to") or ""
    if p:
        return slug_from_filename(Path(p).stem)
    return ""


def load_manifest_index(input_dir: Path) -> dict:
    mpath = input_dir / "manifest.json"
    if not mpath.exists():
        return {}
    try:
        data = json.loads(mpath.read_text())
    except json.JSONDecodeError:
        return {}
    return {
        Path(e["downloaded_to"]).name: e
        for e in data.get("results", [])
        if e.get("downloaded_to")
    }


# ---------- dedup ----------

try:
    import imagehash  # type: ignore
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


def _ahash_fallback(img: Image.Image) -> int:
    """8x8 average hash → 64-bit int."""
    gs = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(gs.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def perceptual_hash(img: Image.Image):
    if HAS_IMAGEHASH:
        return ("phash", imagehash.phash(img))
    return ("ahash", _ahash_fallback(img))


def hashes_close(h1, h2, threshold: int) -> bool:
    kind1, v1 = h1
    kind2, v2 = h2
    if kind1 != kind2:
        return False
    if kind1 == "phash":
        return (v1 - v2) <= threshold
    return bin(v1 ^ v2).count("1") <= threshold


# ---------- main pipeline ----------

def process_one(src: Path, out_dir: Path, caption: str, min_size: int) -> dict:
    info = {
        "src": str(src), "dst": None, "status": None, "reason": None,
        "bucket": None, "in_dims": None,
    }
    try:
        with Image.open(src) as raw:
            img = ImageOps.exif_transpose(raw)
            w, h = img.size
            info["in_dims"] = f"{w}x{h}"
            if w < min_size or h < min_size:
                info["status"] = "skipped"
                info["reason"] = f"strict min-size: need both ≥ {min_size}, got {w}x{h}"
                return info
            if img.mode != "RGB":
                if img.mode in ("RGBA", "LA", "P"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    rgba = img.convert("RGBA")
                    bg.paste(rgba, mask=rgba.split()[-1])
                    img = bg
                else:
                    img = img.convert("RGB")
            bw, bh = closest_bucket(w, h)
            info["bucket"] = f"{bw}x{bh}"
            out = ImageOps.fit(
                img, (bw, bh), Image.Resampling.LANCZOS, centering=(0.5, 0.5)
            )
            base_stem = re.sub(r"^\d+_", "", src.stem)
            dst = out_dir / f"{base_stem}.png"
            n = 1
            while dst.exists():
                dst = out_dir / f"{base_stem}_{n}.png"
                n += 1
            out.save(dst, "PNG", optimize=True)
            dst.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
            info["dst"] = str(dst)
            info["status"] = "ok"
    except Exception as e:
        info["status"] = "error"
        info["reason"] = f"{type(e).__name__}: {e}"
    return info


def process_dir(
    input_dir,
    output_dir=None,
    min_size: int = 512,
    dedup: bool = True,
    dedup_threshold: int = 5,
) -> dict:
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"{input_dir} is not a directory")
    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (input_dir / "processed").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_idx = load_manifest_index(input_dir)
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    sys.stderr.write(f"🪣 preprocessing {len(files)} images → {out_dir}\n")
    if dedup:
        method = "phash" if HAS_IMAGEHASH else "ahash-fallback"
        sys.stderr.write(f"   dedup: ON ({method}, threshold={dedup_threshold})\n")
    else:
        sys.stderr.write("   dedup: OFF\n")

    dedupe_drops: set[str] = set()
    if dedup:
        kept = []
        for src in files:
            try:
                with Image.open(src) as raw:
                    img = ImageOps.exif_transpose(raw).convert("RGB")
                    h = perceptual_hash(img)
            except Exception:
                continue
            if any(hashes_close(h, kh, dedup_threshold) for kh in kept):
                dedupe_drops.add(src.name)
            else:
                kept.append(h)

    results = []
    counts = {"ok": 0, "skipped": 0, "duplicate": 0, "error": 0}
    for i, src in enumerate(files, 1):
        if src.name in dedupe_drops:
            results.append({
                "src": str(src), "status": "duplicate",
                "reason": "near-duplicate of earlier image",
            })
            counts["duplicate"] += 1
            sys.stderr.write(f"   [{i:02d}/{len(files)}] ⊘   dup       {src.name}\n")
            continue
        entry = manifest_idx.get(src.name, {})
        caption = pick_caption(entry) if entry else slug_from_filename(src.stem)
        info = process_one(src, out_dir, caption, min_size)
        info["caption"] = caption
        results.append(info)
        counts[info["status"]] = counts.get(info["status"], 0) + 1
        marker = {"ok": "✓", "skipped": "·", "error": "✗"}.get(info["status"], "?")
        sys.stderr.write(
            f"   [{i:02d}/{len(files)}] {marker}  {(info['bucket'] or '-'):>9}  {src.name}\n"
        )

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "min_size": min_size,
        "dedup": dedup,
        "dedup_threshold": dedup_threshold,
        "dedup_method": "phash" if HAS_IMAGEHASH else "ahash-fallback",
        "counts": counts,
        "results": results,
    }
    (out_dir / "preprocess.json").write_text(json.dumps(report, indent=2))
    sys.stderr.write(
        f"\n✅ ok={counts['ok']}  duplicate={counts['duplicate']}  "
        f"skipped={counts['skipped']}  errors={counts['error']}\n   {out_dir}\n"
    )
    return report


def main():
    p = argparse.ArgumentParser(
        description="SDXL LoRA-ready preprocessing for fished images."
    )
    p.add_argument("--input-dir", "-i", required=True,
                   help="folder of source images (e.g. a fisher output)")
    p.add_argument("--output-dir", "-o", default=None,
                   help="output folder (default: <input>/processed)")
    p.add_argument("--min-size", type=int, default=512,
                   help="strict: BOTH dims must be ≥ this px (default 512)")
    p.add_argument("--no-dedup", action="store_true",
                   help="disable perceptual-hash dedup")
    p.add_argument("--dedup-threshold", type=int, default=5,
                   help="Hamming threshold for dedup (default 5)")
    args = p.parse_args()
    process_dir(
        Path(args.input_dir),
        Path(args.output_dir) if args.output_dir else None,
        args.min_size,
        dedup=not args.no_dedup,
        dedup_threshold=args.dedup_threshold,
    )


if __name__ == "__main__":
    main()
