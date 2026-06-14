#!/usr/bin/env python3
"""Single-image empirical terminal render — Humean impressions mode.

Each pixel is treated as a discrete empirical impression (Hume's "ideas of
sensation"), and the terminal's 24-bit color space is the canvas. No mass
scraping, no LoRA pipeline — just one image as the subject of study,
rendered at the terminal's full chromatic depth, with the act of cataloging
made visible.

Workflow:
  1. DuckDuckGo image search → first downloadable candidate held in memory.
  2. Render the image in true 24-bit color (half-block mode).
  3. Empirically log: total pixels, unique colors, top-N dominant hexes with
     live ANSI swatches.
  4. Render an HSV color wheel + grey/R/G/B ramps below so the empirical
     image and the terminal's gamut sit side by side.

By default nothing is written to disk. Use --save <dir> to keep the subject
image + a small JSON summary.
"""
import argparse
import colorsys
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

try:
    from ddgs import DDGS
except ImportError:
    sys.stderr.write("missing ddgs: pip install ddgs\n")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
import view as _view


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")
HEADERS = {"User-Agent": UA}


def fetch_first_image(query: str, candidates: int = 8) -> tuple[dict | None, bytes | None]:
    """Return (result_dict, image_bytes) for the first downloadable DDG result."""
    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=candidates))
    for r in results:
        url = r.get("image")
        if not url:
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return r, resp.content
        except (requests.RequestException, ValueError):
            continue
    return None, None


def color_wheel_image(size: int = 96) -> Image.Image:
    """HSV wheel on the left, value ramps (grey/R/G/B) on the right."""
    img = Image.new("RGB", (size * 2, size), (10, 10, 10))
    px = img.load()
    cx, cy = size // 2, size // 2
    r_max = size // 2 - 1
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= r_max:
                hue = (math.atan2(dy, dx) / (2 * math.pi) + 0.5) % 1.0
                sat = dist / r_max
                r, g, b = colorsys.hsv_to_rgb(hue, sat, 1.0)
                px[x, y] = (int(r * 255), int(g * 255), int(b * 255))
    band_h = max(1, size // 4)
    for x in range(size):
        v = int(x / (size - 1) * 255)
        for y in range(band_h):
            px[size + x, y] = (v, v, v)
        for y in range(band_h, 2 * band_h):
            px[size + x, y] = (v, 0, 0)
        for y in range(2 * band_h, 3 * band_h):
            px[size + x, y] = (0, v, 0)
        for y in range(3 * band_h, size):
            px[size + x, y] = (0, 0, v)
    return img


SPARK_BARS = "▁▂▃▄▅▆▇█"


def row_profiles(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean luminance (Rec. 709) and per-row unique color count.

    The row-by-row Humean impression: each row is a discrete perceptual event
    whose chromatic content is empirically recorded. Vectorized via numpy.
    """
    arr = np.asarray(img.convert("RGB"))                       # H × W × 3 uint8
    luma = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) / 255.0
    row_mean_luma = luma.mean(axis=1)                          # H, in [0, 1]
    packed = ((arr[..., 0].astype(np.uint32) << 16) |
              (arr[..., 1].astype(np.uint32) << 8) |
              arr[..., 2].astype(np.uint32))                   # H × W uint32
    row_unique = np.fromiter(
        (np.unique(row).size for row in packed),
        dtype=np.int32, count=packed.shape[0],
    )
    return row_mean_luma, row_unique


def downsample_to(values: np.ndarray, width: int) -> np.ndarray:
    """Average-pool a 1D array down to `width` bins. Preserves shape of profile."""
    n = len(values)
    if n <= width:
        return values.astype(np.float64)
    edges = np.linspace(0, n, width + 1, dtype=np.int64)
    return np.array([values[edges[i]:edges[i + 1]].mean() for i in range(width)])


def sparkline(values: np.ndarray, colorize: bool = False) -> str:
    """▁▂▃▄▅▆▇█ rendering of a 1D series. Optionally colored per-bar by value."""
    if len(values) == 0:
        return ""
    lo, hi = float(values.min()), float(values.max())
    span = (hi - lo) or 1e-9
    n = len(SPARK_BARS)
    parts = []
    for v in values:
        norm = (float(v) - lo) / span
        idx = max(0, min(n - 1, int(norm * (n - 1))))
        ch = SPARK_BARS[idx]
        if colorize:
            shade = int(60 + norm * 195)
            parts.append(f"\033[38;2;{shade};{shade};{shade}m{ch}")
        else:
            parts.append(ch)
    if colorize:
        parts.append("\033[0m")
    return "".join(parts)


def summarize(img: Image.Image, top_n: int = 8) -> dict:
    arr = np.asarray(img.convert("RGB")).reshape(-1, 3)
    n = int(arr.shape[0])
    packed = ((arr[:, 0].astype(np.uint32) << 16) |
              (arr[:, 1].astype(np.uint32) << 8) |
              arr[:, 2].astype(np.uint32))
    unique_vals, counts = np.unique(packed, return_counts=True)
    unique_count = int(unique_vals.size)
    top_idx = np.argsort(-counts)[:top_n]
    dominant = []
    for i in top_idx:
        c = int(unique_vals[i])
        r, g, b = (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF
        count = int(counts[i])
        dominant.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "count": count,
            "pct": round(count / n * 100, 2),
        })
    return {
        "total_pixels": n,
        "unique_colors": unique_count,
        "gamut_coverage_pct": round(unique_count / 16_777_216 * 100, 6),
        "dominant": dominant,
    }


def print_summary(summary: dict) -> None:
    sys.stdout.write("\n━━ empirical pixel log ━━\n")
    sys.stdout.write(f"  total pixels:    {summary['total_pixels']:,}\n")
    sys.stdout.write(f"  unique colors:   {summary['unique_colors']:,}\n")
    sys.stdout.write(
        f"  gamut coverage:  {summary['gamut_coverage_pct']}% of 16,777,216 (24-bit space)\n"
    )
    sys.stdout.write(f"  dominant {len(summary['dominant'])}:\n")
    for d in summary["dominant"]:
        r, g, b = d["rgb"]
        swatch = f"\033[48;2;{r};{g};{b}m      \033[0m"
        sys.stdout.write(
            f"    {swatch}  {d['hex']}  {d['pct']:5.2f}%  ({d['count']:,} px)\n"
        )
    sys.stdout.write("\n")


def main():
    p = argparse.ArgumentParser(
        description="single-image empirical terminal render — Humean impressions mode"
    )
    p.add_argument("query", nargs="*",
                   help="DuckDuckGo image query (omit if using --image or --only-wheel)")
    p.add_argument("--image", type=str, default=None,
                   help="use a local image path instead of fetching from DDG")
    p.add_argument("--only-wheel", action="store_true",
                   help="just render the terminal color gamut wheel + ramps")
    p.add_argument("--no-wheel", action="store_true",
                   help="skip the color wheel below the empirical image")
    p.add_argument("--save", type=str, default=None,
                   help="save subject image + summary JSON to this directory")
    args = p.parse_args()

    if args.only_wheel:
        sys.stdout.write(
            "━━ terminal color gamut (HSV wheel · grey / R / G / B ramps) ━━\n"
        )
        sys.stdout.write(_view.render_blocks_from_image(color_wheel_image()) + "\n")
        return

    img: Image.Image | None = None
    result_meta: dict = {}

    if args.image:
        img = Image.open(args.image)
        result_meta = {"source": "local", "path": args.image}
    else:
        if not args.query:
            sys.stderr.write("query required (or pass --image / --only-wheel)\n")
            sys.exit(1)
        query = " ".join(args.query)
        sys.stderr.write(f"🐟 fetching one image for: {query!r}\n")
        result, content = fetch_first_image(query)
        if not result or not content:
            sys.stderr.write("   no fetchable result\n")
            sys.exit(1)
        img = Image.open(io.BytesIO(content))
        result_meta = {
            "source": "duckduckgo",
            "query": query,
            "title": result.get("title"),
            "img_url": result.get("image"),
            "source_url": result.get("url"),
        }
        sys.stderr.write(f"   got: {(result_meta['title'] or '')[:70]!r}\n")
        sys.stderr.write(f"   dims: {img.size[0]}x{img.size[1]}\n")

    img = ImageOps.exif_transpose(img).convert("RGB")

    sys.stdout.write("\n━━ empirical render (24-bit, half-block) ━━\n")
    sys.stdout.write(_view.render_blocks_from_image(img) + "\n")

    # Per-row Humean impressions: luma + chromatic complexity, sparkline-compressed
    cols = max(40, min(120, _view.get_term_size()[0] - 16))
    luma, unique_per_row = row_profiles(img)
    luma_ds = downsample_to(luma, cols)
    unique_ds = downsample_to(unique_per_row.astype(np.float64), cols)
    sys.stdout.write("\n━━ row profile (each cell = one empirical row of the image) ━━\n")
    sys.stdout.write(f"  luma (Rec. 709)  {sparkline(luma_ds, colorize=True)}\n")
    sys.stdout.write(f"  chromatic Δ      {sparkline(unique_ds, colorize=True)}\n")

    summary = summarize(img)
    print_summary(summary)

    if not args.no_wheel:
        sys.stdout.write(
            "━━ terminal color gamut (HSV wheel · grey / R / G / B ramps) ━━\n"
        )
        sys.stdout.write(_view.render_blocks_from_image(color_wheel_image()) + "\n")

    if args.save:
        out_dir = Path(args.save).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        img_path = out_dir / f"subject_{ts}.png"
        img.save(img_path, "PNG", optimize=True)
        summary_path = out_dir / f"summary_{ts}.json"
        summary_path.write_text(
            json.dumps({"meta": result_meta, "summary": summary}, indent=2)
        )
        sys.stderr.write(f"💾 saved {img_path}\n💾 saved {summary_path}\n")


if __name__ == "__main__":
    main()
