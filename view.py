#!/usr/bin/env python3
"""Render images in the terminal — pointillism by default.

The terminal grid is the canvas. Each cell is a "dab" whose density encodes
luminance and whose color is a 24-bit ANSI escape from the source pixel.

Modes
-----
  pointillism  Default. Maps each pixel to a character from one of three
               palettes (dots / ascii / impasto). With --organic, character,
               bucket, and color are stochastically jittered so the result
               reads like a painted canvas rather than ASCII art.
  blocks       Unicode half-blocks ▀ with fg/bg color, packing 2 vertical
               pixels per cell — photographic-leaning, no jitter.
  iterm        iTerm2 inline image protocol (true bitmap render).
  kitty        Kitty graphics protocol (true bitmap render).
  auto         Detect iterm/kitty terminal, else fall back to pointillism.

Palettes (index 0 = densest/darkest → last = sparsest/lightest)
---------------------------------------------------------------
  dots     [["●","⬤"], ["•"], ["∙","·"], ["◦","○"], [" "]]
  ascii    [["@"], ["#"], ["*"], ["+"], ["="], ["-"], [":"], ["."], [" "]]
  impasto  [["█","■"], ["▞","▚"], ["░","▒"], ["*","¤"], ["+"], ["-"], [" "]]
"""
import argparse
import base64
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps


PALETTES = {
    "dots":    [["●", "⬤"], ["•"], ["∙", "·"], ["◦", "○"], [" "]],
    "ascii":   [["@"], ["#"], ["*"], ["+"], ["="], ["-"], [":"], ["."], [" "]],
    "impasto": [["█", "■"], ["▞", "▚"], ["░", "▒"], ["*", "¤"], ["+"], ["-"], [" "]],
}

HALF_BLOCK = "▀"


# ----- terminal detection -----

def detect_terminal() -> str:
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        return "iterm"
    if os.environ.get("TERM") == "xterm-kitty" or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    return "pointillism"


def get_term_size() -> tuple[int, int]:
    try:
        s = os.get_terminal_size()
        return s.columns, s.lines
    except OSError:
        return 80, 24


def fit_to_grid(img: Image.Image, max_cols: int, max_rows: int, cell_aspect: float = 2.0) -> Image.Image:
    """Resize so the image fits inside (max_cols × max_rows) cells.
    cell_aspect = pixel_height / pixel_width of one cell (~2 for monospace)."""
    iw, ih = img.size
    target_w = max_cols
    target_h = max(1, int(target_w * (ih / iw) / cell_aspect))
    if target_h > max_rows:
        scale = max_rows / target_h
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


# ----- jitter helpers -----

def jitter_color(r: int, g: int, b: int, amount: int) -> tuple[int, int, int]:
    """Stochastic RGB jitter — pigment variation between adjacent paint dabs."""
    if amount <= 0:
        return r, g, b
    rj = max(0, min(255, r + random.randint(-amount, amount)))
    gj = max(0, min(255, g + random.randint(-amount, amount)))
    bj = max(0, min(255, b + random.randint(-amount, amount)))
    return rj, gj, bj


def jitter_bucket(idx: int, n: int, prob: float = 0.10) -> int:
    """Small chance to bump the brightness bucket ±1 — dab size varies."""
    if random.random() < prob:
        idx += random.choice((-1, 1))
    return max(0, min(n - 1, idx))


def pick_char(bucket: list[str], do_char_jitter: bool) -> str:
    """Choose canonical char or randomly from the sub-list."""
    if do_char_jitter and len(bucket) > 1:
        return random.choice(bucket)
    return bucket[0]


# ----- pointillism renderer -----

def render_pointillism_from_image(
    img: Image.Image,
    palette: str = "dots",
    max_cols: int | None = None,
    max_rows: int | None = None,
    mono: bool = False,
    jitter_chars: bool = False,
    jitter_buckets: bool = False,
    color_jitter: int = 0,
    seed: int | None = None,
) -> str:
    """Render a PIL image directly — no file I/O. Used by the streaming callback."""
    if seed is not None:
        random.seed(seed)

    cols, rows = get_term_size()
    max_cols = max_cols or min(cols - 2, 96)
    max_rows = max_rows or max(8, rows - 4)

    palette_buckets = PALETTES[palette]
    n_buckets = len(palette_buckets)

    img = ImageOps.exif_transpose(img).convert("RGB")
    img = fit_to_grid(img, max_cols, max_rows, cell_aspect=2.0)
    px = img.load()
    w, h = img.size

    out_lines = []
    for y in range(h):
        parts = []
        for x in range(w):
            r, g, b = px[x, y]
            lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0  # Rec. 709
            idx = int(lum * (n_buckets - 1))
            if jitter_buckets:
                idx = jitter_bucket(idx, n_buckets)
            ch = pick_char(palette_buckets[idx], jitter_chars)
            if mono:
                parts.append(ch)
            else:
                jr, jg, jb = jitter_color(r, g, b, color_jitter)
                parts.append(f"\033[38;2;{jr};{jg};{jb}m{ch}")
        if not mono:
            parts.append("\033[0m")
        out_lines.append("".join(parts))
    return "\n".join(out_lines)


def render_pointillism(
    path: Path,
    palette: str = "dots",
    max_cols: int | None = None,
    max_rows: int | None = None,
    mono: bool = False,
    jitter_chars: bool = False,
    jitter_buckets: bool = False,
    color_jitter: int = 0,
    seed: int | None = None,
) -> str:
    img = Image.open(path)
    return render_pointillism_from_image(
        img, palette, max_cols, max_rows, mono,
        jitter_chars, jitter_buckets, color_jitter, seed,
    )


# ----- other renderers -----

def render_blocks_from_image(img: Image.Image, max_cols: int | None = None, max_rows: int | None = None) -> str:
    """Half-block 24-bit render of an in-memory PIL image — packs 2 vertical px per cell."""
    cols, rows = get_term_size()
    max_cols = max_cols or min(cols - 2, 120)
    max_rows = max_rows or max(8, rows - 4)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = fit_to_grid(img, max_cols, max_rows * 2, cell_aspect=1.0)
    w, h = img.size
    if h % 2:
        pad = Image.new("RGB", (w, h + 1), (0, 0, 0))
        pad.paste(img, (0, 0))
        img = pad
        h += 1
    px = img.load()
    out_lines = []
    for y in range(0, h, 2):
        parts = []
        for x in range(w):
            r1, g1, b1 = px[x, y]
            r2, g2, b2 = px[x, y + 1]
            parts.append(
                f"\033[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m{HALF_BLOCK}"
            )
        parts.append("\033[0m")
        out_lines.append("".join(parts))
    return "\n".join(out_lines)


def render_blocks(path: Path, max_cols: int | None = None, max_rows: int | None = None) -> str:
    return render_blocks_from_image(Image.open(path), max_cols, max_rows)


def render_iterm(path: Path) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"\033]1337;File=inline=1;preserveAspectRatio=1:{b64}\a"


def render_kitty(path: Path) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data)
    chunks = [b64[i:i + 4096] for i in range(0, len(b64), 4096)]
    out = []
    for i, chunk in enumerate(chunks):
        m = 1 if i < len(chunks) - 1 else 0
        prefix = "\033_Ga=T,f=100," if i == 0 else "\033_G"
        out.append(f"{prefix}m={m};{chunk.decode()}\033\\")
    return "".join(out)


# ----- dispatch -----

def view(
    path: Path,
    mode: str = "auto",
    palette: str = "dots",
    max_cols=None,
    max_rows=None,
    mono: bool = False,
    jitter_chars: bool = False,
    jitter_buckets: bool = False,
    color_jitter: int = 0,
    seed: int | None = None,
) -> str:
    if mode == "auto":
        mode = detect_terminal()
    if mode == "iterm":
        return render_iterm(path)
    if mode == "kitty":
        return render_kitty(path)
    if mode == "blocks":
        return render_blocks(path, max_cols, max_rows)
    return render_pointillism(
        path, palette, max_cols, max_rows, mono,
        jitter_chars, jitter_buckets, color_jitter, seed,
    )


def collect_images(target: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in exts)
    return [target]


def main():
    p = argparse.ArgumentParser(
        description="Render images in the terminal — pointillism by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  view.py taco.png\n"
            "  view.py taco.png --palette ascii\n"
            "  view.py taco.png --palette impasto --organic\n"
            "  view.py taco.png --mode blocks\n"
            "  view.py ./folder --top 3 --organic --seed 42\n"
        ),
    )
    p.add_argument("path", help="image file or directory")
    p.add_argument("--mode", choices=["auto", "pointillism", "blocks", "iterm", "kitty"],
                   default="pointillism", help="render mode (default: pointillism)")
    p.add_argument("--palette", choices=list(PALETTES.keys()), default="dots",
                   help="pointillism palette (default: dots)")
    p.add_argument("--organic", action="store_true",
                   help="shortcut: enable character + bucket + color jitter (canvas feel)")
    p.add_argument("--jitter-chars", action="store_true",
                   help="randomly pick alternates inside each brightness bucket")
    p.add_argument("--jitter-buckets", action="store_true",
                   help="small chance to shift the brightness bucket ±1 per cell")
    p.add_argument("--jitter-color", type=int, default=0, metavar="N",
                   help="per-cell RGB jitter ±N (try 6–10; 0 disables)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed the RNG for reproducible organic renders")
    p.add_argument("--cols", type=int, default=None, help="max columns (default: terminal width)")
    p.add_argument("--rows", type=int, default=None, help="max rows (default: terminal height - 4)")
    p.add_argument("--mono", action="store_true", help="[pointillism] disable color")
    p.add_argument("--top", type=int, default=None,
                   help="if path is a dir, render only the first N images")
    args = p.parse_args()

    # --organic = all three jitters on with sensible defaults
    jitter_chars = args.jitter_chars or args.organic
    jitter_buckets = args.jitter_buckets or args.organic
    color_jitter = args.jitter_color
    if args.organic and color_jitter == 0:
        color_jitter = 8

    target = Path(args.path).expanduser()
    imgs = collect_images(target)
    if args.top:
        imgs = imgs[:args.top]
    if not imgs:
        sys.stderr.write(f"no images found in {target}\n")
        sys.exit(1)

    for img in imgs:
        if len(imgs) > 1:
            print(f"\n━━ {img.name} ━━")
        print(view(
            img,
            mode=args.mode,
            palette=args.palette,
            max_cols=args.cols,
            max_rows=args.rows,
            mono=args.mono,
            jitter_chars=jitter_chars,
            jitter_buckets=jitter_buckets,
            color_jitter=color_jitter,
            seed=args.seed,
        ))


if __name__ == "__main__":
    main()
