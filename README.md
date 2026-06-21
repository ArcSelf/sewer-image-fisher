# sewer-image-fisher
<div align="center">
<img width="306" height="565" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/57727095-0496-429f-8bcb-ce0fb3da58d0" />
</div>

A four-stage image studio for the terminal: **fish** images from the open web (SEO-ranked), **preprocess** them into SDXL LoRA-ready datasets with Kohya caption sidecars, **generate** locally via HuggingFace `diffusers` (SDXL + LoRA), and **view** any result as terminal pointillism — palettes `dots` / `ascii` / `impasto` with optional jitter, or live denoising animation via `--stream`.

A fifth, lightweight mode (**empirical**) treats each pixel as a Humean impression: fetches a single image, renders it in true 24-bit color, logs per-row luminance and chromatic complexity as sparklines, and paints the terminal's own HSV gamut wheel below for reference.

[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

This is a [Claude Code](https://github.com/anthropics/claude-code) skill — drop it into `~/.claude/skills/image-fisher/` and Claude will auto-invoke it when you ask things like *"fish me images of a taco"* or *"generate me an SDXL of a rusted vespa"*. Standalone usage from the CLI works just as well.

---

## What's in the box

```
fish.py          DuckDuckGo image search, SEO-ranked (alt text, og:title, schema,
                 filename slug, dimensions, domain quality) via BeautifulSoup
                 page verification. Optional --preprocess hand-off.

preprocess.py    SDXL aspect-ratio bucketing (9 native buckets), perceptual-hash
                 dedup, EXIF orientation + RGB normalization, PNG re-encoding,
                 Kohya-format caption sidecars (.txt per image).

generate.py      Local SDXL via HuggingFace diffusers. Auto-detects CUDA → MPS →
                 CPU. Optional LoRA injection. With --stream, decodes mid-flight
                 latents through the VAE every Nth step and paints pointillism in
                 the terminal with ANSI cursor overdraw — animated denoising.

view.py          Terminal pointillism engine. Three palettes (dots / ascii /
                 impasto). --organic enables character, bucket, and color
                 jitter so the result reads as a painted canvas. Also supports
                 unicode half-blocks (24-bit) and iTerm2 / Kitty inline graphics.

empirical.py     Lightweight single-image mode. One image from DDG → 24-bit
                 half-block render → per-row luma + chromatic Δ sparklines →
                 dominant-colors log with live ANSI swatches → HSV gamut wheel
                 for terminal capability reference.
```

---

## Requirements

Python 3.10+ and a truecolor-capable terminal (iTerm2, WezTerm, Ghostty, Kitty, Alacritty, or modern macOS Terminal.app with `COLORTERM=truecolor` set).

Core dependencies:
```bash
pip install -r requirements.txt
```

For local SDXL generation (`generate.py` only), install the ML stack separately so the rest of the skill stays light:
```bash
pip install diffusers transformers accelerate safetensors
pip install torch  # Apple Silicon → MPS by default; Linux/CUDA → use the pytorch.org selector
```

---

## Quick start

### Fish + preprocess in one shot (SDXL LoRA pipeline)
```bash
python3 fish.py "rusted vespa" -k 20 --preprocess
```

### Generate locally with streaming pointillism
```bash
python3 generate.py \
  --prompt "a rusted vespa parked at golden hour, cinematic" \
  --stream --stream-interval 5 --stream-palette impasto \
  --preview-palette dots --seed 42 --steps 30
```

### View any image as pointillism
```bash
python3 view.py ./photo.jpg --palette impasto --organic
```

### Single-image empirical mode (no disk writes)
```bash
python3 empirical.py "rusted vespa golden hour"
python3 empirical.py --only-wheel       # just the terminal's color gamut
python3 empirical.py --image ./photo.jpg
```

---

## Using as a Claude Code skill

Drop the folder into `~/.claude/skills/image-fisher/`. Claude will auto-invoke when you ask things like:

| You say | What runs |
|---|---|
| *"fish me images of X"* | `fish.py` |
| *"prep these for a LoRA"* | `preprocess.py` |
| *"generate / draw / synthesize / render me an SDXL of X"* | `generate.py` |
| *"show me X as pointillism"* / *"paint it in the terminal"* | `view.py` |
| *"empirical render of X"* / *"test the terminal's color"* | `empirical.py` |

See [SKILL.md](SKILL.md) for the full auto-invocation contract and flag reference.

---

## Aesthetic note

The whole skill is built around an idea: the terminal is a canvas, not just a control surface. Pointillism is the bridge — each cell is a discrete dab of paint, each dab is a discrete empirical impression. Half-block 24-bit rendering is the most pixel-faithful mode any terminal can produce. `--organic` adds Seurat-style stochastic variation. `empirical.py` is the Humean slice: row-by-row, pixel-by-pixel, the terminal's own gamut shown beside the image as the reference canvas.

Brilliantly archaic, modernly precise.

---

## License

Licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for third-party attributions.

## Contributing

PRs welcome. By submitting, you agree your contributions are licensed under Apache 2.0 including the patent grant (Section 3).
