---
name: image-fisher
description: A closed-loop image studio. (1) Fish the web for images via DuckDuckGo + BeautifulSoup, SEO-ranked. (2) Preprocess into SDXL LoRA-ready PNGs with Kohya caption sidecars. (3) Generate locally via HuggingFace diffusers (SDXL + optional LoRA injection). (4) Render any result as terminal pointillism — palettes dots/ascii/impasto with --organic jitter, or live denoising animation via --stream. Invoke when the user wants to fish/find/get/source images ("fish me images of X"), preprocess a folder for LoRA training, generate images locally from a prompt ("generate / draw / synthesize / render me an SDXL of X"), or view any image in the terminal ("show me X as pointillism" / "paint it in the terminal").
---

# image-fisher

Find images on the open web that **empirically match a query** — ranked the way an SEO-aware crawler would rank them. The header "taco" should produce an image whose alt text, page title, filename slug, and Open Graph metadata all converge on "taco".

## When to invoke

- "Fish me images of X" / "find images of X" / "get a picture of X"
- The user describes a concept and wants its empirical visual form
- Image sourcing for slides, references, mockups, mood boards

## When NOT to invoke

- The user has a single known URL — just fetch it directly
- The user wants to *generate* an image (use a generation tool, not this)
- The user wants Figma assets (use the Figma MCP)

## How to run it

```bash
# default: 30 candidates, keep top 10
python3 ~/.claude/skills/image-fisher/fish.py "taco"

# keep top 5
python3 ~/.claude/skills/image-fisher/fish.py "vintage typewriter" -k 5

# fast mode (skip page fetches — less accurate but ~10x faster)
python3 ~/.claude/skills/image-fisher/fish.py "art deco poster" --no-verify

# fish + preprocess in one shot (SDXL LoRA-ready PNGs + captions + dedup)
python3 ~/.claude/skills/image-fisher/fish.py "rusted vespa" -k 20 --preprocess

# emit manifest JSON to stdout (for piping)
python3 ~/.claude/skills/image-fisher/fish.py "labrador puppy" --json
```

## Preprocessing (SDXL LoRA pipeline)

`preprocess.py` is also a standalone CLI — point it at any directory of images
(including a previous fish run, or your own pile):

```bash
python3 ~/.claude/skills/image-fisher/preprocess.py -i ~/Pythonic/image-fisher/out/taco_20260614_015000
python3 ~/.claude/skills/image-fisher/preprocess.py -i ./my-refs -o ./my-refs-processed
python3 ~/.claude/skills/image-fisher/preprocess.py -i ./pile --no-dedup --min-size 768
```

Per image, it:
1. Perceptual-hash dedups (default ON; `--no-dedup` to disable). Uses `imagehash.phash` if installed, else falls back to an inline 8×8 average hash.
2. Strict min-size filter — **both dims must be ≥ 512** by default (`--min-size N` to change).
3. Honors EXIF orientation.
4. Converts CMYK / RGBA / grayscale / P → RGB (white matte under transparency).
5. Picks the closest SDXL aspect-ratio bucket (1024², 1152×896, 1216×832, … 640×1536) and does a center-crop fit in LANCZOS.
6. Writes the result as PNG + a matching `.txt` caption sidecar (Kohya format).

Caption fallback chain (first non-empty wins): DDG title → page `<img alt>` → `og:title` → page `<h1>` → schema.org `ImageObject` name → cleaned filename slug. All captions get lowercased, web suffixes stripped (` | Site`, ` - Blog`, ` — WordPress`, ` at Foo blog`, `(1920x1080)`), and phrase separators normalized to commas.

Output layout (when run from fish or standalone):
```
<input_dir>/
  processed/
    <slug>.png
    <slug>.txt        # Kohya caption sidecar
    ...
    preprocess.json   # per-image report: status, bucket, caption, drop reasons
```

## Viewing in the terminal — pointillism

`view.py` paints images directly into the terminal grid as pointillism: each cell is a "dab" whose character density encodes luminance and whose color is a 24-bit ANSI escape from the source pixel. Step back from the screen (or zoom out) and the dots optically merge into the image.

```bash
# default: dots palette, no jitter
python3 ~/.claude/skills/image-fisher/view.py taco.png

# the canvas feel: char + bucket + color jitter (all three at once)
python3 ~/.claude/skills/image-fisher/view.py taco.png --palette impasto --organic

# the three palettes
python3 ~/.claude/skills/image-fisher/view.py taco.png --palette dots      # ●⬤•∙·◦○  — Seurat stipple
python3 ~/.claude/skills/image-fisher/view.py taco.png --palette ascii     # @#*+=-:. — classic weights
python3 ~/.claude/skills/image-fisher/view.py taco.png --palette impasto   # █■▞▚░▒*¤ — chunky brushstroke

# inline graphics (true bitmap) if you're in iTerm2 or Kitty
python3 ~/.claude/skills/image-fisher/view.py taco.png --mode iterm
python3 ~/.claude/skills/image-fisher/view.py taco.png --mode blocks       # half-block ▀ true-color

# a folder, top 3, organic, reproducible
python3 ~/.claude/skills/image-fisher/view.py ./folder --top 3 --organic --seed 42
```

Jitter flags (all imply `--palette pointillism`-family):
- `--organic` — one-flag shortcut: enables all three jitters with sensible defaults
- `--jitter-chars` — randomly pick alternates inside each brightness bucket
- `--jitter-buckets` — ~10% chance to bump the bucket ±1 per cell (dab size varies)
- `--jitter-color N` — per-cell RGB jitter ±N (try 6-10; pigment mixing variance)
- `--seed N` — reproducible "organic" renders

When showing a fished image to the user, default to `--palette dots --organic` — that's the brilliantly-archaic-yet-modern look.

## Local SDXL generation (closed loop)

`generate.py` runs SDXL locally via HuggingFace `diffusers`, then hands the final PNG straight into `view.py` for a pointillism preview. With `--stream`, every Nth denoising step is decoded mid-flight through the VAE and painted as pointillism — you watch denoising resolve, frame by frame, with ANSI cursor overdraw.

Trigger keywords (auto-invoke): *generate*, *draw me*, *synthesize*, *render me an SDXL of*, *paint me a*, *make an image of* (no web — purely model output).

```bash
# basic gen → final pointillism preview
python3 ~/.claude/skills/image-fisher/generate.py --prompt "a rusted vespa in golden hour"

# stream denoising live (impasto palette during steps, dots for the final)
python3 ~/.claude/skills/image-fisher/generate.py \
  --prompt "a rusted vespa in golden hour" \
  --stream --stream-interval 5 --stream-palette impasto --preview-palette dots

# test a LoRA you trained
python3 ~/.claude/skills/image-fisher/generate.py \
  --prompt "a rusted vespa in golden hour, <lora_token>" \
  --lora ~/loras/vespa_v1.safetensors --lora-scale 0.85

# reproducible + smaller for quick iteration
python3 ~/.claude/skills/image-fisher/generate.py \
  --prompt "..." --seed 42 --width 768 --height 768 --steps 20
```

Device handling is automatic: CUDA → MPS → CPU. fp16 on GPU, fp32 on CPU.

**Heavy dependencies** (only needed for `generate.py`; commented in `requirements.txt`):
```bash
pip install diffusers transformers accelerate safetensors
# torch: install the right wheel for your backend (MPS on Mac, CUDA on Linux)
pip install torch  # Apple Silicon: this gives you MPS-capable torch by default
```

First run downloads ~7GB of SDXL base weights to `~/.cache/huggingface/`. On MPS expect ~60-90s per 1024² generation; streaming previews add a per-VAE-decode tax (~2-3s every 5 steps).

## When NOT to invoke generate.py

- The user wants to fish *existing* images from the web (use `fish.py`)
- The user wants a Figma asset or a UI mockup (use the Figma MCP)
- The user wants a sketch / SVG / chart (different tools)

## Empirical (Humean) single-image mode

`empirical.py` is the lightweight, storage-free counterpart to the full fishing pipeline. Treats each pixel as a discrete empirical impression (Hume's "ideas of sensation") and exercises the terminal's true 24-bit color gamut. Single image, in-memory only, no disk writes unless `--save` is passed.

Trigger keywords: *empirical*, *Humean*, *single image*, *test the terminal's color*, *show me one*, *log every pixel*.

```bash
# fetch one image from DDG and render with empirical log + row sparklines + color wheel
python3 ~/.claude/skills/image-fisher/empirical.py "rusted vespa"

# local image instead of DDG
python3 ~/.claude/skills/image-fisher/empirical.py --image ./photo.jpg

# just the terminal's color wheel + gamut ramps (no image)
python3 ~/.claude/skills/image-fisher/empirical.py --only-wheel

# save the subject + summary JSON
python3 ~/.claude/skills/image-fisher/empirical.py "rusted vespa" --save ~/empirical
```

Output:
1. **Empirical render** — half-block 24-bit (most pixel-faithful terminal mode)
2. **Row profile sparklines** — per-row mean luma (Rec. 709) + per-row unique-color count, colored `▁▂▃▄▅▆▇█` bars; one bin per image row, average-pooled to terminal width
3. **Pixel log** — total pixels, unique colors, % of 24-bit gamut covered, top-8 dominant colors with live ANSI swatches + hex codes
4. **Color gamut wheel** — programmatic HSV wheel + grey/R/G/B ramps for terminal capability reference (skip with `--no-wheel`)

If `ddgs` / `bs4` are missing:
```bash
pip install -r ~/.claude/skills/image-fisher/requirements.txt
```

Or reuse the getting-news venv:
```bash
source /Users/ArcSelf/Pythonic/getting-news/.venv/bin/activate 2>/dev/null && \
  python3 ~/.claude/skills/image-fisher/fish.py "<query>"
```

## What the SEO scoring looks at

For each candidate image, `fish.py`:

1. Hits the source page with BeautifulSoup
2. Extracts these signals and tokenizes them against the query:
   - **alt** on the matching `<img>` tag
   - **`<title>`** and **`<h1>`** of the page
   - **`og:title`** meta tag (and `og:image` to confirm canonical)
   - **schema.org `ImageObject.name` / `.caption`** in JSON-LD
   - **filename slug** of the image URL
3. Weights signals so that *concurring* signals (alt + og:title + h1 all say "taco") rank highest
4. Penalizes: thumbnail dimensions, known low-quality hotlink domains

The full scoring weights are in `fish.py` — adjust if a category over- or under-fires.

## Output

```
~/Pythonic/image-fisher/out/<slug>_<YYYYMMDD_HHMMSS>/
  01_<slug>.jpg
  02_<slug>.png
  ...
  manifest.json    # ranked, with per-candidate signal breakdown
```

## Reporting back to the user

After running, tell the user:
- The output folder path (and offer to `open` it in Finder)
- Top 3-5 picks with a one-line reason each (e.g. *"alt text and og:title both say 'taco', 1200×800"*)

Do NOT dump the raw manifest JSON unless asked — summarize.
