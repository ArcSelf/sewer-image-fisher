#!/usr/bin/env python3
"""Image fisher: SEO-ranked DuckDuckGo image search + BeautifulSoup page verification.

For each candidate image, fetches its source page and scores how strongly the page's
SEO signals (alt text, <title>, <h1>, og:title, schema.org ImageObject, filename slug)
converge on the query. Top results are downloaded with a manifest.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    sys.stderr.write(
        "Missing deps. Install with:\n"
        "  pip install -r ~/.claude/skills/image-fisher/requirements.txt\n"
    )
    sys.exit(1)


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
HEADERS = {"User-Agent": UA}
PAGE_TIMEOUT = 8
DOWNLOAD_TIMEOUT = 15
RATE_LIMIT_SEC = 0.35
STOPWORDS = {"a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "with", "by", "at", "from"}

LOW_QUALITY_DOMAINS = (
    "pinterest.",
    "i.pinimg.",
    "lookaside.fbsbx",
    "encrypted-tbn",
    "gstatic.",
    "googleusercontent.",
)

SIGNAL_WEIGHTS = {
    "alt_match":        1.5,
    "filename_match":   1.0,
    "page_title_match": 1.2,
    "page_h1_match":    1.2,
    "og_title_match":   1.5,
    "schema_match":     1.0,
    "page_alt_match":   1.3,
    "dim_quality":      0.8,
    "domain_quality":   0.5,
}


def tokenize(s: str) -> set:
    if not s:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOPWORDS and len(t) > 1}


def overlap(query_tokens: set, signal_text: str) -> float:
    if not query_tokens:
        return 0.0
    sig = tokenize(signal_text)
    return len(query_tokens & sig) / len(query_tokens)


def dim_score(width: int, height: int) -> float:
    if width >= 1000 and height >= 700:
        return 1.0
    if width >= 600 and height >= 400:
        return 0.75
    if width >= 300 and height >= 200:
        return 0.4
    return 0.1


def domain_score(src_url: str) -> float:
    domain = urlparse(src_url).netloc.lower()
    if any(bad in domain for bad in LOW_QUALITY_DOMAINS):
        return 0.3
    return 1.0


def fetch_page_signals(src_url: str, img_url: str) -> dict:
    signals = {
        "title": "",
        "h1": "",
        "og_title": "",
        "og_image": "",
        "schema_name": "",
        "img_alt": "",
        "fetched": False,
    }
    if not src_url:
        return signals
    try:
        r = requests.get(src_url, headers=HEADERS, timeout=PAGE_TIMEOUT)
        r.raise_for_status()
    except (requests.RequestException, ValueError):
        return signals

    signals["fetched"] = True
    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return signals

    if soup.title and soup.title.string:
        signals["title"] = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        signals["h1"] = h1.get_text(strip=True)

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        signals["og_title"] = og_title["content"]
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        signals["og_image"] = og_image["content"]

    # Best-effort: find the <img> tag whose src matches our candidate
    img_fname = Path(urlparse(img_url).path).name.lower() if img_url else ""
    for img in soup.find_all("img"):
        src = (img.get("src") or img.get("data-src") or "").lower()
        if not src:
            continue
        if img_fname and img_fname in src:
            signals["img_alt"] = img.get("alt", "") or ""
            break
    if not signals["img_alt"]:
        # fallback: first <img> with alt
        first = soup.find("img", alt=True)
        if first:
            signals["img_alt"] = first.get("alt", "") or ""

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "ImageObject":
                signals["schema_name"] = (item.get("name") or item.get("caption") or "").strip()
                if signals["schema_name"]:
                    break
        if signals["schema_name"]:
            break

    return signals


def score_candidate(query_tokens: set, result: dict, page: dict) -> tuple[float, dict]:
    img_url = result.get("image") or ""
    src_url = result.get("url") or ""
    width = int(result.get("width") or 0)
    height = int(result.get("height") or 0)

    filename = Path(unquote(urlparse(img_url).path)).stem.replace("-", " ").replace("_", " ")

    sigs = {
        "alt_match":        overlap(query_tokens, result.get("title") or ""),
        "filename_match":   overlap(query_tokens, filename),
        "page_title_match": overlap(query_tokens, page.get("title", "")),
        "page_h1_match":    overlap(query_tokens, page.get("h1", "")),
        "og_title_match":   overlap(query_tokens, page.get("og_title", "")),
        "schema_match":     overlap(query_tokens, page.get("schema_name", "")),
        "page_alt_match":   overlap(query_tokens, page.get("img_alt", "")),
        "dim_quality":      dim_score(width, height),
        "domain_quality":   domain_score(src_url),
    }
    score = sum(sigs[k] * SIGNAL_WEIGHTS[k] for k in SIGNAL_WEIGHTS)
    return score, sigs


def search_images(query: str, n_search: int) -> list:
    with DDGS() as ddgs:
        return list(ddgs.images(query, max_results=n_search))


def download_image(img_url: str, out_dir: Path, slug: str, idx: int) -> Path | None:
    try:
        r = requests.get(img_url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT, stream=True)
        r.raise_for_status()
    except (requests.RequestException, ValueError):
        return None
    ct = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext_by_ct = {
        "image/jpeg": ".jpg",
        "image/jpg":  ".jpg",
        "image/png":  ".png",
        "image/webp": ".webp",
        "image/gif":  ".gif",
        "image/avif": ".avif",
    }
    ext = ext_by_ct.get(ct) or (Path(urlparse(img_url).path).suffix.lower() or ".jpg")
    if len(ext) > 5 or "/" in ext:
        ext = ".jpg"
    path = out_dir / f"{idx:02d}_{slug}{ext}"
    try:
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    except OSError:
        return None
    return path


def fish(query: str, n_search: int, n_keep: int, out_root: Path | None, verify: bool) -> tuple[list, Path]:
    query_tokens = tokenize(query)
    if not query_tokens:
        sys.stderr.write("Empty / unscorable query.\n")
        sys.exit(1)

    sys.stderr.write(f"🐟 fishing for: {query!r}\n")
    results = search_images(query, n_search)
    sys.stderr.write(f"   {len(results)} raw candidates from DuckDuckGo\n")

    ranked = []
    for i, r in enumerate(results, 1):
        page = fetch_page_signals(r.get("url", ""), r.get("image", "")) if verify else {}
        if verify:
            time.sleep(RATE_LIMIT_SEC)
        score, sigs = score_candidate(query_tokens, r, page)
        ranked.append({"result": r, "page": page, "score": score, "signals": sigs})
        title_excerpt = (r.get("title") or "")[:54].replace("\n", " ")
        sys.stderr.write(f"   [{i:02d}/{len(results)}] {score:5.2f}  {title_excerpt}\n")

    ranked.sort(key=lambda x: -x["score"])
    top = ranked[:n_keep]

    slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40] or "query"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_root or (Path.home() / "Pythonic" / "image-fisher" / "out")
    out_dir = base / f"{slug}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    saved = 0
    for idx, item in enumerate(top, 1):
        r = item["result"]
        path = download_image(r.get("image", ""), out_dir, slug, idx)
        if path:
            saved += 1
        manifest.append({
            "rank": idx,
            "score": round(item["score"], 3),
            "signals": {k: round(v, 3) for k, v in item["signals"].items()},
            "title": r.get("title"),
            "img_url": r.get("image"),
            "source_url": r.get("url"),
            "width": r.get("width"),
            "height": r.get("height"),
            "page_signals": item["page"],
            "downloaded_to": str(path) if path else None,
        })

    (out_dir / "manifest.json").write_text(
        json.dumps({"query": query, "ts": ts, "n_search": n_search, "n_keep": n_keep, "results": manifest}, indent=2)
    )

    sys.stderr.write(f"\n✅ saved {saved}/{len(top)} images\n   {out_dir}\n")
    return manifest, out_dir


def main():
    p = argparse.ArgumentParser(description="Fish the web for SEO-ranked images.")
    p.add_argument("query", nargs="*", help="search query (prompts if omitted)")
    p.add_argument("-n", "--n-search", type=int, default=30, help="raw candidates to fetch (default 30)")
    p.add_argument("-k", "--n-keep", type=int, default=10, help="top-K to download (default 10)")
    p.add_argument("-o", "--out", type=str, default=None, help="output root (default ~/Pythonic/image-fisher/out)")
    p.add_argument("--no-verify", action="store_true", help="skip page fetches (fast, less accurate)")
    p.add_argument("--json", action="store_true", help="print manifest JSON to stdout")
    p.add_argument("--preprocess", action="store_true",
                   help="after download, run preprocess.py (SDXL LoRA-ready PNGs + captions)")
    p.add_argument("--min-size", type=int, default=512,
                   help="[--preprocess] strict min dim, both must be ≥ this (default 512)")
    p.add_argument("--no-dedup", action="store_true",
                   help="[--preprocess] disable perceptual-hash dedup")
    args = p.parse_args()

    query = " ".join(args.query).strip() if args.query else input("query: ").strip()
    if not query:
        sys.exit(1)

    out_root = Path(args.out).expanduser() if args.out else None
    manifest, out_dir = fish(query, args.n_search, args.n_keep, out_root, not args.no_verify)

    if args.preprocess:
        sys.path.insert(0, str(Path(__file__).parent))
        import preprocess as _pre
        _pre.process_dir(out_dir, min_size=args.min_size, dedup=not args.no_dedup)

    if args.json:
        print(json.dumps({"out_dir": str(out_dir), "results": manifest}, indent=2))


if __name__ == "__main__":
    main()
