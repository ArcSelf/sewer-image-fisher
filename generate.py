#!/usr/bin/env python3
"""SDXL local generation via HuggingFace diffusers, with pointillism terminal preview.

Generates an image, saves it, and renders pointillism in the terminal afterward.
With --stream, every Nth denoising step is decoded through the VAE mid-flight
and painted into the terminal with ANSI cursor-up overdraw — animated denoising
as pointillism.

Device handling: auto-detects CUDA → MPS → CPU. fp16 on CUDA/MPS, fp32 on CPU.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline


# ---------- device pick ----------

def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ---------- streaming callback ----------

CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


def make_stream_callback(total_steps: int, interval: int, palette: str, seed):
    """Closure: VAE-decode partial latents every Nth step and paint as pointillism."""
    state = {"last_lines": 0}

    def cb(pipe, step, timestep, callback_kwargs):
        # diffusers calls cb after each step; render every Nth, skip step 0 (pure noise)
        emit_at_step = step + 1  # 1-indexed display
        if step == 0 or emit_at_step % interval != 0:
            return callback_kwargs
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        with torch.no_grad():
            scaled = latents / pipe.vae.config.scaling_factor
            decoded = pipe.vae.decode(scaled.to(pipe.vae.dtype)).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            arr = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
            img = pipe.image_processor.numpy_to_pil(arr)[0]

        import view as _view  # skill dir is on sys.path
        rendered = _view.render_pointillism_from_image(
            img,
            palette=palette,
            jitter_chars=True,
            jitter_buckets=True,
            color_jitter=10,
            seed=seed,
        )
        header = f"━━ denoising step {emit_at_step}/{total_steps} ━━"
        body = f"{header}\n{rendered}"

        # Move cursor up to overwrite the previous frame, then clear-below
        if state["last_lines"] > 0:
            sys.stdout.write(f"\033[{state['last_lines']}A\r\033[J")
        sys.stdout.write(body + "\n")
        sys.stdout.flush()
        state["last_lines"] = body.count("\n") + 1
        return callback_kwargs

    return cb


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="SDXL local generation with terminal pointillism preview."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="low quality, bad anatomy")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to local LoRA weights (.safetensors)")
    parser.add_argument("--lora-scale", type=float, default=0.75)
    parser.add_argument("--model", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output-dir", type=str,
                        default="~/Pythonic/image-fisher/generations")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the final terminal pointillism preview")
    parser.add_argument("--preview-palette", choices=["dots", "ascii", "impasto"],
                        default="dots", help="Palette for the post-gen render")
    parser.add_argument("--stream", action="store_true",
                        help="Animate denoising: paint pointillism every Nth step")
    parser.add_argument("--stream-interval", type=int, default=5,
                        help="[--stream] render every Nth step (default 5)")
    parser.add_argument("--stream-palette", choices=["dots", "ascii", "impasto"],
                        default="impasto",
                        help="[--stream] palette during denoising (default impasto)")
    args = parser.parse_args()

    # Make view.py importable from the same skill dir
    skill_dir = Path(__file__).parent
    sys.path.insert(0, str(skill_dir))

    # Paths
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"gen_{timestamp}.png"

    # Device + dtype
    device, dtype = pick_device()
    sys.stderr.write(f"🌀 device={device}  dtype={dtype}\n")

    # Load pipeline
    sys.stderr.write(f"   loading {args.model}...\n")
    pipe_kwargs: dict = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        pipe_kwargs["variant"] = "fp16"
    pipeline = DiffusionPipeline.from_pretrained(args.model, **pipe_kwargs).to(device)

    # LoRA
    if args.lora and os.path.exists(args.lora):
        sys.stderr.write(f"   injecting LoRA: {args.lora} (scale={args.lora_scale})\n")
        pipeline.load_lora_weights(args.lora)
        pipeline.fuse_lora(lora_scale=args.lora_scale)

    # RNG
    generator = None
    if args.seed is not None:
        gen_device = device if device != "mps" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(args.seed)

    # Streaming callback
    pipe_extra: dict = {}
    if args.stream:
        sys.stderr.write(
            f"   stream: ON  palette={args.stream_palette}  every {args.stream_interval} steps\n"
        )
        pipe_extra["callback_on_step_end"] = make_stream_callback(
            args.steps, args.stream_interval, args.stream_palette, args.seed,
        )
        pipe_extra["callback_on_step_end_tensor_inputs"] = ["latents"]
        sys.stdout.write(CURSOR_HIDE)
        sys.stdout.flush()

    sys.stderr.write(
        f"   generating {args.width}x{args.height} over {args.steps} steps "
        f"(cfg={args.cfg})...\n"
    )

    try:
        image = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=generator,
            **pipe_extra,
        ).images[0]
    finally:
        if args.stream:
            sys.stdout.write(CURSOR_SHOW)
            sys.stdout.flush()

    image.save(out_path)
    sys.stderr.write(f"✅ saved {out_path}\n")

    if args.no_preview:
        return

    sys.stderr.write("\n━━ final pointillism preview ━━\n")
    try:
        import view as _view
        print(_view.view(
            out_path,
            mode="pointillism",
            palette=args.preview_palette,
            jitter_chars=True,
            jitter_buckets=True,
            color_jitter=8,
        ))
    except Exception as e:
        sys.stderr.write(f"   preview failed ({type(e).__name__}: {e}); CLI fallback\n")
        os.system(
            f"python3 '{skill_dir / 'view.py'}' '{out_path}' "
            f"--palette {args.preview_palette} --organic"
        )


if __name__ == "__main__":
    main()
