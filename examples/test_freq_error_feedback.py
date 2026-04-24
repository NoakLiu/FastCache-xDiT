#!/usr/bin/env python3
"""
Test frequency-domain event score + spectral error-feedback caching (FastCache path).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from PIL import Image

try:
    from diffusers import FluxPipeline
except ImportError:
    print("Please install diffusers>=0.30.0")
    sys.exit(1)


def _get_transformer(pipeline):
    if hasattr(pipeline, "transformer"):
        return pipeline.transformer
    unet = getattr(pipeline, "unet", None)
    if unet is not None and hasattr(unet, "transformer"):
        return unet.transformer
    return None


def main():
    parser = argparse.ArgumentParser(description="Test frequency-domain error-feedback FastCache")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--prompt", type=str, default="a serene landscape with mountains and a lake")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05)
    parser.add_argument("--motion_threshold", type=float, default=0.1)
    parser.add_argument("--freq_event_gamma", type=float, default=2.0)
    parser.add_argument("--freq_error_ema_decay", type=float, default=0.85)
    parser.add_argument("--enable_adacorrection", action="store_true", help="Also blend with AdaCorrection weights (max with freq)")
    parser.add_argument("--adacorr_gamma", type=float, default=1.0)
    parser.add_argument("--adacorr_lambda", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="freq_error_feedback_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer

    gen = torch.Generator(device="cuda").manual_seed(42)

    def load_pipe():
        p = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
        tr = _get_transformer(p)
        if tr is None:
            raise RuntimeError("No transformer on pipeline")
        return p, tr

    def run(pipe, tag: str):
        t0 = time.time()
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                generator=gen,
            )
        dt = time.time() - t0
        path = os.path.join(args.output_dir, f"{tag}.png")
        result.images[0].save(path)
        return dt, path

    pipe, transformer = load_pipe()
    apply_cache_on_transformer(
        transformer,
        use_cache="Fast",
        rel_l1_thresh=args.cache_ratio_threshold,
        motion_threshold=args.motion_threshold,
        enable_adacorrection=False,
        enable_freq_error_feedback=False,
        return_hidden_states_first=False,
        num_steps=args.num_inference_steps,
    )
    t_base, p_base = run(pipe, "fastcache_baseline")
    del pipe
    torch.cuda.empty_cache()

    pipe, transformer = load_pipe()
    apply_cache_on_transformer(
        transformer,
        use_cache="Fast",
        rel_l1_thresh=args.cache_ratio_threshold,
        motion_threshold=args.motion_threshold,
        enable_adacorrection=args.enable_adacorrection,
        adacorr_gamma=args.adacorr_gamma,
        adacorr_lambda=args.adacorr_lambda,
        enable_freq_error_feedback=True,
        freq_event_gamma=args.freq_event_gamma,
        freq_error_ema_decay=args.freq_error_ema_decay,
        return_hidden_states_first=False,
        num_steps=args.num_inference_steps,
    )
    t_fb, p_fb = run(pipe, "fastcache_freq_error_feedback")

    base_im = Image.open(p_base)
    fb_im = Image.open(p_fb)
    comp = Image.new("RGB", (base_im.width * 2, base_im.height))
    comp.paste(base_im, (0, 0))
    comp.paste(fb_im, (base_im.width, 0))
    comp.save(os.path.join(args.output_dir, "comparison.png"))

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "baseline_sec": t_base,
                "freq_feedback_sec": t_fb,
                "freq_event_gamma": args.freq_event_gamma,
                "freq_error_ema_decay": args.freq_error_ema_decay,
                "enable_adacorrection": args.enable_adacorrection,
            },
            f,
            indent=2,
        )
    print(f"Baseline: {t_base:.2f}s -> {p_base}")
    print(f"+Freq feedback: {t_fb:.2f}s -> {p_fb}")


if __name__ == "__main__":
    main()
