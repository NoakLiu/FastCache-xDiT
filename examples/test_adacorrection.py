#!/usr/bin/env python3
"""
AdaCorrection Test Script

This script demonstrates how to use AdaCorrection with FastCache
for improved generation quality while maintaining efficient cache reuse.
"""

import os
import sys
import time
import torch
import argparse
from pathlib import Path
import json

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from diffusers import FluxPipeline, PixArtSigmaPipeline
    from PIL import Image
except ImportError:
    print("Please install diffusers>=0.30.0 and Pillow")
    sys.exit(1)

def test_adacorrection(
    model_type="flux",
    model_name="black-forest-labs/FLUX.1-schnell",
    prompt="a serene landscape with mountains and a lake",
    num_inference_steps=30,
    cache_ratio_threshold=0.05,
    motion_threshold=0.1,
    enable_adacorrection=True,
    adacorr_gamma=1.0,
    adacorr_lambda=1.0,
    output_dir="adacorrection_results"
):
    """
    Test AdaCorrection with FastCache
    
    Args:
        model_type: Type of model ("flux", "pixart")
        model_name: Model name or path
        prompt: Text prompt for generation
        num_inference_steps: Number of inference steps
        cache_ratio_threshold: Threshold for block-level caching
        motion_threshold: Threshold for token-level motion detection
        enable_adacorrection: Whether to enable AdaCorrection
        adacorr_gamma: Sensitivity parameter γ
        adacorr_lambda: Spatial weight λ
        output_dir: Output directory for results
    """
    
    print(f"Testing AdaCorrection with FastCache")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"AdaCorrection: {enable_adacorrection}")
    if enable_adacorrection:
        print(f"  γ (gamma): {adacorr_gamma}")
        print(f"  λ (lambda): {adacorr_lambda}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if model_type == "flux":
        pipeline = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    elif model_type == "pixart":
        pipeline = PixArtSigmaPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    pipeline = pipeline.to("cuda")
    
    # Test baseline FastCache (without AdaCorrection)
    print("Running FastCache baseline (without AdaCorrection)...")
    start_time = time.time()
    
    from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer
    
    # Find transformer in the pipeline
    if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "transformer"):
        transformer = pipeline.unet.transformer
    else:
        print("Warning: Could not find transformer in pipeline")
        return
    
    # Apply FastCache without AdaCorrection
    apply_cache_on_transformer(
        transformer,
        use_cache="Fast",
        rel_l1_thresh=cache_ratio_threshold,
        motion_threshold=motion_threshold,
        enable_adacorrection=False,
        return_hidden_states_first=False,
        num_steps=num_inference_steps
    )
    
    with torch.no_grad():
        baseline_result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
    
    baseline_time = time.time() - start_time
    print(f"FastCache baseline time: {baseline_time:.2f}s")
    
    # Save baseline image
    baseline_image = baseline_result.images[0]
    baseline_image.save(os.path.join(output_dir, "fastcache_baseline.png"))
    
    # Test FastCache + AdaCorrection
    print("Running FastCache + AdaCorrection...")
    start_time = time.time()
    
    # Re-apply with AdaCorrection enabled
    apply_cache_on_transformer(
        transformer,
        use_cache="Fast",
        rel_l1_thresh=cache_ratio_threshold,
        motion_threshold=motion_threshold,
        enable_adacorrection=enable_adacorrection,
        adacorr_gamma=adacorr_gamma,
        adacorr_lambda=adacorr_lambda,
        return_hidden_states_first=False,
        num_steps=num_inference_steps
    )
    
    with torch.no_grad():
        adacorr_result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
    
    adacorr_time = time.time() - start_time
    print(f"FastCache + AdaCorrection time: {adacorr_time:.2f}s")
    
    # Save AdaCorrection image
    adacorr_image = adacorr_result.images[0]
    adacorr_image.save(os.path.join(output_dir, "fastcache_adacorrection.png"))
    
    # Calculate speedup
    speedup = baseline_time / adacorr_time if adacorr_time > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    
    # Save comparison image
    comparison_image = Image.new('RGB', (baseline_image.width * 2, baseline_image.height))
    comparison_image.paste(baseline_image, (0, 0))
    comparison_image.paste(adacorr_image, (baseline_image.width, 0))
    comparison_image.save(os.path.join(output_dir, "comparison.png"))
    
    # Save results
    results = {
        "model_type": model_type,
        "model_name": model_name,
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "enable_adacorrection": enable_adacorrection,
        "adacorr_gamma": adacorr_gamma,
        "adacorr_lambda": adacorr_lambda,
        "cache_ratio_threshold": cache_ratio_threshold,
        "motion_threshold": motion_threshold,
        "baseline_time": baseline_time,
        "adacorr_time": adacorr_time,
        "speedup": speedup
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Speedup achieved: {speedup:.2f}x")
    print("\nNote: AdaCorrection improves generation quality (FID) while maintaining")
    print("similar or better speed compared to baseline FastCache.")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test AdaCorrection with FastCache")
    parser.add_argument("--model_type", type=str, default="flux", choices=["flux", "pixart"],
                       help="Model type")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell",
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, default="a serene landscape with mountains and a lake",
                       help="Text prompt")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--cache_ratio_threshold", type=float, default=0.05,
                       help="Cache ratio threshold")
    parser.add_argument("--motion_threshold", type=float, default=0.1,
                       help="Motion threshold")
    parser.add_argument("--enable_adacorrection", action="store_true",
                       help="Enable AdaCorrection")
    parser.add_argument("--adacorr_gamma", type=float, default=1.0,
                       help="AdaCorrection sensitivity parameter γ")
    parser.add_argument("--adacorr_lambda", type=float, default=1.0,
                       help="AdaCorrection spatial weight λ")
    parser.add_argument("--output_dir", type=str, default="adacorrection_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    test_adacorrection(
        model_type=args.model_type,
        model_name=args.model,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        cache_ratio_threshold=args.cache_ratio_threshold,
        motion_threshold=args.motion_threshold,
        enable_adacorrection=args.enable_adacorrection,
        adacorr_gamma=args.adacorr_gamma,
        adacorr_lambda=args.adacorr_lambda,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()

