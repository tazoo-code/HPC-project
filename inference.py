"""
inference.py
------------
Age progression/regression inference using the fine-tuned SD 1.5 + LoRA model.

Usage:
    python inference.py \
        --input_image  /path/to/face.jpg \
        --target_age   70 \
        --lora_weights /path/to/lora-test-run \
        --output_dir   inference-test-results
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


def age_to_prompt(age: int) -> str:
    if age <= 2:
        desc = "a toddler"
    elif age <= 12:
        desc = "a child"
    elif age <= 19:
        desc = "a teenager"
    elif age <= 35:
        desc = "a young adult"
    elif age <= 55:
        desc = "a middle-aged person"
    elif age <= 70:
        desc = "an older adult"
    else:
        desc = "an elderly person"
    return f"a portrait photo of {desc}, aged {age}, realistic, high quality"


def main():
    parser = argparse.ArgumentParser(description="Age progression inference")
    parser.add_argument("--input_image",    type=str,   required=True)
    parser.add_argument("--target_age",     type=int,   required=True)
    parser.add_argument("--lora_weights",   type=str,   required=True)
    parser.add_argument("--base_model",     type=str,   default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output_dir",     type=str,   default="inference-test-results")
    parser.add_argument("--strength",       type=float, default=0.55)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_steps",      type=int,   default=30)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    # flush=True ensures every print appears immediately, even if output is piped
    def log(msg): print(msg, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[1/5] Device: {device}")
    if device == "cuda":
        log(f"      GPU: {torch.cuda.get_device_name(0)}")
        log(f"      VRAM free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # ── Load pipeline ──────────────────────────────────────────────────────────
    log(f"[2/5] Loading pipeline from: {args.base_model}")
    log(f"      (uses HF cache — no download if already cached)")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=False,
    )
    log("      Pipeline loaded.")

    # ── Load LoRA weights ──────────────────────────────────────────────────────
    log(f"[3/5] Loading LoRA weights from: {args.lora_weights}")
    pipe.load_lora_weights(args.lora_weights)
    pipe = pipe.to(device)
    log("      LoRA loaded and moved to device.")

    # Memory optimisations
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("      xformers enabled.")
    except Exception:
        log("      xformers not available, skipping.")

    if device == "cuda":
        log(f"      VRAM after load: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free")

    # ── Load input image ───────────────────────────────────────────────────────
    log(f"[4/5] Input image: {args.input_image}")
    input_image = Image.open(args.input_image).convert("RGB")
    input_image = input_image.resize((512, 512), Image.LANCZOS)

    prompt          = age_to_prompt(args.target_age)
    negative_prompt = "cartoon, anime, painting, blurry, low quality, deformed, ugly"
    log(f"      Prompt: {prompt}")
    log(f"      Strength: {args.strength} | Steps: {args.num_steps} | CFG: {args.guidance_scale}")

    # ── Generate ───────────────────────────────────────────────────────────────
    log(f"[5/5] Generating... (progress bar below)")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        generator=generator,
    )
    output_image = result.images[0]

    # ── Save ───────────────────────────────────────────────────────────────────
    stem        = Path(args.input_image).stem
    output_path = output_dir / f"{stem}_age{args.target_age}.jpg"
    output_image.save(output_path, quality=95)
    log(f"      Saved: {output_path}")

    comparison = Image.new("RGB", (1024, 512))
    comparison.paste(input_image,  (0, 0))
    comparison.paste(output_image, (512, 0))
    comparison_path = output_dir / f"{stem}_age{args.target_age}_comparison.jpg"
    comparison.save(comparison_path, quality=95)
    log(f"      Saved comparison: {comparison_path}")
    log("Done!")


if __name__ == "__main__":
    main()