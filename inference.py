import argparse
import os
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
    parser.add_argument("--input_image",    type=str,   default="")
    parser.add_argument("--target_ages",     type=str,   default="3,9,16,25,40,60,80")
    parser.add_argument("--lora_weights",   type=str,   default="")
    parser.add_argument("--base_model",     type=str,   default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output_dir",     type=str,   default="inference-test-results")
    parser.add_argument("--strength",       type=float, default=0.55)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_steps",      type=int,   default=30)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    if args.input_image == "":
        val_imgs = sorted(os.listdir("data/val/images"))
        midpoint = len(val_imgs) // 2
        args.input_image = f"data/val/images/{val_imgs[midpoint]}"

    if args.lora_weights == "":
        ckpt_dirs = sorted(os.listdir("train_output"))
        ckpt_epochs = [int(cp.split("-")[1]) for cp in ckpt_dirs]
        last = max(ckpt_epochs)
        args.lora_weights = f"train_output/checkpoint-{last}"

    # flush=True ensures every print appears immediately, even if output is piped
    def log(msg): print(msg, flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=False,
    )
    pipe.load_lora_weights(args.lora_weights)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    input_image = Image.open(args.input_image).convert("RGB")
    input_image = input_image.resize((512, 512), Image.LANCZOS)

    negative_prompt = "cartoon, anime, painting, blurry, low quality, deformed, ugly"
    for target_age in args.target_ages.split(","):
        target_age = int(target_age.strip())
        prompt = age_to_prompt(target_age)

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

        stem = Path(args.input_image).stem
        output_path = output_dir / f"{stem}_age{target_age}.jpg"
        output_image.save(output_path, quality=95)

        comparison = Image.new("RGB", (1024, 512))
        comparison.paste(input_image,  (0, 0))
        comparison.paste(output_image, (512, 0))
        comparison_path = output_dir / f"{stem}_age{target_age}_comparison.jpg"
        comparison.save(comparison_path, quality=95)


if __name__ == "__main__":
    main()