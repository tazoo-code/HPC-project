import os
import json
import argparse
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def age_to_caption(age: int) -> str:
    """Convert an integer age to a natural language text caption."""
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


def parse_filename(filename: str):
    """
    Parse UTKFace filename: [age]_[gender]_[race]_[datetime].jpg
    Returns (age, gender, race) or None if malformed.
    """
    stem = Path(filename).stem  # strip .jpg
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        age = int(parts[0])
        return age
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess UTKFace for SD LoRA fine-tuning")
    parser.add_argument("--data_dir",   type=str, default="dataset/UTKFace", help="Path to UTKFace raw images")
    parser.add_argument("--output_dir", type=str, default="data/UTKFace_processed", help="Output directory")
    parser.add_argument("--image_size", type=int, default=512,   help="Target image size (default: 512)")
    parser.add_argument("--val_split",  type=float, default=0.05, help="Fraction for validation (default: 0.05)")
    parser.add_argument("--max_age",    type=int, default=99,    help="Discard images with age >= this (default: 99)")
    parser.add_argument("--seed",       type=int, default=42,    help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Gather all jpg files
    all_images = sorted(data_dir.glob("*.jpg"))
    print(f"Found {len(all_images)} images in {data_dir}")

    # Parse and filter
    valid = []
    skipped = 0
    for img_path in all_images:
        age = parse_filename(img_path.name)
        if age is None or age > args.max_age or age < 0:
            skipped += 1
            continue
        valid.append((img_path, age))

    print(f"Valid images: {len(valid)} | Skipped (malformed/out of range): {skipped}")

    # Shuffle and split
    random.shuffle(valid)
    val_n  = max(1, int(len(valid) * args.val_split))
    splits = {
        "val":   valid[:val_n],
        "train": valid[val_n:],
    }
    print(f"Split — train: {len(splits['train'])} | val: {len(splits['val'])}")

    # Process each split
    for split_name, items in splits.items():
        img_out_dir = output_dir / split_name / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / split_name / "metadata.jsonl"

        with open(meta_path, "w") as meta_f:
            for img_path, age in tqdm(items, desc=f"Processing {split_name}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"  WARNING: Could not open {img_path.name}: {e}")
                    continue

                # Centre-crop to square then resize
                w, h  = img.size
                min_d = min(w, h)
                left  = (w - min_d) // 2
                top   = (h - min_d) // 2
                img   = img.crop((left, top, left + min_d, top + min_d))
                img   = img.resize((args.image_size, args.image_size), Image.LANCZOS)

                out_name = img_path.name
                img.save(img_out_dir / out_name, quality=95)

                caption = age_to_caption(age)
                meta_f.write(json.dumps({
                    "file_name": f"images/{out_name}",
                    "text":      caption,
                    "age":       age,        # keep raw age for evaluation later
                }) + "\n")

    print(f"\nDone! Processed data written to: {output_dir}")


if __name__ == "__main__":
    main()
