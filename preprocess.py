import argparse
import kagglehub
import os
import json
import random
from PIL import Image


def age_to_caption(age):
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
    return f"a portrait photo of {desc}, aged {age}, realistic, high quality", desc


classes = [
    "a toddler",
    "a child",
    "a teenager",
    "a young adult",
    "a middle-aged person",
    "an older adult",
    "an elderly person"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction for validation (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=100,
        help="Number of samples per class"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Target image size (default: 512)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("Downloading the dataset...")
    os.makedirs("data", exist_ok=True)
    kagglehub.dataset_download("jangedoo/utkface-new", output_dir="data", force_download=True)
    os.system(f"rm -rf data/.complete")
    os.system(f"rm -rf data/crop_part1")
    os.system(f"rm -rf data/utkface_aligned_cropped")

    print("Preprocessing...")
    metadata = {}
    for img_name in sorted(os.listdir(f"data/UTKFace")):
        if len(img_name.split("_")) != 4 or " " in img_name:
            print(f"Image {img_name} has an unexpected filename format, pruning...")
            os.system(f"rm 'data/UTKFace/{img_name}'")
            continue
        age, gender, race, _ = img_name.split("_")
        prompt, desc = age_to_caption(int(age))
        metadata[img_name] = {
            "age": int(age),
            "gender": int(gender),
            "race": int(race),
            "prompt": prompt,
            "desc": desc,
        }

    img_names = list(metadata.keys())
    random.shuffle(img_names)
    val_size = int(len(img_names) * args.val_split)
    val_names = img_names[:val_size]
    train_names = img_names[val_size:]

    train_class_counts = {cl: 0 for cl in classes}

    os.makedirs("data/train/images", exist_ok=True)
    with open("data/train/metadata.jsonl", "w") as f:
        for img_name in train_names:
            file_name = img_name
            age = metadata[img_name]["age"]
            gender = metadata[img_name]["gender"]
            race = metadata[img_name]["race"]
            prompt = metadata[img_name]["prompt"]
            desc = metadata[img_name]["desc"]
            if train_class_counts[desc] < args.num_samples_per_class:
                train_class_counts[desc] += 1
                # Load image, centre-crop to square then resize, save to train dir
                img = Image.open(f"data/UTKFace/{img_name}").convert("RGB")
                w, h  = img.size
                min_d = min(w, h)
                left  = (w - min_d) // 2
                top   = (h - min_d) // 2
                img   = img.crop((left, top, left + min_d, top + min_d))
                img   = img.resize((args.image_size, args.image_size), Image.LANCZOS)
                img.save(f"data/train/images/{img_name}", quality=95)
                # Save metadata
                f.write(json.dumps({
                    "file_name": f"images/{img_name}",
                    "age": metadata[img_name]["age"],
                    "gender": metadata[img_name]["gender"],
                    "race": metadata[img_name]["race"],
                    "prompt": metadata[img_name]["prompt"],
                    "desc": metadata[img_name]["desc"],
                }) + "\n")

    val_class_counts = {cl: 0 for cl in classes}

    os.makedirs("data/val/images", exist_ok=True)
    with open("data/val/metadata.jsonl", "w") as f:
        for img_name in val_names:
            file_name = img_name
            age = metadata[img_name]["age"]
            gender = metadata[img_name]["gender"]
            race = metadata[img_name]["race"]
            prompt = metadata[img_name]["prompt"]
            desc = metadata[img_name]["desc"]
            if val_class_counts[desc] < int(args.num_samples_per_class * args.val_split):
                val_class_counts[desc] += 1
                # Load image, centre-crop to square then resize, save to val dir
                img = Image.open(f"data/UTKFace/{img_name}").convert("RGB")
                w, h  = img.size
                min_d = min(w, h)
                left  = (w - min_d) // 2
                top   = (h - min_d) // 2
                img   = img.crop((left, top, left + min_d, top + min_d))
                img   = img.resize((args.image_size, args.image_size), Image.LANCZOS)
                img.save(f"data/val/images/{img_name}", quality=95)
                # Save metadata
                f.write(json.dumps({
                    "file_name": f"images/{img_name}",
                    "age": metadata[img_name]["age"],
                    "gender": metadata[img_name]["gender"],
                    "race": metadata[img_name]["race"],
                    "prompt": metadata[img_name]["prompt"],
                    "desc": metadata[img_name]["desc"],
                }) + "\n")

    os.system(f"rm -rf data/UTKFace")