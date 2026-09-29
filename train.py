"""
train.py
--------
One-shot training pipeline for the ArtStyle Predictor. Does everything
modelling.ipynb did, but as a single script you run once from start to
finish -- no Jupyter, no cell-by-cell decisions.

USAGE (defaults match a standard project layout with the Kaggle
'resized.zip' images already extracted to data/resized):

    pip install -r requirements.txt
    python train.py
    streamlit run ArtStyle.py

If your raw images live somewhere else:

    python train.py --raw-images-dir path/to/images --artists-csv data/artists.csv

What it does, in order:
  1. Builds dataset/train|validation|test/<style>/ from your raw images
     (skipped automatically if dataset/ already exists -- delete it first
     if you want to rebuild from scratch, e.g. after changing --cap-per-artist).
  2. Cleans the resulting folders (corrupt/duplicate/tiny-image removal).
  3. Runs a FAST architecture comparison (EfficientNet-B0, ResNet18,
     MobileNetV3-Small) on a small data subset, 2 epochs each -- a few
     minutes regardless of your full dataset size.
  4. Trains the winning architecture for real (up to 15 epochs, early
     stopping). If artifacts/model.pth already exists from a previous run,
     it's reused instead of retraining, unless --force-retrain is passed.
  5. Evaluates on the test set, saves a confusion matrix image.
  6. Saves class_names.json and model_config.json so ArtStyle.py never
     has to hardcode either.
"""

import argparse
import hashlib
import json
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display available when run as a plain script
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import timm
from PIL import Image, UnidentifiedImageError


# =====================================================================
# Data preparation (raw images -> dataset/train|validation|test/<style>/)
# =====================================================================

TARGET_CLASSES = {
    "Baroque", "Cubism", "Expressionism", "Impressionism", "Pop Art",
    "Post-Impressionism", "Primitivism", "Renaissance", "Romanticism",
    "Suprematism", "Surrealism", "Symbolism",
}

GENRE_ALIASES = {
    "Northern Renaissance": "Renaissance",
    "Early Renaissance": "Renaissance",
    "High Renaissance": "Renaissance",
    "Proto Renaissance": "Renaissance",
    "Abstract Expressionism": "Expressionism",
}

MANUAL_FILENAME_PREFIXES = {
    "Albrecht Dürer": "albrecht_du",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def resolve_genre(raw_genre: str):
    for tag in (t.strip() for t in str(raw_genre).split(",")):
        normalized = GENRE_ALIASES.get(tag, tag)
        if normalized in TARGET_CLASSES:
            return normalized
    return None


def find_artist_folder(raw_dir: Path, artist_name: str):
    candidates = [artist_name.replace(" ", "_"), normalize(artist_name).replace(" ", "_")]
    existing = {p.name: p for p in raw_dir.iterdir() if p.is_dir()}
    for cand in candidates:
        if cand in existing:
            return existing[cand]
    target_norm = normalize(artist_name).lower().replace(" ", "_")
    for folder_name, path in existing.items():
        if normalize(folder_name).lower() == target_norm:
            return path
    return None


def build_flat_file_index(raw_dir: Path):
    return [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def match_flat_files_for_artist(all_files, artist_name: str):
    if artist_name in MANUAL_FILENAME_PREFIXES:
        prefix = MANUAL_FILENAME_PREFIXES[artist_name]
        return [p for p in all_files if p.name.lower().startswith(prefix)]

    candidates = {
        artist_name.replace(" ", "_").lower() + "_",
        normalize(artist_name).replace(" ", "_").lower() + "_",
    }
    matched = [p for p in all_files if any(p.name.lower().startswith(c) for c in candidates)]
    if matched:
        return matched

    surname = normalize(artist_name.split()[-1]).lower()
    if len(surname) >= 4:
        matched = [p for p in all_files if surname in normalize(p.stem).lower()]
    return matched


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_image_list(paths):
    valid, seen_hashes = [], set()
    for path in sorted(paths):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB")
        except (UnidentifiedImageError, OSError):
            continue
        h = file_hash(path)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        valid.append(path)
    return valid


def split_paths(paths, seed: int):
    import random
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["validation"])
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def prepare_dataset(raw_images_dir: Path, artists_csv: Path, output_dir: Path, cap_per_artist: int, seed: int = 42):
    import shutil

    artists = pd.read_csv(artists_csv)
    has_subfolders = any(p.is_dir() for p in raw_images_dir.iterdir())
    flat_file_index = None
    if has_subfolders:
        print(f"Detected per-artist subfolders under {raw_images_dir}")
    else:
        print(f"Detected a flat folder under {raw_images_dir} -- matching by filename prefix.")
        flat_file_index = build_flat_file_index(raw_images_dir)
        print(f"Found {len(flat_file_index)} image files to match against {len(artists)} artists.")

    included, excluded_genre, excluded_notfound = 0, 0, 0
    class_counts_after = defaultdict(lambda: Counter())

    for _, row in artists.iterrows():
        name, raw_genre = row["name"], row["genre"]
        genre = resolve_genre(raw_genre)
        if genre is None:
            excluded_genre += 1
            continue

        if has_subfolders:
            folder = find_artist_folder(raw_images_dir, name)
            if folder is None:
                excluded_notfound += 1
                print(f"[WARNING] Could not find images for '{name}' -- skipping.")
                continue
            raw_images = sorted(folder.glob("*"))
        else:
            raw_images = match_flat_files_for_artist(flat_file_index, name)
            if not raw_images:
                excluded_notfound += 1
                print(f"[WARNING] Could not find images for '{name}' -- skipping.")
                continue

        cleaned = clean_image_list(raw_images)
        if len(cleaned) > cap_per_artist:
            import random
            rng = random.Random(seed)
            cleaned = rng.sample(cleaned, cap_per_artist)

        splits = split_paths(cleaned, seed=seed)
        for split_name, split_list in splits.items():
            dest_dir = output_dir / split_name / genre
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_list:
                dest_name = f"{name.replace(' ', '_')}__{src.name}"
                shutil.copy2(src, dest_dir / dest_name)
            class_counts_after[split_name][genre] += len(split_list)

        included += 1

    print(f"\nArtists included: {included} | excluded (genre): {excluded_genre} | excluded (not found): {excluded_notfound}")
    for split_name in ["train", "validation", "test"]:
        print(f"  {split_name}: {dict(sorted(class_counts_after[split_name].items()))}")


# =====================================================================
# Cleaning pass over already-built dataset/ folders
# =====================================================================

MIN_SIDE_PX = 64


def clean_split(split_dir: Path):
    removed = {"corrupt": 0, "duplicate": 0, "too_small": 0}
    counts = Counter()
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        seen_hashes = set()
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
            except (UnidentifiedImageError, OSError):
                removed["corrupt"] += 1
                img_path.unlink()
                continue
            if min(w, h) < MIN_SIDE_PX:
                removed["too_small"] += 1
                img_path.unlink()
                continue
            hh = file_hash(img_path)
            if hh in seen_hashes:
                removed["duplicate"] += 1
                img_path.unlink()
                continue
            seen_hashes.add(hh)
            counts[class_dir.name] += 1
    return counts, removed


# =====================================================================
# Model / train / eval helpers
# =====================================================================

def build_model(arch_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(arch_name, pretrained=True, num_classes=num_classes)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return running_loss / len(loader.dataset), acc, macro_f1


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-images-dir", type=Path, default=Path("data/resized"))
    parser.add_argument("--artists-csv", type=Path, default=Path("data/artists.csv"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cap-per-artist", type=int, default=120)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--comparison-epochs", type=int, default=2)
    parser.add_argument("--comparison-subset-fraction", type=float, default=0.2)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--force-retrain", action="store_true",
                         help="Retrain even if artifacts/model.pth already exists")
    parser.add_argument("--skip-comparison", action="store_true",
                         help="Skip the architecture comparison, train efficientnet_b0 directly")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.artifacts_dir.mkdir(exist_ok=True)

    # ---- 1. Data preparation ----
    if not (args.dataset_dir / "train").exists():
        print("\n=== Step 1: Building dataset/ from raw images ===")
        prepare_dataset(args.raw_images_dir, args.artists_csv, args.dataset_dir, args.cap_per_artist)
    else:
        print("\n=== Step 1: dataset/ already exists, skipping preparation ===")
        print(f"    (delete {args.dataset_dir} first if you want to rebuild it)")

    # ---- 2. Cleaning pass ----
    print("\n=== Step 2: Cleaning dataset/ folders ===")
    all_removed = {"corrupt": 0, "duplicate": 0, "too_small": 0}
    for split in ["train", "validation", "test"]:
        split_dir = args.dataset_dir / split
        if not split_dir.exists():
            print(f"[WARNING] {split_dir} missing")
            continue
        counts, removed = clean_split(split_dir)
        for k in all_removed:
            all_removed[k] += removed[k]
        print(f"  {split}: {sum(counts.values())} valid images across {len(counts)} classes")
    print(f"  Removed during cleaning: {all_removed}")

    # ---- 3. Transforms & loaders ----
    print("\n=== Step 3: Building transforms & loaders ===")
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(args.dataset_dir / "train", transform=train_transform)
    val_dataset = ImageFolder(args.dataset_dir / "validation", transform=eval_transform)
    test_dataset = ImageFolder(args.dataset_dir / "test", transform=eval_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    train_counts = Counter(train_dataset.targets)
    total = sum(train_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * train_counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ---- 4. Architecture comparison (fast, on a subset) ----
    if args.skip_comparison:
        best_arch = "efficientnet_b0"
        print("\n=== Step 4: Skipping comparison (--skip-comparison) -- using efficientnet_b0 ===")
    else:
        print("\n=== Step 4: Architecture comparison (fast subset) ===")
        rng = np.random.default_rng(42)
        comp_train_idx = rng.choice(len(train_dataset), size=int(len(train_dataset) * args.comparison_subset_fraction), replace=False)
        comp_val_idx = rng.choice(len(val_dataset), size=int(len(val_dataset) * args.comparison_subset_fraction), replace=False)
        comp_train_loader = DataLoader(Subset(train_dataset, comp_train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2)
        comp_val_loader = DataLoader(Subset(val_dataset, comp_val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"Comparison subset: {len(comp_train_idx)} train / {len(comp_val_idx)} val images")

        comparison_results = []
        for arch in ["efficientnet_b0", "resnet18", "mobilenetv3_small_100"]:
            print(f"\n--- {arch} ---")
            model = build_model(arch, num_classes).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            start = time.time()
            for epoch in range(args.comparison_epochs):
                _, train_acc = train_one_epoch(model, comp_train_loader, optimizer, criterion, device)
                _, val_acc, val_f1 = evaluate(model, comp_val_loader, criterion, device)
                print(f"  epoch {epoch+1}/{args.comparison_epochs}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  val_macro_f1={val_f1:.3f}")
            comparison_results.append({"architecture": arch, "val_macro_f1": val_f1, "seconds": time.time() - start})

        comparison_df = pd.DataFrame(comparison_results).sort_values("val_macro_f1", ascending=False)
        print("\nComparison results:")
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(args.artifacts_dir / "architecture_comparison.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        comparison_df.set_index("architecture")["val_macro_f1"].plot(kind="bar", ax=ax, title="Validation macro-F1 (comparison subset)")
        plt.tight_layout()
        plt.savefig(args.artifacts_dir / "architecture_comparison.png")
        plt.close()

        best_arch = comparison_df.iloc[0]["architecture"]
        print(f"\nBest architecture: {best_arch} (saved comparison table/plot to artifacts/)")

    # ---- 5. Full training (with checkpoint-reuse safety) ----
    checkpoint_path = args.artifacts_dir / "model.pth"
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    reused_from_checkpoint = False
    if checkpoint_path.exists() and not args.force_retrain:
        print(f"\n=== Step 5: Checking existing checkpoint at {checkpoint_path} ===")
        # The comparison above picks a "winner" from a noisy 2-epoch/small-subset
        # signal -- not a good enough reason to discard an already-trained model.
        # Instead, find out which architecture the existing checkpoint actually
        # matches (by trying each candidate) and reuse that if any fits.
        state_dict = torch.load(checkpoint_path, map_location=device)
        for candidate_arch in ["efficientnet_b0", "resnet18", "mobilenetv3_small_100"]:
            try:
                candidate_model = build_model(candidate_arch, num_classes).to(device)
                candidate_model.load_state_dict(state_dict, strict=True)
                model = candidate_model
                best_arch = candidate_arch
                reused_from_checkpoint = True
                print(f"Existing checkpoint matches architecture '{best_arch}' -- reusing it, "
                      f"ignoring the comparison's suggestion (use --force-retrain to override).")
                break
            except RuntimeError:
                continue
        if not reused_from_checkpoint:
            print("Existing checkpoint doesn't match any known architecture -- training fresh instead.")

    if reused_from_checkpoint:
        model.eval()
        _, _, best_val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Loaded checkpoint's validation macro-F1: {best_val_f1:.3f}")
    else:
        print(f"\n=== Step 5: Full training ({best_arch}) ===")
        model = build_model(best_arch, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        best_val_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(args.num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_f1)
            print(f"Epoch {epoch+1}/{args.num_epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  New best model saved (val_macro_f1={val_f1:.3f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"  Stopping early -- no improvement in {args.patience} epochs.")
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # ---- 6. Test evaluation ----
    print("\n=== Step 6: Test-set evaluation ===")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    test_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    test_macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    print(f"Test accuracy: {test_acc*100:.2f}%  |  Test macro-F1: {test_macro_f1:.3f}\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (row-normalized)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(args.artifacts_dir / "confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to {args.artifacts_dir / 'confusion_matrix.png'}")

    # ---- 7. Save class names + config ----
    with open(args.artifacts_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    with open(args.artifacts_dir / "model_config.json", "w") as f:
        json.dump({
            "architecture": best_arch,
            "num_classes": num_classes,
            "image_size": args.img_size,
            "test_accuracy": test_acc,           
            "test_macro_f1": test_macro_f1,
        }, f, indent=2)

    print("\n=== Done ===")
    print(f"artifacts/ now contains: {sorted(p.name for p in args.artifacts_dir.iterdir())}")
    print("Run: streamlit run ArtStyle.py")


if __name__ == "__main__":
    main()