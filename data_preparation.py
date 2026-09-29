"""
data_preparation.py
--------------------
Cleans and organizes the raw Kaggle "Best Artworks of All Time" dataset
(https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)
into a style-labeled train/validation/test folder structure that
notebook/modelling.ipynb can load directly with ImageFolder.

WHY THIS EXISTS
---------------
The raw dataset is organized by ARTIST, not by STYLE. To train a style
classifier we need to:
  1. Map each artist to one of the 12 target style classes (via genre).
  2. Drop artists whose genre doesn't map cleanly to a target class.
  3. Clean the images themselves (corrupt/unreadable files, exact
     duplicates, non-RGB-convertible files).
  4. Cap images per artist so no single prolific artist (e.g. Van Gogh
     has 800+ paintings) dominates a class.
  5. Split into train/validation/test *per artist* (so the same painting
     never leaks across splits) and copy into dataset/<split>/<style>/.

Run this BEFORE opening modelling.ipynb.

    python data_preparation.py \
        --raw-images-dir /path/to/kaggle/images/images \
        --artists-csv data/artists.csv \
        --output-dir dataset \
        --cap-per-artist 120

The Kaggle download usually unzips to a folder called "images/images/"
containing one subfolder per artist, named like "Vincent_van_Gogh"
(underscores, no accents stripped). Point --raw-images-dir at that
subfolder.
"""

import argparse
import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

# The 12 classes the classifier predicts. Order doesn't matter here --
# ImageFolder will re-sort alphabetically at training time, and the
# training notebook saves that exact order to artifacts/class_names.json
# so the app never has to guess it again.
TARGET_CLASSES = {
    "Baroque",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Pop Art",
    "Post-Impressionism",
    "Primitivism",
    "Renaissance",
    "Romanticism",
    "Suprematism",
    "Surrealism",
    "Symbolism",
}

# artists.csv genre values are sometimes a comma-separated list
# (e.g. "Expressionism,Abstractionism"), and sometimes a more specific
# sub-period than our 12 classes (e.g. "Northern Renaissance"). This maps
# those variants onto one of the 12 target classes.
GENRE_ALIASES = {
    "Northern Renaissance": "Renaissance",
    "Early Renaissance": "Renaissance",
    "High Renaissance": "Renaissance",
    "Proto Renaissance": "Renaissance",
    "Abstract Expressionism": "Expressionism",
}

# Known filename-encoding casualties in the Kaggle 'resized' zip: accented
# characters sometimes get mangled by mojibake during extraction (e.g.
# "Dürer" -> "DuΓòá├¬rer") in a way accent-stripping can't undo, since the
# underlying bytes are actually corrupted, not just differently encoded.
# The prefix before the accented character usually survives intact, so we
# match on that directly for these specific known cases.
MANUAL_FILENAME_PREFIXES = {
    "Albrecht Dürer": "albrecht_du",
}


def resolve_genre(raw_genre: str) -> str | None:
    """
    An artist's genre field may list several tags. Take the FIRST tag
    (after alias normalization) that matches one of our 12 target
    classes -- Wikipedia infoboxes list the primary genre first, so this
    is a deterministic, defensible tie-break rather than silently
    dropping every artist whose genre field isn't a single exact match.
    Returns None if no tag maps to a target class (genuinely excluded).
    """
    for tag in (t.strip() for t in str(raw_genre).split(",")):
        normalized = GENRE_ALIASES.get(tag, tag)
        if normalized in TARGET_CLASSES:
            return normalized
    return None

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def normalize(name: str) -> str:
    """Strip accents/diacritics so 'Dürer' can match a folder like 'Durer'."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_artist_folder(raw_dir: Path, artist_name: str) -> Path | None:
    """
    Resolve an artists.csv 'name' value (e.g. "Albrecht Dürer") to the
    actual folder on disk. Kaggle folders use underscores instead of
    spaces (e.g. "Albrecht_Dürer" or sometimes an ASCII-folded variant).
    Only relevant for the 'images.zip' layout (one subfolder per artist).
    """
    candidates = [
        artist_name.replace(" ", "_"),
        normalize(artist_name).replace(" ", "_"),
    ]
    existing = {p.name: p for p in raw_dir.iterdir() if p.is_dir()}

    for cand in candidates:
        if cand in existing:
            return existing[cand]

    # Fallback: case-insensitive / accent-insensitive fuzzy match
    target_norm = normalize(artist_name).lower().replace(" ", "_")
    for folder_name, path in existing.items():
        if normalize(folder_name).lower() == target_norm:
            return path

    return None


def build_flat_file_index(raw_dir: Path) -> list[Path]:
    """
    For the 'resized.zip' layout: every image sits directly in raw_dir,
    named like 'Vincent_van_Gogh_10.jpg' (artist name + underscore +
    number). Build the list once so matching against 50 artists is cheap.
    """
    return [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def match_flat_files_for_artist(all_files: list[Path], artist_name: str) -> list[Path]:
    """Find every file in a flat folder whose name starts with this artist's
    underscored name, e.g. 'Vincent_van_Gogh_' matches 'Vincent_van_Gogh_10.jpg'."""
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

    # Fallback for encoding quirks (accented filenames don't always survive
    # zip extraction the same way the CSV text does): match on the
    # accent-stripped surname (last word) as a substring instead of an
    # exact prefix.
    surname = normalize(artist_name.split()[-1]).lower()
    if len(surname) >= 4:  # avoid over-matching on very short surnames
        matched = [p for p in all_files if surname in normalize(p.stem).lower()]

    return matched


def suggest_similar_filenames(all_files: list[Path], artist_name: str, limit: int = 5) -> list[str]:
    """When no match is found at all, surface a few filenames that share the
    artist's first name/initial letters, so the actual on-disk spelling is
    visible instead of guessing blind."""
    first_token = normalize(artist_name.split()[0]).lower()
    hits = [p.name for p in all_files if normalize(p.stem).lower().startswith(first_token[:4])]
    return hits[:limit]


def file_hash(path: Path) -> str:
    """MD5 of file bytes, used to catch exact-duplicate images."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_image_list(paths: list[Path]) -> list[Path]:
    """
    Validate a list of image paths belonging to one artist:
      - must open with PIL without error
      - must be convertible to RGB
      - drop exact-duplicate files (same content hash)
    Returns the list of valid, de-duplicated image paths.
    Works regardless of whether the paths came from a per-artist folder
    or were matched out of a flat all-images-in-one-folder layout.
    """
    valid = []
    seen_hashes = set()

    for path in sorted(paths):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(path) as img:
                img.verify()  # cheap corruption check
            with Image.open(path) as img:
                img.convert("RGB")  # confirm it's actually decodable
        except (UnidentifiedImageError, OSError):
            print(f"  [skip] unreadable/corrupt: {path.name}")
            continue

        h = file_hash(path)
        if h in seen_hashes:
            print(f"  [skip] duplicate: {path.name}")
            continue
        seen_hashes.add(h)
        valid.append(path)

    return valid


def split_paths(paths: list[Path], seed: int) -> dict[str, list[Path]]:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-images-dir", type=Path, required=True,
                         help="Path to the Kaggle 'images/images' folder (one subfolder per artist)")
    parser.add_argument("--artists-csv", type=Path, default=Path("data/artists.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--cap-per-artist", type=int, default=120,
                         help="Max images kept per artist, to stop prolific artists from dominating their class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artists = pd.read_csv(args.artists_csv)

    # Auto-detect layout: 'images.zip' has one subfolder per artist,
    # 'resized.zip' has every image flat in one folder.
    has_subfolders = any(p.is_dir() for p in args.raw_images_dir.iterdir())
    flat_file_index = None
    if has_subfolders:
        print(f"Detected per-artist subfolders under {args.raw_images_dir}")
    else:
        print(f"Detected a flat folder (no subfolders) under {args.raw_images_dir} "
              f"-- matching images by filename prefix instead.")
        flat_file_index = build_flat_file_index(args.raw_images_dir)
        print(f"Found {len(flat_file_index)} image files to match against {len(artists)} artists.")

    report = {
        "excluded_genre_not_in_target_classes": [],
        "excluded_folder_not_found": [],
        "included": [],
    }
    class_counts_before = Counter()
    class_counts_after = defaultdict(lambda: Counter())

    for _, row in artists.iterrows():
        name, raw_genre = row["name"], row["genre"]
        genre = resolve_genre(raw_genre)

        if genre is None:
            report["excluded_genre_not_in_target_classes"].append((name, raw_genre))
            continue

        if has_subfolders:
            folder = find_artist_folder(args.raw_images_dir, name)
            if folder is None:
                report["excluded_folder_not_found"].append(name)
                print(f"[WARNING] Could not find image folder for '{name}' -- skipping. "
                      f"Check --raw-images-dir and the folder naming.")
                continue
            raw_images = sorted(folder.glob("*"))
        else:
            raw_images = match_flat_files_for_artist(flat_file_index, name)
            if not raw_images:
                report["excluded_folder_not_found"].append(name)
                similar = suggest_similar_filenames(flat_file_index, name)
                print(f"[WARNING] Could not find any images for '{name}' -- skipping.")
                if similar:
                    print(f"          Files that start similarly (check the actual spelling): {similar}")
                else:
                    print(f"          No filenames even start with '{name.split()[0]}' -- "
                          f"double-check --raw-images-dir points at the right folder.")
                continue

        class_counts_before[genre] += len([p for p in raw_images if p.suffix.lower() in IMAGE_EXTS])

        print(f"\nCleaning {name} ({genre}) -- {len(raw_images)} raw files found ...")
        cleaned = clean_image_list(raw_images)

        if len(cleaned) > args.cap_per_artist:
            rng = random.Random(args.seed)
            cleaned = rng.sample(cleaned, args.cap_per_artist)

        splits = split_paths(cleaned, seed=args.seed)

        for split_name, split_paths_list in splits.items():
            dest_dir = args.output_dir / split_name / genre
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths_list:
                # Prefix with artist name to avoid filename collisions
                # between different artists mapped to the same class.
                dest_name = f"{name.replace(' ', '_')}__{src.name}"
                shutil.copy2(src, dest_dir / dest_name)
            class_counts_after[split_name][genre] += len(split_paths_list)

        report["included"].append({
            "artist": name,
            "raw_genre": raw_genre,
            "resolved_genre": genre,
            "raw_count": len(raw_images),
            "cleaned_count": len(cleaned),
        })

    # ---- Summary report ----
    print("\n" + "=" * 70)
    print("DATA CLEANING SUMMARY")
    print("=" * 70)
    print(f"Artists included:                 {len(report['included'])}")
    print(f"Excluded (genre outside 12 classes): {len(report['excluded_genre_not_in_target_classes'])}")
    for name, genre in report["excluded_genre_not_in_target_classes"]:
        print(f"   - {name} ({genre})")
    print(f"Excluded (image folder not found): {len(report['excluded_folder_not_found'])}")
    for name in report["excluded_folder_not_found"]:
        print(f"   - {name}")

    print("\nPer-class image counts after cleaning + capping:")
    for split_name in ["train", "validation", "test"]:
        print(f"\n  {split_name}:")
        for cls, count in sorted(class_counts_after[split_name].items()):
            print(f"    {cls:20s} {count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {args.output_dir / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()