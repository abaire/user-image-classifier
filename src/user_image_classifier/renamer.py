from __future__ import annotations

# ruff: noqa: T201 `print` found
# ruff: noqa: DTZ007 Naive datetime constructed using `datetime.datetime.strptime()` without %z
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


def get_image_datetime(image_path) -> datetime | None:
    """
    Extracts the best available timestamp from an image's EXIF data and
    returns it as a datetime object. It checks for 'DateTimeOriginal',
    'DateTimeDigitized', and 'DateTime' in that order.
    """
    image = Image.open(image_path)
    exif_data = image.getexif()

    if not exif_data:
        return None

    tag_dict = {TAGS[key]: val for key, val in exif_data.items() if key in TAGS}

    date_tags = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
    date_str = None

    for tag in date_tags:
        if tag in tag_dict:
            date_str = tag_dict[tag]
            break

    if not date_str:
        return None

    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")


def _get_class_counts(label_file: Path, class_map: dict[str, str]) -> str | None:
    """
    Parses a label file and returns a string with the counts of each class.

    Args:
        label_file: The path to the label file (.txt or .json).
        class_map: A dictionary mapping keys to class names.

    Returns:
        A string in the format '1coyote_2deer', or None if no classes are found.
    """
    counts: Counter[str] = Counter()
    if label_file.suffix == ".json":
        with open(label_file) as f:
            data = json.load(f)
        for class_name, bboxes in data.items():
            counts[class_name] = len(bboxes)

    elif label_file.suffix == ".txt":
        id_to_class = dict(enumerate(sorted(class_map.values())))
        with open(label_file) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_name = id_to_class.get(class_id)
                if class_name:
                    counts[class_name] += 1

    if not counts:
        return None

    return "_".join(f"{count}{name}" for name, count in sorted(counts.items()))


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in (".jpg", ".jpeg")


def rename_files(
    input_dir: str,
    class_map: dict[str, str],
    *,
    dry_run: bool = False,
    remove_empty: bool = False,
    move_empty: bool = False,
) -> None:
    """
    Renames image and label files in a directory based on EXIF data and label content.

    Args:
        input_dir: The directory containing the files to rename.
        class_map: A dictionary mapping keys to class names.
        dry_run: If True, print the changes without renaming files.
        remove_empty: If True, delete files with no labels.
        move_empty: If True, move files with no labels to a new 'empty' directory.
    """
    input_path = Path(input_dir)
    empty_dir = input_path / "empty"

    for image_path in input_path.glob("*.*"):
        if not _is_image_file(image_path):
            continue

        label_path = None
        for ext in (".txt", ".json"):
            if (input_path / f"{image_path.stem}{ext}").exists():
                label_path = input_path / f"{image_path.stem}{ext}"
                break

        if not label_path.is_file():
            print(f"⚠️  No label file found for {image_path.name}, skipping.")
            continue

        class_counts = _get_class_counts(label_path, class_map)

        if not class_counts:
            if remove_empty:
                print(f"Removing empty file: {image_path.name}")
                print(f"Removing empty label file: {label_path.name}")
                if not dry_run:
                    os.remove(image_path)
                    os.remove(label_path)
                continue

            if move_empty:
                print(f"Moving empty file: {image_path.name} to {empty_dir}")
                print(f"Moving empty label file: {label_path.name} to {empty_dir}")
                if not dry_run:
                    empty_dir.mkdir(exist_ok=True)
                    os.rename(image_path, empty_dir / image_path.name)
                    os.rename(label_path, empty_dir / label_path.name)
                continue

            class_counts = "unlabeled"

        timestamp = get_image_datetime(image_path)
        if not timestamp:
            print(f"⚠️  No EXIF date found for {image_path.name}, skipping.")
            continue

        new_filename = f"{timestamp}_{class_counts}--{image_path.name}"
        new_image_path = input_path / new_filename
        new_label_path = input_path / f"{new_image_path.stem}{label_path.suffix}"

        _perform_rename(image_path, new_image_path, label_path, new_label_path, dry_run=dry_run)


def undo_rename(input_dir: str, *, dry_run: bool = False) -> None:
    """
    Reverts a previous renaming operation.

    Args:
        input_dir: The directory containing the files to rename.
        dry_run: If True, print the changes without renaming files.
    """
    input_path = Path(input_dir)
    for image_path in input_path.glob("*--*.*"):
        if image_path.stem.count("--") != 1:
            continue

        if not _is_image_file(image_path):
            continue

        original_filename = image_path.name.split("--", 1)[1]
        new_image_path = input_path / original_filename

        label_path = None
        for ext in (".txt", ".json"):
            # The label file would have been renamed to match the image file (without the image extension)
            potential_label_name = f"{image_path.stem}{ext}"
            if (input_path / potential_label_name).exists():
                label_path = input_path / potential_label_name
                break

        if not label_path:
            print(f"⚠️  No label file found for {image_path.name}, skipping.")
            continue

        new_label_path = input_path / f"{new_image_path.stem}{label_path.suffix}"

        _perform_rename(image_path, new_image_path, label_path, new_label_path, dry_run=dry_run)


def _perform_rename(
    image_path: Path, new_image_path: Path, label_path: Path, new_label_path: Path, *, dry_run: bool
) -> None:
    print(f"Renaming '{image_path.name}' -> '{new_image_path.name}'")
    print(f"Renaming '{label_path.name}' -> '{new_label_path.name}'")

    if not dry_run:
        os.rename(image_path, new_image_path)
        os.rename(label_path, new_label_path)
