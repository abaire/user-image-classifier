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

    for file_path in input_path.glob("*.*"):
        if file_path.suffix.lower() not in (".jpg", ".jpeg"):
            continue

        label_file = None
        for ext in (".txt", ".json"):
            if (input_path / f"{file_path.stem}{ext}").exists():
                label_file = input_path / f"{file_path.stem}{ext}"
                break

        if not label_file:
            print(f"⚠️  No label file found for {file_path.name}, skipping.")
            continue

        class_counts = _get_class_counts(label_file, class_map)

        if not class_counts:
            if remove_empty:
                print(f"Removing empty file: {file_path.name}")
                print(f"Removing empty label file: {label_file.name}")
                if not dry_run:
                    os.remove(file_path)
                    os.remove(label_file)
            elif move_empty:
                if not dry_run:
                    empty_dir.mkdir(exist_ok=True)
                print(f"Moving empty file: {file_path.name} to {empty_dir}")
                print(f"Moving empty label file: {label_file.name} to {empty_dir}")
                if not dry_run:
                    os.rename(file_path, empty_dir / file_path.name)
                    os.rename(label_file, empty_dir / label_file.name)
            else:
                class_counts = "unlabeled"

        if class_counts:
            timestamp = get_image_datetime(file_path)
            if not timestamp:
                print(f"⚠️  No EXIF date found for {file_path.name}, skipping.")
                continue

            new_filename = f"{timestamp}_{class_counts}--{file_path.name}"
            new_image_path = input_path / new_filename
            new_label_path = input_path / f"{new_image_path.stem}{label_file.suffix}"

            print(f"Renaming '{file_path.name}' -> '{new_image_path.name}'")
            print(f"Renaming '{label_file.name}' -> '{new_label_path.name}'")

            if not dry_run:
                os.rename(file_path, new_image_path)
                os.rename(label_file, new_label_path)
