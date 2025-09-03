from __future__ import annotations

# ruff: noqa: T201 `print` found
import hashlib
import itertools
import os
from pathlib import Path

SUPPORTED_FORMATS = {"jpg", "jpeg"}


def find_images(
    input_dirs: list[str], ignore_dirs: list[str] | None = None, extensions: set[str] | None = None
) -> set[Path]:
    """Recursively finds all images in the given input_dirs."""
    if not extensions:
        extensions = SUPPORTED_FORMATS

    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]

    combined_results = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)
    all_files = set(combined_results)

    if ignore_dirs is None:
        ignore_dirs = []
    ignored_paths = [Path(ignored) for ignored in ignore_dirs]

    def keep_file(filename: Path) -> bool:
        if any(filename.is_relative_to(ignored) for ignored in ignored_paths):
            return False

        if not filename.is_file():
            return False

        return filename.suffix[1:].lower() in extensions

    return {filename for filename in all_files if keep_file(filename)}


def hash_file(path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cleanup_images(
    input_dirs: list[str],
    *,
    dry_run: bool = False,
    golden_dirs: list[str] | None = None,
) -> int:
    """
    Finds and removes duplicate images with metadata from the given directories.

    Args:
        input_dirs: A list of directories to search for images.
        dry_run: If True, prints the actions that would be taken without
                 actually deleting any files.
        golden_dirs: List of directories to be used as reference images when deleting images. These will never be deleted but may cause identical images in input_dirs to be removed.

    Returns:
        The number of duplicates found.
    """

    hashes: dict[tuple[int, str], Path] = {}

    def _calculate_key(image_path: Path) -> tuple[int, str] | None:
        try:
            file_size = image_path.stat().st_size
            file_hash = hash_file(image_path)
        except FileNotFoundError:
            # This can happen if a file is deleted during the process
            # (e.g. it was a duplicate of another file that was processed earlier)
            return None

        return (file_size, file_hash)

    if golden_dirs:
        print(f"Scanning golden images in {golden_dirs}...")
        for image_path in find_images(golden_dirs):
            key = _calculate_key(image_path)
            if not key:
                continue
            hashes[key] = image_path

    print(f"Scanning for duplicate images in {input_dirs}...")
    images = find_images(input_dirs)
    duplicates_found = 0

    for image_path in sorted(images):
        metadata_file = image_path.with_suffix(".json")
        if not metadata_file.is_file():
            continue

        key = _calculate_key(image_path)
        if not key:
            continue

        if key in hashes:
            if dry_run:
                print(f"Duplicate found: {image_path} is a duplicate of {hashes[key]} (would be deleted)")
            else:
                print(f"Duplicate found: {image_path} is a duplicate of {hashes[key]}")
                image_path.unlink()
                metadata_file.unlink()

            duplicates_found += 1
        else:
            hashes[key] = image_path

    if dry_run:
        print(f"Scan complete. Found {duplicates_found} duplicate images that would be removed.")
    else:
        print(f"Scan complete. Found and removed {duplicates_found} duplicate images.")

    return duplicates_found
