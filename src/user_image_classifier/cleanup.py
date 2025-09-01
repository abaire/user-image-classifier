from __future__ import annotations

# ruff: noqa: T201 `print` found
import hashlib
import itertools
import os
from pathlib import Path

SUPPORTED_FORMATS = {"jpg", "jpeg"}


def _calculate_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()


def cleanup_images(
    input_dirs: list[str],
    *,
    dry_run: bool = False,
) -> list[str]:
    """
    Finds and removes duplicate images from the given directories.

    Args:
        input_dirs: A list of directories to search for images.
        dry_run: If True, prints the actions that would be taken without
                 actually deleting any files.

    Returns:
        A list of file paths that were (or would be) deleted.
    """
    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]
    all_files = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)

    seen_hashes = {}
    deleted_files = []

    for file_path in sorted(all_files):
        if not file_path.is_file() or file_path.suffix[1:].lower() not in SUPPORTED_FORMATS:
            continue

        str_path = str(file_path)
        file_hash = _calculate_hash(str_path)

        if file_hash in seen_hashes:
            if dry_run:
                print(f"Would delete '{str_path}' (duplicate of '{seen_hashes[file_hash]}')")
            else:
                print(f"Deleting '{str_path}' (duplicate of '{seen_hashes[file_hash]}')")
                os.remove(str_path)
            deleted_files.append(str_path)
        else:
            seen_hashes[file_hash] = str_path

    return deleted_files
