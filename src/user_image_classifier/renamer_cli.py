from __future__ import annotations

import argparse
import sys

from user_image_classifier.config import load_key_map
from user_image_classifier.renamer import rename_files


def main() -> int:
    """
    Main function to parse arguments and run the renamer.
    """
    parser = argparse.ArgumentParser(
        description="A utility to rename image files based on EXIF data and labels.",
    )
    parser.add_argument("-c", "--config", help="Path to the JSON configuration file for key mappings.")
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="The source directory to search for JPGs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the changes without renaming files.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--remove-empty",
        action="store_true",
        help="Delete files with no labels.",
    )
    group.add_argument(
        "--move-empty",
        action="store_true",
        help="Move files with no labels to a new 'empty' directory.",
    )

    args = parser.parse_args()
    key_map = load_key_map(args.config)

    rename_files(
        args.dir,
        key_map,
        dry_run=args.dry_run,
        remove_empty=args.remove_empty,
        move_empty=args.move_empty,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
