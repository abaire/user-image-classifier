from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import sys

from user_image_classifier.cleanup import cleanup_images


def main() -> int:
    """
    Main function to parse arguments and run the cleanup script.
    """
    parser = argparse.ArgumentParser(
        description="A utility to find and remove duplicate images.",
    )
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="+",
        required=True,
        help="One or more source directories to search for JPGs recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the changes without deleting files.",
    )

    args = parser.parse_args()

    deleted_files = cleanup_images(args.dirs, dry_run=args.dry_run)

    if deleted_files:
        print(f"\n{len(deleted_files)} duplicate files found.")
    else:
        print("No duplicate files found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
