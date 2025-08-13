from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import itertools
import json
import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from user_image_classifier.config import DEFAULT_CONFIG


class ImageClassifierGUI:
    """
    A GUI application for classifying images using keyboard shortcuts.
    """

    def __init__(self, root: tk.Tk, image_paths: set[str], key_map: dict[str, str], output_root: str):
        """
        Initializes the classifier GUI.

        Args:
            root: The root tkinter window.
            image_paths: A set of absolute paths to the images to be classified.
            key_map: A dictionary mapping keyboard keys to directory names.
        """
        self.root = root
        self.image_paths = sorted(image_paths)
        self.key_map = key_map
        self.last_move = None
        self.output_root = output_root

        self.root.title("Image Classifier")

        banner_text = " | ".join([f"'{key}': {folder}" for key, folder in self.key_map.items()])
        self.banner_label = tk.Label(self.root, text=banner_text, font=("Helvetica", 14), pady=5)
        self.banner_label.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)
        self.root.bind("<Key>", self.handle_key_press)

        self.current_index = 0
        self.last_move: tuple[str, str, int] | None = None

        self.display_image()

    def display_image(self):
        """Loads and displays the next image in the queue."""

        if not self.image_paths:
            messagebox.showinfo("Done!", "All images have been classified.")
            self.root.quit()
            return

        self.current_path = self.image_paths[self.current_index]
        filename = os.path.basename(self.current_path)
        self.root.title(f"Image Classifier - {len(self.image_paths)} left - [{filename}]")

        image = Image.open(self.current_path)

        screen_width = self.root.winfo_screenwidth() * 0.8
        screen_height = self.root.winfo_screenheight() * 0.8
        img_width, img_height = image.size

        ratio = min(screen_width / img_width, screen_height / img_height)
        if ratio < 1:
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def handle_key_press(self, event):
        key = event.keysym.lower()

        if key == "escape":
            self.root.destroy()
        elif key == "backspace":
            self.undo_last_move()
        elif key == "right":
            self.navigate(1)
        elif key == "left":
            self.navigate(-1)
        elif event.char.lower() in self.key_map:
            self.move_image(event.char.lower())

    def navigate(self, delta: int):
        """Navigates through the image list by a given delta."""
        if not self.image_paths:
            return
        new_index = (self.current_index + delta) % len(self.image_paths)
        self.current_index = new_index
        self.display_image()

    def move_image(self, key: str):
        """Moves the current image to the directory mapped to the given key."""
        source_path = self.image_paths.pop(self.current_index)
        filename = os.path.basename(source_path)
        dest_dir = self.key_map[key]
        dest_path_dir = os.path.join(self.output_root, dest_dir)
        dest_path = os.path.join(dest_path_dir, filename)

        suffix = 1
        filename_without_ext, filename_ext = os.path.splitext(filename)
        while os.path.isfile(dest_path):
            new_filename = f"filename_without_ext_{suffix:4d}.{filename_ext}"
            dest_path = os.path.join(dest_path_dir, new_filename)

        shutil.move(source_path, dest_path)
        print(f"✅ Moved: '{filename}' -> '{dest_dir}'")
        self.last_move = (dest_path, source_path, self.current_index)

        if self.image_paths and self.current_index >= len(self.image_paths):
            self.current_index = len(self.image_paths) - 1

        self.display_image()

    def undo_last_move(self):
        """Undoes the last file move operation."""
        if not self.last_move:
            print("❗️ No move to undo.")
            return

        moved_path, original_path, original_index = self.last_move
        shutil.move(moved_path, original_path)

        self.image_paths.insert(original_index, original_path)
        self.current_index = original_index
        print(f"↩️ UNDO: Moved '{os.path.basename(moved_path)}' back.")

        self.last_move = None
        self.display_image()


def _load_key_map(config_path: str | None) -> dict[str, str]:
    if not config_path:
        return DEFAULT_CONFIG

    with open(config_path) as f:
        return json.load(f)


def _find_sources(input_dirs: list[str], output_dir: str) -> set[str]:
    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]

    combined_results = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)
    all_files = set(combined_results)

    output_path = Path(output_dir)

    def keep_file(filename: Path) -> bool:
        if filename.is_relative_to(output_path):
            return False

        if not filename.is_file():
            return False

        return filename.suffix[1:].lower() in {"jpg", "jpeg"}

    return {filename for filename in all_files if keep_file(filename)}


def _run_gui(image_paths: set[str], key_map: dict[str, str], output_root: str):
    root = tk.Tk()
    ImageClassifierGUI(root, image_paths, key_map, output_root)
    root.update_idletasks()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f"+{center_x}+{center_y}")

    root.mainloop()


def main() -> int:
    """
    Main function to parse arguments, find images, and start the GUI.
    """
    parser = argparse.ArgumentParser(
        description="A utility to classify JPG images using keyboard shortcuts.",
        epilog="Example: python image_sorter.py --config map.json --dirs ./photos ./downloads",
    )
    parser.add_argument("-c", "--config", help="Path to the JSON configuration file for key mappings.")
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="+",
        required=True,
        help="One or more source directories to search for JPGs recursively.",
    )
    parser.add_argument("--output", "-o", help="Base directory into which classified files should be moved")
    args = parser.parse_args()

    key_map = _load_key_map(args.config)

    if not args.output:
        args.output = "."
    args.output = os.path.abspath(os.path.expanduser(args.output))

    for target_dir in key_map.values():
        os.makedirs(os.path.join(args.output, target_dir), exist_ok=True)

    image_paths = _find_sources(args.dirs, args.output)
    if not image_paths:
        print("No JPG images found in the specified directories. Exiting.")
        return 0

    print(f"Found {len(image_paths)} images to classify.\n")
    print("--- Controls ---")
    for key, folder in key_map.items():
        print(f"  Press '{key}' -> move to '{folder}/'")
    print("  Press 'u' -> Undo last move")
    print("  Press 'space' -> Skip image (moves to end of queue)")
    print("  Press 'q' or 'esc' -> Quit")
    print("----------------\n")

    _run_gui(image_paths, key_map, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
