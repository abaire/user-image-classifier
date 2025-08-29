from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import itertools
import json
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from user_image_classifier.config import DEFAULT_CONFIG

CONFIDENCE_SUBSTRING = r"C(\d+)"
CLASS_SUBSTRING = r"([a-zA-Z_-]+)"

MULTICLASS_PREFIX_RE = re.compile(
    f"^{CONFIDENCE_SUBSTRING}_{CLASS_SUBSTRING}_{CONFIDENCE_SUBSTRING}_{CLASS_SUBSTRING}_(.*)"
)
SINGLECLASS_PREFIX_RE = re.compile(f"^{CONFIDENCE_SUBSTRING}_(.*)")
MULTICLASS_SUFFIX_RE = re.compile(
    rf"(.*?)_{CONFIDENCE_SUBSTRING}_{CLASS_SUBSTRING}_{CONFIDENCE_SUBSTRING}_{CLASS_SUBSTRING}(\..*)?$"
)
SINGLECLASS_SUFFIX_RE = re.compile(rf"(.+?)_{CONFIDENCE_SUBSTRING}(\..*)?$")


def _remove_confidence_substring(filename: str) -> str:
    match = MULTICLASS_PREFIX_RE.match(filename)
    if match:
        return match.group(5)

    match = SINGLECLASS_PREFIX_RE.match(filename)
    if match:
        return match.group(2)

    match = MULTICLASS_SUFFIX_RE.match(filename)
    if match:
        return f"{match.group(1)}{match.group(6) or ''}"

    match = SINGLECLASS_SUFFIX_RE.match(filename)
    if match:
        return f"{match.group(1)}{match.group(3) or ''}"

    return filename


def _get_confidences(filename: str) -> tuple[int | None, int | None]:
    match = MULTICLASS_PREFIX_RE.match(filename)
    if match:
        return int(match.group(1)), int(match.group(3))

    match = SINGLECLASS_PREFIX_RE.match(filename)
    if match:
        return int(match.group(1)), None

    match = MULTICLASS_SUFFIX_RE.match(filename)
    if match:
        return int(match.group(2)), int(match.group(4))

    match = SINGLECLASS_SUFFIX_RE.match(filename)
    if match:
        return int(match.group(2)), None

    return None, None


def _get_confidence(filename: str, *, use_secondary: bool = False) -> int | None:
    primary, secondary = _get_confidences(filename)
    if use_secondary:
        return secondary
    return primary


class ImageClassifierGUI:
    """
    A GUI application for classifying images using keyboard shortcuts.
    """

    def __init__(
        self,
        root: tk.Tk,
        image_paths: set[str],
        key_map: dict[str, str],
        output_root: str,
        min_confidence_threshold: int | None = None,
        max_confidence_threshold: int | None = None,
        *,
        strip_confidence: bool = False,
        use_secondary_confidence: bool = False,
        yolo: bool = True,
    ):
        """
        Initializes the classifier GUI.

        Args:
            root: The root tkinter window.
            image_paths: A set of absolute paths to the images to be classified.
            key_map: A dictionary mapping keyboard keys to directory names.
            output_root: Directory into which classified files should be placed.
            strip_confidence: If True, strip "Cxx" confidence prefix/suffix from filenames.
            use_secondary_confidence: If True, use the secondary confidence score for filtering.
            yolo: If True, output annotations in YOLO format.
        """
        self.root = root
        self.image_paths = sorted(image_paths)
        self.key_map = key_map
        self.output_root = output_root
        self.strip_confidence = strip_confidence
        self.yolo = yolo
        self.class_to_id = {name: i for i, name in enumerate(sorted(self.key_map.values()))}

        self.root.title("Image Classifier")

        if min_confidence_threshold is not None or max_confidence_threshold is not None:
            min_thresh = min_confidence_threshold if min_confidence_threshold is not None else 0
            max_thresh = max_confidence_threshold if max_confidence_threshold is not None else 100

            def _keep_filename(filename: str) -> bool:
                confidence = _get_confidence(os.path.basename(filename), use_secondary=use_secondary_confidence)
                if confidence is None:
                    return True

                return min_thresh <= confidence <= max_thresh

            self.image_paths = list(filter(_keep_filename, self.image_paths))

        banner_text = " | ".join([f"'{key}': {folder}" for key, folder in self.key_map.items()])
        banner_text += (
            "\nDraw boxes with mouse. Press key to label. Space: Save and Next. Backspace: Undo Box. ESC: Quit"
        )
        self.banner_label = tk.Label(self.root, text=banner_text, font=("Helvetica", 14), pady=5)
        self.banner_label.pack()

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(padx=10, pady=10)
        self.root.bind("<Key>", self.handle_key_press)

        self.current_index = 0
        self.bboxes = []
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.image_width = 0
        self.image_height = 0

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.display_image()

    def display_image(self):
        """Loads and displays the next image in the queue."""
        self.bboxes = []
        self.canvas.delete("all")

        if not self.image_paths:
            messagebox.showinfo("Done!", "All images have been classified.")
            self.root.quit()
            return

        self.current_path = self.image_paths[self.current_index]
        filename = os.path.basename(self.current_path)
        self.root.title(f"Image Classifier - {len(self.image_paths)} left - [{filename}]")

        image = Image.open(self.current_path)
        self.image_width, self.image_height = image.size

        screen_width = self.root.winfo_screenwidth() * 0.8
        screen_height = self.root.winfo_screenheight() * 0.8
        img_width, img_height = image.size

        ratio = min(screen_width / img_width, screen_height / img_height)
        if ratio < 1:
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(image)
        self.canvas.config(width=img_tk.width(), height=img_tk.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def handle_key_press(self, event):
        key = event.keysym.lower()

        if key == "escape":
            self.root.destroy()
        elif key == "backspace":
            self.undo_last_bbox()
        elif key == "right":
            self.navigate(1)
        elif key == "left":
            self.navigate(-1)
        elif key == "space":
            self.save_and_next()
        elif event.char.lower() in self.key_map:
            self.add_label(event.char.lower())

    def navigate(self, delta: int):
        """Navigates through the image list by a given delta."""
        if not self.image_paths:
            return
        new_index = (self.current_index + delta) % len(self.image_paths)
        self.current_index = new_index
        self.display_image()

    def _update_after_removal(self):
        if self.image_paths and self.current_index >= len(self.image_paths):
            self.current_index = len(self.image_paths) - 1

        self.display_image()

    def _save_yolo_format(self, filename: str):
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        dest_path = os.path.join(self.output_root, txt_filename)
        lines = []
        for bbox in self.bboxes:
            label = bbox["label"]
            if not label:
                continue

            class_id = self.class_to_id[label]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            box_width = x2 - x1
            box_height = y2 - y1
            x_center = x1 + box_width / 2
            y_center = y1 + box_height / 2

            x_center_norm = x_center / self.image_width
            y_center_norm = y_center / self.image_height
            width_norm = box_width / self.image_width
            height_norm = box_height / self.image_height

            lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        with open(dest_path, "w") as f:
            f.writelines(lines)
        print(f"✅ Saved: '{txt_filename}'")

    def _save_json_format(self, filename: str):
        json_filename = os.path.splitext(filename)[0] + ".json"
        dest_path = os.path.join(self.output_root, json_filename)
        output_data = {}
        for bbox in self.bboxes:
            label = bbox["label"]
            if not label:
                continue

            if label not in output_data:
                output_data[label] = []

            output_data[label].append(
                {
                    "x1": bbox["x1"],
                    "y1": bbox["y1"],
                    "x2": bbox["x2"],
                    "y2": bbox["y2"],
                }
            )

        with open(dest_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Saved: '{json_filename}'")

    def save_and_next(self):
        """Saves the bounding boxes and moves to the next image."""
        if not self.bboxes:
            # If no bboxes, just go to next image without saving anything.
            self.image_paths.pop(self.current_index)
            self._update_after_removal()
            return

        source_path = self.image_paths.pop(self.current_index)
        filename = os.path.basename(source_path)

        if self.yolo:
            self._save_yolo_format(filename)
        else:
            self._save_json_format(filename)

        self._update_after_removal()

    def add_label(self, key: str):
        if not self.bboxes:
            return

        # Find the last unlabeled bbox
        for bbox in reversed(self.bboxes):
            if bbox["label"] is None:
                bbox["label"] = self.key_map[key]
                x1, y1 = bbox["x1"], bbox["y1"]
                label_item = self.canvas.create_text(x1, y1 - 5, text=bbox["label"], fill="red", anchor=tk.SW)
                bbox["label_item"] = label_item
                break

    def undo_last_bbox(self):
        """Undoes the last bounding box."""
        if not self.bboxes:
            print("❗️ No bounding box to undo.")
            return

        bbox = self.bboxes.pop()
        self.canvas.delete(bbox["rect"])
        if bbox["label_item"]:
            self.canvas.delete(bbox["label_item"])

        print("↩️ UNDO: Removed last bounding box.")

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        bbox = {
            "x1": self.start_x,
            "y1": self.start_y,
            "x2": end_x,
            "y2": end_y,
            "label": None,
            "rect": self.rect,
            "label_item": None,
        }
        self.bboxes.append(bbox)


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


def _run_gui(
    image_paths: set[str],
    key_map: dict[str, str],
    output_root: str,
    min_confidence_threshold: int | None,
    max_confidence_threshold: int | None,
    *,
    strip_confidence: bool = False,
    use_secondary_confidence: bool = False,
    yolo: bool = False,
):
    root = tk.Tk()
    ImageClassifierGUI(
        root,
        image_paths,
        key_map,
        output_root,
        min_confidence_threshold=min_confidence_threshold,
        max_confidence_threshold=max_confidence_threshold,
        strip_confidence=strip_confidence,
        use_secondary_confidence=use_secondary_confidence,
        yolo=yolo,
    )
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
    parser.add_argument(
        "--strip-confidence",
        "-S",
        action="store_true",
        help="Strip Cxx prefix/suffix from filenames with confidence scores.",
    )
    parser.add_argument(
        "--min-confidence-threshold",
        "-T",
        type=int,
        metavar="confidence_percent",
        help="Skip files with confidence less than <confidence_percent>",
    )
    parser.add_argument(
        "--max-confidence-threshold",
        "-X",
        type=int,
        metavar="confidence_percent",
        help="Skip files with confidence greater than <confidence_percent>",
    )
    parser.add_argument("--output", "-o", help="Base directory into which classified files should be moved")
    parser.add_argument(
        "--use-secondary-confidence",
        "-U",
        action="store_true",
        help="Use the secondary confidence score (if present) for confidence thresholding.",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Do not emit YOLOv8 format output, use the default JSON format instead.",
    )
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

    _run_gui(
        image_paths,
        key_map,
        args.output,
        min_confidence_threshold=args.min_confidence_threshold,
        max_confidence_threshold=args.max_confidence_threshold,
        strip_confidence=args.strip_confidence,
        use_secondary_confidence=args.use_secondary_confidence,
        yolo=not args.no_yolo,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
