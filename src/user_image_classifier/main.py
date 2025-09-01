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

from user_image_classifier.config import load_key_map

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


_ZOOM_OUT_SCALE = 0.9
_ZOOM_IN_SCALE = 1.1
_MOUSE_WHEEL_DELTA = 120
_MOUSE_BUTTON_4 = 4
_MOUSE_BUTTON_5 = 5

_MAX_UNDO_FILES = 16


class ImageClassifierGUI:
    """
    A GUI application for classifying images using keyboard shortcuts.
    """

    def __init__(
        self,
        root: tk.Tk,
        image_paths: set[str],
        key_map: dict[str, str],
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
        self.strip_confidence = strip_confidence
        self.yolo = yolo
        self.class_to_id = {name: i for i, name in enumerate(sorted(self.key_map.values()))}
        self.id_to_class = {i: name for name, i in self.class_to_id.items()}
        self.colors = [
            "red",
            "green",
            "light sky blue",
            "yellow",
            "cyan",
            "magenta",
            "spring green",
            "light cyan",
            "light slate blue",
            "gold",
            "orchid",
            "deep pink",
            "MediumPurple1",
            "OliveDrab1",
        ]

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
        banner_text += "\n+/-/=: Zoom | Shift+Arrows: Pan"
        self.banner_label = tk.Label(self.root, text=banner_text, font=("Helvetica", 14), pady=5)
        self.banner_label.pack()

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(padx=10, pady=10)
        self.root.bind("<Key>", self.handle_key_press)

        self.current_index = 0
        self.bboxes = []
        self.rect = None
        self.rect_bg = None
        self.start_x = None
        self.start_y = None
        self.image_width = 0
        self.image_height = 0
        self.original_image = None
        self.photo_image = None
        self.image_on_canvas = None
        self.zoom_level = 1.0
        self.is_drawing = False
        self.crosshair_v = None
        self.crosshair_h = None
        self.undo_stack = []
        self._undo_clears_all_bounding_boxes = False

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<ButtonPress-2>", self.pan_start)
        self.canvas.bind("<B2-Motion>", self.pan_move)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)
        self.canvas.bind("<Motion>", self.handle_mouse_move)
        self.canvas.bind("<Leave>", self.handle_mouse_leave)

        self.display_image()

    def handle_right_click(self, event):
        del event
        self.undo_last_bbox()

    def _delete_crosshair(self):
        if self.crosshair_v:
            self.canvas.delete(self.crosshair_v)
            self.crosshair_v = None
        if self.crosshair_h:
            self.canvas.delete(self.crosshair_h)
            self.crosshair_h = None

    def handle_mouse_leave(self, event):
        del event
        self._delete_crosshair()

    def handle_mouse_move(self, event):
        if self.is_drawing:
            return

        if self.crosshair_v:
            self.canvas.delete(self.crosshair_v)
        if self.crosshair_h:
            self.canvas.delete(self.crosshair_h)

        if self.image_on_canvas:
            img_x1, img_y1, img_x2, img_y2 = self.canvas.bbox(self.image_on_canvas)
            self.crosshair_v = self.canvas.create_line(event.x, img_y1, event.x, img_y2, fill="red")
            self.crosshair_h = self.canvas.create_line(img_x1, event.y, img_x2, event.y, fill="red")

    def pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _redraw_canvas(self, x, y):
        if self.original_image is None:
            return

        # Resize image
        new_width = int(self.image_width * self.zoom_level)
        new_height = int(self.image_height * self.zoom_level)

        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized_image)

        # Update canvas
        self.canvas.delete("all")
        self.image_on_canvas = self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Redraw bboxes
        for bbox in self.bboxes:
            x1 = x + bbox["x1"] * self.zoom_level
            y1 = y + bbox["y1"] * self.zoom_level
            x2 = x + bbox["x2"] * self.zoom_level
            y2 = y + bbox["y2"] * self.zoom_level

            color = "grey"
            if bbox["label"]:
                class_id = self.class_to_id[bbox["label"]]
                color = self.colors[class_id % len(self.colors)]

            bbox["rect_bg"] = self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=4, tags="bbox")
            bbox["rect"] = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="bbox")

            if bbox["label"]:
                label_item = self.canvas.create_text(
                    x1,
                    y1 - 5,
                    text=bbox["label"],
                    fill=color,
                    anchor=tk.SW,
                    tags="bbox",
                    font=("Helvetica", 16),
                )
                bbox["label_item"] = label_item
                text_bbox = self.canvas.bbox(label_item)
                bg_rect = self.canvas.create_rectangle(text_bbox, fill="black", outline="black", tags="bbox")
                self.canvas.tag_lower(bg_rect, label_item)
                bbox["label_bg_item"] = bg_rect

    def zoom(self, event=None, scale_factor=None):
        if event:
            if event.num == _MOUSE_BUTTON_5 or event.delta == -_MOUSE_WHEEL_DELTA:
                scale = _ZOOM_OUT_SCALE
            elif event.num == _MOUSE_BUTTON_4 or event.delta == _MOUSE_WHEEL_DELTA:
                scale = _ZOOM_IN_SCALE
            else:
                return
            cx, cy = event.x, event.y
        elif scale_factor:
            scale = scale_factor
            cx = self.canvas.winfo_width() / 2
            cy = self.canvas.winfo_height() / 2
        else:
            return

        img_x, img_y = self.canvas.coords(self.image_on_canvas)
        rel_x, rel_y = cx - img_x, cy - img_y

        orig_x = rel_x / self.zoom_level
        orig_y = rel_y / self.zoom_level

        new_zoom_level = self.zoom_level * scale
        if int(self.image_width * new_zoom_level) < 1:
            return

        self.zoom_level = new_zoom_level

        new_rel_x = orig_x * self.zoom_level
        new_rel_y = orig_y * self.zoom_level

        new_img_x = cx - new_rel_x
        new_img_y = cy - new_rel_y

        self._redraw_canvas(new_img_x, new_img_y)

    def _load_existing_metadata(self):
        image_path = Path(self.current_path)
        base_filename = image_path.stem
        parent_dir = image_path.parent

        json_path = parent_dir / f"{base_filename}.json"
        if json_path.exists():
            self._load_json_metadata(json_path)
            return

        yolo_path = parent_dir / f"{base_filename}.txt"
        if yolo_path.exists():
            self._load_yolo_metadata(yolo_path)

    def display_image(self):
        """Loads and displays the next image in the queue."""
        self.bboxes = []
        self.canvas.delete("all")
        self.zoom_level = 1.0

        if not self.image_paths:
            messagebox.showinfo("Done!", "All images have been classified.")
            self.root.quit()
            return

        self.current_path = self.image_paths[self.current_index]
        filename = os.path.basename(self.current_path)
        self.root.title(f"Image Classifier - {len(self.image_paths)} left - [{filename}]")

        self.original_image = Image.open(self.current_path)
        self.image_width, self.image_height = self.original_image.size

        self._load_existing_metadata()
        self._undo_clears_all_bounding_boxes = bool(self.bboxes)

        screen_width = self.root.winfo_screenwidth() * 0.8
        screen_height = self.root.winfo_screenheight() * 0.8
        ratio = min(screen_width / self.image_width, screen_height / self.image_height)
        if ratio < 1:
            self.zoom_level = ratio

        # Set canvas size once
        canvas_width = int(self.image_width * self.zoom_level)
        canvas_height = int(self.image_height * self.zoom_level)
        self.canvas.config(width=canvas_width, height=canvas_height)

        self._redraw_canvas(0, 0)

    def handle_key_press(self, event):
        key = event.keysym.lower()

        if key == "escape":
            self.root.destroy()
        elif key == "backspace":
            if self.bboxes:
                self.undo_last_bbox()
            else:
                self.undo_last_save()
        elif key == "right":
            if event.state & 0x0001:  # Shift key
                self.canvas.xview_scroll(1, "units")
            else:
                self.navigate(1)
        elif key == "left":
            if event.state & 0x0001:  # Shift key
                self.canvas.xview_scroll(-1, "units")
            else:
                self.navigate(-1)
        elif key == "up":
            if event.state & 0x0001:  # Shift key
                self.canvas.yview_scroll(-1, "units")
        elif key == "down":
            if event.state & 0x0001:  # Shift key
                self.canvas.yview_scroll(1, "units")
        elif key in ("plus", "equal", "kp_add") or event.char == "+":
            self.zoom(scale_factor=_ZOOM_IN_SCALE)
        elif key in ("minus", "kp_subtract") or event.char == "-":
            self.zoom(scale_factor=_ZOOM_OUT_SCALE)
        elif key == "equals" or event.char == "=":
            self.zoom_level = 1.0
            self._redraw_canvas(0, 0)
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

    def _load_yolo_metadata(self, yolo_path: Path):
        with open(yolo_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            x_center = x_center_norm * self.image_width
            y_center = y_center_norm * self.image_height
            half_box_width = width_norm * self.image_width * 0.5
            half_box_height = height_norm * self.image_height * 0.5

            x1 = round(x_center - half_box_width)
            y1 = round(y_center - half_box_height)
            x2 = round(x_center + half_box_width)
            y2 = round(y_center + half_box_height)

            label = self.id_to_class.get(class_id)
            if label is None:
                print(f"⚠️ Warning: Unknown class ID {class_id} in {yolo_path}")
                continue

            self.bboxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                }
            )

    def _save_yolo_format(self, filename: str, output_dir: Path):
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        dest_path = output_dir / txt_filename
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
        if not lines:
            print(f"✅ Saved empty: '{txt_filename}'")
        else:
            print(f"✅ Saved: '{txt_filename}'")

    def _load_json_metadata(self, json_path: Path):
        with open(json_path) as f:
            data = json.load(f)

        for label, bboxes in data.items():
            for bbox in bboxes:
                self.bboxes.append(
                    {
                        "x1": bbox["x1"],
                        "y1": bbox["y1"],
                        "x2": bbox["x2"],
                        "y2": bbox["y2"],
                        "label": label,
                    }
                )

    def _save_json_format(self, filename: str, output_dir: Path):
        json_filename = os.path.splitext(filename)[0] + ".json"
        dest_path = output_dir / json_filename
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

        if not output_data:
            print(f"✅ Saved empty: '{json_filename}'")
        else:
            print(f"✅ Saved: '{json_filename}'")

    def save_and_next(self):
        """Saves the bounding boxes and moves to the next image."""
        source_path = Path(self.image_paths.pop(self.current_index))
        self.undo_stack.append(source_path)
        if len(self.undo_stack) > _MAX_UNDO_FILES:
            self.undo_stack.pop(0)

        filename = source_path.name
        output_dir = source_path.parent

        if self.yolo:
            self._save_yolo_format(filename, output_dir)
        else:
            self._save_json_format(filename, output_dir)

        self._update_after_removal()

    def add_label(self, key: str):
        if not self.bboxes:
            return

        # Find the last unlabeled bbox
        for bbox in reversed(self.bboxes):
            if bbox["label"] is None:
                bbox["label"] = self.key_map[key]
                x, y = self.canvas.coords(self.image_on_canvas)
                self._redraw_canvas(x, y)
                break

    def undo_last_bbox(self):
        """Undoes the last bounding box."""
        if not self.bboxes:
            print("❗️ No bounding box to undo.")
            return

        if self._undo_clears_all_bounding_boxes:
            self._undo_clears_all_bounding_boxes = False
            self.bboxes.clear()
        else:
            self.bboxes.pop()

        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)
        print("↩️ UNDO: Removed last bounding box.")

    def undo_last_save(self):
        """Undoes the last save operation."""
        if not self.undo_stack:
            print("❗️ No save to undo.")
            return

        last_saved_path = str(self.undo_stack.pop())
        self.image_paths.insert(self.current_index, last_saved_path)
        self.display_image()
        print(f"↩️ UNDO: Reopened '{os.path.basename(last_saved_path)}'")

    def on_button_press(self, event):
        if self.bboxes and self.bboxes[-1]["label"] is None:
            self.undo_last_bbox()

        self.is_drawing = True
        self._delete_crosshair()
        self.start_x = event.x
        self.start_y = event.y
        self.rect_bg = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="black", width=4
        )
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="grey", width=2
        )

    def on_mouse_drag(self, event):
        if not self.image_on_canvas:
            cur_x, cur_y = (event.x, event.y)
        else:
            img_x1, img_y1, img_x2, img_y2 = self.canvas.bbox(self.image_on_canvas)
            cur_x = max(img_x1, min(event.x, img_x2))
            cur_y = max(img_y1, min(event.y, img_y2))

        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
        self.canvas.coords(self.rect_bg, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.is_drawing = False
        if not self.image_on_canvas:
            return

        img_x, img_y = self.canvas.coords(self.image_on_canvas)
        start_x_orig = (self.start_x - img_x) / self.zoom_level
        start_y_orig = (self.start_y - img_y) / self.zoom_level
        end_x_orig = (event.x - img_x) / self.zoom_level
        end_y_orig = (event.y - img_y) / self.zoom_level

        # Clamp to image dimensions
        start_x_clamped = max(0, min(start_x_orig, self.image_width))
        start_y_clamped = max(0, min(start_y_orig, self.image_height))
        end_x_clamped = max(0, min(end_x_orig, self.image_width))
        end_y_clamped = max(0, min(end_y_orig, self.image_height))

        bbox = {
            "x1": min(start_x_clamped, end_x_clamped),
            "y1": min(start_y_clamped, end_y_clamped),
            "x2": max(start_x_clamped, end_x_clamped),
            "y2": max(start_y_clamped, end_y_clamped),
            "label": None,
        }

        # Ignore zero-sized boxes
        if bbox["x1"] == bbox["x2"] or bbox["y1"] == bbox["y2"]:
            x, y = self.canvas.coords(self.image_on_canvas)
            self._redraw_canvas(x, y)
            return

        self.bboxes.append(bbox)
        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)  # Redraw to show the new box scaled correctly


def _find_sources(input_dirs: list[str], *, edit: bool = False) -> set[str]:
    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]

    combined_results = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)
    all_files = set(combined_results)

    def keep_file(filename: Path) -> bool:
        if not filename.is_file():
            return False

        if filename.suffix[1:].lower() not in {"jpg", "jpeg"}:
            return False

        # Skip if a label file already exists
        base_filename = filename.stem
        parent_dir = filename.parent
        has_label = (parent_dir / f"{base_filename}.json").exists() or (parent_dir / f"{base_filename}.txt").exists()
        if edit:
            return has_label
        return not has_label

    return {str(filename) for filename in all_files if keep_file(filename)}


def _run_gui(
    image_paths: set[str],
    key_map: dict[str, str],
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
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Edit existing classifications.",
    )
    args = parser.parse_args()

    key_map = load_key_map(args.config)

    image_paths = _find_sources(args.dirs, edit=args.edit)
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
        min_confidence_threshold=args.min_confidence_threshold,
        max_confidence_threshold=args.max_confidence_threshold,
        strip_confidence=args.strip_confidence,
        use_secondary_confidence=args.use_secondary_confidence,
        yolo=not args.no_yolo,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
