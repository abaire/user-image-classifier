from __future__ import annotations

# ruff: noqa: T201 `print` found
import argparse
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import tkinter as tk
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Any

from PIL import Image, ImageTk

from user_image_classifier.config import load_key_map

_ZOOM_OUT_SCALE = 0.9
_ZOOM_IN_SCALE = 1.1
_MOUSE_WHEEL_DELTA = 120
_MOUSE_BUTTON_4 = 4
_MOUSE_BUTTON_5 = 5

_MAX_UNDO_FILES = 16


class PendingDeletions:
    """Manages files that are pending hard deletion."""

    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.files_to_delete = set()

    def add(self, path: Path):
        """Adds a file to the list of pending deletions."""
        self.files_to_delete.add(path)

    def clear(self):
        """Deletes all pending files."""
        for path in self.files_to_delete:
            path.unlink(missing_ok=True)
        self.files_to_delete.clear()
        self.temp_dir.cleanup()

    def __del__(self):
        self.clear()


class UndoableAction(ABC):
    description: str = ""

    @abstractmethod
    def undo(self, gui: ImageClassifierGUI) -> None: ...

    def finalize(self, gui: ImageClassifierGUI) -> None:
        """Finalizes the action. By default, do nothing."""


@dataclass
class AddBoundingBoxAction(UndoableAction):
    bbox: dict[str, Any]

    def __post_init__(self):
        self.description = "add bounding box"

    def undo(self, gui: ImageClassifierGUI) -> None:
        gui.bboxes.remove(self.bbox)
        gui.redraw_canvas_on_undo()


@dataclass
class DeleteBoundingBoxAction(UndoableAction):
    bbox: dict[str, Any]
    index: int

    def __post_init__(self):
        self.description = "delete bounding box"

    def undo(self, gui: ImageClassifierGUI) -> None:
        gui.bboxes.insert(self.index, self.bbox)
        gui.redraw_canvas_on_undo()


@dataclass
class AddLabelAction(UndoableAction):
    bbox: dict[str, Any]
    old_label: str | None

    def __post_init__(self):
        self.description = "add label"

    def undo(self, gui: ImageClassifierGUI) -> None:
        self.bbox["label"] = self.old_label
        gui.redraw_canvas_on_undo()


@dataclass
class SaveFileAction(UndoableAction):
    source_path: Path
    dest_path: Path

    def __post_init__(self):
        self.description = "save file"

    def undo(self, gui: ImageClassifierGUI) -> None:
        shutil.move(self.dest_path, self.source_path)
        gui.image_paths.insert(gui.current_index, str(self.source_path))
        gui.display_image()


@dataclass
class SkipFileAction(UndoableAction):
    source_path: Path

    def __post_init__(self):
        self.description = "skip file"

    def undo(self, gui: ImageClassifierGUI) -> None:
        gui.image_paths.insert(gui.current_index, str(self.source_path))
        gui.display_image()


@dataclass
class DeleteFileAction(UndoableAction):
    original_path: Path
    renamed_path: Path
    original_json_path: Path | None
    renamed_json_path: Path | None
    is_hard_delete: bool = False

    def __post_init__(self):
        self.description = "hard delete file" if self.is_hard_delete else "soft delete file"

    def undo(self, gui: ImageClassifierGUI) -> None:
        shutil.move(self.renamed_path, self.original_path)
        if self.renamed_json_path and self.original_json_path:
            shutil.move(self.renamed_json_path, self.original_json_path)
        gui.image_paths.insert(gui.current_index, str(self.original_path))
        gui.display_image()

    def finalize(self, gui: ImageClassifierGUI) -> None:
        """Permanently deletes the file if it was a hard delete."""
        if self.is_hard_delete:
            print(f"🔥 Scheduling final deletion of {self.original_path.name}")
            gui.pending_deletions.add(self.renamed_path)
            if self.renamed_json_path:
                gui.pending_deletions.add(self.renamed_json_path)


class UndoManager:
    """Manages an undo stack for user actions."""

    def __init__(self, gui: ImageClassifierGUI, max_size: int = 20):
        self.gui = gui
        self.stack = deque(maxlen=max_size)

    def register_action(self, action):
        """Registers an undoable action."""
        if len(self.stack) == self.stack.maxlen:
            self.stack[0].finalize(self.gui)
        self.stack.append(action)

    def undo(self, gui):
        """Undoes the last action."""
        if not self.stack:
            print("❗️ No action to undo.")
            return

        action = self.stack.pop()
        action.undo(gui)
        print(f"↩️ UNDO: {action.description}")

    def clear(self):
        """Clears the undo stack."""
        self.stack.clear()

    def is_empty(self) -> bool:
        """Returns True if the undo stack is empty."""
        return not self.stack


class ImageClassifierGUI:
    """
    A GUI application for classifying images using keyboard shortcuts.
    """

    def __init__(
        self,
        root: tk.Tk,
        image_paths: set[str],
        key_map: dict[str, str],
        *,
        fixup_output_dir: Path | None = None,
        really_delete: bool = False,
        copy: bool = False,
    ):
        """
        Initializes the classifier GUI.

        Args:
            root: The root tkinter window.
            image_paths: A set of absolute paths to the images to be classified.
            key_map: A dictionary mapping keyboard keys to directory names.
            output_root: Directory into which classified files should be placed.
            fixup_output_dir: If provided, move committed images and metadata to this directory.
            really_delete: If True, permanently delete files.
            copy: If True, copy images instead of moving them.
        """
        self.root = root
        self.image_paths = sorted(image_paths)
        self.key_map = key_map
        self.output_dir = fixup_output_dir.expanduser() if fixup_output_dir else None
        self.copy = copy
        self.really_delete = really_delete
        self.class_to_id = {name: i for i, name in enumerate(sorted(self.key_map.values()))}
        self.id_to_class = {i: name for name, i in self.class_to_id.items()}
        self.colors = [
            "firebrick1",
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
            "thistle2",
            "MediumPurple1",
            "OliveDrab1",
            "cyan3",
            "bisque",
        ]

        self.root.title("Image Classifier")

        banner_text = " | ".join([f"'{key}': {folder}" for key, folder in self.key_map.items()])
        banner_text += "\nDraw boxes with mouse. Press key to label. Space: Save and next. `: Skip and next. Backspace: Undo. ESC: Quit"
        banner_text += "\n+/-/=: Zoom | Shift+Arrows: Pan | F5/F6: Cycle Boxes"
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
        self.bbox_undo_manager = UndoManager(self)
        self.file_undo_manager = UndoManager(self)
        self.pending_deletions = PendingDeletions()
        self._undo_clears_all_bounding_boxes = False
        self.selected_bbox_index = None

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
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.display_image()

    def on_closing(self):
        """Handles the window closing event."""
        self.pending_deletions.clear()
        self.root.destroy()

    def handle_right_click(self, event):
        del event
        if self.is_drawing:
            return
        if not self.delete_selected_bbox():
            self.bbox_undo_manager.undo(self)

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
        for i, bbox in enumerate(self.bboxes):
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

            if i == self.selected_bbox_index:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, stipple="gray25", width=0, tags="bbox")
                self.canvas.create_rectangle(
                    x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline="white", width=2, dash=(4, 4), tags="bbox"
                )

            if bbox["label"]:
                # Adjust label position if the box is near the top of the image
                font_size = 16
                y_offset = 5
                label_y = y1 - y_offset
                anchor = tk.SW
                if y1 < (font_size + y_offset):
                    label_y = y2 + y_offset
                    anchor = tk.NW

                label_item = self.canvas.create_text(
                    x1,
                    label_y,
                    text=bbox["label"],
                    fill=color,
                    anchor=anchor,
                    tags="bbox",
                    font=("Helvetica", font_size),
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

    def redraw_canvas_on_undo(self):
        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)

    def display_image(self):
        """Loads and displays the next image in the queue."""
        self.bbox_undo_manager.clear()
        self.bboxes = []
        self.canvas.delete("all")
        self.zoom_level = 1.0
        self.selected_bbox_index = None

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
            self.on_closing()
        elif key in ("f5", "f6"):
            if key == "f5":
                self.cycle_bbox_selection(1)
            else:
                self.cycle_bbox_selection(-1)
        elif key == "backspace":
            if self.is_drawing:
                return
            if not self.bbox_undo_manager.is_empty():
                self.bbox_undo_manager.undo(self)
            else:
                self.file_undo_manager.undo(self)
        elif key == "delete":
            self.handle_delete_key()
        elif key == "space":
            self.save_and_next()
        elif key in {"`", "~", "grave"}:
            self.skip_and_next()
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
        elif event.char.lower() in self.key_map:
            self.add_label(event.char.lower())

    def navigate(self, delta: int):
        """Navigates through the image list by a given delta."""
        if not self.image_paths:
            return
        new_index = (self.current_index + delta) % len(self.image_paths)
        self.current_index = new_index
        self.display_image()

    def cycle_bbox_selection(self, delta: int):
        """Cycles through the bounding box selection."""
        if not self.bboxes:
            return

        if self.selected_bbox_index is None:
            self.selected_bbox_index = 0 if delta > 0 else len(self.bboxes) - 1
        else:
            self.selected_bbox_index = (self.selected_bbox_index + delta) % len(self.bboxes)

        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)

    def _update_after_removal(self):
        if self.image_paths and self.current_index >= len(self.image_paths):
            self.current_index = len(self.image_paths) - 1

        self.display_image()

    def _load_json_metadata(self, json_path: Path):
        with open(json_path) as f:
            data = json.load(f)

        for label, bboxes in data.items():
            for bbox in bboxes:
                new_bbox = {
                    "x1": bbox["x1"],
                    "y1": bbox["y1"],
                    "x2": bbox["x2"],
                    "y2": bbox["y2"],
                    "label": label,
                }
                self.bboxes.append(new_bbox)

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

        filename = source_path.name
        if self.output_dir:
            output_dir = self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # Strip the __<object_count><class_name> suffixes from the filename
            name, ext = os.path.splitext(filename)
            new_name = re.sub(r"__\d+[a-zA-Z_]+", "", name)
            output_filename = new_name + ext

            dest_path = output_dir / output_filename
            if self.copy:
                try:
                    shutil.copy2(source_path, dest_path)
                except PermissionError:
                    shutil.copy(source_path, dest_path)
            else:
                shutil.move(source_path, dest_path)
            self.file_undo_manager.register_action(SaveFileAction(source_path, dest_path))
            self._save_json_format(output_filename, output_dir)
        else:
            output_dir = source_path.parent
            self._save_json_format(filename, output_dir)

        self._update_after_removal()

    def skip_and_next(self):
        """Skips the current image and removes it from the queue without modification."""
        source_path = Path(self.image_paths.pop(self.current_index))

        self.file_undo_manager.register_action(SkipFileAction(source_path))
        self._update_after_removal()

    def add_label(self, key: str):
        if not self.bboxes:
            return

        bbox_to_label = None
        if self.selected_bbox_index is not None:
            bbox_to_label = self.bboxes[self.selected_bbox_index]
            self.selected_bbox_index = None
        else:
            # Find the last unlabeled bbox
            for bbox in reversed(self.bboxes):
                if bbox["label"] is None:
                    bbox_to_label = bbox
                    break

        if bbox_to_label:
            old_label = bbox_to_label["label"]
            bbox_to_label["label"] = self.key_map[key]
            self.bbox_undo_manager.register_action(AddLabelAction(bbox_to_label, old_label))

        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)

    def delete_selected_bbox(self) -> bool:
        """Deletes the selected bounding box."""
        if self.selected_bbox_index is None:
            return False

        bbox_to_delete = self.bboxes[self.selected_bbox_index]
        self.bbox_undo_manager.register_action(DeleteBoundingBoxAction(bbox_to_delete, self.selected_bbox_index))
        del self.bboxes[self.selected_bbox_index]
        self.selected_bbox_index = None
        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)
        return True

    def handle_delete_key(self):
        """Handles the delete key press for soft or hard deletion of an image."""
        if not self.image_paths:
            return

        image_path = Path(self.image_paths.pop(self.current_index))
        json_path = image_path.with_suffix(".json")
        json_exists = json_path.exists()

        if self.really_delete:
            # Hard delete: move to temp dir and register undo action
            temp_dir = self.pending_deletions.temp_dir.name

            new_image_path = Path(temp_dir) / image_path.name
            print(f"🔥 Moving {image_path.name} to temp dir for deletion.")
            shutil.move(image_path, new_image_path)

            new_json_path = None
            if json_exists:
                new_json_path = Path(temp_dir) / json_path.name
                print(f"🔥 Moving {json_path.name} to temp dir for deletion.")
                shutil.move(json_path, new_json_path)

            action = DeleteFileAction(
                original_path=image_path,
                renamed_path=new_image_path,
                original_json_path=json_path if json_exists else None,
                renamed_json_path=new_json_path,
                is_hard_delete=True,
            )
            self.file_undo_manager.register_action(action)

        else:
            # Soft delete (rename)
            new_image_name = f"_DELETE__{image_path.name}"
            new_image_path = image_path.with_name(new_image_name)
            print(f"🗑️  Renaming to {new_image_name}")
            shutil.move(image_path, new_image_path)

            new_json_path = None
            if json_exists:
                new_json_name = f"_DELETE__{json_path.name}"
                new_json_path = json_path.with_name(new_json_name)
                print(f"🗑️  Renaming to {new_json_name}")
                shutil.move(json_path, new_json_path)

            action = DeleteFileAction(
                original_path=image_path,
                renamed_path=new_image_path,
                original_json_path=json_path if json_exists else None,
                renamed_json_path=new_json_path,
                is_hard_delete=False,
            )
            self.file_undo_manager.register_action(action)

        self._update_after_removal()

    def on_button_press(self, event):
        if self.bboxes and self.bboxes[-1]["label"] is None:
            self.bbox_undo_manager.undo(self)

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
        self.bbox_undo_manager.register_action(AddBoundingBoxAction(bbox))
        x, y = self.canvas.coords(self.image_on_canvas)
        self._redraw_canvas(x, y)  # Redraw to show the new box scaled correctly


def _find_sources(input_dirs: list[str], *, edit: bool = False, process_all: bool = False) -> set[str]:
    input_dirs = [Path(os.path.expanduser(input_dir)) for input_dir in input_dirs]

    combined_results = itertools.chain.from_iterable(base_path.rglob("*.*") for base_path in input_dirs)
    all_files = set(combined_results)

    def keep_file(filename: Path) -> bool:
        if not filename.is_file():
            return False

        if filename.name[0] == ".":
            return False

        if filename.suffix[1:].lower() not in {"jpg", "jpeg"}:
            return False

        if process_all:
            return True

        # Skip if a label file already exists
        base_filename = filename.stem
        parent_dir = filename.parent
        has_label = (parent_dir / f"{base_filename}.json").exists()
        if edit:
            return has_label
        return not has_label

    return {str(filename) for filename in all_files if keep_file(filename)}


def _run_gui(
    image_paths: set[str],
    key_map: dict[str, str],
    *,
    fixup_output_dir: Path | None = None,
    really_delete: bool = False,
    copy: bool = False,
):
    root = tk.Tk()
    ImageClassifierGUI(
        root,
        image_paths,
        key_map,
        fixup_output_dir=fixup_output_dir,
        really_delete=really_delete,
        copy=copy,
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
        "dirs",
        nargs="+",
        help="One or more source directories to search for JPGs recursively.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--edit",
        action="store_true",
        help="Edit existing classifications.",
    )
    group.add_argument(
        "--fixup",
        metavar="OUTPUT_DIR",
        help="Edit existing classifications and save to a new directory.",
    )
    parser.add_argument("--copy", action="store_true", help="In fixup mode, do not move or delete original files.")
    parser.add_argument(
        "--process-all-images", "-A", action="store_true", help="Process images without metadata in edit/fixup mode."
    )
    parser.add_argument(
        "--really-delete",
        action="store_true",
        help="Permanently delete files when the Delete key is pressed.",
    )
    args = parser.parse_args()

    key_map = load_key_map(args.config)

    image_paths = _find_sources(args.dirs, edit=args.edit or args.fixup, process_all=args.process_all_images)
    if not image_paths:
        print("No JPG images found in the specified directories. Exiting.")
        return 0

    print(f"Found {len(image_paths)} images to classify.\n")
    print("----------------\n")

    _run_gui(
        image_paths,
        key_map,
        fixup_output_dir=Path(args.fixup) if args.fixup else None,
        really_delete=args.really_delete,
        copy=args.copy,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
