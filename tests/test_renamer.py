from __future__ import annotations

# ruff: noqa: DTZ001 `datetime.datetime()` called without a `tzinfo` argument
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

from user_image_classifier.renamer import rename_files, undo_rename

if TYPE_CHECKING:
    from pathlib import Path


def test_rename_and_undo(tmp_path: Path):
    # 1. Setup: Create dummy files
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create dummy image file
    image_path = input_dir / "test_image.jpg"
    image_path.touch()

    # Create dummy label file
    label_path = input_dir / "test_image.txt"
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.1 0.1\n")  # class 0

    class_map = {"0": "cat", "1": "dog"}

    # Mock get_image_datetime to return a fixed timestamp
    mock_datetime = datetime(2025, 1, 1, 12, 0, 0)

    # 2. Run rename_files
    with patch("user_image_classifier.renamer.get_image_datetime", return_value=mock_datetime):
        rename_files(str(input_dir), class_map)

    # 3. Assert rename_files results
    expected_new_filename_stem = f"{mock_datetime}_1cat--test_image"
    renamed_image_path = input_dir / f"{expected_new_filename_stem}.jpg"
    renamed_label_path = input_dir / f"{expected_new_filename_stem}.txt"

    assert renamed_image_path.exists()
    assert renamed_label_path.exists()
    assert not image_path.exists()
    assert not label_path.exists()

    # 4. Run undo_rename
    undo_rename(str(input_dir))

    # 5. Assert undo_rename results
    assert image_path.exists()
    assert label_path.exists()
    assert not renamed_image_path.exists()
    assert not renamed_label_path.exists()
