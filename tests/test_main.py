import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from user_image_classifier.config import DEFAULT_CONFIG
from user_image_classifier.main import ImageClassifierGUI, _load_key_map

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from user_image_classifier.main import _find_sources


def test_find_sources_no_images(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (input_dir / "a.txt").touch()
    (input_dir / "b.png").touch()

    found_files = _find_sources([str(input_dir)], str(output_dir))
    assert found_files == set()


def test_find_sources(tmp_path: Path):
    # Setup directories
    input_dir1 = tmp_path / "input1"
    input_dir1.mkdir()
    input_dir1_sub = input_dir1 / "sub"
    input_dir1_sub.mkdir()

    input_dir2 = tmp_path / "input2"
    input_dir2.mkdir()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create dummy files
    (input_dir1 / "image1.jpg").touch()
    (input_dir1_sub / "image2.jpeg").touch()
    (input_dir1 / "not_an_image.txt").touch()
    (input_dir2 / "image3.JPG").touch()  # Test case-insensitivity
    (output_dir / "image4.jpg").touch()  # Should be ignored because it is in the output dir
    (input_dir1 / "image5.png").touch()  # Should be ignored because of extension
    (input_dir1 / "an_actual_dir").mkdir()  # Should be ignored because it is a directory

    # Expected files
    expected_files = {
        Path(input_dir1 / "image1.jpg"),
        Path(input_dir1_sub / "image2.jpeg"),
        Path(input_dir2 / "image3.JPG"),
    }

    # Run the function
    # The function expects string paths as input
    found_files = _find_sources([str(input_dir1), str(input_dir2)], str(output_dir))

    # Assert the result
    assert found_files == expected_files


def test_load_key_map_with_config_file(tmp_path: Path):
    config_data = {"a": "dir_a", "b": "dir_b"}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    key_map = _load_key_map(str(config_file))
    assert key_map == config_data


def test_load_key_map_no_config_file():
    key_map = _load_key_map(None)
    assert key_map == DEFAULT_CONFIG


@pytest.fixture
def mock_gui(tmp_path: Path):
    """Fixture to set up a mock ImageClassifierGUI instance."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()

    image_paths = set()
    for i in range(3):
        p = image_dir / f"image{i}.jpg"
        # Create a dummy file
        with open(p, "w") as f:
            f.write("dummy image data")
        image_paths.add(str(p))

    key_map = {"a": "class_a", "b": "class_b"}
    for folder in key_map.values():
        (output_root / folder).mkdir()

    with patch("tkinter.Tk"), patch("tkinter.Label"), patch("PIL.ImageTk.PhotoImage"), patch(
        "PIL.Image.open"
    ) as mock_open:
        mock_image = MagicMock()
        mock_image.size = (100, 100)
        mock_image.resize.return_value = mock_image
        mock_open.return_value = mock_image

        root = MagicMock()
        root.winfo_screenwidth.return_value = 800
        root.winfo_screenheight.return_value = 600

        gui = ImageClassifierGUI(root, image_paths, key_map, str(output_root))
        yield gui, tmp_path


def test_gui_initialization(mock_gui):
    gui, _ = mock_gui
    assert gui.root.title.call_count > 0
    assert len(gui.image_paths) == 3
    assert gui.current_index == 0


def test_navigate(mock_gui):
    gui, _ = mock_gui

    # Navigate right
    gui.navigate(1)
    assert gui.current_index == 1

    # Navigate right again
    gui.navigate(1)
    assert gui.current_index == 2

    # Navigate right (wrap around)
    gui.navigate(1)
    assert gui.current_index == 0

    # Navigate left
    gui.navigate(-1)
    assert gui.current_index == 2


def test_move_image(mock_gui):
    gui, tmp_path = mock_gui
    initial_image_count = len(gui.image_paths)

    # current image path before move
    current_image_path = Path(gui.image_paths[gui.current_index])

    gui.move_image("a")

    assert len(gui.image_paths) == initial_image_count - 1

    dest_dir = Path(tmp_path / "output" / "class_a")
    moved_file_path = dest_dir / current_image_path.name
    assert moved_file_path.exists()
    assert not current_image_path.exists()


def test_abandon_file(mock_gui):
    gui, _ = mock_gui
    initial_image_count = len(gui.image_paths)

    gui.abandon_file()

    assert len(gui.image_paths) == initial_image_count - 1
    assert gui.last_move is not None
    assert gui.last_move[0] is None  # No destination path for abandon


def test_undo_last_move(mock_gui):
    gui, tmp_path = mock_gui

    current_image_path = Path(gui.image_paths[gui.current_index])
    gui.move_image("a")

    assert not current_image_path.exists()

    gui.undo_last_move()

    assert len(gui.image_paths) == 3
    assert current_image_path.exists()
    assert gui.last_move is None


def test_undo_after_abandon(mock_gui):
    gui, _ = mock_gui
    initial_image_count = len(gui.image_paths)
    original_path = gui.image_paths[gui.current_index]

    gui.abandon_file()
    assert len(gui.image_paths) == initial_image_count - 1

    gui.undo_last_move()
    assert len(gui.image_paths) == initial_image_count
    assert gui.image_paths[gui.current_index] == original_path
    assert gui.last_move is None


@pytest.mark.parametrize(
    ("min_confidence", "max_confidence", "use_secondary", "expected_images"),
    [
        (None, None, False, ["C50_a.jpg", "C80_fox_C20_bear_b.jpg", "c.jpg"]),
        (60, None, False, ["C80_fox_C20_bear_b.jpg", "c.jpg"]),
        (None, 60, False, ["C50_a.jpg", "c.jpg"]),
        (None, None, True, ["C50_a.jpg", "C80_fox_C20_bear_b.jpg", "c.jpg"]),
        (10, 30, True, ["C50_a.jpg", "C80_fox_C20_bear_b.jpg", "c.jpg"]),
        (90, None, True, ["C50_a.jpg", "c.jpg"]),
    ],
)
def test_confidence_filtering(tmp_path: Path, min_confidence, max_confidence, use_secondary, expected_images):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()

    image_paths = set()
    for filename in ["C50_a.jpg", "C80_fox_C20_bear_b.jpg", "c.jpg"]:
        p = image_dir / filename
        p.touch()
        image_paths.add(str(p))

    with patch("tkinter.Tk"), patch("tkinter.Label"), patch("PIL.ImageTk.PhotoImage"), patch(
        "PIL.Image.open"
    ) as mock_open:
        mock_image = MagicMock()
        mock_image.size = (100, 100)
        mock_image.resize.return_value = mock_image
        mock_open.return_value = mock_image

        root = MagicMock()
        root.winfo_screenwidth.return_value = 800
        root.winfo_screenheight.return_value = 600

        gui = ImageClassifierGUI(
            root,
            image_paths,
            {},
            str(output_root),
            min_confidence_threshold=min_confidence,
            max_confidence_threshold=max_confidence,
            use_secondary_confidence=use_secondary,
        )

        assert len(gui.image_paths) == len(expected_images)
        assert {os.path.basename(p) for p in gui.image_paths} == set(expected_images)
