from __future__ import annotations

# ruff: noqa: SLF001 Private member accessed
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from user_image_classifier.config import DEFAULT_CONFIG, load_key_map
from user_image_classifier.main import AddBoundingBoxAction, ImageClassifierGUI, _find_sources


def test_find_sources_no_images(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (input_dir / "a.txt").touch()
    (input_dir / "b.png").touch()

    found_files = _find_sources([str(input_dir)])
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
        str(Path(input_dir1 / "image1.jpg")),
        str(Path(input_dir1_sub / "image2.jpeg")),
        str(Path(input_dir2 / "image3.JPG")),
    }

    # Run the function
    # The function expects string paths as input
    found_files = _find_sources([str(input_dir1), str(input_dir2)])

    # Assert the result
    assert found_files == expected_files


def test_find_sources_skips_existing(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    (input_dir / "image1.jpg").touch()
    (input_dir / "image2.jpg").touch()
    (input_dir / "image2.json").touch()  # Pre-existing label for image2

    found_files = _find_sources([str(input_dir)])
    assert found_files == {str(Path(input_dir / "image1.jpg"))}


def test_load_key_map_with_config_file(tmp_path: Path):
    config_data = {"a": "dir_a", "b": "dir_b"}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    key_map = load_key_map(str(config_file))
    assert key_map == config_data


def test_load_key_map_no_config_file():
    key_map = load_key_map(None)
    assert key_map == DEFAULT_CONFIG


@pytest.fixture
def mock_gui(tmp_path: Path):
    """Fixture to set up a mock ImageClassifierGUI instance."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    image_paths = set()
    for i in range(3):
        p = image_dir / f"image{i}.jpg"
        # Create a dummy file
        with open(p, "w") as f:
            f.write("dummy image data")
        image_paths.add(str(p))

    key_map = {"a": "class_a", "b": "class_b"}

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

        gui = ImageClassifierGUI(root, image_paths, key_map)
        yield gui, image_dir


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


def test_save_and_next(mock_gui):
    gui, image_dir = mock_gui
    initial_image_count = len(gui.image_paths)
    current_image_path = Path(gui.image_paths[gui.current_index])
    gui.canvas.delete = MagicMock()

    # Simulate drawing and labeling two boxes
    gui.bboxes = [
        {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "class_a"},
        {"x1": 60, "y1": 60, "x2": 100, "y2": 100, "label": "class_b"},
        {"x1": 20, "y1": 20, "x2": 40, "y2": 40, "label": "class_a"},
    ]

    gui.save_and_next()

    assert len(gui.image_paths) == initial_image_count - 1

    json_filename = current_image_path.stem + ".json"
    output_path = image_dir / json_filename
    assert output_path.exists()

    with open(output_path) as f:
        data = json.load(f)

    expected_data = {
        "class_a": [{"x1": 10, "y1": 10, "x2": 50, "y2": 50}, {"x1": 20, "y1": 20, "x2": 40, "y2": 40}],
        "class_b": [{"x1": 60, "y1": 60, "x2": 100, "y2": 100}],
    }
    assert data == expected_data


def test_save_and_next_empty_json(mock_gui):
    gui, image_dir = mock_gui
    initial_image_count = len(gui.image_paths)
    current_image_path = Path(gui.image_paths[gui.current_index])
    gui.bboxes = []

    gui.save_and_next()

    assert len(gui.image_paths) == initial_image_count - 1
    json_filename = current_image_path.stem + ".json"
    output_path = image_dir / json_filename
    assert output_path.exists()
    assert output_path.read_text() == "{}"


def test_keyboard_pan(mock_gui):
    gui, _ = mock_gui
    gui.canvas.xview_scroll = MagicMock()
    gui.canvas.yview_scroll = MagicMock()

    # Mock the event object
    mock_event = MagicMock()
    mock_event.state = 0x0001  # Shift key pressed

    mock_event.keysym = "Right"
    gui.handle_key_press(mock_event)
    gui.canvas.xview_scroll.assert_called_with(1, "units")

    mock_event.keysym = "Up"
    gui.handle_key_press(mock_event)
    gui.canvas.yview_scroll.assert_called_with(-1, "units")


def test_keyboard_zoom(mock_gui):
    gui, _ = mock_gui
    gui.image_on_canvas = 1
    gui.canvas.coords = MagicMock(return_value=(0, 0))
    gui.canvas.winfo_width = MagicMock(return_value=200)
    gui.canvas.winfo_height = MagicMock(return_value=200)
    gui._redraw_canvas = MagicMock()

    # Mock the event object
    mock_event = MagicMock()
    mock_event.char = "+"
    mock_event.keysym = ""

    gui.handle_key_press(mock_event)
    assert gui.zoom_level == 1.1
    gui._redraw_canvas.assert_called_once()


def test_coordinate_saving_with_zoom(mock_gui):
    gui, _ = mock_gui
    gui.image_width = 100
    gui.image_height = 100
    gui.zoom_level = 2.0  # Zoom in 2x
    gui.image_on_canvas = 1
    gui.canvas.coords = MagicMock(return_value=(-50, -50))  # Simulate pan
    gui._redraw_canvas = MagicMock()

    # Mock the event object for drawing
    mock_event = MagicMock()
    mock_event.x = 20  # View coordinates
    mock_event.y = 20
    gui.on_button_press(mock_event)
    mock_event.x = 40
    mock_event.y = 60
    gui.on_button_release(mock_event)

    assert len(gui.bboxes) == 1
    bbox = gui.bboxes[0]
    # Expected: (view_coord - img_coord) / zoom
    assert bbox["x1"] == (20 - (-50)) / 2.0  # 35.0
    assert bbox["y1"] == (20 - (-50)) / 2.0  # 35.0
    assert bbox["x2"] == (40 - (-50)) / 2.0  # 45.0
    assert bbox["y2"] == (60 - (-50)) / 2.0  # 55.0


def test_drawing_new_box_discards_unlabeled(mock_gui):
    gui, _ = mock_gui
    gui.image_width = 100
    gui.image_height = 100
    gui.image_on_canvas = 1
    gui.canvas.coords = MagicMock(return_value=(0, 0))
    gui._redraw_canvas = MagicMock()
    mock_event = MagicMock()

    # Setup: one unlabeled box exists
    bbox = {"x1": 10, "y1": 10, "x2": 20, "y2": 20, "label": None}
    gui.bboxes = [bbox]
    gui.bbox_undo_manager.register_action(AddBoundingBoxAction(bbox))
    assert len(gui.bboxes) == 1

    # Mock the undo method
    gui.bbox_undo_manager.undo = MagicMock()

    # Action: Start drawing a new box
    mock_event.x, mock_event.y = 30, 30
    gui.on_button_press(mock_event)

    # Assertions
    gui.bbox_undo_manager.undo.assert_called_once_with(gui)


def test_drawing_new_box_retains_labeled(mock_gui):
    gui, _ = mock_gui
    gui.image_width = 100
    gui.image_height = 100
    gui.image_on_canvas = 1
    gui.canvas.coords = MagicMock(return_value=(0, 0))
    gui._redraw_canvas = MagicMock()
    mock_event = MagicMock()

    # Draw first box
    mock_event.x, mock_event.y = 10, 10
    gui.on_button_press(mock_event)
    mock_event.x, mock_event.y = 20, 20
    gui.on_button_release(mock_event)
    assert len(gui.bboxes) == 1

    # Label the first box
    gui.add_label("a")
    assert gui.bboxes[0]["label"] == "class_a"

    # Draw second box
    mock_event.x, mock_event.y = 30, 30
    gui.on_button_press(mock_event)
    assert len(gui.bboxes) == 1  # Should not have discarded the labeled box


def test_zoom_centering_logic(mock_gui):
    gui, _ = mock_gui
    gui.image_on_canvas = 1  # Mock canvas item ID
    gui.canvas.coords = MagicMock(return_value=(0, 0))
    gui._redraw_canvas = MagicMock()

    # Mock the event object for mouse zoom
    mock_event = MagicMock()
    mock_event.x = 100
    mock_event.y = 100
    mock_event.num = 4  # Scroll up
    mock_event.delta = 120

    gui.zoom(event=mock_event)

    # Check that _redraw_canvas is called with the correct new coordinates
    # Expected: new_img_x = cx - ( (cx - img_x) / zoom_level * new_zoom_level )
    # cx=100, img_x=0, zoom_level=1.0, new_zoom_level=1.1
    # new_img_x = 100 - ( (100 - 0) / 1.0 * 1.1 ) = 100 - 110 = -10
    gui._redraw_canvas.assert_called_with(pytest.approx(-10.0), pytest.approx(-10.0))


def test_drawing_state_and_crosshair(mock_gui):
    gui, _ = mock_gui
    gui.image_width = 100
    gui.image_height = 100
    gui.image_on_canvas = 1
    gui.canvas.coords = MagicMock(return_value=(0, 0))
    gui._delete_crosshair = MagicMock()
    gui._redraw_canvas = MagicMock()  # Mock this to prevent it from running
    mock_event = MagicMock()
    mock_event.x = 1
    mock_event.y = 2

    # Initial state
    assert not gui.is_drawing

    # Press button
    gui.on_button_press(mock_event)
    assert gui.is_drawing
    gui._delete_crosshair.assert_called_once()

    # Release button
    gui.on_button_release(mock_event)
    assert not gui.is_drawing
