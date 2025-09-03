from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from user_image_classifier.main import (
    AddBoundingBoxAction,
    AddLabelAction,
    DeleteBoundingBoxAction,
    DeleteFileAction,
    ImageClassifierGUI,
    PendingDeletions,
    SaveFileAction,
    UndoManager,
)


@pytest.fixture
def mock_gui():
    """Fixture to set up a mock ImageClassifierGUI instance."""
    gui = MagicMock(spec=ImageClassifierGUI)
    gui.bboxes = []
    gui.image_paths = []
    gui.current_index = 0
    gui.bbox_undo_manager = UndoManager(gui)
    gui.file_undo_manager = UndoManager(gui)
    gui.pending_deletions = MagicMock(spec=PendingDeletions)
    return gui


def test_undo_manager_register(mock_gui):
    manager = UndoManager(mock_gui)
    action = MagicMock()
    manager.register_action(action)
    assert len(manager.stack) == 1
    assert manager.stack[0] == action


def test_undo_manager_undo(mock_gui):
    manager = UndoManager(mock_gui)
    action = MagicMock()
    manager.register_action(action)
    manager.undo(mock_gui)
    action.undo.assert_called_once_with(mock_gui)
    assert len(manager.stack) == 0


def test_undo_manager_clear(mock_gui):
    manager = UndoManager(mock_gui)
    manager.register_action(MagicMock())
    manager.clear()
    assert len(manager.stack) == 0


def test_undo_manager_max_size(mock_gui):
    manager = UndoManager(mock_gui, max_size=2)
    action1 = MagicMock()
    action2 = MagicMock()
    action3 = MagicMock()
    manager.register_action(action1)
    manager.register_action(action2)
    manager.register_action(action3)
    assert len(manager.stack) == 2
    assert manager.stack[0] == action2
    assert manager.stack[1] == action3


def test_add_bbox_action_undo(mock_gui):
    bbox = {"x1": 0, "y1": 0, "x2": 1, "y2": 1}
    mock_gui.bboxes.append(bbox)
    action = AddBoundingBoxAction(bbox)
    action.undo(mock_gui)
    assert bbox not in mock_gui.bboxes
    mock_gui.redraw_canvas_on_undo.assert_called_once()


def test_delete_bbox_action_undo(mock_gui):
    bbox = {"x1": 0, "y1": 0, "x2": 1, "y2": 1}
    action = DeleteBoundingBoxAction(bbox, 0)
    action.undo(mock_gui)
    assert mock_gui.bboxes[0] == bbox
    mock_gui.redraw_canvas_on_undo.assert_called_once()


def test_add_label_action_undo(mock_gui):
    bbox = {"label": "new_label"}
    action = AddLabelAction(bbox, "old_label")
    action.undo(mock_gui)
    assert bbox["label"] == "old_label"
    mock_gui.redraw_canvas_on_undo.assert_called_once()


def test_save_file_action_undo(mock_gui, tmp_path):
    source_path = tmp_path / "source.txt"
    dest_path = tmp_path / "dest.txt"
    dest_path.touch()

    action = SaveFileAction(source_path, dest_path)
    action.undo(mock_gui)

    assert source_path.exists()
    assert not dest_path.exists()
    assert mock_gui.image_paths[0] == str(source_path)
    mock_gui.display_image.assert_called_once()


def test_delete_file_action_undo(mock_gui, tmp_path):
    original_path = tmp_path / "original.txt"
    renamed_path = tmp_path / "renamed.txt"
    original_json_path = tmp_path / "original.json"
    renamed_json_path = tmp_path / "renamed.json"

    renamed_path.touch()
    renamed_json_path.touch()

    action = DeleteFileAction(original_path, renamed_path, original_json_path, renamed_json_path)
    action.undo(mock_gui)

    assert original_path.exists()
    assert original_json_path.exists()
    assert not renamed_path.exists()
    assert not renamed_json_path.exists()
    assert mock_gui.image_paths[0] == str(original_path)
    mock_gui.display_image.assert_called_once()


def test_delete_file_action_finalize_hard_delete(mock_gui):
    original_path = MagicMock()
    renamed_path = MagicMock()
    original_json_path = MagicMock()
    renamed_json_path = MagicMock()
    action = DeleteFileAction(
        original_path,
        renamed_path,
        original_json_path,
        renamed_json_path,
        is_hard_delete=True,
    )
    mock_gui.pending_deletions = MagicMock()

    action.finalize(mock_gui)

    mock_gui.pending_deletions.add.assert_any_call(renamed_path)
    mock_gui.pending_deletions.add.assert_any_call(renamed_json_path)


def test_delete_file_action_finalize_soft_delete(mock_gui):
    original_path = MagicMock()
    renamed_path = MagicMock()
    original_json_path = MagicMock()
    renamed_json_path = MagicMock()
    action = DeleteFileAction(
        original_path,
        renamed_path,
        original_json_path,
        renamed_json_path,
        is_hard_delete=False,
    )
    mock_gui.pending_deletions = MagicMock()

    action.finalize(mock_gui)

    mock_gui.pending_deletions.add.assert_not_called()


def test_pending_deletions():
    with patch("tempfile.TemporaryDirectory") as mock_temp_dir:
        mock_path = MagicMock()

        manager = PendingDeletions()
        manager.add(mock_path)

        assert mock_path in manager.files_to_delete

        manager.clear()

        mock_path.unlink.assert_called_once_with(missing_ok=True)
        assert not manager.files_to_delete
        mock_temp_dir.return_value.cleanup.assert_called_once()
