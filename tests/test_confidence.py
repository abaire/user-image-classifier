import pytest

from user_image_classifier.main import (
    _get_confidences,
    _remove_confidence_substring,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("C99_myfile.jpg", (99, None)),
        ("C99_foxes_C88_empty_myfile.jpg", (99, 88)),
        ("C99_mountain_lions_C88_empty_myfile.jpg", (99, 88)),
        ("C99_mountain-lions_C88_empty_myfile.jpg", (99, 88)),
        ("myfile_C77.jpg", (77, None)),
        ("myfile_C77_foxes_C66_empty.jpg", (77, 66)),
        ("myfile.jpg", (None, None)),
    ],
)
def test_get_confidences(filename, expected):
    assert _get_confidences(filename) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("C99_myfile.jpg", "myfile.jpg"),
        ("C99_foxes_C88_empty_myfile.jpg", "myfile.jpg"),
        ("C99_mountain_lions_C88_empty_myfile.jpg", "myfile.jpg"),
        ("C99_mountain-lions_C88_empty_myfile.jpg", "myfile.jpg"),
        ("myfile_C77.jpg", "myfile.jpg"),
        ("myfile_C77_foxes_C66_empty.jpg", "myfile.jpg"),
        ("myfile.jpg", "myfile.jpg"),
    ],
)
def test_remove_confidence_substring(filename, expected):
    assert _remove_confidence_substring(filename) == expected
