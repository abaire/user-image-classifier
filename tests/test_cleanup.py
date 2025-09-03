from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from user_image_classifier.cleanup import cleanup_images

if TYPE_CHECKING:
    from pathlib import Path


def create_dummy_image(path: Path, size: tuple[int, int] = (10, 10)):
    """Creates a dummy image file."""
    img = Image.new("RGB", size, color="red")
    img.save(path, "JPEG")

    metadata = path.with_suffix(".json")
    with open(metadata, "w") as outfile:
        outfile.write("{}")


def test_cleanup_images_dry_run(tmp_path: Path):
    """
    Tests that cleanup_images with dry_run=True identifies but does not
    delete duplicate files.
    """
    # 1. Setup: Create dummy files and duplicates
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    dir2 = tmp_path / "dir2"
    dir2.mkdir()

    image_path1 = dir1 / "image1.jpg"
    image_path2 = dir2 / "image2.jpg"
    image_path3 = dir1 / "image3.jpg"  # a different image

    create_dummy_image(image_path1)
    create_dummy_image(image_path2)
    create_dummy_image(image_path3, size=(5, 5))

    # 2. Run cleanup_images with dry_run
    deleted_files = cleanup_images([str(tmp_path)], dry_run=True)

    # 3. Assert results
    assert deleted_files == 1
    assert image_path1.exists()
    assert image_path2.exists()
    assert image_path3.exists()


def test_cleanup_images_deletes_duplicates(tmp_path: Path):
    """
    Tests that cleanup_images correctly identifies and deletes duplicate files.
    """
    # 1. Setup: Create dummy files and duplicates
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    dir2 = tmp_path / "dir2"
    dir2.mkdir()

    image_path1 = dir1 / "image1.jpg"
    image_path2 = dir2 / "image2.jpg"
    image_path3 = dir1 / "image3.jpg"  # a different image

    create_dummy_image(image_path1)
    create_dummy_image(image_path2)
    create_dummy_image(image_path3, size=(5, 5))

    # 2. Run cleanup_images
    deleted_files = cleanup_images([str(tmp_path)], dry_run=False)

    # 3. Assert results
    assert deleted_files == 1
    assert image_path1.exists()
    assert not image_path2.exists()
    assert image_path3.exists()
