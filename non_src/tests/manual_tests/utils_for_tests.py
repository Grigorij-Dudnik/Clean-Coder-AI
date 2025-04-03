"""Universal utility functions for manual (and not only) tests."""

import os
import shutil
from pathlib import Path
from src.utilities.objects import CodeFile


# def setup_work_dir(project_files_folder):
def setup_work_dir(test_files_dir: Path, manual_tests_folder: Path) -> None:
    """Sets up a temporary project directory from template."""
    if manual_tests_folder.exists():
        cleanup_work_dir(manual_tests_folder=manual_tests_folder)
    manual_tests_folder.mkdir()
    shutil.copytree(str(test_files_dir), str(manual_tests_folder), dirs_exist_ok=True)


def cleanup_work_dir(manual_tests_folder: Path) -> None:
    """Removes a project directory."""
    shutil.rmtree(str(manual_tests_folder))


def get_files_in_folder(manual_tests_folder: Path) -> set[CodeFile]:
    """Lists files in directory."""
    # Initialize an empty set to store filenames
    files = set()

    # List all files in the directory
    for filename in os.listdir(manual_tests_folder):
        file_path = os.path.join(manual_tests_folder, filename)
        if Path(file_path).is_file():
            # Add the filename to the set
            files.add(CodeFile(filename))

    return files
