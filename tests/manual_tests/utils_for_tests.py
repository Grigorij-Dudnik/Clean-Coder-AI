"""Universal utility functions for manual (and not only) tests."""
import os
import shutil


# def setup_work_dir(project_files_folder):
def setup_work_dir(project_files_dir: str) -> None:
    """Sets up a temporary project directory from template."""
    if os.path.exists("sandbox_work_dir"):
        cleanup_work_dir()
    os.makedirs("sandbox_work_dir")
    shutil.copytree(project_files_dir, "sandbox_work_dir", dirs_exist_ok=True)


def cleanup_work_dir():
    shutil.rmtree("sandbox_work_dir")


def get_filenames_in_folder(project_files_dir: str):
    # Initialize an empty set to store filenames
    filenames = set()

    # List all files in the directory
    for file in os.listdir(project_files_dir):
        file_path = os.path.join(project_files_dir, file)
        if os.path.isfile(file_path):
            # Add the filename to the set
            filenames.add(file)

    return filenames
