"""
Place here functions that should be called when clean coder is started.
"""

import os
import fnmatch
from termcolor import colored
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern


def read_frontend_feedback_story():
    frontend_feedback_story_path = os.path.join(Work.dir(), ".clean_coder", "frontend_feedback_story.txt")
    with open(frontend_feedback_story_path, "r") as file:
        return file.read()


def file_folder_ignored(path):
    path = path.rstrip("/")  # Remove trailing slash if present
    spec = PathSpec.from_lines(GitWildMatchPattern, CoderIgnore.get_forbidden())
    if spec.match_file(path):
        return True
    # old way of checking, to remove in future. For now still needed for checking exact folder matches
    for pattern in CoderIgnore.get_forbidden():
        pattern = pattern.rstrip("/")  # Remove trailing slash from pattern if present
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(f"{path}/", f"{pattern}/"):
            return True

    return False


class CoderIgnore:
    forbidden_files_and_folders = None

    @staticmethod
    def read_coderignore():
        coderignore_path = os.path.join(Work.dir(), '.clean_coder', '.coderignore')
        try:
            with open(coderignore_path, 'r') as file:
                return [line.strip() for line in file if line.strip() and not line.startswith('#')]
        except FileNotFoundError:
            print(f"File .codeignore was not found in {coderignore_path}. No files will be ignored.")
            return [] 

    @staticmethod
    def get_forbidden():
        if CoderIgnore.forbidden_files_and_folders is None:
            CoderIgnore.forbidden_files_and_folders = CoderIgnore.read_coderignore()
        return CoderIgnore.forbidden_files_and_folders


class Work:
    work_dir = None

    @staticmethod
    def read_work_dir():
        try:
            return os.environ["WORK_DIR"]
        except KeyError:
            raise Exception("Please set up your project folder as WORK_DIR parameter in .env")

    @staticmethod
    def dir():
        if Work.work_dir is None:
            Work.work_dir = Work.read_work_dir()
        return Work.work_dir


def print_ascii_logo():
    with open("non_src/assets/ascii-art.txt", "r") as f:
        logo = f.read()
    with open("non_src/assets/Clean_Coder_writing.txt", "r") as f:
        writing = f.read()
    print(colored(logo, color="yellow"))
    print(colored(writing, color="white"))


if __name__ == "__main__":
    read_frontend_feedback_story()
