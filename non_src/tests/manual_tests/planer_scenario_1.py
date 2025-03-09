import pathlib
import sys

repo_directory = pathlib.Path(__file__).parents[3].resolve()
sys.path.append(str(repo_directory))

from dotenv import find_dotenv, load_dotenv

from src.agents.planner_agent import planning
from non_src.tests.manual_tests.utils_for_tests import cleanup_work_dir, get_filenames_in_folder, setup_work_dir

load_dotenv(find_dotenv())

folder_with_project_files = repo_directory.joinpath("non_src/tests/manual_tests/projects_files", "planner_scenario_1_files")
tmp_folder = pathlib.Path(__file__).parent.resolve().joinpath("sandbox_work_dir")
setup_work_dir(manual_tests_folder=tmp_folder, test_files_dir=folder_with_project_files)

task = "Make form wider, with green background. Improve styling."
files = get_filenames_in_folder(manual_tests_folder=tmp_folder)

planning(task, files, image_paths={},work_dir=str(tmp_folder))
cleanup_work_dir(manual_tests_folder=tmp_folder)
