import pathlib
import sys

repo_directory = pathlib.Path(__file__).parents[2].resolve()
sys.path.append(str(repo_directory))

from dotenv import find_dotenv, load_dotenv

from src.agents.planner_agent import planning
from tests.manual_tests.utils_for_tests import cleanup_work_dir, get_filenames_in_folder, setup_work_dir

load_dotenv(find_dotenv())

folder_with_project_files = str(repo_directory.joinpath("tests", "manual_tests", "projects_files", "planner_scenario_1_files"))
setup_work_dir(folder_with_project_files)

task = "Make form wider, with green background. Improve styling."
files = get_filenames_in_folder(folder_with_project_files)

planning(task, files, image_paths={},work_dir="sandbox_work_dir")
cleanup_work_dir()
