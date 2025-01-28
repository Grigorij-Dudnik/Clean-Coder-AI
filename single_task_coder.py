if __name__ == "__main__":
    from src.utilities.start_work_functions import print_ascii_logo
    print_ascii_logo()
from dotenv import find_dotenv
from src.utilities.set_up_dotenv import set_up_env_coder_pipeline
if not find_dotenv():
    set_up_env_coder_pipeline()

from src.agents.researcher_agent import Researcher
from src.agents.doc_harvester import DocHarvester
from src.agents.planner_agent import planning
from src.agents.executor_agent import Executor
from src.agents.debugger_agent import Debugger
from src.agents.frontend_feedback import write_screenshot_codes
import os
from src.utilities.exceptions import MissingEnvironmentVariableError
from src.utilities.user_input import user_input
from src.utilities.start_project_functions import set_up_dot_clean_coder_dir
from src.utilities.util_functions import create_frontend_feedback_story
from concurrent.futures import ThreadPoolExecutor


use_frontend_feedback = bool(os.getenv("FRONTEND_URL"))


def run_clean_coder_pipeline(task: str, work_dir: str, doc_harvest: bool = False) -> None:
    """Single run of clean code pipeline to address a task., from Researcher to Debugger."""
    researcher = Researcher(work_dir)
    file_paths, image_paths = researcher.research_task(task)
    documentation = None
    if doc_harvest:
        harvester = DocHarvester()
        documentation = harvester.find_documentation(task)

    plan = planning(task, file_paths, image_paths, work_dir, documentation=documentation)

    executor = Executor(file_paths, work_dir)

    playwright_codes = None
    if use_frontend_feedback:
        create_frontend_feedback_story()
        with ThreadPoolExecutor() as executor_thread:
            future = executor_thread.submit(write_screenshot_codes, task, plan, work_dir)
            file_paths = executor.do_task(task, plan)
            playwright_codes = future.result()
    else:
        file_paths = executor.do_task(task, plan)

    human_message = user_input("Please test app and provide commentary if debugging/additional refinement is needed. ")
    if human_message in ["o", "ok"]:
        return
    debugger = Debugger(
        file_paths, work_dir, human_message,image_paths,  playwright_codes)
    debugger.do_task(task, plan)


if __name__ == "__main__":
    work_dir = os.getenv("WORK_DIR")
    set_up_dot_clean_coder_dir(work_dir)
    task = user_input("Provide task to be executed. ")
    if not work_dir:
        msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
        raise MissingEnvironmentVariableError(msg)
    run_clean_coder_pipeline(task, work_dir)
