if __name__ == "__main__":
    from src.utilities.start_work_functions import print_ascii_logo
    print_ascii_logo()

from dotenv import find_dotenv
from src.utilities.set_up_dotenv import set_up_env_coder_pipeline

if not find_dotenv():
    set_up_env_coder_pipeline()

from concurrent.futures import ThreadPoolExecutor
import os
from src.agents.researcher_agent import Researcher

from src.agents.planner_agent import planning
from src.agents.executor_agent import Executor
from src.agents.debugger_agent import Debugger
from src.agents.frontend_feedback import write_screenshot_codes
from src.utilities.user_input import user_input
from src.utilities.start_project_functions import set_up_dot_clean_coder_dir
from src.utilities.util_functions import create_frontend_feedback_story, join_paths
from src.utilities.script_execution_utils import run_script_in_env, format_log_message
from src.tools.rag.rag_utils import update_descriptions
from src.tools.rag.index_file_descriptions import prompt_index_project_files
from src.linters.static_analisys import python_static_analysis


os.environ["TOKENIZERS_PARALLELISM"] = "false"
use_frontend_feedback = bool(os.getenv("FRONTEND_URL"))
execute_file_name = os.getenv("EXECUTE_FILE_NAME")


def run_clean_coder_pipeline(task: str, work_dir: str, task_id: str=None):
    """Execute the complete Clean Coder pipeline including research, planning, execution, and debugging phases."""
    researcher = Researcher(task_id=task_id)
    files, image_paths = researcher.research_task(task)

    plan = planning(task, files, image_paths, work_dir)

    executor = Executor(files, work_dir)

    playwright_codes = None
    if use_frontend_feedback:
        create_frontend_feedback_story()
        with ThreadPoolExecutor() as executor_thread:
            future = executor_thread.submit(write_screenshot_codes, task, plan, work_dir)
            files = executor.do_task(task, plan)
            playwright_codes = future.result()
    else:
        files = executor.do_task(task, plan)

    # Execute the script and collect logs
    execution_message = None
    if execute_file_name and os.path.exists(join_paths(work_dir, execute_file_name)):
        stdout, stderr = run_script_in_env(work_dir, execute_file_name, silent_setup=True)
        execution_message = format_log_message(stdout=stdout, stderr=stderr)

    # static analysis
    files_to_check = [file for file in files if file.filename.endswith(".py") and file.is_modified]
    analysis_result = python_static_analysis(files_to_check)
    if analysis_result:
        # Automatically proceed to debugger with static analysis results
        human_message = analysis_result
        if execution_message:
            human_message = execution_message + "\n\n" + human_message
    else:
        # If we have logs of the script execution, add them to the message
        if execution_message:
            print(execution_message)

        # No static analysis issues - ask for user input
        human_message = user_input(
            "Please test app and provide commentary if debugging/additional refinement is needed. "
        )
        if human_message in ["o", "ok"]:
            update_descriptions([file for file in files if file.is_modified])
            return

        if execution_message:
            human_message = execution_message + "\n\n" + human_message


    debugger = Debugger(files, work_dir, human_message, image_paths, playwright_codes)
    files = debugger.do_task(task, plan)
    update_descriptions([file for file in files if file.is_modified])



if __name__ == "__main__":
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        raise Exception("WORK_DIR variable is not provided. Please add WORK_DIR to .env file")
    set_up_dot_clean_coder_dir(work_dir)
    prompt_index_project_files()
    task = user_input("Provide task to be executed. ")
    run_clean_coder_pipeline(task, work_dir)
