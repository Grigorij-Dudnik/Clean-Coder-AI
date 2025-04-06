from langchain.tools import tool
from typing_extensions import Annotated
from todoist_api_python.api import TodoistAPI
import os
from src.utilities.print_formatters import print_formatted, print_text_snippet
from src.utilities.manager_utils import actualize_progress_description_file
from src.utilities.user_input import user_input
from src.utilities.graphics import task_completed_animation
from src.utilities.util_functions import join_paths
from dotenv import load_dotenv, find_dotenv
from single_task_coder import run_clean_coder_pipeline
import uuid
import requests
from requests.exceptions import HTTPError
import json


load_dotenv(find_dotenv())
load_dotenv()

work_dir = os.getenv("WORK_DIR")
load_dotenv(join_paths(work_dir, ".clean_coder/.env"))
todoist_api_key = os.getenv("TODOIST_API_KEY")
todoist_api = TodoistAPI(todoist_api_key)


@tool
def add_task(
    task_name: Annotated[
        str,
        "Name of the task. Good name is descriptive, starts with a verb and usually could be fitted in formula 'To complete this task, I need to $TASK_NAME'",
    ],
    task_description: Annotated[
        str,
        "Detailed description of what needs to be done in order to implement task. Good description includes: - Definition of done (required) - section, describing what need to be done with acceptance criteria. - Resources (optional) - Include here all information that will be helpful for developer to complete task",
    ],
    order: Annotated[int, "Order of the task in project"],
):
    """Add new task to Todoist.
    Think very carefully before adding a new task to know what do you want exactly. Explain in detail what needs to be
    done in order to execute task.
    Avoid creating new tasks that have overlapping scope with old ones - modify or delete old tasks first.
    """
    human_message = user_input("Type (o)k to agree or provide commentary.")
    if human_message not in ["o", "ok"]:
        return f"Action wasn't executed because of human interruption. He said: {human_message}"

    try:
        todoist_api.add_task(
            project_id=os.getenv("TODOIST_PROJECT_ID"),
            content=task_name,
            description=task_description,
            order=order,
        )
    except HTTPError:
        raise Exception(f"Are you sure Todoist project (ID: {os.getenv('TODOIST_PROJECT_ID')}) exists?")
    return "Task added successfully"


@tool
def modify_task(
    task_id: Annotated[str, "ID of the task"],
    new_task_name: Annotated[str, "New name of the task (optional)"] = None,
    new_task_description: Annotated[
        str, "New detailed description of what needs to be done in order to implement task (optional)"
    ] = None,
    delete: Annotated[bool, "If True, task will be deleted"] = False,
):
    """Modify task in project management platform (Todoist)."""
    try:
        task_name = todoist_api.get_task(task_id).content
    except HTTPError:
        raise Exception(
            f"Are you sure Todoist project (ID: {os.getenv('TODOIST_PROJECT_ID')}) and task (ID: {task_id}) exist?"
        )
    human_message = user_input(
        f"I want to {'delete' if delete else 'modify'} task '{task_name}'. Type (o)k or provide commentary. "
    )
    if human_message not in ["o", "ok"]:
        return f"Action wasn't executed because of human interruption. He said: {human_message}"

    update_data = {}
    if new_task_name:
        update_data["content"] = new_task_name
    if new_task_description:
        update_data["description"] = new_task_description
    if update_data:
        todoist_api.update_task(task_id=task_id, **update_data)

    if delete:
        todoist_api.delete_task(task_id=task_id)
        return "Task deleted successfully"

    return "Task modified successfully"


@tool
def reorder_tasks(
    task_items: Annotated[
        list,
        "List of dictionaries with 'id' (str) and 'child_order' (int) keys. Example: [{'id': '123', 'child_order': 0}, {'id': '456', 'child_order': 1}]",
    ],
):
    """Reorder tasks in project management platform (Todoist)."""
    command = {"type": "item_reorder", "uuid": str(uuid.uuid4()), "args": {"items": task_items}}
    commands_json = json.dumps([command])
    requests.post(
        "https://api.todoist.com/sync/v9/sync",
        headers={"Authorization": f"Bearer {todoist_api_key}"},
        data={"commands": commands_json},
    )
    return "Tasks reordered successfully"


@tool
def finish_project_planning(dummy: Annotated[str, "Type 'ok' to proceed."]):
    """Call that tool to fire execution of top task from list. Use tool when all task in Todoist correctly reflect work. No extra tasks or tasks with
    overlapping scope allowed.
    """
    human_message = user_input(
        "Project planning finished. Provide your proposition of changes in task list or type (o)k to continue...\n"
    )
    if human_message not in ["o", "ok"]:
        return f"Human: {human_message}"
    # Get first task and it's name and description
    task = todoist_api.get_tasks(project_id=os.getenv("TODOIST_PROJECT_ID"))[0]
    task_name_description = f"{task.content}\n\n{task.description}"

    # Execute the main pipeline to implement the task
    print_formatted("Asked programmer to execute task:", color="light_blue")
    print_text_snippet(task.description, title=task.content)
    run_clean_coder_pipeline(task_name_description, work_dir)

    # ToDo: git upload

    actualize_progress_description_file(task_name_description)

    # Mark task as done
    todoist_api.close_task(task_id=task.id)

    task_completed_animation()
