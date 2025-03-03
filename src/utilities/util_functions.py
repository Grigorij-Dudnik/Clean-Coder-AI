import re
import os
import xml.etree.ElementTree as ET
import base64
import mimetypes
import requests
import subprocess
from src.utilities.start_work_functions import file_folder_ignored, CoderIgnore, Work
from src.utilities.print_formatters import print_formatted
from dotenv import load_dotenv, find_dotenv
from todoist_api_python.api import TodoistAPI
from langchain_core.messages import HumanMessage, ToolMessage
import click


load_dotenv(find_dotenv())
work_dir = os.getenv("WORK_DIR")
log_file_path = os.getenv("LOG_FILE")
todoist_api = TodoistAPI(os.getenv('TODOIST_API_KEY'))
PROJECT_ID = os.getenv('TODOIST_PROJECT_ID')


TOOL_NOT_EXECUTED_WORD = "Tool not been executed. "
WRONG_TOOL_CALL_WORD = "Wrong tool call. "

storyfile_template = """<This is the story of your project for a frontend feedback agent. Modify it according to commentaries provided in <> brackets.>

App we working on is ... <describe what your project about here>.

How to write your playwright code:

If you want to test changes that does not require to be logged in, just go straight away to the page you want to see:
```python
page.goto(f'{frontend_url}/your_endpoint_to_test')
```

If it required to be logged in, use next code first: <adjust login code according to login page of your app>.
```python
page.goto(f'{frontend_url}/login')
page.fill('input[type="email"]', username)
page.fill('input[type="password"]', password)
page.click('button[type="submit"]')
page.wait_for_url('**/')
```
For being logged in as usual user, use username="frontend.feedback@user", password="123". <exchange your login credentials for example user>
For being logged in as admin user, use username="frontend.feedback@admin", password="456". <exchange your login credentials for example user>
"""


def check_file_contents(files, work_dir, line_numbers=True):
    file_contents = str()
    for file_name in files:
        file_content = watch_file(file_name, work_dir, line_numbers)
        file_contents += file_content + "\n\n###\n\n"

    return file_contents


def watch_file(filename, work_dir, line_numbers=True):
    if file_folder_ignored(filename):
        return "You are not allowed to work with this file."
    
    file_path = join_paths(work_dir, filename)
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Check if the file is not a text file
    if mime_type and not mime_type.startswith('text'):
        return f"File {filename} is a binary file and cannot be read as text."
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return "File does not exist."
    except UnicodeDecodeError:
        return f"File {filename} could not be decoded as UTF-8. It may be a binary file."
    
    if line_numbers:
        formatted_lines = [f"{i + 1}|{line[:-1]}\n" for i, line in enumerate(lines)]
    else:
        formatted_lines = [f"{line[:-1]}\n" for line in lines]
    
    file_content = "".join(formatted_lines)
    file_content = f"{filename}:\n\n{file_content}"
    return file_content

def find_tool_xml(input_str):
    match = re.search('```xml(.*?)```', input_str, re.DOTALL)
    if match:
        root = ET.fromstring(match.group(1).strip())
        tool = root.find('tool').text.strip()
        tool_input_element = root.find('tool_input')
        tool_input = {}
        for child in tool_input_element:
            child.text = child.text.strip()
            if list(child):
                tool_input[child.tag] = [item.text for item in child]
            else:
                tool_input[child.tag] = child.text
        # output = {child.tag: child.text for child in root}
        return {"tool": tool, "tool_input": tool_input}
    else:
        return None


def check_application_logs():
    """Check out logs to see if application works correctly."""
    try:
        with open(log_file_path, 'r') as file:
            logs = file.read()
        if logs.strip().endswith("No messages found"):
            print("Logs are correct")
            return "Logs are correct"
        else:
            return logs
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def see_image(filename, work_dir):
    with open(join_paths(work_dir, filename), 'rb') as image_file:
        img_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return img_encoded


def convert_images(image_paths):
    images = []
    for image_path in image_paths:
        if not os.path.exists(join_paths(work_dir, image_path)):
            continue
        images.extend([
                 {"type": "text", "text": f"I###\n{image_path}"},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{see_image(image_path, work_dir)}"}}
             ])

    return images


def join_paths(*args):
    leading_slash = '/' if args[0].startswith('/') else ''
    joined = leading_slash + '/'.join(p.strip('/') for p in args)
    return os.path.normpath(joined)


def get_joke():
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Programming?type=single")
        # response = requests.get("https://uselessfacts.jsph.pl//api/v2/facts/random")
        joke = response.json()["joke"] + "\n"
    except Exception as e:
        joke = f"Failed to receive joke :/"
    return joke


def list_directory_tree(work_dir):
    tree = []
    for root, dirs, files in os.walk(work_dir):
        # Filter out forbidden directories and files
        dirs[:] = [d for d in dirs if not file_folder_ignored(d, CoderIgnore.get_forbidden())]
        files = [f for f in files if not file_folder_ignored(f, CoderIgnore.get_forbidden())]
        rel_path = os.path.relpath(root, work_dir)
        depth = rel_path.count(os.sep)
        indent = "│ " * depth

        # Add current directory to the tree
        tree.append(f"{indent}{'└──' if depth > 0 else ''}📁 {os.path.basename(root)}")

        # Check if the total number of items exceeds the threshold
        total_items = len(dirs) + len(files)
        if total_items > 30:
            file_indent = "│ " * (depth + 1)
            tree.append(f"{file_indent}Too many files/folders to display ({total_items} items)")
            dirs.clear()
            continue
        elif total_items == 0:
            file_indent = "│ " * (depth + 1)
            tree.append(f"{file_indent}<Directory is empty>")
            dirs.clear()
            continue

        # Add files to the tree
        file_indent = "│ " * (depth + 1)
        for i, file in enumerate(files):
            connector = "└── " if i == len(files) - 1 else "├── "
            tree.append(f"{file_indent}{connector}{file}")

    return "Content of directory tree:\n" + "\n".join(tree)


def invoke_tool_native(tool_call, tools):
    # convert string to real function
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call["name"]
    requested_tool = tool_name_to_tool[name]
    args = tool_call["args"]
    tool_output = requested_tool.invoke(args)
    return ToolMessage(tool_output, tool_call_id=tool_call["id"])


def exchange_file_contents(state, files, work_dir):
    # Remove old one
    state["messages"] = [msg for msg in state["messages"] if not hasattr(msg, "contains_file_contents")]
    # Add new file contents
    file_contents = check_file_contents(files, work_dir)
    file_contents = f"Find most actual file contents here:\n\n{file_contents}\nTake a look at line numbers before introducing changes."
    file_contents_msg = HumanMessage(content=file_contents, contains_file_contents=True)
    state["messages"].insert(2, file_contents_msg)  # insert after the system and plan msgs
    return state


def bad_tool_call_looped(state):
    last_tool_messages = [m for m in state["messages"] if m.type == "tool"][-4:]
    tool_not_executed_msgs = [
        m for m in last_tool_messages if isinstance(m.content, str) and m.content.startswith(WRONG_TOOL_CALL_WORD)
    ]
    if len(tool_not_executed_msgs) == 4:
        print_formatted("Seems like AI been looped. Please suggest it how to introduce change correctly:", color="yellow")
        return True


def create_frontend_feedback_story():
    frontend_feedback_story_path = os.path.join(Work.dir(), '.clean_coder', 'frontend_feedback_story.txt')
    if not os.path.exists(frontend_feedback_story_path):
        with open(frontend_feedback_story_path, 'w') as file:
            file.write(storyfile_template)
        click.launch(frontend_feedback_story_path)
        input("Fulfill file with informations needed for a frontend feedback agent to know. Save file and hit Enter.")


def read_coderrules():
    project_rules_path = os.path.join(Work.dir(), '.coderrules')
    if not os.path.exists(project_rules_path):
        return create_coderrules(project_rules_path)
    with open(project_rules_path, 'r') as file:
        return file.read()


def create_coderrules(coderrules_path):
    print_formatted("(Optional) Describe your project rules and structure to give AI more context about it. Learn how to do it: https://clean-coder.dev/features/coderrules/. ", color="light_blue")
    rules = input()
    with open(coderrules_path, 'w', encoding='utf-8') as file:
        file.write(rules)
    print_formatted(f"Project rules saved. You can edit it in .coderrules file.", color="green")
    return rules
