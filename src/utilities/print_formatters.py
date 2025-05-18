import json
import textwrap
from termcolor import colored
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Console
from rich.padding import Padding
from pygments.util import ClassNotFound
from pygments.lexers import get_lexer_by_name


def print_formatted_content_planner(content):
    """
    Prints output of planner module. Highlights code snippets in diff.
    """
    parts = content.split("```")
    outside_texts = parts[::2]
    code_snippets = parts[1::2]

    print_formatted(content=outside_texts[0], color="dark_grey")
    for i, snippet in enumerate(code_snippets):
        # Split snippet into filename and content
        lines = snippet.split("\n", 1)
        filename = lines[0]
        snippet_content = lines[1].strip()
        # Place outside text after this snippet
        outside_text = outside_texts[i + 1].strip()
        extension = "diff" if "." in filename else filename
        print_code_snippet(code=snippet_content, title=filename, extension=extension)
        print_formatted(content=outside_text, color="dark_grey")


def print_formatted_content(response):
    if isinstance(response.content, str):
        print_formatted(content=response.content, color="dark_grey")
        ordered_calls = _order_tool_calls(response.tool_calls, "args")
        for call in ordered_calls:
            print_tool_message(tool_name=call["name"], tool_input=call["args"])
    else:
        tool_calls = []
        for response_part in response.content:
            if response_part["type"] == "text":
                print_formatted(content=response_part["text"], color="dark_grey")
            elif response_part["type"] == "tool_use":
                tool_calls.append(response_part)
        ordered_calls = _order_tool_calls(tool_calls, "input")
        for call in ordered_calls:
            print_tool_message(tool_name=call["name"], tool_input=call["input"])


def _order_tool_calls(tool_calls, args_key):
    with_start_line = []
    without_start_line = []
    for call in tool_calls:
        args = call.get(args_key, {})
        if isinstance(args, dict) and "start_line" in args:
            with_start_line.append(call)
        else:
            without_start_line.append(call)
    with_start_line.sort(key=lambda c: c[args_key]["start_line"], reverse=True)
    return with_start_line + without_start_line


def print_formatted(content, width=None, color=None, on_color=None, bold=False, end="\n"):
    if width:
        lines = content.split("\n")
        lines = [textwrap.fill(line, width=width) for line in lines]
        content = "\n".join(lines)
    if bold:
        content = f"\033[1m{content}\033[0m"
    if color:
        content = colored(content, color, on_color=on_color, force_color=True)

    print(content, end=end)


def get_lexer(extension):
    try:
        lexer = get_lexer_by_name(extension)
    except ClassNotFound:
        if extension in ["tsx", "svelte"]:
            lexer = get_lexer_by_name("jsx")
        else:
            lexer = get_lexer_by_name("text")
    return lexer


def print_code_snippet(code, extension, start_line=1, title=None):
    console = Console()

    lexer = get_lexer(extension)

    syntax = Syntax(
        code,
        lexer,
        line_numbers=True,
        start_line=start_line,
        theme="monokai",
        word_wrap=True,
        padding=(1, 1),
    )

    snippet_title = title or f"{extension.capitalize()} Snippet"
    if len(snippet_title) > 100:
        snippet_title = f"..{snippet_title[-95:]}"

    styled_code = Panel(syntax, border_style="bold yellow", title=snippet_title, expand=False)
    console.print(Padding(styled_code, 1))


def print_text_snippet(text, title=None):
    console = Console()

    # Create styled text with background
    styled_text = Panel(
        Padding(
            text,
            1,
            style="bright_white on #272822",  # bright white text on monokai background
        ),
        border_style="bold green",
        title=title,
        expand=False,
    )

    console.print(Padding(styled_text, 1))


def print_error(message: str) -> None:
    print_formatted(content=message, color="red", bold=False)


def print_tool_message(tool_name, tool_input=None):
    if tool_name == "ask_human_tool":
        message = "I have a question for you:"
        print_text_snippet(tool_input["prompt"], title=message)
    elif tool_name == "see_file":
        message = "Looking at the file content..."
        print_formatted(content=message, color="blue", bold=True)
        print_formatted(content=tool_input, color="cyan", bold=True)
    elif tool_name == "list_dir":
        message = "Listing files in a directory..."
        print_formatted(content=message, color="blue", bold=True)
        print_formatted(content=tool_input, color="cyan", bold=True)
    elif tool_name == "create_file_with_code":
        message = "Let's create new file..."
        extension = tool_input["filename"].split(".")[-1]
        print_formatted(content=message, color="blue", bold=True)
        print_code_snippet(code=tool_input["code"], extension=extension, title=tool_input["filename"])
    elif tool_name == "insert_code":
        message = f"Let's insert code after line {tool_input['start_line']}"
        extension = tool_input["filename"].split(".")[-1]
        print_formatted(content=message, color="blue", bold=True)
        print_code_snippet(
            code=tool_input["code"],
            extension=extension,
            start_line=tool_input["start_line"] + 1,
            title=tool_input["filename"],
        )
    elif tool_name == "replace_code":
        message = f"Let's replace code in lines {tool_input['start_line']} to {tool_input['end_line']} for:"
        extension = tool_input["filename"].split(".")[-1]
        print_formatted(content=message, color="blue", bold=True)
        print_code_snippet(
            code=tool_input["code"],
            extension=extension,
            start_line=tool_input["start_line"],
            title=tool_input["filename"],
        )

    elif tool_name == "add_task":
        message = "Let's add a task..."
        print_formatted(content=message, color="blue", bold=True)
        print_text_snippet(tool_input["task_description"], tool_input["task_name"])
    elif tool_name == "modify_task":
        if "delete" in tool_input:
            print_text_snippet("I want to delete task")
        else:
            message = "Let's modify a task..."
            print_formatted(content=message, color="blue", bold=True)
            title = tool_input["new_task_name"] if "new_task_name" in tool_input else f"ID: {tool_input['task_id']}"
            if "new_task_description" in tool_input:
                print_text_snippet(tool_input["new_task_description"], title=title)
    elif tool_name == "final_response_researcher":
        json_string = json.dumps(tool_input, indent=2)
        print_code_snippet(code=json_string, extension="json", title="Files:")
    elif tool_name == "final_response_executor":
        message = "Hurray! The work is DONE!"
        print_formatted(content=message, color="cyan", bold=True)
        if isinstance(tool_input, str):
            print_text_snippet(tool_input, title="Instruction:")
        else:
            print_text_snippet(tool_input["test_instruction"], title="Instruction:")
    elif tool_name == "final_response_debugger":
        if isinstance(tool_input, str):
            print_text_snippet(tool_input, title="Instruction:")
        else:
            print_text_snippet(tool_input["test_instruction"], title="Instruction:")
        print_formatted(
            "Clean Coder helped? Leave a star to support authors https://github.com/Grigorij-Dudnik/Clean-Coder-AI ⭐",
            color="green",
        )
    elif tool_name == "finish_project_planning":
        message = "Planning finished! Time to execute first task from the list."
        print_text_snippet(message)
    else:
        message = f"Calling {tool_name} tool..."
        print_formatted(content=message, color="blue", bold=True)
        print_formatted(content=tool_input, color="blue", bold=True)


if __name__ == "__main__":
    text = """
Here’s the complete plan, split into two separate steps:
```profileEditStyles.css
-   max-width: 300px;
+   max-width: 600px;
-   background-color: #f9f9f9;
+   background-color: #daf5da;
```
Step 2:
```profileEditStyles.css
-   background-color: #218838;
+   background-color: #34a853;
```
"""

    print_formatted_content_planner(text)
