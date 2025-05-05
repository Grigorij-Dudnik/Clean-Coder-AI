from langchain_core.tools import tool
from typing_extensions import Annotated
import os
from dotenv import load_dotenv, find_dotenv
from src.linters.syntax_checker_functions import check_syntax
from src.utilities.start_work_functions import file_folder_ignored
from src.utilities.util_functions import join_paths, WRONG_TOOL_CALL_WORD, TOOL_NOT_EXECUTED_WORD
from src.utilities.user_input import user_input
from src.tools.rag.retrieval import retrieve


load_dotenv(find_dotenv())

syntax_error_insert_code = """
Changes can cause next error: {error_response}. Probably you:
- Provided a wrong line number to insert code, or
- Forgot to add an indents on beginning of code.
Please analyze which place is correct to introduce the code before calling a tool.
"""
syntax_error_modify_code = """
Changes can cause next error: {error_response}. Probably you:
- Provided a wrong end or beginning line number, or
- Forgot to add an indents on beginning of code.
Think step by step which function/code block you want to change before proposing improved change.
"""


def prepare_list_dir_tool(work_dir):
    @tool
    def list_dir(
        directory: Annotated[str, "Directory to list files in."],
    ):
        """
        List files in directory. Use only for dirs content of which is hidden in the project tree.
        """
        try:
            if file_folder_ignored(directory):
                return f"You are not allowed to work with directory {directory}."
            files = os.listdir(join_paths(work_dir, directory))

            return f"Content of directory {directory}:\n" + "\n".join(files)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    return list_dir


def prepare_see_file_tool(work_dir):
    @tool
    def see_file(filename: Annotated[str, "Name and path of file to check."]):
        """
        Check contents of code file.
        """
        try:
            if file_folder_ignored(filename):
                return f"You are not allowed to work with {filename}."
            with open(join_paths(work_dir, filename), "r", encoding="utf-8") as file:
                lines = file.readlines()
            formatted_lines = [f"{i+1}|{line[:-1]}|{i+1}\n" for i, line in enumerate(lines)]
            file_content = "".join(formatted_lines)
            file_content = filename + ":\n\n" + file_content

            return file_content
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    return see_file

@tool
def see_image(filename: Annotated[str, "Name and path of image file to check."]):
    """
    Check contents of image file.
    """
    if file_folder_ignored(filename):
        return "You are not allowed to work with this file."
    return "Image added to context."



@tool
def retrieve_files_by_semantic_query(
    query: Annotated[
        str,
        "Semantic query describing subject you looking for in one sentence. Ask for a singe thing only. Explain here thing you look only: good query is '<Thing I'm looking for>', bad query is 'Find a files containing <thing I'm looking for>'",
    ],
):
    """
    Use that function to find files or folders in the app by text search.
    You can search for example for common styles, endpoint with user data, etc.
    Useful, when you know what do you look for, but don't know where.

    Use that function at least once BEFORE calling final response to ensure you found all appropriate files.
    """
    return retrieve(query)


def prepare_insert_code_tool(work_dir):
    @tool
    def insert_code(
        filename: Annotated[str, "Name and path of file to change."],
        start_line: Annotated[int, "Line number to insert new code after."],
        code: Annotated[
            str,
            "Code to insert into the file. Without backticks around. Start it with appropriate indentation if needed.",
        ],
    ):
        """
        Insert new piece of code into provided file. Use when new code need to be added without replacing old one.
        Proper indentation is important.
        """
        try:
            with open(join_paths(work_dir, filename), "r+", encoding="utf-8") as file:
                file_contents = file.readlines()
                file_contents.insert(start_line, code + "\n")
                file_contents = "".join(file_contents)
                check_syntax_response = check_syntax(file_contents, filename)
                if check_syntax_response != "Valid syntax":
                    print("Wrong syntax provided, asking to correct.")
                    return WRONG_TOOL_CALL_WORD + syntax_error_insert_code.format(error_response=check_syntax_response)
                message = "Never accept changes you don't understand. Type (o)k if you accept or provide commentary. "
                human_message = user_input(message)
                if human_message not in ["o", "ok"]:
                    return TOOL_NOT_EXECUTED_WORD + f"Human: '{human_message}'"
                file.seek(0)
                file.truncate()
                file.write(file_contents)
            return "Code inserted."
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    return insert_code


def prepare_replace_code_tool(work_dir):
    @tool
    def replace_code(
        filename: Annotated[str, "Name and path of file to change."],
        start_line: Annotated[int, "Start line number to replace (inclusive)."],
        code: Annotated[
            str,
            "New piece of code to replace old one. Without backticks around. Start it with appropriate indentation if needed.",
        ],
        end_line: Annotated[int, "End line number to replace (inclusive)."],
    ):
        """
        Replace old piece of code between start_line and end_line with new one. Proper indentation is important.
        Exchange entire functions or code blocks at once. Avoid changing functions partially.
        """
        try:
            with open(join_paths(work_dir, filename), "r+", encoding="utf-8") as file:
                file_contents = file.readlines()
                file_contents[start_line - 1 : end_line] = [code + "\n"]
                file_contents = "".join(file_contents)
                check_syntax_response = check_syntax(file_contents, filename)
                if check_syntax_response != "Valid syntax":
                    print(check_syntax_response)
                    return WRONG_TOOL_CALL_WORD + syntax_error_modify_code.format(error_response=check_syntax_response)
                message = "Never accept changes you don't understand. Type (o)k if you accept or provide commentary. "
                human_message = user_input(message)
                if human_message not in ["o", "ok"]:
                    return TOOL_NOT_EXECUTED_WORD + f"Human: '{human_message}'"
                file.seek(0)
                file.truncate()
                file.write(file_contents)
            return "Code modified."
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    return replace_code


def prepare_create_file_tool(work_dir):
    @tool
    def create_file_with_code(
        filename: Annotated[str, "Name and path of file to create."],
        code: Annotated[str, "Code to write in the file."],
    ):
        """
        Create new file with provided code. If you need to create directory, all directories in provided path will be
        automatically created.
        Do not write files longer than 1000 words. If you need to create big files, start small, and next add new functions
        with another tools.
        """
        try:
            message = "Never accept changes you don't understand. Type (o)k if you accept or provide commentary. "
            human_message = user_input(message)
            if human_message not in ["o", "ok"]:
                return TOOL_NOT_EXECUTED_WORD + f"Human: '{human_message}'"

            full_path = join_paths(work_dir, filename)
            directory = os.path.dirname(full_path)

            # Create directories if they don't exist
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(full_path, "w", encoding="utf-8") as file:
                file.write(code)
            return "File been created successfully."
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    return create_file_with_code


@tool
def ask_human_tool(prompt: Annotated[str, "Prompt you want to show to human."]):
    """
    Ask human to do project setup/debug actions you're not available to do or provide observations of how does program works.
    """
    human_message = user_input("Type ")
    return human_message
