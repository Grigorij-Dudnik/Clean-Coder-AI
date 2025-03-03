import datetime
import os
import subprocess
from langchain_core.messages import HumanMessage


def setup_virtual_env(work_dir):
    work_dir = os.path.abspath(work_dir)
    env_path = os.path.join(work_dir, "env")
    if not os.path.exists(env_path):
        import venv

        venv.create(env_path, with_pip=True)
        subprocess.run([os.path.join(env_path, "bin", "pip"), "install", "-U", "pip"], check=True)
    return os.path.join(env_path, "bin", "python")


def run_script_in_env(script_path, work_dir):
    work_dir = os.path.abspath(work_dir)
    python_path = setup_virtual_env(work_dir)
    req_file = os.path.join(work_dir, "requirements.txt")
    if os.path.exists(req_file):
        pip_path = os.path.join(os.path.dirname(python_path), "pip")
        subprocess.run([pip_path, "install", "-r", req_file], check=True)
    try:
        result = subprocess.run([python_path, script_path], capture_output=True, text=True, check=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr


def format_log_message(work_dir, script_path, is_error=False, stdout="", stderr="", error_msg=""):
    """Format log message content"""
    message = "\n[SCRIPT EXECUTION {}]\n".format("ERROR" if is_error else "INFO")
    message += f"Script: {script_path}\n"
    message += f"Timestamp: {datetime.datetime.now()}\n"

    if is_error:
        message += f"Error: {error_msg}\n"
        message += f"STDOUT:\n{stdout}\n"
        message += f"STDERR:\n{stderr}\n"
    else:
        message += f"Executed in env: {work_dir}/env\n"
        message += f"STDOUT:\n{stdout}\n"
        message += f"STDERR:\n{stderr}\n"

    return message


def write_and_append_log(state: dict, message: str, logs_file: str) -> dict:
    """Write log to file and append to state"""
    with open(logs_file, "w") as log_file:
        log_file.write(message)

    state["messages"].append(HumanMessage(content=message))
    return state


def get_executed_filename(state: dict) -> str:
    """Get filename of executed script from messages"""
    for message in state.get("messages", []):
        if isinstance(message.content, str) and "File contents:" in message.content:
            header_line = message.content.splitlines()[0]
            if header_line.startswith("File contents:"):
                parts = header_line.split("File contents:")[-1].split(":")
                filename = parts[0].strip()
                return filename
    return ""
