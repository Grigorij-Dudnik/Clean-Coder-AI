import os
import subprocess
from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.debugger_agent import Debugger
from src.utilities.util_functions import format_log_message, run_script_in_env, write_and_append_log


@pytest.fixture
def debugger_agent(monkeypatch):
    monkeypatch.setenv("LOG_FILE", "/tmp/test_log.log")
    agent = Debugger(
        files=set(), work_dir="/tmp/test_work", human_feedback="Test feedback", image_paths=[], playwright_code=None,
    )
    return agent


def test_run_script_success(debugger_agent: Debugger):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("os.path.isfile") as mock_isfile,
    ):
        mock_exists.side_effect = lambda x: x == os.path.join(debugger_agent.work_dir, "env")
        mock_isfile.return_value = False  # No requirements.txt
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Test output", stderr="")

        stdout, stderr = run_script_in_env("test_script.py", debugger_agent.work_dir)

        assert stdout == "Test output"
        assert stderr == ""
        mock_run.assert_called_once_with(
            [os.path.join(debugger_agent.work_dir, "env/bin/python"), "test_script.py"],
            capture_output=True,
            check=True,
            text=True,
        )


def test_run_script_error(debugger_agent: Debugger):
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=[], output="Error output", stderr="Error details",
        )

        stdout, stderr = run_script_in_env("failing_script.py", debugger_agent.work_dir)

        assert "Error output" in stdout
        assert "Error details" in stderr


def test_logs_from_running_script_file_not_found(debugger_agent: Debugger):
    with (
        patch("src.agents.debugger_agent.get_executed_filename") as mock_get_filename,
        patch("os.path.exists") as mock_exists,
    ):
        mock_get_filename.return_value = "non_existent.py"
        mock_exists.return_value = False
        result_state = debugger_agent.logs_from_running_script({"messages": []})
        assert "File not found" in result_state["messages"][-1].content


def test_logs_from_running_script_success(debugger_agent: Debugger):
    with (
        patch("src.agents.debugger_agent.get_executed_filename") as mock_get_filename,
        patch("src.agents.debugger_agent.run_script_in_env") as mock_run_script,
        patch("src.utilities.util_functions.subprocess.run") as mock_subprocess_run,
        patch("os.path.exists") as mock_exists,
    ):
        mock_get_filename.return_value = "existing_script.py"
        mock_run_script.return_value = ("Success output", "")
        mock_subprocess_run.return_value = Mock(
            spec=subprocess.CompletedProcess,
            returncode=0,
        )
        mock_exists.return_value = True
        result_state = debugger_agent.logs_from_running_script({"messages": []})

        assert "existing_script.py" in result_state["messages"][-1].content
        assert "Success output" in result_state["messages"][-1].content


def test_write_and_append_log():
    test_message = "Test log message"
    test_logs_file = "/tmp/test_work/logs.txt"

    with patch("builtins.open", mock_open()) as mocked_file:
        initial_state = {"messages": []}
        new_state = write_and_append_log(initial_state, test_message, test_logs_file)

        mocked_file.assert_called_once_with(test_logs_file, "w")
        mocked_file().write.assert_called_once_with(test_message)
        assert len(new_state["messages"]) == 1
        assert isinstance(new_state["messages"][0], HumanMessage)
        assert test_message in new_state["messages"][0].content


def test_logs_from_running_script_empty_filename(debugger_agent: Debugger):
    with (
        patch("src.agents.debugger_agent.get_executed_filename") as mock_get_filename,
        patch("os.path.exists") as mock_exists,
        patch("src.agents.debugger_agent.run_script_in_env") as mock_run_script,
    ):
        mock_get_filename.return_value = ""
        mock_exists.return_value = True
        mock_run_script.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=[], output="", stderr="Error: Not a valid script",
        )

        result_state = debugger_agent.logs_from_running_script({"messages": []})
        assert "Error: Not a valid script" in result_state["messages"][-1].content


def test_logs_from_running_script_run_script_error(debugger_agent: Debugger):
    with (
        patch("src.agents.debugger_agent.get_executed_filename") as mock_get_filename,
        patch("os.path.exists") as mock_exists,
        patch("src.agents.debugger_agent.run_script_in_env") as mock_run_script,
    ):
        mock_get_filename.return_value = "existing_script.py"
        mock_exists.return_value = True
        mock_run_script.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=[], output="Error output", stderr="Error details",
        )

        result_state = debugger_agent.logs_from_running_script({"messages": []})
        assert "Error output" in result_state["messages"][-1].content
        assert "Error details" in result_state["messages"][-1].content


def test_run_script_in_env_with_requirements(debugger_agent: Debugger):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("os.path.isfile") as mock_isfile,
    ):
        mock_exists.side_effect = lambda x: True
        mock_isfile.side_effect = lambda x: x.endswith("requirements.txt")
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Installed", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Test output", stderr=""),
        ]
        stdout, stderr = run_script_in_env("test_script.py", debugger_agent.work_dir)
        assert stdout == "Test output"
        assert stderr == ""
        mock_run.assert_any_call(
            [
                os.path.join(debugger_agent.work_dir, "env/bin/pip"),
                "install",
                "-r",
                os.path.join(debugger_agent.work_dir, "requirements.txt"),
            ],
            check=True,
        )
        mock_run.assert_any_call(
            [os.path.join(debugger_agent.work_dir, "env/bin/python"), "test_script.py"],
            capture_output=True,
            check=True,
            text=True,
        )


def test_format_log_message(debugger_agent: Debugger):
    message = format_log_message(
        work_dir=debugger_agent.work_dir,
        script_path="test_script.py",
        is_error=False,
        stdout="Success output",
    )
    assert "test_script.py" in message
    assert "Success output" in message

    message = format_log_message(
        work_dir=debugger_agent.work_dir, script_path="test_script.py", is_error=True, error_msg="File not found",
    )
    assert "test_script.py" in message
    assert "File not found" in message
