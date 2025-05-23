import os
import subprocess
from unittest.mock import ANY, Mock, mock_open, patch

import pytest
from src.utilities.script_execution_utils import format_log_message, run_script_in_env, logs_from_running_script, create_script_execution_env


@pytest.fixture
def temp_work_dir():
    """Fixture providing a temporary work directory path."""
    return "/tmp/test_work"


def test_run_script_success(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("src.utilities.script_execution_utils.create_script_execution_env") as mock_create_env,
    ):
        mock_exists.side_effect = lambda x: x == os.path.join(temp_work_dir, "env")
        mock_create_env.return_value = os.path.join(temp_work_dir, "env/bin/python")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Test output", stderr="")
        stdout, stderr = run_script_in_env("test_script.py", temp_work_dir)
        assert stdout == "Test output"
        assert stderr == ""
        mock_run.assert_called_once_with(
            [os.path.join(temp_work_dir, "env/bin/python"), "test_script.py"],
            capture_output=True,
            check=True,
            text=True,
        )


def test_run_script_error(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("src.utilities.script_execution_utils.create_script_execution_env") as mock_create_env,
    ):
        mock_create_env.return_value = os.path.join(temp_work_dir, "env/bin/python")
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[],
            output="Error output",
            stderr="Error details",
        )
        stdout, stderr = run_script_in_env("failing_script.py", temp_work_dir)
        assert "Error output" in stdout
        assert "Error details" in stderr


def test_logs_from_running_script_file_not_found(temp_work_dir):
    with (
        patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script,
    ):
        mock_run_script.side_effect = FileNotFoundError(f"No such file: 'non_existent.py'")
        result = logs_from_running_script(temp_work_dir, "non_existent.py")
        assert "No such file" in result


def test_logs_from_running_script_success(temp_work_dir):
    with (
        patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script,
    ):
        mock_run_script.return_value = ("Success output", "")
        result = logs_from_running_script(temp_work_dir, "test_script.py")
        assert "Success output" in result
        assert "[SCRIPT EXECUTION INFO]" in result
        mock_run_script.assert_called_once_with(
            ANY,
            temp_work_dir, 
            silent_setup=True
        )

def test_logs_from_running_script_run_script_error(temp_work_dir):
    with (
        patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script,
    ):
        mock_run_script.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=[],
            output="Error output",
            stderr="Error details",
        )
        result = logs_from_running_script(temp_work_dir, "failing_script.py")
        assert "Error output" in result
        assert "Error details" in result


def test_run_script_in_env_with_requirements(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("platform.system") as mock_system,
        patch("src.utilities.script_execution_utils.create_script_execution_env") as mock_create_env,
    ):
        mock_system.return_value = "Windows"
        mock_exists.side_effect = lambda x: True
        mock_create_env.return_value = os.path.join(temp_work_dir, "env", "Scripts", "python.exe")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Test output", stderr="")
        stdout, stderr = run_script_in_env("test_script.py", temp_work_dir)
        assert stdout == "Test output"
        assert stderr == ""
        mock_run.assert_any_call(
            [
                ANY,
                "install",
                "-r",
                ANY,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mock_run.assert_any_call(
            [ANY, "test_script.py"],
            capture_output=True,
            check=True,
            text=True,
        )


def test_format_log_message():
    message = format_log_message(
        is_error=False,
        stdout="Success output",
    )
    assert "[SCRIPT EXECUTION INFO]" in message
    assert "Success output" in message
    message = format_log_message(
        is_error=True,
        error_msg="File not found",
    )
    assert "[SCRIPT EXECUTION ERROR]" in message
    assert "File not found" in message


def test_logs_from_running_script_with_custom_filename(temp_work_dir):
    with patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script:
        mock_run_script.return_value = ("Custom file output", "")
        result = logs_from_running_script(temp_work_dir, "custom_script.py")
        mock_run_script.assert_called_once_with(
            ANY,
            temp_work_dir, 
            silent_setup=True
        )
        args, kwargs = mock_run_script.call_args
        assert args[0].endswith("custom_script.py")
        
        assert "Custom file output" in result

def test_logs_from_running_script_with_generic_exception(temp_work_dir):
    with patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script:
        mock_run_script.side_effect = PermissionError("Permission denied")
        result = logs_from_running_script(temp_work_dir, "main.py")
        assert "[SCRIPT EXECUTION ERROR]" in result
        assert "Permission denied" in result


def test_logs_from_running_script_with_empty_output(temp_work_dir):
    with patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script:
        mock_run_script.return_value = ("", "")
        result = logs_from_running_script(temp_work_dir, "empty_output.py")
        assert "[SCRIPT EXECUTION INFO]" in result
        assert "STDOUT:\n\n" not in result
        assert "STDERR:\n\n" not in result


def test_format_log_message_with_all_parameters():
    message = format_log_message(
        stdout="Standard output", stderr="Standard error", is_error=True, error_msg="Custom error message"
    )
    assert "[SCRIPT EXECUTION ERROR]" in message
    assert "Error: Custom error message" in message
    assert "STDOUT:\nStandard output" in message
    assert "STDERR:\nStandard error" in message


def test_logs_from_running_script_with_silent_setup_false(temp_work_dir):
    with patch("src.utilities.script_execution_utils.run_script_in_env") as mock_run_script:
        mock_run_script.return_value = ("Output with setup logs", "")
        result = logs_from_running_script(temp_work_dir, "test_script.py", silent_setup=False)
        mock_run_script.assert_called_once_with(
            ANY,
            temp_work_dir,
            silent_setup=False
        )
        args, kwargs = mock_run_script.call_args
        assert args[0].endswith("test_script.py")
        assert kwargs["silent_setup"] == False
        assert "Output with setup logs" in result

def test_create_script_execution_env_silent_parameter(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("venv.create") as mock_venv_create,
        patch("platform.system", return_value="Linux") as mock_system,
    ):
        mock_exists.return_value = False
        create_script_execution_env(temp_work_dir)
        mock_run.assert_called_once_with(
            [ANY, "install", "-U", "pip"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        mock_run.reset_mock()
        create_script_execution_env(temp_work_dir, silent=False)
        mock_run.assert_called_once_with(
            [ANY, "install", "-U", "pip"],
            check=True,
            stdout=None,
            stderr=None,
        )

def test_run_script_in_env_without_requirements(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("src.utilities.script_execution_utils.create_script_execution_env") as mock_create_env,
    ):
        mock_exists.return_value = False
        mock_create_env.return_value = os.path.join(temp_work_dir, "env/bin/python")
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="Test output", stderr="")

        stdout, stderr = run_script_in_env("test_script.py", temp_work_dir)

        assert stdout == "Test output"
        assert stderr == ""
        assert not any("install -r" in str(call) for call in mock_run.call_args_list)
        mock_run.assert_called_once_with(
            [os.path.join(temp_work_dir, "env/bin/python"), "test_script.py"],
            capture_output=True,
            check=True,
            text=True,
        )

@pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
def test_windows_paths(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("os.path.join", side_effect=lambda *args: "\\".join(args)) as mock_join,
        patch("os.path.dirname", side_effect=lambda path: "\\".join(path.split("\\")[:-1])) as mock_dirname,
        patch("venv.create") as mock_venv_create,
        patch("platform.system", return_value="Windows") as mock_system,
    ):
        mock_exists.return_value = False
        python_path = create_script_execution_env(temp_work_dir)
        assert "\\env\\Scripts\\python.exe" in python_path
        mock_run.assert_called_once_with(
            [ANY, "install", "-U", "pip"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        args, kwargs = mock_run.call_args
        assert "pip.exe" in args[0][0]
        assert "Scripts" in args[0][0]