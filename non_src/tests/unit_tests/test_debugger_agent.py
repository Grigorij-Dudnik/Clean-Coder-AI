import os
import subprocess
from unittest.mock import Mock, mock_open, patch

import pytest
from src.utilities.script_execution_utils import format_log_message, run_script_in_env, logs_from_running_script


@pytest.fixture
def temp_work_dir():
    """Fixture providing a temporary work directory path."""
    return "/tmp/test_work"


def test_run_script_success(temp_work_dir):
    with (
        patch("subprocess.run") as mock_run,
        patch("os.path.exists") as mock_exists,
        patch("os.path.isfile") as mock_isfile,
    ):
        mock_exists.side_effect = lambda x: x == os.path.join(temp_work_dir, "env")
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
    with patch("subprocess.run") as mock_run:
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
        patch("os.path.isfile") as mock_isfile,
    ):
        mock_exists.side_effect = lambda x: True
        mock_isfile.side_effect = lambda x: x.endswith("requirements.txt")
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Installed", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Test output", stderr=""),
        ]
        stdout, stderr = run_script_in_env("test_script.py", temp_work_dir)
        assert stdout == "Test output"
        assert stderr == ""
        mock_run.assert_any_call(
            [
                os.path.join(temp_work_dir, "env/bin/pip"),
                "install",
                "-r",
                os.path.join(temp_work_dir, "requirements.txt"),
            ],
            check=True,
        )
        mock_run.assert_any_call(
            [os.path.join(temp_work_dir, "env/bin/python"), "test_script.py"],
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

        mock_run_script.assert_called_once_with(os.path.join(temp_work_dir, "custom_script.py"), temp_work_dir)
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
