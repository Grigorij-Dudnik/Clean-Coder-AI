import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import HumanMessage

repo_directory = Path(__file__).parents[3].resolve()
sys.path.append(str(repo_directory))
# Third-party imports
from dotenv import find_dotenv, load_dotenv  # noqa: E402

# Local imports
from non_src.tests.manual_tests.utils_for_tests import cleanup_work_dir, setup_work_dir  # noqa: E402
from src.agents.debugger_agent import Debugger  # noqa: E402
from src.utilities.start_work_functions import file_folder_ignored  # noqa: E402
from src.utilities.script_execution_utils import logs_from_running_script

# Constants
CODERIGNORE_PATTERNS = ("*.log", "*.pyc", "__pycache__")
ENV_DIR_NAME = "env"

# Type aliases
PathLike = Path | str

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(Path(__file__).name)


def create_coderignore(target_dir: Path) -> Path:
    """Create .coderignore file with standard patterns.

    Args:
        target_dir (Path): Directory where .coderignore should be created.

    Returns:
        Path: Path to the created .coderignore file.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    try:
        clean_coder_dir = target_dir / ".clean_coder"
        clean_coder_dir.mkdir(parents=True, exist_ok=True)
        ignore_file = clean_coder_dir / ".coderignore"
        ignore_file.write_text("\n".join(CODERIGNORE_PATTERNS))
        logger.info("Created .coderignore at: %s", ignore_file)
        return ignore_file
    except OSError as e:
        logger.exception("Failed to create .coderignore: %s", e)
        raise


@contextmanager
def managed_workspace(test_files: Path, work_dir: Path) -> Generator[Path, None, None]:
    """Context manager for temporary test workspace.

    Args:
        test_files (Path): Directory containing test files.
        work_dir (Path): Directory to use as the temporary workspace.

    Yields:
        Path: The path to the temporary workspace.

    Raises:
        Exception: If workspace setup fails.
    """
    try:
        setup_work_dir(test_files_dir=test_files, manual_tests_folder=work_dir)
        logger.info("Workspace set up with files from: %s", test_files)
        create_coderignore(work_dir)
        yield work_dir
    except Exception as e:
        logger.exception("Error during workspace setup: %s", e)
        raise
    finally:
        if work_dir.exists():
            cleanup_work_dir(manual_tests_folder=work_dir)
            logger.info("Cleaned up workspace: %s", work_dir)


def collect_project_files(root_dir: Path) -> set[str]:
    """Gather project files respecting .coderignore rules.

    Args:
        root_dir (Path): Root directory to scan for files.

    Returns:
        set[str]: Set of relative file paths.
    """
    files = set()
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = Path(root) / filename
            relative_path = full_path.relative_to(root_dir)
            if not file_folder_ignored(str(relative_path)):
                files.add(str(relative_path))
                logger.debug("Added file to list: %s", relative_path)
            else:
                logger.debug("Skipped file: %s (ignored)", relative_path)
    return files


def configure_environment() -> tuple[Path, Path]:
    """Set up repository paths and environment variables.

    Returns:
        tuple[Path, Path]: Paths to project files and temporary working directory.
    """
    repo_root = Path(__file__).parents[3].resolve()
    sys.path.append(str(repo_root))

    load_dotenv(find_dotenv())
    logger.info("Environment variables loaded")

    project_files = repo_root / "non_src" / "tests" / "manual_tests" / "projects_files" / "debugger_scenario_2_files"
    work_dir = Path(__file__).parent.resolve() / "sandbox_work_dir"
    return project_files, work_dir


def main_test_flow() -> None:
    """Main execution flow for the debugger test scenario."""
    project_files_dir, workspace_dir = configure_environment()

    with managed_workspace(project_files_dir, workspace_dir) as workspace:
        files = collect_project_files(workspace)
        logger.info("Workspace contains %d files: %s", len(files), files)

        # Check if main.py exists and create it if needed
        main_py_path = workspace / "main.py"
        if not main_py_path.exists():
            main_py_path.write_text("print('Hello World!')")
            logger.info("Created main.py with simple 'Hello World' script")

        logger.info("Starting Test 1: Running script with silent setup (default behavior)")
        test_state_silent = {
            "messages": [
                HumanMessage(content=f"File contents: main.py:\n\n{main_py_path.read_text()}\n"),
            ],
        }
        
        message_silent = logs_from_running_script(str(workspace), "main.py", silent_setup=True)
        test_state_silent["messages"].append(HumanMessage(content=message_silent))
        logger.info("Completed silent setup test")
        
        logger.info("Starting Test 2: Running script with verbose setup logs")
        test_state_verbose = {
            "messages": [
                HumanMessage(content=f"File contents: main.py:\n\n{main_py_path.read_text()}\n"),
            ],
        }
        
        # Remove environment to force recreation with logs
        env_path = workspace / ENV_DIR_NAME
        if env_path.exists():
            import shutil
            logger.info("Removing existing virtual environment to force recreation")
            shutil.rmtree(env_path)
        
        message_verbose = logs_from_running_script(str(workspace), "main.py", silent_setup=False)
        test_state_verbose["messages"].append(HumanMessage(content=message_verbose))
        logger.info("Completed verbose setup test")

        # Verification checks
        logger.info("Running verification checks")
        assert env_path.exists(), "Virtual environment not created"
        logger.info("Virtual environment exists as expected")
        
        # Check if silent mode doesn't contain pip installation logs
        if "pip" not in message_silent and "install" not in message_silent:
            logger.info("Silent mode correctly suppresses setup logs")
        else:
            logger.warning("Silent mode may not be suppressing setup logs properly")
        assert "pip" not in message_silent or "install" not in message_silent, "Setup logs should be silent"
        
        # Check if verbose mode contains pip installation logs (if they were performed)
        if "pip" in message_verbose and "install" in message_verbose:
            logger.info("Verbose mode correctly shows pip installation logs")
        else:
            logger.warning("Verbose mode doesn't show installation logs, possibly because environment already existed")
        
        # Check if script output is visible in both cases
        if "Hello World!" in message_silent:
            logger.info("Script output correctly visible in silent mode")
        else:
            logger.error("Script output missing in silent mode")
        assert "Hello World!" in message_silent, "Missing expected output in silent mode"
        
        if "Hello World!" in message_verbose:
            logger.info("Script output correctly visible in verbose mode")
        else:
            logger.error("Script output missing in verbose mode")
        assert "Hello World!" in message_verbose, "Missing expected output in verbose mode"
        
        logger.info("All tests passed successfully")

if __name__ == "__main__":
    main_test_flow()
