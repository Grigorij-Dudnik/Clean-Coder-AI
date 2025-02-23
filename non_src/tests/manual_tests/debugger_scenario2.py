import logging
import os
import shutil
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
from src.agents.debugger_agent import Debugger  # noqa: E402
from src.utilities.start_work_functions import file_folder_ignored  # noqa: E402

# Constants
CODERIGNORE_PATTERNS = ("*.log", "*.pyc", "__pycache__")
ENV_DIR_NAME = "env"
LOGS_FILE_NAME = "logs.txt"

# Type aliases
PathLike = Path | str

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(Path(__file__).name)


class FileOperations:
    """Encapsulates file system operations for test environment"""
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Safely create directory structure.

        Args:
            path (Path): Directory path to create.
        """
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: %s", path)
        except OSError:
            logger.exception("Failed to create directory.")
            raise

    @staticmethod
    def copy_tree(src: Path, dest: Path) -> None:
        """Recursively copy directory contents with logging.

        Args:
            src (Path): Source directory to copy from.
            dest (Path): Destination directory to copy to.
        """
        try:
            for item in src.glob("**/*"):
                if item.is_file():
                    relative = item.relative_to(src)
                    target = dest / relative
                    FileOperations.ensure_directory(target.parent)
                    target.write_bytes(item.read_bytes())
                    logger.info("Copied: %s ➔ %s", item, target)
        except OSError:
            logger.exception("Failed to copy files.")
            raise

    @staticmethod
    def create_coderignore(target_dir: Path) -> Path:
        """Create .coderignore file with standard patterns.

        Args:
            target_dir (Path): Directory where .coderignore should be created.

        Returns:
            Path: Path to the created .coderignore file.
        """
        clean_coder_dir = target_dir / ".clean_coder"
        FileOperations.ensure_directory(clean_coder_dir)
        ignore_file = clean_coder_dir / ".coderignore"
        ignore_file.write_text("\n".join(CODERIGNORE_PATTERNS))
        logger.info("Created .coderignore at: %s", ignore_file)
        return ignore_file


@contextmanager
def managed_workspace(test_files: Path, work_dir: Path) -> Generator[Path, None, None]:
    """Context manager for temporary test workspace.

    Args:
        test_files (Path): Directory containing test files.
        work_dir (Path): Directory to use as the temporary workspace.

    Yields:
        Path: The path to the temporary workspace.
    """
    try:
        FileOperations.ensure_directory(work_dir)
        FileOperations.copy_tree(test_files, work_dir)
        FileOperations.create_coderignore(work_dir)
        yield work_dir
    except Exception:
        logging.exception("Error during workspace setup.")
        raise
    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)
            logger.info("Cleaned up workspace: %s", work_dir)


def collect_project_files(root_dir: Path) -> set[str]:
    """Gather project files respecting .coderignore rules.

    Args:
        root_dir (Path): Root directory to scan for files.

    Returns:
        Set[str]: Set of relative file paths.
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
    """Main execution flow for the debugger test scenario"""
    project_files_dir, workspace_dir = configure_environment()
    
    with managed_workspace(project_files_dir, workspace_dir) as workspace:
        files = collect_project_files(workspace)
        logger.info("Workspace contains %d files: %s", len(files), files)
        
        debugger = Debugger(files, str(workspace), "Please ensure the script runs correctly and logs its output.", [],
        )
        
        test_state = {
            "messages": [
                HumanMessage(content="File contents: sample_test.py:\n\nprint('Hello World!')\n"),
            ],
        }
        
        updated_state = debugger.logs_from_running_script(test_state)
        
        # Verification checks
        env_path = workspace / ENV_DIR_NAME
        logs_path = workspace / LOGS_FILE_NAME
        assert env_path.exists(), "Virtual environment not created"  # noqa: S101
        assert logs_path.exists(), "Log file missing"  # noqa: S101
        
        logger.info(f"Updated state: {updated_state}")
        assert "Hello World!" in updated_state["messages"][-1].content, "Missing expected output"  # noqa: S101


if __name__ == "__main__":
    main_test_flow()
