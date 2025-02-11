"""Integration tests for running single coder pipeline with no documentation context added."""

import logging
import os
import pathlib
import subprocess

import pytest

from single_task_coder import run_clean_coder_pipeline
from tests.manual_tests.utils_for_tests import cleanup_work_dir, setup_work_dir

logger = logging.getLogger()
logger.level = logging.INFO


@pytest.mark.integration
def test_llm_no_context(tmp_path: pathlib.Path) -> None:
# def test_llm_no_context() -> None:
    """Test that the LLM hallucinates and produces incorrect import statement without documentation context."""
    # Given the task for the LLM
    task = '''populate main_dummy.py with code to pull information with PubMedAPIWrapper from langchain,
        to load results to the query "cancer research". Use API key "123412367"'''
    # and given a test work directory as well as .py file
    # folder_with_project_files = "test_llm_no_context"
    # setup_work_dir(folder_with_project_files)
    work_dir = tmp_path / "trial"
    work_dir.mkdir()
    py_file = work_dir / "main_dummy.py"
    content = 'print("hello world")'
    py_file.write_text(content, encoding="utf-8")
    
    os.environ["WORK_DIR"] = str(work_dir)
    # When starting single coder pipeline and making the LLM call
    run_clean_coder_pipeline(task, str(work_dir))
    # Then assert that main_dummy.py was modified by the agents
    assert py_file.read_text(encoding="utf-8") != content
    # Then assert that the response is not runnable
    command = ["python", py_file]
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        subprocess.run(command, check=True)
    assert excinfo.value.returncode != 0
    cleanup_work_dir()


# @pytest.mark.integration
# # def test_llm_rag_context(tmp_path: pathlib.Path) -> None:
# def test_llm_rag_context() -> None:
#     """Test that an LLM with RAG documentation makes a correct implementation of what is requested."""
#     # Given initial request for the LLM
#     task = '''populate main_dummy.py with code to pull information with PubMedAPIWrapper from langchain,
#         to load results to the query "cancer research". Use API key "123412367"'''
#     # and given a test work directory as well as .py file
#     work_dir = tmp_path / "trial"
#     py_file = work_dir / "main_dummy.py"
#     content = 'print("hello world")'
#     py_file.write_text(content, encoding="utf-8")
#     work_dir.mkdir()
#     os.environ["WORK_DIR"] = str(work_dir)
#     # When starting single coder pipeline and making the LLM call, with RAG
#     run_clean_coder_pipeline(task, str(work_dir),doc_harvest=True)
#     # Then assert that main_dummy.py was modified by the agents
#     assert py_file.read_text(encoding="utf-8") != content
#     # Then assert that the response is not runnable
#     command = ["python", py_file]
#     with pytest.raises(subprocess.CalledProcessError) as excinfo:
#         subprocess.run(command, check=True)
#     assert excinfo.value.returncode == 0
#     cleanup_work_dir()
