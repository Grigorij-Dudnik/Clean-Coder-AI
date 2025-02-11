"""Documentation harvester pulls relevant docs for the task set by the user of the pipeline."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.models import CrawlResult
from dotenv import find_dotenv, load_dotenv

from src.tools.rag.write_descriptions import produce_descriptions, upload_descriptions_to_vdb
from src.tools.tools_doc_harvester import PythonLibraries
from src.utilities.exceptions import MissingEnvironmentVariableError, ModuleImportedButNotLocatedError
from src.utilities.llms import init_llms_mini
from src.utilities.util_functions import join_paths

load_dotenv(find_dotenv())

# playwright install --with-deps chromium

async def pull_webpage(url: str) -> CrawlResult:
    """Pulls URL information."""
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(
            url=url,
        )


class DocHarvester:
    """
    Agent for collecting documentation relevant to user's task. Requires internet access.

    More description.

    Attributes
    ----------
    work_dir: str
        Location of the project that Clean Coder pipeline operates on.

    Methods
    -------
        find_documentation(task: str)

    Examples
    --------
        dh = DocHarvester()
        task = "prepare a scraper of a website"
        dh.find_documentation(task=task)
    """

    def __init__(self) -> None:
        """Initial information to help harvest documentation."""
        work_dir = os.getenv("WORK_DIR")
        if not work_dir:
            msg = "WORK_DIR variable not provided. Please add WORK_DIR to .env file"
            raise MissingEnvironmentVariableError(msg)
        self.work_dir = work_dir
        llms_mini = init_llms_mini(run_name="DocHarvester")
        self.llm_mini = llms_mini[0]

    def identify_libraries(self, task: str) -> list[str]:
        """Library names relevant for user's task. An LLM task."""
        # TODO: generalise to cross-language
        # TODO: use google search engine. Good prompts.
        structured_llm = self.llm_mini.with_structured_output(PythonLibraries)
        return structured_llm.invoke(task).libraries

    def locate_module_files(self, lib: str) -> Path:
        """Identify locations where module scripts are stored."""
        imported = importlib.import_module(lib)
        if imported.__file__:
            return Path(imported.__file__).parent
        msg = f"'{lib}' imported but not found."
        raise ModuleImportedButNotLocatedError(msg)

    def identify_documentation(self, libraries: list[str]) -> dict[str, str]:
        """Find files of software packages useful for the task, including docstrings."""
         # TODO: generalise to cross-language. Package managers for key languages. Browser-based for other languages.
         # UnimplementedError for languages not supported.
        installed = {pkg.metadata["name"] for pkg in importlib.metadata.distributions()}
        missing = set(libraries) - installed
        if missing:
            # install lib
            python = sys.executable
            subprocess.check_call([python, "-m", "pip", "install", *missing], stdout=subprocess.DEVNULL)
        lib_documentation = {}
        for lib in libraries:
            lib_documentation[lib] = self.locate_module_files(lib=lib)
        return lib_documentation

    def indexed_data(self, rag_input: dict[str, str]) -> None:
        """Prepare RAG-ready data from scripts in directories indicated in the input."""
        file_description_dir = join_paths(self.work_dir, ".clean_coder/lib_documentation_descriptions")
        file_extension_constraint = {
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".vue",
            ".py",
            ".rb",
            ".php",
            ".java",
            ".c",
            ".cpp",
            ".cs",
            ".go",
            ".swift",
            ".kt",
            ".rs",
            ".htm",
            ".html",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".prompt",
        }
        ignore = {".clean_coder", ".coderrules"}
        produce_descriptions(
            directories_with_files_to_describe=list(rag_input.values()),
            file_description_dir=file_description_dir,
            work_dir=self.work_dir,
            file_extension_constraint=file_extension_constraint,
            ignore=ignore,
            )
        chroma_collection_name = f"clean_coder_{Path(self.work_dir).name}_lib_documentation_descriptions"
        upload_descriptions_to_vdb(
            chroma_collection_name=chroma_collection_name,
            work_dir=self.work_dir,
            file_description_dir=file_description_dir,
        )


    def rag_documentation(self, task: str) -> None | list[str]:
        """Returns documentation relevant for the task set by human user, a list of files."""
        libraries = self.identify_libraries(task=task)
        rag_input = self.identify_documentation(libraries=libraries)
        return self.indexed_data(rag_input=rag_input)
